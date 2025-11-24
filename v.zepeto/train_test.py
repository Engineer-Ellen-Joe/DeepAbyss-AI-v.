"""
Training and testing script for the CEPTA-based Transformer LM.

Dataset:
    Expects text files located under Z:\\Final_project\\data_set\\After_aling
    named 01.txt ... 07.txt. Files are concatenated and tokenized, then split
    into contiguous blocks for language modeling.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from embedding import CeptaEmbeddingConfig
from module_layer import CeptaTransformerLM
from tokenizer import get_deepseek_v3_tokenizer


def read_corpus(data_dir: Path) -> List[str]:
    """Read all text files sorted by name."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    texts: List[str] = []
    for path in sorted(data_dir.glob("*.txt")):
        texts.append(path.read_text(encoding="utf-8"))
    if not texts:
        raise ValueError(f"No .txt files found in {data_dir}")
    return texts


def tokenize_corpus(tokenizer, texts: List[str]) -> torch.Tensor:
    """Tokenize and concatenate a list of texts into a single 1-D tensor of token IDs."""
    token_ids: List[torch.Tensor] = []
    for txt in texts:
        enc = tokenizer(
            txt,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=False,
            max_length=None,
        )
        token_ids.append(enc["input_ids"].squeeze(0))
    return torch.cat(token_ids, dim=0)


class LMDataset(Dataset):
    """Contiguous block language modeling dataset."""

    def __init__(self, tokens: torch.Tensor, block_size: int):
        if tokens.dim() != 1:
            raise ValueError("tokens must be a 1-D tensor.")
        if block_size <= 1:
            raise ValueError("block_size must be > 1.")
        self.tokens = tokens
        self.block_size = block_size
        self.n_blocks = (len(tokens) - 1) // block_size
        if self.n_blocks <= 0:
            raise ValueError("Not enough tokens to form a single block.")

    def __len__(self) -> int:
        return self.n_blocks

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.block_size
        end = start + self.block_size
        x = self.tokens[start:end]
        y = self.tokens[start + 1 : end + 1]
        return x, y


def split_tokens(tokens: torch.Tensor, train_ratio: float = 0.9) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split token stream into train and test portions."""
    n_train = int(len(tokens) * train_ratio)
    if n_train <= 0 or n_train >= len(tokens):
        raise ValueError("Invalid train split; adjust train_ratio.")
    return tokens[:n_train], tokens[n_train:]


def build_model(vocab_size: int, max_seq_len: int, dtype_store: str = "bf16") -> CeptaTransformerLM:
    emb_cfg = CeptaEmbeddingConfig(
        vocab_size=vocab_size,
        P=512,
        alpha=4,
        P_r=64,
        d_model=1024,
        max_seq_len=max_seq_len,
        dtype_store=dtype_store,
    )
    model = CeptaTransformerLM(emb_cfg, n_layers=6)
    return model


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_tokens = 0
    for input_ids, labels in dataloader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * labels.numel()
        total_tokens += labels.numel()
    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for input_ids, labels in dataloader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        logits = model(input_ids)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100
        )
        total_loss += loss.item() * labels.numel()
        total_tokens += labels.numel()
    return total_loss / max(total_tokens, 1)


def main():
    parser = argparse.ArgumentParser(description="Train/Test CEPTA Transformer on After_aling dataset.")
    parser.add_argument("--data_dir", type=Path, default=Path("Z:/Final_project/data_set/After_aling"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dtype_store", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--save_path", type=Path, default=Path("cepta_lm.pt"))
    args = parser.parse_args()

    tokenizer = get_deepseek_v3_tokenizer()
    # Avoid tokenizer max_length warnings; we will chunk into blocks ourselves.
    tokenizer.model_max_length = int(1e9)
    texts = read_corpus(args.data_dir)
    tokens_all = tokenize_corpus(tokenizer, texts)
    train_tokens, test_tokens = split_tokens(tokens_all, train_ratio=args.train_ratio)

    train_ds = LMDataset(train_tokens, block_size=args.block_size)
    test_ds = LMDataset(test_tokens, block_size=args.block_size)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, drop_last=True)

    model = build_model(tokenizer.vocab_size, max_seq_len=args.block_size, dtype_store=args.dtype_store)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        test_loss = evaluate(model, test_loader, device)
        print(f"Epoch {epoch}: train loss {train_loss:.4f}, test loss {test_loss:.4f}")

    torch.save(model.state_dict(), args.save_path)
    print(f"Saved trained model to {args.save_path}")


if __name__ == "__main__":
    main()
