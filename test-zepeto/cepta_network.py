# -*- coding: utf-8 -*-
"""
cepta_network.py

구성:
- AutoTokenizer 로더
- 위치 정보: 절대 사인·코사인
- 임베딩 모듈: CEPTA 토큰 임베딩(CeptaEmbedding 사용, use_index=True)
- 트랜스포머 스택: Pre-LN(RMSNorm) + 경로기반 CeptaBlock × N + 최종 RMSNorm + LM Head
- AMP 및 CE 손실 학습 스텝

의존: perceptron_cepta.py (CeptaEmbedding, CeptaRouting, CeptaConfig)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler

from perceptron_cepta import CeptaEmbedding, CeptaRouting, CeptaConfig

# ------------------------------------------------------------
# 토크나이저
# ------------------------------------------------------------

def build_tokenizer(name_or_path: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name_or_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


# ------------------------------------------------------------
# RMSNorm (fp32 연산)
# ------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig = x.dtype
        x32 = x.float()
        rms = x32.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        y = x32 * rms * self.weight.float()
        return y.to(orig)


# ------------------------------------------------------------
# 절대 사인-코사인 위치 부여
# ------------------------------------------------------------
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 262144):
        super().__init__()
        pe = torch.zeros(max_len, dim, dtype=torch.float32)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        return (x.float() + self.pe[:T].unsqueeze(0)).to(x.dtype)


# ------------------------------------------------------------
# MLP (SwiGLU)
# ------------------------------------------------------------
class SwiGLU(nn.Module):
    def __init__(self, dim: int, ff_mult: float = 3.5, dropout: float = 0.0):
        super().__init__()
        hid = int(math.ceil(dim * ff_mult))
        self.w12 = nn.Linear(dim, 2 * hid)
        self.w3 = nn.Linear(hid, dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.w12.weight)
        nn.init.xavier_uniform_(self.w3.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.w12(x).chunk(2, dim=-1)
        y = F.silu(a) * b
        y = self.w3(y)
        return self.dropout(y) if self.training else y


# ------------------------------------------------------------
# 토큰 임베딩: input_ids → (B,T,D)
#   - 내부: CEPTA 임베딩(use_index=True)로 U,F,Y 생성 후 Y(P·α)→D 투영
# ------------------------------------------------------------
class CeptaTokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, P: int, alpha: int, D: int,
                 *, gate: Literal['hard','ste_band','ste_sigmoid'] = 'hard',
                 dtype_store: Literal['bf16','fp16','fp32'] = 'bf16',
                 dropout: float = 0.0):
        super().__init__()
        self.P, self.alpha, self.D = P, alpha, D
        self.embed = CeptaEmbedding(CeptaConfig(P=P, d_or_vocab=vocab_size, alpha=alpha,
                                                use_index=True, gate=gate, dtype_store=dtype_store))
        self.proj = nn.Linear(P * alpha, D)
        nn.init.xavier_uniform_(self.proj.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor):
        U, Fhard, Y = self.embed(input_ids=input_ids)
        x = self.proj(Y.reshape(Y.size(0), Y.size(1), -1))
        x = self.dropout(x) if self.training else x
        return x, (U, Fhard, Y)


# ------------------------------------------------------------
# 경로기반 블록: Pre-LN → CEPTA 경로 → 잔차 → Pre-LN → MLP → 잔차
# ------------------------------------------------------------
class CeptaBlock(nn.Module):
    def __init__(self, D: int, P: int, alpha: int,
                 *, route_mode: Literal['softmax','topk'] = 'softmax', topk: int = 8,
                 dropout: float = 0.0,
                 gate: Literal['hard','ste_band','ste_sigmoid'] = 'hard',
                 dtype_store: Literal['bf16','fp16','fp32'] = 'bf16'):
        super().__init__()
        self.rms1 = RMSNorm(D)
        self.to_P = nn.Linear(D, P)
        self.embed = CeptaEmbedding(CeptaConfig(P=P, d_or_vocab=P, alpha=alpha,
                                                use_index=False, gate=gate, dtype_store=dtype_store))
        self.route = CeptaRouting(P_in=P, P_out=P, mode=route_mode, topk=topk)
        self.from_P = nn.Linear(P, D)
        nn.init.xavier_uniform_(self.to_P.weight)
        nn.init.xavier_uniform_(self.from_P.weight)

        self.rms2 = RMSNorm(D)
        self.mlp = SwiGLU(D, ff_mult=3.5, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.rms1(x)
        U, Fhard, Y = self.embed(X_dense=self.to_P(h1))
        t = (Fhard.float() * U.float())
        routed = self.route(t)
        x = x + self.from_P(routed)
        h2 = self.rms2(x)
        x = x + self.mlp(h2)
        return x


# ------------------------------------------------------------
# 모델
# ------------------------------------------------------------
@dataclass
class ModelCfg:
    vocab_size: int
    d_model: int
    n_layers: int
    P: int
    alpha: int
    route_mode: Literal['softmax','topk'] = 'softmax'
    topk: int = 8
    dropout: float = 0.0
    gate: Literal['hard','ste_band','ste_sigmoid'] = 'hard'
    dtype_store: Literal['bf16','fp16','fp32'] = 'bf16'


class DeepSeekCeptaTransformer(nn.Module):
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = CeptaTokenEmbedding(cfg.vocab_size, cfg.P, cfg.alpha, cfg.d_model,
                                           gate=cfg.gate, dtype_store=cfg.dtype_store, dropout=cfg.dropout)
        self.pos = SinusoidalPositionalEncoding(cfg.d_model)
        self.blocks = nn.ModuleList([
            CeptaBlock(D=cfg.d_model, P=cfg.P, alpha=cfg.alpha,
                       route_mode=cfg.route_mode, topk=cfg.topk, dropout=cfg.dropout,
                       gate=cfg.gate, dtype_store=cfg.dtype_store)
            for _ in range(cfg.n_layers)
        ])
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        nn.init.xavier_uniform_(self.lm_head.weight)

    def forward(self, input_ids: torch.Tensor, *, return_pack: bool = False):
        x, pack0 = self.tok_emb(input_ids)
        x = self.pos(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return (logits, pack0) if return_pack else logits


# ------------------------------------------------------------
# 학습 스텝
# ------------------------------------------------------------
@dataclass
class TrainCfg:
    use_bf16: bool = True
    use_fp16: bool = False
    clip_norm: float = 1.0


def train_step(model: DeepSeekCeptaTransformer,
               input_ids: torch.Tensor,
               targets: torch.Tensor,
               optimizer: torch.optim.Optimizer,
               cfg: TrainCfg = TrainCfg()) -> torch.Tensor:
    device = next(model.parameters()).device
    amp_dtype = torch.bfloat16 if cfg.use_bf16 else (torch.float16 if cfg.use_fp16 else None)
    scaler = GradScaler('cuda', enabled=cfg.use_fp16 and not cfg.use_bf16)
    autocast_enabled = (amp_dtype is not None)

    def _forward():
        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=autocast_enabled):
            logits = model(input_ids.to(device))
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.to(device).reshape(-1))
        return loss

    optimizer.zero_grad(set_to_none=True)
    if scaler.is_enabled():
        loss = _forward()
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_norm)
        scaler.step(optimizer)
        scaler.update()
        return loss.detach()
    else:
        loss = _forward()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_norm)
        optimizer.step()
        return loss.detach()


# ------------------------------------------------------------
# 예시 실행
# ------------------------------------------------------------
if __name__ == "__main__":
    import os
    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1) 토크나이저
    name_or_path = os.environ.get('TOKENIZER_NAME', 'gpt2')
    tok = build_tokenizer(name_or_path)

    # 2) 모델
    cfg = ModelCfg(vocab_size=len(tok), d_model=768, n_layers=6, P=256, alpha=4,
                   route_mode='topk', topk=8, dropout=0.0, gate='hard', dtype_store='bf16')
    model = DeepSeekCeptaTransformer(cfg).to(device)

    # 3) 더미 배치
    texts = ["hello world", "deepseek cepta routing"]
    batch = tok(texts, return_tensors='pt', padding=True, truncation=True)
    input_ids = batch['input_ids'].to(device)
    targets = input_ids.clone()

    # 4) 학습 스텝
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss = train_step(model, input_ids, targets, opt,
                      cfg=TrainCfg(use_bf16=torch.cuda.is_available()))
    print("loss:", float(loss))
