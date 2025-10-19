
# -*- coding: utf-8 -*-
# 파일명: multi_layer_network_B.py
"""
멀티 레이어 신경망 실시간 시뮬레이터 (GPU 선택적 가속)
- 배치 처리, 퍼셉트론 병렬
- 가중치 클리핑 [-w_max, w_max]
- 중심화 Hebbian + 임계값 홈오스테이시스 + 바이폴라 전달
- (신규) 1층 퍼셉트론에 매 스텝 서로 다른 강도의 외부 구동 신호 주입
  * --drive-k: 매 스텝 활성화할 퍼셉트론 수
  * --drive-min/--drive-max: 강도 범위(>=0)
  * --drive-mode: cycle|random, cycle은 퍼셉트론을 순환하며 교대
"""
from __future__ import annotations
import sys, time, argparse, logging, logging.handlers
from dataclasses import dataclass
from typing import Optional, Tuple, List

# --- Backend 선택: CuPy 있으면 GPU, 없으면 CPU(Numpy) ---
try:
    import cupy as cp  # type: ignore
    try:
        _GPU_COUNT = cp.cuda.runtime.getDeviceCount()
        _GPU_OK = _GPU_COUNT > 0
    except Exception:
        _GPU_OK = False
except Exception:
    cp = None  # type: ignore
    _GPU_OK = False

import numpy as np

def _get_xp(use_gpu: Optional[bool]):
    if use_gpu is True and _GPU_OK and cp is not None:
        return cp, True
    return np, False

# --- CUDA 커널 (ASCII) ---
if _GPU_OK and cp is not None:
    threshold_kernel = cp.ElementwiseKernel(
        'float32 u, float32 t', 'float32 f',
        'f = (u >= t) ? 1.0f : 0.0f;',
        'threshold_kernel'
    )
    clip_kernel = cp.ElementwiseKernel(
        'float32 x, float32 wmin, float32 wmax', 'float32 y',
        'y = fminf(fmaxf(x, wmin), wmax);',
        'clip_kernel'
    )
    weight_update_src = r'''
    extern "C" __global__
    void weight_update(const float* __restrict__ w,
                       const float* __restrict__ delta,
                             float* __restrict__ w_out,
                       const float  lr,
                       const int    n) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i < n) {
            w_out[i] = w[i] + lr * delta[i];
        }
    }''';
    weight_update_kernel = cp.RawKernel(weight_update_src, 'weight_update')
else:
    threshold_kernel = None  # type: ignore
    clip_kernel = None       # type: ignore
    weight_update_kernel = None  # type: ignore

# --- 로깅 설정 (회전 파일) ---
def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("sim")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=5_000_000, backupCount=3, encoding='utf-8'
    )
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

# --- 데이터 클래스 ---
@dataclass
class LayerConfig:
    n_in: int
    n_out: int
    threshold: float = 0.0

@dataclass
class SimConfig:
    seed: int = 18267
    batch_size: int = 64
    lr: float = 0.01
    stim_interval: float = 10.0
    vis_interval: float = 0.5
    sleep_interval: float = 0.01
    w_scale: float = 0.1
    w_min: float = -1.0
    w_max: float = 1.0
    percentile_vmin: float = 1.0
    percentile_vmax: float = 99.0
    use_gpu: Optional[bool] = None
    out_mode: str = "bipolar"    # 'binary' or 'bipolar'
    thresh_lr: float = 0.05
    target_rate: float = 0.5
    # 신호 주입
    drive_k: int = 16
    drive_min: float = 0.1
    drive_max: float = 1.0
    drive_mode: str = "cycle"    # 'cycle' or 'random'

class MultiLayerNetwork:
    def __init__(self, layers: Tuple[LayerConfig, ...], cfg: SimConfig):
        self.cfg = cfg
        self.xp, self.gpu = _get_xp(cfg.use_gpu)
        self.rng = np.random.default_rng(cfg.seed)
        self.layers = layers
        self.W: List = []
        for lc in layers:
            w = self.rng.standard_normal((lc.n_out, lc.n_in), dtype=np.float32) * cfg.w_scale
            if self.gpu:
                w = cp.asarray(w)  # type: ignore
                w = clip_kernel(w, np.float32(cfg.w_min), np.float32(cfg.w_max))  # type: ignore
            else:
                np.clip(w, cfg.w_min, cfg.w_max, out=w)
            self.W.append(w)
        self.last_F = [self.zeros((lc.n_out,)) for lc in layers]
        # 레이어별 적응 임계값
        self.thresh = [float(lc.threshold) for lc in layers]
        # 1층 구동 신호 상태
        n0 = layers[0].n_out
        self.drive = self.zeros((n0,))              # shape (n_out,)
        self._perm = np.arange(n0, dtype=np.int64)  # CPU에서 유지
        self._perm_ptr = 0

    # 편의 래퍼
    def zeros(self, shape):
        return (self.xp.zeros(shape, dtype=self.xp.float32))

    def randn(self, shape, scale=1.0):
        arr = self.rng.standard_normal(shape, dtype=np.float32) * scale
        return cp.asarray(arr) if self.gpu else arr  # type: ignore

    def randu(self, low, high, size):
        arr = self.rng.uniform(low, high, size=size).astype(np.float32)
        return cp.asarray(arr) if self.gpu else arr  # type: ignore

    def threshold(self, U, t):
        if self.gpu and threshold_kernel is not None:
            return threshold_kernel(U.astype(self.xp.float32), self.xp.float32(t))
        return (U >= t).astype(self.xp.float32)

    def clip_(self, W):
        if self.gpu and clip_kernel is not None:
            return clip_kernel(W, np.float32(self.cfg.w_min), np.float32(self.cfg.w_max))
        np.clip(W, self.cfg.w_min, self.cfg.w_max, out=W)
        return W

    def weight_update(self, W, Delta, lr):
        if self.gpu and weight_update_kernel is not None:
            n = int(W.size)
            threads = 256
            blocks = (n + threads - 1) // threads
            weight_update_kernel((blocks,), (threads,),
                                 (W.ravel(), Delta.ravel(), W.ravel(), np.float32(lr), n))
            self.clip_(W)
            return W
        W += lr * Delta
        self.clip_(W)
        return W

    # ---- 1층 구동 신호 갱신 ----
    def _update_drive(self):
        n0 = self.drive.shape[0]
        k = max(1, min(self.cfg.drive_k, n0))
        # 초기화
        if self.gpu:
            self.drive.fill(0)  # type: ignore
        else:
            self.drive[...] = 0.0
        # 대상 인덱스 선택
        if self.cfg.drive_mode == "random":
            idx = self.rng.choice(n0, size=k, replace=False)
        else:  # cycle
            if self._perm_ptr == 0:
                self.rng.shuffle(self._perm)
            start = self._perm_ptr
            end = start + k
            if end <= n0:
                idx = self._perm[start:end]
            else:
                idx = np.concatenate([self._perm[start:], self._perm[:end - n0]])
            self._perm_ptr = (end) % n0
        # 강도 생성
        amp = self.randu(self.cfg.drive_min, self.cfg.drive_max, size=(k,))
        # 할당
        if self.gpu:
            self.drive[idx] = amp  # type: ignore
        else:
            self.drive[idx] = amp

    def forward(self, X_batch):
        X_list = []
        F_list = []
        X = X_batch
        for li, (lc, W) in enumerate(zip(self.layers, self.W)):
            X_list.append(X)
            U = X @ W.T
            if li == 0:
                # 배치 모든 샘플에 동일한 퍼셉트론별 외부 구동 추가
                U = U + self.drive[None, :]
            F = self.threshold(U, self.thresh[li])
            F_list.append(F)
            # 다음 레이어 입력 선택
            if self.cfg.out_mode == 'bipolar':
                X = (2.0 * F - 1.0).astype(self.xp.float32)
            else:
                X = F
            self.last_F[li] = F.mean(axis=0)
        return X, X_list, F_list

    def train_step(self, X_batch):
        # 스텝마다 1층 구동 신호를 새로운 퍼셉트론 집합에 서로 다른 강도로 주입
        self._update_drive()
        _, X_list, F_list = self.forward(X_batch)
        for li, (W, X_in, F) in enumerate(zip(self.W, X_list, F_list)):
            # 중심화 Hebbian
            Xc = X_in - X_in.mean(axis=0, keepdims=True)
            Fc = F - F.mean(axis=0, keepdims=True)
            Delta = (Fc.T @ Xc) / X_in.shape[0]
            self.W[li] = self.weight_update(W, Delta, self.cfg.lr)
            # 발화율 홈오스테이시스
            m = float(F.mean())
            self.thresh[li] += self.cfg.thresh_lr * (m - self.cfg.target_rate)

    def get_weight_image(self, li: int):
        W = self.W[li]
        if self.gpu:
            W = cp.asnumpy(W)  # type: ignore
        return W

    def get_fire_vector(self, li: int):
        F = self.last_F[li]
        if self.gpu and hasattr(F, 'get'):
            F = cp.asnumpy(F)  # type: ignore
        return F

# ---- 시각화 ----
def setup_figure(layers: Tuple[LayerConfig, ...]):
    import numpy as _np
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(layers), 2, figsize=(8, 3*len(layers)))
    if len(layers) == 1:
        axes = _np.array([axes])
    w_images, f_images = [], []
    for li, lc in enumerate(layers):
        ax_w, ax_f = axes[li, 0], axes[li, 1]
        w_im = ax_w.imshow(_np.zeros((lc.n_out, lc.n_in), dtype=_np.float32),
                           aspect='auto', interpolation='nearest')
        ax_w.set_title(f'W L{li}')
        fig.colorbar(w_im, ax=ax_w, fraction=0.046, pad=0.04)
        f_im = ax_f.imshow(_np.zeros((lc.n_out,), dtype=_np.float32)[None, :],
                           aspect='auto', interpolation='nearest')
        ax_f.set_title(f'F mean L{li}')
        fig.colorbar(f_im, ax=ax_f, fraction=0.046, pad=0.04)
        w_images.append(w_im); f_images.append(f_im)
    fig.tight_layout(); plt.ion(); fig.show()
    return fig, w_images, f_images

def update_figure(fig, net: MultiLayerNetwork, w_images, f_images, cfg: SimConfig):
    import numpy as _np
    for li, (w_im, f_im) in enumerate(zip(w_images, f_images)):
        W = net.get_weight_image(li)
        F = net.get_fire_vector(li)[None, :]
        vmin, vmax = _np.percentile(W, [cfg.percentile_vmin, cfg.percentile_vmax])
        if vmin == vmax: vmax = vmin + 1e-3
        w_im.set_data(W); w_im.set_clim(vmin=vmin, vmax=vmax)
        f_im.set_data(F); f_im.set_clim(vmin=0.0, vmax=1.0)
    fig.canvas.draw_idle(); fig.canvas.flush_events()

# ---- 메인 ----
def main():
    parser = argparse.ArgumentParser(description="멀티 레이어 네트워크 시뮬레이터 (GPU 선택적)")
    parser.add_argument('--layers', type=str, default='auto',
                        help='"n_in:n_out[,n_in:n_out,...]" 또는 auto')
    parser.add_argument('--inputs', type=int, default=128, help='auto일 때 입력 차원')
    parser.add_argument('--width', type=int, default=128, help='auto일 때 각 레이어 폭')
    parser.add_argument('--depth', type=int, default=6, help='auto일 때 총 레이어 수')
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--threshold', type=float, default=0.0)
    parser.add_argument('--stim-interval', type=float, default=10.0)
    parser.add_argument('--vis-interval', type=float, default=0.5)
    parser.add_argument('--sleep', type=float, default=0.01)
    parser.add_argument('--w-scale', type=float, default=0.1)
    parser.add_argument('--w-min', type=float, default=-1.0)
    parser.add_argument('--w-max', type=float, default=1.0)
    parser.add_argument('--out', dest='out_mode', choices=['binary','bipolar'], default='bipolar')
    parser.add_argument('--thresh-lr', type=float, default=0.05)
    parser.add_argument('--target-rate', type=float, default=0.5)
    # 신호 주입 옵션
    parser.add_argument('--drive-k', type=int, default=16)
    parser.add_argument('--drive-min', type=float, default=0.1)
    parser.add_argument('--drive-max', type=float, default=1.0)
    parser.add_argument('--drive-mode', choices=['cycle','random'], default='cycle')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--log', type=str, default='firing_log.txt')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.gpu and args.cpu:
        print("GPU와 CPU 동시 지정은 허용되지 않습니다.", file=sys.stderr); sys.exit(2)
    if args.w_min >= args.w_max:
        print("--w-min 은 --w-max 보다 작아야 합니다.", file=sys.stderr); sys.exit(2)
    if args.drive_min < 0 or args.drive_max < 0 or args.drive_min > args.drive_max:
        print("drive 강도 범위가 잘못되었습니다.", file=sys.stderr); sys.exit(2)

    use_gpu = True if args.gpu else (False if args.cpu else None)
    # 레이어 구성
    if args.layers != 'auto':
        layer_specs = []
        for token in args.layers.split(','):
            n_in, n_out = token.split(':')
            layer_specs.append(LayerConfig(int(n_in), int(n_out), threshold=args.threshold))
    else:
        layer_specs = []
        n_in = args.inputs
        for _ in range(args.depth):
            layer_specs.append(LayerConfig(n_in=n_in, n_out=args.width, threshold=args.threshold))
            n_in = args.width

    cfg = SimConfig(seed=args.seed, batch_size=args.batch, lr=args.lr,
                    stim_interval=args.stim_interval, vis_interval=args.vis_interval,
                    sleep_interval=args.sleep, w_scale=args.w_scale,
                    w_min=args.w_min, w_max=args.w_max, use_gpu=use_gpu,
                    out_mode=args.out_mode, thresh_lr=args.thresh_lr, target_rate=args.target_rate,
                    drive_k=args.drive_k, drive_min=args.drive_min, drive_max=args.drive_max, drive_mode=args.drive_mode)

    logger = setup_logger(args.log)
    logger.info("start | depth=%d width=%d out=%s target=%.2f tlr=%.3g drive=(k=%d,%.2f~%.2f,%s)",
                args.depth, args.width, args.out_mode, args.target_rate, args.thresh_lr,
                args.drive_k, args.drive_min, args.drive_max, args.drive_mode)

    net = MultiLayerNetwork(tuple(layer_specs), cfg)

    # 시각화
    try:
        fig, w_images, f_images = setup_figure(tuple(layer_specs)); have_fig = True
    except Exception as e:
        print("시각화 비활성:", e, file=sys.stderr); have_fig = False

    stim = net.randn((cfg.batch_size, layer_specs[0].n_in), scale=1.0)
    last_stim = time.time(); last_vis = 0.0

    try:
        while True:
            now = time.time()
            if now - last_stim >= cfg.stim_interval:
                stim = net.randn((cfg.batch_size, layer_specs[0].n_in), scale=1.0); last_stim = now
            net.train_step(stim)
            if have_fig and (now - last_vis >= cfg.vis_interval):
                update_figure(fig, net, w_images, f_images, cfg); last_vis = now
            time.sleep(cfg.sleep_interval)
    except KeyboardInterrupt:
        print("\n중단")

if __name__ == '__main__':
    main()
