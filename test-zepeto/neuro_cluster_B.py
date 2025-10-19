
# -*- coding: utf-8 -*-
# 파일명: neuro_cluster_B.py
"""
퍼셉트론 클러스터 실시간 시뮬레이터 (GPU 선택적 가속)
- NumPy/CPU 또는 CuPy/GPU 자동 선택
- CUDA 커널: ElementwiseKernel, RawKernel (ASCII 소스)
- 배치 입력과 퍼셉트론 병렬 처리
- (신규) 가중치 하한/상한(--w-min/--w-max) 강제 클리핑
"""
from __future__ import annotations

import sys, time, argparse, logging, logging.handlers
import numpy as np

# --- Backend 선택 ---
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

def _get_xp(use_gpu: bool|None):
    if use_gpu is True and _GPU_OK and cp is not None:
        return cp, True
    return np, False

# 커널들
if _GPU_OK and cp is not None:
    thresh_kernel = cp.ElementwiseKernel(
        'float32 u, float32 t', 'float32 f',
        'f = (u >= t) ? 1.0f : 0.0f;',
        'thresh_kernel'
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
    thresh_kernel = None  # type: ignore
    clip_kernel = None    # type: ignore
    weight_update_kernel = None  # type: ignore

def setup_logger(path: str) -> logging.Logger:
    logger = logging.getLogger("cluster")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    h = logging.handlers.RotatingFileHandler(path, maxBytes=5_000_000, backupCount=3, encoding='utf-8')
    f = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    h.setFormatter(f); logger.addHandler(h)
    return logger

class PerceptronCluster:
    def __init__(self, n_perceptrons: int, n_inputs: int, batch: int, lr: float,
                 threshold: float, seed: int, use_gpu: bool|None,
                 w_min: float, w_max: float):
        self.n_p = n_perceptrons
        self.n_in = n_inputs
        self.batch = batch
        self.lr = lr
        self.thresh = threshold
        self.w_min = w_min
        self.w_max = w_max
        self.xp, self.gpu = _get_xp(use_gpu)
        self.rng = np.random.default_rng(seed)
        W = self.rng.standard_normal((n_perceptrons, n_inputs), dtype=np.float32) * 0.1
        if self.gpu and clip_kernel is not None:
            self.W = clip_kernel(cp.asarray(W), np.float32(w_min), np.float32(w_max))  # type: ignore
        else:
            np.clip(W, w_min, w_max, out=W); self.W = W
        self.F_mean = self.xp.zeros((n_perceptrons,), dtype=self.xp.float32)

    def randn(self, shape, scale=1.0):
        arr = self.rng.standard_normal(shape, dtype=np.float32) * scale
        return cp.asarray(arr) if self.gpu else arr

    def threshold(self, U):
        if self.gpu and thresh_kernel is not None:
            return thresh_kernel(U.astype(self.xp.float32), self.xp.float32(self.thresh))
        return (U >= self.thresh).astype(self.xp.float32)

    def clip_(self):
        if self.gpu and clip_kernel is not None:
            self.W = clip_kernel(self.W, np.float32(self.w_min), np.float32(self.w_max))  # type: ignore
        else:
            np.clip(self.W, self.w_min, self.w_max, out=self.W)

    def weight_update(self, Delta):
        if self.gpu and weight_update_kernel is not None:
            n = int(self.W.size)
            threads = 256
            blocks = (n + threads - 1) // threads
            weight_update_kernel((blocks,), (threads,),
                                 (self.W.ravel(), Delta.ravel(), self.W.ravel(),
                                  np.float32(self.lr), n))
            self.clip_()
        else:
            self.W += self.lr * Delta
            self.clip_()

    def train_step(self, X):
        U = X @ self.W.T
        F = self.threshold(U)  # (B, P)
        self.F_mean = F.mean(axis=0)
        Delta = (F.T @ X) / X.shape[0]
        self.weight_update(Delta)

    def get_W_cpu(self):
        return cp.asnumpy(self.W) if self.gpu else self.W

    def get_F_cpu(self):
        f = self.F_mean
        return cp.asnumpy(f) if self.gpu else f

def setup_figure(n_perceptrons, n_inputs):
    import matplotlib.pyplot as plt
    fig, (ax_w, ax_f) = plt.subplots(1, 2, figsize=(8, 3))
    w_im = ax_w.imshow(np.zeros((n_perceptrons, n_inputs), dtype=np.float32),
                       aspect='auto', interpolation='nearest')
    ax_w.set_title('W'); fig.colorbar(w_im, ax=ax_w, fraction=0.046, pad=0.04)
    f_im = ax_f.imshow(np.zeros((1, n_perceptrons), dtype=np.float32),
                       aspect='auto', interpolation='nearest')
    ax_f.set_title('F mean'); fig.colorbar(f_im, ax=ax_f, fraction=0.046, pad=0.04)
    fig.tight_layout(); plt.ion(); fig.show()
    return fig, w_im, f_im

def update_figure(fig, w_im, f_im, cluster: PerceptronCluster, pmin=1, pmax=99):
    import numpy as np
    W = cluster.get_W_cpu(); F = cluster.get_F_cpu()[None, :]
    vmin, vmax = np.percentile(W, [pmin, pmax]);  vmax = vmin + 1e-3 if vmin == vmax else vmax
    w_im.set_data(W); w_im.set_clim(vmin=vmin, vmax=vmax)
    f_im.set_data(F); f_im.set_clim(vmin=0.0, vmax=1.0)
    fig.canvas.draw_idle(); fig.canvas.flush_events()

def main():
    parser = argparse.ArgumentParser(description="퍼셉트론 클러스터 (GPU 선택적)")
    parser.add_argument('--perceptrons', type=int, default=64)
    parser.add_argument('--inputs', type=int, default=128)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--threshold', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--stim-interval', type=float, default=10.0)
    parser.add_argument('--vis-interval', type=float, default=0.5)
    parser.add_argument('--sleep', type=float, default=0.01)
    parser.add_argument('--w-min', type=float, default=-1.0)
    parser.add_argument('--w-max', type=float, default=1.0)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--log', type=str, default='cluster.log')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.gpu and args.cpu:
        print("GPU와 CPU 동시 지정 불가", file=sys.stderr); sys.exit(2)
    if args.w_min >= args.w_max:
        print("--w-min 은 --w-max 보다 작아야 합니다.", file=sys.stderr); sys.exit(2)

    use_gpu = True if args.gpu else (False if args.cpu else None)
    logger = setup_logger(args.log)
    logger.info("start P=%d D=%d B=%d lr=%.3g wmin=%.3g wmax=%.3g gpu=%s",
                args.perceptrons, args.inputs, args.batch, args.lr, args.w_min, args.w_max, str(use_gpu))

    cluster = PerceptronCluster(args.perceptrons, args.inputs, args.batch,
                                args.lr, args.threshold, args.seed, use_gpu,
                                w_min=args.w_min, w_max=args.w_max)

    try:
        fig, w_im, f_im = setup_figure(args.perceptrons, args.inputs); have_fig = True
    except Exception as e:
        print("시각화 비활성:", e, file=sys.stderr); have_fig = False

    stim = cluster.randn((args.batch, args.inputs), scale=1.0)
    last_stim = time.time(); last_vis = 0.0

    try:
        while True:
            now = time.time()
            if now - last_stim >= args.stim_interval:
                stim = cluster.randn((args.batch, args.inputs), scale=1.0); last_stim = now
            cluster.train_step(stim)
            if have_fig and (now - last_vis >= args.vis_interval):
                update_figure(fig, w_im, f_im, cluster); last_vis = now
            time.sleep(args.sleep)
    except KeyboardInterrupt:
        print("\n종료")

if __name__ == '__main__':
    main()
