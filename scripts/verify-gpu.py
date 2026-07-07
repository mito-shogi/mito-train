"""GPU / MPS / CPU の動作確認と簡易ベンチ。

期待値 (10 回 matmul の合計):
- RTX 4070 Ti: 20-30 ms
- RTX 5060: 30-50 ms
- M3 Max (MPS): 50-100 ms
- CPU: 5,000 ms+
"""
from __future__ import annotations

import time

import torch


def main() -> None:
    if torch.cuda.is_available():
        device = "cuda"
        print(f"CUDA: {torch.cuda.get_device_name(0)}")
        print(f"Capability: {torch.cuda.get_device_capability(0)}")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("MPS (Apple Silicon)")
    else:
        device = "cpu"
        print("CPU only")

    x = torch.randn(4096, 4096, device=device)
    y = torch.randn(4096, 4096, device=device)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        z = x @ y  # noqa: F841
    if device == "cuda":
        torch.cuda.synchronize()
    print(f"{device}: {(time.perf_counter() - t0) * 100:.1f} ms/matmul")


if __name__ == "__main__":
    main()
