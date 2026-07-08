import torch

if torch.cuda.is_available():
    print(f"CUDA: {torch.cuda.get_device_name(0)}")
    print(f"Capability: {torch.cuda.get_device_capability(0)}")
elif torch.backends.mps.is_available():
    print("MPS (Apple Silicon)")
else:
    print("CPU only")

# 簡単な matmul で計測
import time
device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
x = torch.randn(4096, 4096, device=device)
y = torch.randn(4096, 4096, device=device)
if device == 'cuda': torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(10):
    z = x @ y
if device == 'cuda': torch.cuda.synchronize()
print(f"{device}: {(time.perf_counter() - t0) * 100:.1f} ms/matmul")
