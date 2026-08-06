# ============================================================
# GPU Latency + VRAM Benchmark for mobilenetv3_best.pth
# Paste this as a new cell in final-mobilenet (1).ipynb on Kaggle
# (run it AFTER the model has been trained/loaded, T4 GPU on)
# ============================================================

import time
import csv

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE, "-", torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "no GPU")

# ------------------------------------------------------------
# Rebuild the exact architecture and load the trained checkpoint
# (same as Cell 5/8 in this notebook)
# ------------------------------------------------------------
model = mobilenet_v3_large(weights=None)
in_features = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(in_features, 2)

checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()
print(f"Loaded checkpoint from {MODEL_SAVE_PATH}")

# ------------------------------------------------------------
# Build 100 single-image batches from the existing test_loader
# (falls back to random tensors if test_loader isn't in scope)
# ------------------------------------------------------------
bench_inputs = []
try:
    for images, _ in test_loader:
        for i in range(images.size(0)):
            bench_inputs.append(images[i:i+1].to(DEVICE))
            if len(bench_inputs) >= 100:
                break
        if len(bench_inputs) >= 100:
            break
    print(f"Using {len(bench_inputs)} real test images for the benchmark.")
except NameError:
    print("test_loader not in scope - using random tensors instead.")
    bench_inputs = [torch.randn(1, 3, 224, 224).to(DEVICE) for _ in range(100)]

# ------------------------------------------------------------
# Warmup (5 runs, discarded - excludes CUDA context/kernel
# compilation overhead from the timed results)
# ------------------------------------------------------------
with torch.no_grad():
    for t in bench_inputs[:5]:
        model(t)
    torch.cuda.synchronize()

# ------------------------------------------------------------
# Reset peak-memory tracking AFTER warmup, so it only reflects
# steady-state inference memory, not one-time allocator setup
# ------------------------------------------------------------
torch.cuda.reset_peak_memory_stats(DEVICE)

# ------------------------------------------------------------
# Timed inference loop - single-image batches, matching the
# real /predict serving pattern (one image per request)
# ------------------------------------------------------------
times_ms = []
with torch.no_grad():
    for t in bench_inputs:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(t)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000)

import statistics
mean_ms = statistics.mean(times_ms)
std_ms = statistics.stdev(times_ms)
min_ms = min(times_ms)
max_ms = max(times_ms)
throughput = 1000.0 / mean_ms

peak_vram_mib = torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 2)

print("=" * 60)
print("GPU Latency Benchmark Results")
print("=" * 60)
print(f"N images         : {len(bench_inputs)}")
print(f"Mean latency      : {mean_ms:.3f} ms")
print(f"Std latency       : {std_ms:.3f} ms")
print(f"Min latency       : {min_ms:.3f} ms")
print(f"Max latency       : {max_ms:.3f} ms")
print(f"Throughput        : {throughput:.2f} images/sec")
print(f"Peak VRAM (steady-state inference): {peak_vram_mib:.2f} MiB")

# ------------------------------------------------------------
# Save to CSV - download this from the Kaggle Output tab and
# hand it back for Sections 4.4 / 9.2 / 9.3
# ------------------------------------------------------------
with open("/kaggle/working/gpu_latency_benchmark.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Device", "N_images", "mean_ms", "std_ms", "min_ms", "max_ms", "throughput_img_per_sec", "peak_vram_mib"])
    w.writerow(["GPU (Tesla T4)", len(bench_inputs), f"{mean_ms:.3f}", f"{std_ms:.3f}", f"{min_ms:.3f}", f"{max_ms:.3f}", f"{throughput:.2f}", f"{peak_vram_mib:.2f}"])
print("\nSaved: /kaggle/working/gpu_latency_benchmark.csv")
