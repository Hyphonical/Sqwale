# Execution Providers ⚡

**Let Sqwale use your GPU (or CPU).**

Sqwale runs inference via [ONNX Runtime](https://onnxruntime.ai/) and supports multiple execution providers—backends that handle the actual compute. This document explains what they are, which one you're using, and what to install to unlock the faster options.

---

## What's an execution provider?

An execution provider is ONNX Runtime's abstraction for "where the math happens." On Windows, it could be TensorRT or DirectML. On Mac, it's CoreML. On Linux, it could be CUDA or XNNPACK. Sqwale picks the fastest one automatically, but you can override it with `--provider`.

---

## Automatic selection

By default, Sqwale uses `--provider auto`. ONNX Runtime probes your hardware at session creation time and picks the fastest available provider. The one that was actually used is printed in the model info:

```
·  Loaded  4xLSDIRCompactv2.onnx  via CUDA
```

This happens once per run and is fast (< 1 second typically).

---

## Platform priority matrix

Sqwale auto-selects in this order (highest to lowest priority):

| Platform | Selection order |
|---|---|
| **Windows** | TensorRT → CUDA → DirectML → CPU |
| **Linux** | TensorRT → CUDA → XNNPACK → CPU |
| **macOS** | CoreML → CPU |

| Provider | Windows | Linux | macOS | Notes |
|---|---|---|---|---|
| **TensorRT** | ✓ | ✓ | — | NVIDIA GPUs only; requires CUDA toolkit |
| **CUDA** | ✓ | ✓ | — | NVIDIA GPUs; widely compatible |
| **DirectML** | ✓ | — | — | Any DirectX 12 GPU (AMD, NVIDIA, Intel) |
| **CoreML** | — | — | ✓ | Apple Silicon (M1/M2/M3) and some Intel Macs |
| **XNNPACK** | — | ✓ | — | Fast CPU mode for x86/ARM (Linux only) |
| **CPU** | ✓ | ✓ | ✓ | Fallback; always available |

---

## Manual provider override

Force a specific provider with `--provider <type>`:

```bash
# Use CUDA
sqwale upscale photo.jpg --provider cuda

# Force CPU mode
sqwale interpolate video.mp4 --provider cpu

# Try TensorRT
sqwale interpolate video.mp4 --provider tensorrt
```

Accepted values: `auto`, `cpu`, `cuda`, `tensorrt`, `directml`, `coreml`, `xnnpack` (also `trt` and `dml` as aliases).

---

## Provider setup & requirements

### CUDA & TensorRT

**For NVIDIA GPUs.** TensorRT is slightly faster than CUDA but both are very quick.

**Requirements:**
- NVIDIA GPU (GTX series or newer)
- CUDA toolkit (11.8 or later)
- cuDNN (optional, but recommended)

ONNX Runtime bundles CUDA and TensorRT libraries; you don't need to install them separately if using prebuilt ORT packages. Verify your GPU is available:

```bash
nvidia-smi
```

**First-run performance:** TensorRT JIT-compiles kernels on first run, which can take several minutes per model. Compiled kernels are cached—subsequent runs are fast.

> [!TIP]
> If you have an NVIDIA GPU and want maximum speed, use `--provider tensorrt`. On first run with a new model, grab a coffee; on subsequent runs, it'll be blazing fast.

### DirectML (Windows only)

**For any GPU on Windows 10/11.** Works with AMD, NVIDIA, Intel GPUs.

No installation needed; DirectML ships with Windows. Sqwale just uses it automatically. Slightly slower than CUDA but still very fast.

### CoreML (macOS only)

**For Apple Silicon (M1/M2/M3).** Also works on some Intel Macs with a Neural Engine.

No setup required. CoreML activates automatically on compatible hardware. Performance is excellent on Apple Silicon—comparable to CUDA on NVIDIA.

### XNNPACK (Linux only)

**Fast CPU mode.** Provides optimized SIMD kernels for x86-64 and ARM64.

ONNX Runtime bundles XNNPACK. Significantly faster than the vanilla CPU provider for many model architectures. Sqwale selects it automatically on Linux when no GPU is available.

---

## Fallback behavior

If the requested provider is unavailable or fails to initialize, Sqwale logs a warning and falls back to CPU:

```
WARN sqwale::session: CUDA provider failed; falling back to CPU: cuda not found
```

To prevent automatic fallback (useful in automated pipelines), pass an explicit `--provider` rather than `auto`. Sqwale will error instead of silently switching to CPU:

```bash
sqwale upscale photo.jpg --provider cuda  # Will error if CUDA is unavailable
sqwale upscale photo.jpg --provider auto  # Will silently fall back to CPU
```

---

## Debugging provider selection

Set `RUST_LOG=sqwale=debug` to see detailed provider initialization messages:

```bash
RUST_LOG=sqwale=debug sqwale upscale photo.jpg
```

Output includes:
- Which providers ORT detected as available
- Which provider was selected
- Session creation timing (usually 50ms–2s depending on provider)

**Example debug output:**
```
DEBUG sqwale::session: Available providers: ["TensorRT", "CUDA", "CPU"]
DEBUG sqwale::session: Attempting TensorRT... OK (342ms)
```

---

## Performance expectations

| Provider | Speed | Use case |
|---|---|---|
| **TensorRT** | 50–100ms per image | RTX GPUs; best option for serious workloads |
| **CUDA** | 80–150ms per image | Any NVIDIA GPU; reliable and fast |
| **DirectML** | 150–300ms per image | Windows with any GPU; good all-rounder |
| **CoreML** | 80–150ms per image | Apple Silicon; excellent battery life on laptops |
| **XNNPACK** | 200–400ms per image | Linux CPU; surprisingly good for pure compute |
| **CPU** | 500–2000ms per image | Fallback; will work but slow |

> [!TIP]
> More VRAM = faster inference. Reduce `--tile-size` if you're running out of memory, or upgrade to a larger GPU for better performance.

---

## Troubleshooting

**"Provider failed, falling back to CPU"** — The requested provider isn't installed or is incompatible. Run with `RUST_LOG=sqwale=debug` to see why.

**"CUDA provider failed"** — NVIDIA GPU drivers are outdated or CUDA isn't installed. Update NVIDIA drivers or install CUDA 11.8+.

**"TensorRT provider failed"** — TensorRT isn't installed. Option 1: Use `--provider cuda` instead. Option 2: Install TensorRT.

**"Session creation timed out"** — TensorRT is JIT-compiling for the first time. This is normal and takes a few minutes. Subsequent runs are much faster.

**Inference is slow even with GPU** — Could be several things:
- GPU is thermal-throttling (check with `nvidia-smi`)
- Host-to-GPU bandwidth is saturated (reduce batch size or `--tile-size`)
- Provider selection wasn't optimal (check with `RUST_LOG=sqwale=debug`)
