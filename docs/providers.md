# Execution Providers

Sqwale runs inference via [ONNX Runtime](https://onnxruntime.ai/) and supports multiple execution providers (EPs). This document explains how provider selection works, what each provider requires, and how to troubleshoot fallback behaviour.

---

## Automatic selection

When `--provider auto` is used (the default), ORT's `AutoDevicePolicy::MaxPerformance` is applied. ORT probes the available hardware at session creation time and selects the fastest applicable provider. The provider that was actually used is shown in the model info line:

```
·  Loaded  4xLSDIRCompactv2.onnx  via DirectML
```

---

## Platform provider matrix

| Provider | Windows | Linux | macOS | Notes |
|---|---|---|---|---|
| **TensorRT** | ✓ | ✓ | — | Requires CUDA toolkit + TensorRT libraries |
| **CUDA** | ✓ | ✓ | — | Requires CUDA toolkit |
| **DirectML** | ✓ | — | — | Works on any DirectX 12 GPU (AMD, Intel, NVIDIA) |
| **CoreML** | — | — | ✓ | Apple Silicon and Intel Mac with Neural Engine |
| **XNNPACK** | — | ✓ | — | Optimised CPU provider for x86/ARM |
| **CPU** | ✓ | ✓ | ✓ | Always available, no install required |

Auto-selection priority (highest to lowest) by platform:

| Platform | Order |
|---|---|
| Windows | TensorRT → CUDA → DirectML → CPU |
| Linux | TensorRT → CUDA → XNNPACK → CPU |
| macOS | CoreML → CPU |

---

## Manual override

Pass `--provider` to force a specific EP:

```bash
sqwale upscale input.png --provider cuda
sqwale interpolate input.mp4 --provider tensorrt
sqwale upscale input.png --provider cpu
```

Accepted values: `auto`, `cpu`, `cuda`, `tensorrt`, `directml`, `coreml`, `xnnpack` (also `trt` and `dml` as aliases).

---

## Provider setup

### CUDA / TensorRT

Requires a CUDA-capable NVIDIA GPU and the CUDA toolkit. ONNX Runtime bundles CUDA and TensorRT dynamic libraries; you do not need to install them separately if you use the prebuilt ORT packages. Verify GPU availability with:

```bash
nvidia-smi
```

TensorRT performs JIT kernel compilation on first run, which can take several minutes. Compiled kernels are cached and subsequent runs are fast.

### DirectML (Windows only)

DirectML is available on any GPU that supports Direct3D 12, including AMD, Intel, and NVIDIA. No additional install is needed on Windows 10 (version 1903+) or Windows 11.

### CoreML (macOS only)

CoreML is activated automatically on Apple Silicon (M1 and later). On Intel Macs it uses the Neural Engine if present, otherwise falls back to the GPU. No setup required.

### XNNPACK (Linux)

XNNPACK provides optimised SIMD CPU kernels for x86-64 and ARM64. It is faster than the default CPU provider for many model architectures and is selected automatically on Linux when no GPU EP is available.

---

## Fallback behaviour

If the requested provider is unavailable or fails to initialise, Sqwale logs a warning and falls back to the CPU provider:

```
WARN sqwale::session: CUDA provider failed, falling back to CPU: ...
```

To suppress the fallback and treat provider failure as an error (useful in automated pipelines), pass an explicit `--provider` value rather than `auto`.

---

## Debugging provider selection

Set `RUST_LOG=sqwale=debug` to see detailed provider initialisation messages:

```bash
RUST_LOG=sqwale=debug sqwale upscale input.png
```

This prints which provider ORT selected, any fallback decisions, and session creation timing.
