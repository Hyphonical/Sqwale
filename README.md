# Sqwale

**Fast, tiled ONNX super-resolution from the command line.**

Sqwale loads any ONNX upscaling model, splits large images into tiles, runs
inference (GPU or CPU), blends the results, and writes the output — all in a
single command.

---

## Features

| Feature | Details |
|---------|---------|
| **Model inspection** | Read scale, colour space, precision, tiling requirements and op-type histogram without running inference. |
| **Tiled inference** | Automatically splits images into overlapping tiles with cosine-window blending — handles images of any size. |
| **FP16 support** | Transparent fp32 ↔ fp16 conversion for half-precision models. |
| **Batch upscaling** | Glob patterns (`*.jpg`) with dual progress bars (per-tile + per-image). |
| **Multi-backend** | CPU, CUDA, TensorRT, CoreML, XNNPACK — with automatic fallback. |
| **Ctrl+C safe** | Graceful interruption — finishes the current image, then stops. |

## Installation

```bash
cargo install --path .
```

> **Note:** The ONNX Runtime shared library is downloaded automatically during
> the build by the [`ort`](https://crates.io/crates/ort) crate.

## Quick Start

### Inspect a model

```bash
sqwale inspect model.onnx
```

Example output:

```
── inspect ──────────────────────────────────────────────
● model.onnx
 ├─ Scale      2x  via DepthToSpace (PixelShuffle)
 ├─ Color      RGB  in:3 → out:3
 ├─ Precision  float16 → float16
 ├─ Opset      17
 ├─ Tiling     supported  dynamic spatial dims
 │   └─ Alignment  divisible by 16
 ╰─ Ops        866 total nodes
     ├─ 180  Constant
     ├─ 101  Unsqueeze
     ╰─       … 8 more op types
```

### Upscale a single image

```bash
sqwale upscale photo.jpg -m model.onnx
```

### Batch upscale

```bash
sqwale upscale "photos/*.jpg" -m model.onnx -o upscaled/
```

### Choose a provider

```bash
sqwale upscale photo.jpg -m model.onnx -p cuda
```

Valid providers: `auto` (default), `cpu`, `cuda`, `tensorrt`, `coreml`,
`xnnpack`.

## Library Usage

Sqwale can also be used as a Rust library:

```rust
use sqwale::{UpscaleSession, UpscaleOptions, Provider, inspect_model};
use std::path::Path;

// Inspect
let info = inspect_model(Path::new("model.onnx"))?;
println!("{}x upscale, {} input", info.scale, info.input_dtype);

// Upscale
let opts = UpscaleOptions { provider: Provider::Cpu };
let mut session = UpscaleSession::new(Path::new("model.onnx"), &opts)?;
let img = image::open("input.jpg")?;
let result = session.upscale(img, |done, total| {
    println!("tile {done}/{total}");
})?;
result.save("output.png")?;
```

## License

[MIT](LICENSE)
