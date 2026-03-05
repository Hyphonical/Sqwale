# sqwale — Project Plan

## Overview

sqwale is a Rust-based ONNX super-resolution tool. It consists of a single crate with both a
library (`lib.rs`) and a CLI binary (`main.rs`). The library is designed to be embeddable in other
Rust projects, similar to how Spandrel serves as a backend in Chaiinner.

---

## Crate Structure

```
sqwale/
├── src/
│   ├── lib.rs          — public library API
│   ├── main.rs         — CLI entry point
│   ├── cli.rs          — clap definitions (Commands, Args)
│   ├── commands/
│   │   ├── mod.rs
│   │   ├── inspect.rs  — inspect command implementation
│   │   └── upscale.rs  — upscale command implementation
│   ├── inspect/
│   │   ├── mod.rs      — public ModelInfo, ColorSpace, etc.
│   │   ├── detect.rs   — scale / channel / tiling detection logic
│   │   └── proto.rs    — raw ONNX protobuf walker
│   └── upscale/
│       ├── mod.rs      — public UpscaleSession, UpscaleOptions
│       ├── session.rs  — ort session management
│       ├── tiling.rs   — tile grid calculation, overlap, blending
│       ├── image.rs    — image I/O, normalization, color space conversion
│       └── vram.rs     — VRAM/RAM estimation and tile size calculation
├── Cargo.toml
└── PLAN.md
```

---

## Public Library API (`lib.rs`)

```rust
// Inspection
pub use inspect::{inspect_model, ModelInfo, ColorSpace, ScaleSource, TileInfo};

// Upscaling
pub use upscale::{UpscaleSession, UpscaleOptions, UpscaleError};
```

### `inspect_model(path: &Path) -> Result<ModelInfo>`

Parses the ONNX model using tract-onnx for I/O facts and a raw proto walker for graph topology.
No inference session is created. Returns immediately with all detected metadata.

### `ModelInfo`

```rust
pub struct ModelInfo {
    pub scale: u32,
    pub scale_source: ScaleSource,
    pub color_space: ColorSpace,
    pub input_channels: u32,
    pub output_channels: u32,
    pub tile: TileInfo,
    pub input_dtype: String,
    pub output_dtype: String,
    pub opset: u64,
    pub op_fingerprint: Vec<(String, usize)>,
}
```

### `UpscaleSession`

Holds a loaded ort inference session for a single model. Designed to be reused across many images
without reloading the model.

```rust
impl UpscaleSession {
    pub fn new(model_path: &Path, options: &UpscaleOptions) -> Result<Self>;
    pub fn model_info(&self) -> &ModelInfo;
    pub fn upscale(&self, image: DynamicImage) -> Result<DynamicImage>;
}
```

### `UpscaleOptions`

```rust
pub struct UpscaleOptions {
    pub provider: Provider,
    pub tile_size: Option<u32>,   // None = auto-calculate from VRAM
    pub overlap: u32,             // default: 16
    pub optimize: bool,           // default: true, --no-optimize sets false
}
```

---

## CLI Commands

### `sqwale inspect <PATTERN>`

Accepts a path or glob pattern. Prints a tree for each matched `.onnx` file.

```
sqwale inspect ./models/*.onnx
sqwale inspect ./models/mymodel.onnx
```

### `sqwale upscale <INPUT> --model <MODEL> [OPTIONS]`

```
sqwale upscale image.png --model model.onnx
sqwale upscale ./input/*.png --model model.onnx --output ./output/
sqwale upscale image --model model.onnx               # preserves source format
sqwale upscale image.png --model model.onnx           # exports as PNG
sqwale upscale image.png --model model.onnx -o result # exports as source format

Options:
  -m, --model <PATH>         Path to ONNX model file
  -o, --output <PATH>        Output file (single) or directory (batch)
      --provider <NAME>      Inference provider (see below)
      --tile-size <N>        Override automatic tile size
      --overlap <N>          Tile overlap in pixels [default: 16]
      --no-optimize          Disable ort graph optimization (for broken models)
  -v, --verbose              Verbose output
  -q, --quiet                Suppress output
```

---

## Output Naming

### Single image input

| `--output` value | Result |
|---|---|
| `result.png` | saved as `result.png` |
| `result` (no extension) | saved as `result.{source_ext}` |
| not provided | `{stem}_{scale}x.{source_ext}` in same directory |

### Batch input (glob)

| `--output` value | Result |
|---|---|
| `./output/` (directory) | `./output/{stem}_{scale}x.{source_ext}` |
| not provided | `{stem}_{scale}x.{source_ext}` next to each source file |

---

## Image Formats

All formats supported by the `image` crate are accepted as input and output.
Primary formats: PNG, JPEG, WebP, TIFF, BMP, GIF (static), QOI, TGA, AVIF.
Format is determined by file extension. When exporting without extension, source format is preserved.

---

## Inference Providers

Providers are platform-gated at compile time and runtime.

| Platform | Available Providers |
|---|---|
| Windows | CPU, CUDA, TensorRT |
| Linux | CPU, CUDA, TensorRT, XNNPack |
| macOS | CPU, CoreML |

Default provider is always CPU. If a requested provider is unavailable (no GPU, missing runtime),
sqwale falls back to CPU with a warning rather than hard-failing.

Provider is specified as a string argument: `--provider cuda`, `--provider tensorrt`, etc.

---

## Tiling

### Always tiled

Every image goes through the tiling pipeline regardless of model type. This eliminates a code path
and ensures consistent behavior. The only difference between model types:

- **Dynamic-shape model**: tile size is calculated automatically or overridden by `--tile-size`
- **Static-shape model**: tile size is fixed to the model's required input size (`TileInfo::fixed_size`),
  overlap is forced to 0 (padding-based approach instead)

### Automatic tile size calculation (dynamic models)

Goal: use 80–90% of available GPU VRAM (or system RAM for CPU), accounting for model size and
activations, while keeping tiles evenly sized across the image.

The calculation:

```
available_memory = query_provider_memory() * 0.85
model_memory     = model_file_size * ~3      (weights + activations estimate)
tile_budget      = available_memory - model_memory
bytes_per_pixel  = channels * dtype_bytes * scale^2 * ~4  (input + output + intermediates)
raw_tile_size    = sqrt(tile_budget / bytes_per_pixel)
tile_size        = largest power-of-two ≤ raw_tile_size, min 64, max 2048
```

Then the tile grid is computed so all tiles are the same size:
```
cols = ceil(image_w / (tile_size - 2 * overlap))
rows = ceil(image_h / (tile_size - 2 * overlap))
actual_tile_w = ceil(image_w / cols) + 2 * overlap
actual_tile_h = ceil(image_h / rows) + 2 * overlap
```
Both dimensions are then rounded up to the nearest multiple of `TileInfo::alignment` (if present).

If the entire image fits in a single tile, tiling is skipped entirely.

### Overlap and blending

Each tile is extracted with `overlap` pixels of padding on all sides from neighboring tiles.
After inference, tiles are composited onto the output canvas using a cosine weight map:

```
weight(x) = 0.5 * (1 - cos(π * x / overlap))   for x in [0, overlap]
         = 1.0                                    for x in (overlap, tile_size - overlap)
weight(x, y) = weight(x) * weight(y)
```

Accumulate `tile_pixels * weight` and `weight` separately into float32 canvases, then divide
at the end. This produces seamless output with no visible tile boundaries.

### Edge tiles

Edge tiles are zero-padded to full tile size before inference, then cropped after.
For models requiring alignment, padding brings the tile up to the nearest aligned size, not just
the tile size. Crop removes both the alignment pad and the out-of-bounds overlap.

---

## Processing Pipeline (per image)

```
1.  inspect_model(model_path)
    → ModelInfo (tract, instant, no ort session)

2.  UpscaleSession::new(model_path, options)
    → builds ort session with chosen provider + optimization level
    → if provider unavailable, warn and fall back to CPU

3.  load_image(input_path)
    → decode via `image` crate → DynamicImage
    → convert to f32 NCHW tensor, normalize to [0.0, 1.0]
    → color space handling:
        - grayscale model + RGB input   → convert to luma
        - RGB model + RGBA input        → strip alpha, save for re-attachment
        - model channels != image channels → error

4.  compute_tile_grid(image_size, model_info, options)
    → returns Vec<TileRect> with (src_rect, dst_rect, pad) for each tile

5.  allocate output canvas (f32, same channel count as model output, size = input * scale)
    allocate weight canvas (f32, same size)

6.  for each tile:
        a. extract tile pixels from input (with overlap)
        b. pad to tile_size (and alignment if needed)
        c. run ort inference → output tensor
        d. crop output to valid region (remove pad)
        e. accumulate into output canvas with cosine weights

7.  divide output canvas by weight canvas (normalize blending)

8.  if alpha was stripped in step 3:
        → upscale alpha channel separately (bilinear resize × scale)
        → re-attach

9.  denormalize [0.0, 1.0] → u8/u16 depending on output format
    save via `image` crate to output path
```

---

## Error Handling

- Single image: any error is fatal, reported immediately.
- Batch input: errors are collected per-image. Processing continues. At the end, a summary is
  printed listing all failed files with their error messages. Exit code is non-zero if any failed.
- Provider fallback: not an error, but always logged as a warning even with `--quiet`.
- Model load failure in batch: that model is reported as an error, remaining images are skipped
  (since there's nothing to run them through).

---

## Dependencies

| Crate | Purpose |
|---|---|
| `clap` | CLI argument parsing |
| `anyhow` | Error handling |
| `tract-onnx` | Model inspection (graph parsing, I/O facts) |
| `ort` | ONNX inference runtime |
| `image` | Image I/O and format support |
| `glob` | Glob pattern expansion |
| `colored` | Terminal color output |
| `treelog` | Tree-structured terminal output (inspect command) |
| `tracing` + `tracing-subscriber` | Structured logging |

No build dependencies (prost/protobuf removed when ort replaced raw proto fallback).

---

## Implementation Order

1. Fix `out:0 / unknown` output channel fallback (mirror input when output parse fails) (if still present, may have been fixed)
2. `lib.rs` skeleton — wire up public exports, confirm crate compiles as both lib and bin
3. `upscale/image.rs` — image loading, normalization, color space conversion, saving
4. `upscale/session.rs` — ort session with provider selection and optimization flag
5. `upscale/tiling.rs` — tile grid, overlap extraction, cosine blending
6. `upscale/vram.rs` — memory estimation and automatic tile size
7. `commands/upscale.rs` — CLI command wiring, output path logic, batch error collection
8. End-to-end test: single image, CPU, dynamic model (tests/ has pngs with large sizes)
9. End-to-end test: single image, CPU, static model
10. Provider support (CUDA, TensorRT, CoreML, XNNPack)
