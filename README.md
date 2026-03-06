# Sqwale

A fast, cross-platform CLI for upscaling images with ONNX super-resolution models. Automatic tiling, GPU acceleration, and clean terminal output.

## Installation

Requires Rust 1.85 or later:

```bash
git clone https://github.com/Hyphonical/Sqwale.git
cd Sqwale
cargo build --release
```

The binary will be at `target/release/sqwale` (or `sqwale.exe` on Windows).

GPU acceleration requires appropriate drivers and runtimes. Sqwale automatically falls back to CPU if GPU providers are unavailable.

## Usage

### Inspect Models

Analyze ONNX model properties without running inference:

```bash
# Single model
sqwale inspect model.onnx

# Multiple models with glob
sqwale inspect "models/*.onnx"

# Entire directory
sqwale inspect models/
```

```
● 2x-AnimeSharpV2_MoSR_Sharp_fp16.onnx
 ├─ Scale      2x  via DepthToSpace (PixelShuffle)
 ├─ Color      RGB  in:3 → out:3
 ├─ Precision  float16 → float16
 ├─ Opset      17
 ├─ Tiling     supported  dynamic spatial dims
 │   └─ Alignment  divisible by 16
 ╰─ Ops        866  total nodes
      ├─ 180  Constant
      ├─ 101  Unsqueeze
      ├─ 100  Mul
      ├─  80  Conv
      ├─  74  Add
      ├─  52  Softplus
      ├─  52  Tanh
      ├─  49  ReduceMean
      ╰─       … 10 more op types
```

### Upscale Images

```bash
# Single image (output: input_2x.png)
sqwale upscale input.png -m model.onnx

# Explicit output path
sqwale upscale input.png -m model.onnx -o output.png

# Batch processing
sqwale upscale "photos/*.jpg" -m model.onnx -o upscaled/

# Custom tiling
sqwale upscale input.png -m model.onnx --tile-size 768 --tile-overlap 32
```

```
● input.jpg
·  Model 2x · RGB · float32 · dynamic  model.onnx via auto
·  Input 4000×6000
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  117/117 tiles  27s  0.2s/tile
·  Output 8000×12000
✓ input_2x.jpg  27s
```

Batch mode shows dual progress bars — per-image tile progress on top, overall batch progress with ETA on the bottom:

```
● [1/5] photos/img1.jpg
  · Input 5776×3856
  ━━━━━━━━━━━━━━━━━━━━╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌  12/24 tiles  1m 02s  3.9s/tile
  ━━━━━━━━╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌   1/5  images  1m 02s  ~4m 08s remaining
```

### Options

| Option | Description | Default |
|---|---|---|
| `--provider <name>` | Execution provider: `auto`, `cpu`, `cuda`, `tensorrt`, `directml`, `coreml`, `xnnpack` | `auto` |
| `--tile-size <px>` | Tile size in pixels (`0` = no tiling) | `512` |
| `--tile-overlap <px>` | Overlap between adjacent tiles | `16` |

### Provider Selection

Sqwale picks the best available provider automatically:

| Platform | Priority |
|---|---|
| Windows | TensorRT → CUDA → DirectML → CPU |
| Linux | TensorRT → CUDA → XNNPACK → CPU |
| macOS | CoreML → CPU |

Override with `--provider cuda`. Falls back to CPU on failure.

### Tiling

Large images are split into overlapping tiles, each processed independently and blended together with cosine-weighted (Hann) windows for seamless output.

- **Dynamic models** use `--tile-size` (default 512px), respecting model alignment requirements.
- **Fixed-size models** automatically use the model's required tile dimensions.
- **Disable tiling** with `--tile-size 0` to process the entire image at once.

### Ctrl+C Handling

- First press: finishes the current image and skips the rest of the batch.
- Second press: exits immediately.

## Supported Formats

PNG, JPEG, WebP, GIF, TIFF, BMP, ICO, PNM, QOI, HDR — via the [image](https://github.com/image-rs/image) crate.

Output format always matches input format.

## Model Compatibility

Sqwale works with ONNX super-resolution models using:

- NCHW layout, RGB (3-channel) input/output
- float32 or float16 precision
- `[0, 1]` normalization range

Scale factor, tiling support, and data types are detected automatically from the ONNX graph.

**Where to find models:** [OpenModelDB](https://openmodeldb.info) · [upscale.wiki](https://upscale.wiki)

## Environment Variables

| Variable | Effect |
|---|---|
| `NO_COLOR` | Disable colored output |
| `CI` | Disable colors and progress bars |
| `RUST_LOG` | Log verbosity (e.g. `sqwale=debug`) |

## Roadmap

Potential future directions:

- **Video frame upscaling** — extract frames, upscale individually, reassemble. Batch processing already handles the core loop; integration with FFmpeg for decode/encode is the main work.
- **Frame interpolation** — generate intermediate frames for smooth slow-motion or framerate conversion. A different class of models but a natural companion to spatial upscaling.
- **Parallel image processing** — process multiple images concurrently using separate sessions, improving throughput on multi-GPU systems or when tiles leave GPU headroom.

## License

MIT
- [tracing](https://github.com/tokio-rs/tracing) — Structured logging

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please:

1. Run `cargo fmt` and `cargo clippy` before submitting
2. Follow Rust idioms and existing code style
3. Add tests for new features
4. Update PLAN.md if changing architecture or design decisions

## Roadmap

Potential future enhancements:

- [ ] Custom normalization profiles per model (YAML/TOML config)
- [ ] Multi-threading for batch processing
- [ ] Video frame upscaling
- [ ] Alternative color spaces (RGBA, grayscale)
- [ ] Model quantization and optimization
- [ ] Web UI or service mode

---

**Made with ❤️ by [Hyphonical](https://github.com/Hyphonical)**