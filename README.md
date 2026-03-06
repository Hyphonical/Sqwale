# Sqwale

A fast, cross-platform CLI and Rust library for running ONNX super-resolution models with automatic tiling, GPU acceleration, and beautiful progress visualization.

## Features

- 🚀 **High Performance**: GPU-accelerated inference with CUDA, TensorRT, CoreML, and CPU fallback
- 🔍 **Model Inspection**: Analyze ONNX models without running inference—detect scale, color space, precision, tiling support, and more
- 🎯 **Smart Tiling**: Automatic tile-based processing with seamless blending for large images
- 🎨 **Beautiful Output**: Colored, styled terminal output with progress bars and real-time statistics
- 🔧 **Flexible**: Single-image and batch processing with glob pattern support
- 📦 **Broad Format Support**: PNG, JPEG, WebP, GIF, TIFF, BMP, and more via the `image` crate
- 🔬 **Float16 Ready**: Automatic detection and handling of FP16 models
- ⚡ **Cross-Platform**: Windows, Linux, and macOS support

## Installation

### From Source

Requires Rust 1.82 or later:

```bash
git clone https://github.com/Hyphonical/Sqwale.git
cd Sqwale
cargo build --release
```

The binary will be at `target/release/sqwale` (or `sqwale.exe` on Windows).

### System Requirements

- **Windows**: CUDA/TensorRT support (optional)
- **Linux**: CUDA/TensorRT/XNNPACK support (optional)
- **macOS**: CoreML support (optional)

GPU acceleration requires appropriate drivers and runtimes installed. Sqwale will automatically fall back to CPU if GPU providers are unavailable.

## Usage

### Inspect Models

Analyze ONNX model properties without running inference:

```bash
# Single model
sqwale inspect models/4x-UltraSharp.onnx

# Multiple models with glob
sqwale inspect "models/*.onnx"

# Entire directory (recursive)
sqwale inspect models/
```

**Output Example:**

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

Process single images or batches with automatic tiling and GPU acceleration:

```bash
# Single image (auto output path: input_2x.png)
sqwale upscale input.png -m models/2x-model.onnx

# Single image with specific output
sqwale upscale input.png -m models/2x-model.onnx -o output.png

# Batch processing
sqwale upscale "images/*.png" -m models/4x-model.onnx -o upscaled/

# Custom tile size and overlap
sqwale upscale input.png -m models/2x-model.onnx --tile-size 768 --tile-overlap 32
```

**Output Example:**

```
● input.png
  ·  Model   2x · RGB · float16 · dynamic  MoSR_Sharp_fp16.onnx via CUDA
  ·  Input   5776×3856
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  24/24 tiles  2m 34s  3.9s/tile
  ·  Output  11552×7712
✓ output_2x.png  2m 34s
```

### Options

#### Global Options

- `--provider <auto|cpu|cuda|tensorrt|coreml|xnnpack>` — Execution provider (default: `auto`)
- `--tile-size <pixels>` — Tile size for large images (default: `512`, `0` = disable tiling)
- `--tile-overlap <pixels>` — Overlap between tiles for seamless blending (default: `16`)

#### Command-Specific Options

**`inspect`**
- `<PATTERN>` — File path, glob pattern, or directory

**`upscale`**
- `<INPUT>` — Input image path or glob pattern
- `-m, --model <MODEL>` — ONNX model path
- `-o, --output <OUTPUT>` — Output path or directory (optional)

### Provider Selection

Sqwale automatically selects the best available execution provider:

- **Windows**: TensorRT → CUDA → CPU
- **Linux**: TensorRT → CUDA → XNNPACK → CPU
- **macOS**: CoreML → CPU

Override with `--provider`:

```bash
sqwale upscale input.png -m model.onnx --provider cuda
```

If the specified provider fails, Sqwale falls back to CPU with a warning.

### Tiling Strategy

Sqwale automatically tiles large images to fit within GPU memory constraints:

- **Dynamic spatial models** (variable input size):
  - Use configured `--tile-size` (default 512px)
  - Respect model alignment requirements (e.g., divisible by 16)
  - Seamless blending with cosine-weighted overlap

- **Fixed spatial models** (static input size):
  - Automatically use model's required tile size
  - Sliding window with padding and cropping

- **Disable tiling**: `--tile-size 0` (processes entire image at once)

### Batch Processing

Process multiple images efficiently:

```bash
sqwale upscale "photos/*.jpg" -m models/4x-model.onnx -o upscaled/
```

**Features:**
- Progress bars for both per-image tiles and overall batch progress
- Continues on individual file failures
- Ctrl+C handling: first press finishes current image and skips remaining, second press force-exits

### Environment Variables

- `NO_COLOR` — Disable colored output
- `CI` — Disable colors and progress bars (auto-detected in CI environments)
- `RUST_LOG` — Control log verbosity (e.g., `RUST_LOG=sqwale=debug`)
- `ORT_LOG_SEVERITY_LEVEL` — ONNX Runtime log level (overridden by Sqwale to `3` = Error)

## Supported Formats

Input and output formats via the `image` crate:

- PNG, JPEG, WebP
- GIF, TIFF, BMP
- ICO, PNM, QOI, HDR

**Note:** Output format must match input format (no conversion). Use a separate tool like ImageMagick for format conversion.

## Model Requirements

Sqwale supports ONNX models with:

- **Input**: NCHW layout, RGB color space (3 channels), float32 or float16 dtype
- **Output**: Same layout and channels as input
- **Color normalization**: `[0, 1]` range (default)
- **Common architectures**: ESRGAN, Real-ESRGAN, SwinIR, HAT, Compact models, and more

Model detection is automatic—Sqwale infers scale factor, tiling support, and data types from the ONNX graph.

### Where to Find Models

- [OpenModelDB](https://openmodeldb.info) — Comprehensive database of upscaling models
- [upscale.wiki](https://upscale.wiki) — Community wiki with model links and guides

## Library Usage

Sqwale can be used as a Rust library:

```toml
[dependencies]
sqwale = "0.1"
```

```rust
use sqwale::{inspect_model, load_model, upscale_image, ProviderSelection, UpscaleOptions};
use std::path::Path;
use std::sync::{atomic::AtomicBool, Arc};

fn main() -> anyhow::Result<()> {
    // Inspect a model
    let info = inspect_model(Path::new("model.onnx"))?;
    println!("Scale: {}x, Precision: {}", info.scale, info.input_dtype);

    // Load model and create session
    let mut ctx = load_model(Path::new("model.onnx"), ProviderSelection::Auto)?;

    // Load image
    let img = image::open("input.png")?;

    // Upscale with options
    let opts = UpscaleOptions {
        tile_size: 512,
        tile_overlap: 16,
        cancel: Arc::new(AtomicBool::new(false)),
        on_tile_done: None,
    };

    let upscaled = upscale_image(&mut ctx, &ctx.model_info.clone(), &img, &opts)?;

    // Save result
    upscaled.save("output_2x.png")?;

    Ok(())
}
```

## Development

### Build Profiles

- `dev` — Debug build with full symbols
- `fast` — Quick optimized build for testing (`cargo build --profile fast`)
- `release` — Fully optimized production build (`cargo build --release`)

### Code Quality

Run checks before committing:

```bash
cargo fmt              # Format code
cargo clippy           # Lint warnings
cargo test             # Run tests (when added)
cargo check            # Verify compilation
```

## Architecture

- **`src/lib.rs`** — Public library API
- **`src/main.rs`** — CLI entry point and argument parsing
- **`src/inspect/`** — Model analysis and metadata extraction
- **`src/session.rs`** — ONNX Runtime session management and provider selection
- **`src/pipeline.rs`** — High-level upscaling orchestration
- **`src/tiling.rs`** — Tile grid computation and blending
- **`src/imageio.rs`** — Image loading, saving, and format handling
- **`src/commands/`** — CLI command implementations

## Credits

Built with:

- [ort](https://github.com/pykeio/ort) — Safe Rust bindings for ONNX Runtime
- [image](https://github.com/image-rs/image) — Pure Rust image encoding/decoding
- [clap](https://github.com/clap-rs/clap) — Command-line argument parsing
- [indicatif](https://github.com/console-rs/indicatif) — Progress bars and spinners
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