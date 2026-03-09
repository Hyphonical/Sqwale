# Sqwale 🐋

**AI-powered video and image enhancement. Blazingly fast.**

Multiply your video's frame rate, upscale images with cutting-edge ONNX models, inspect model metadata, and do it all on your GPU. No cloud. No APIs. No waiting around. Just you, your media, and some seriously clever neural networks running locally on your machine.

## Table of Contents

- [What's This About?](#whats-this-about)
- [Why?](#why)
- [Features](#features-)
- [Quick Start](#quick-start-)
- [Documentation](#documentation-)
- [Usage](#usage)
  - [`interpolate` - Multiply Video Frame Rate](#interpolate---multiply-video-frame-rate)
  - [`upscale` - Enhance Images & Video](#upscale---enhance-images--video)
  - [`inspect` - Analyze Models](#inspect---analyze-models)
  - [Global Options](#global-options)
- [Hardware Support](#hardware-support-)
- [Contributing](#contributing)
- [License](#license)

## What's This About?

You know that feeling when you have a video shot at 24fps and it feels choppy when you play it back at 60fps? Or when you've got a beautiful photo but it's not quite sharp enough? Or when you find a cool ONNX model online and want to know exactly what it does before using it?

Sqwale fixes that.

Type `sqwale interpolate myvideo.mp4` and boom—it's now twice as smooth with perfectly blended frames. Pipe in `sqwale upscale photo.jpg` and get a beautifully upscaled version, automatically handling any image size with intelligent tiling. Run `sqwale inspect model.onnx` and get a human-readable breakdown of what's inside that black box.

Sqwale uses [RIFE 4.25](https://github.com/megvii-research/RIFE) for frame interpolation (the same tech that Netflix uses to smooth slow-motion content) and supports any ONNX super-resolution model you throw at it. It comes bundled with [4xLSDIRCompactv2](https://openmodeldb.info/models/4x-LSDIRCompact-v2) for upscaling. It automatically detects your GPU and does all the heavy lifting with zero-disk streaming pipelines—everything flows through OS pipes, nothing hits the disk.

## Why?

Look, I could wax poetic about how "video enhancement is underutilized in local workflows" or something. But honestly? It's because I wanted a tool that was **fast**, **local**, and **didn't require reading 10 blog posts to set up**. Something I could just `cargo build --release` and start using immediately.

Also, RIFE is sick. FFmpeg pipes are neat. Rust is cool.

So if you're the kind of person who has videos that need smoothing, images that need sharpening, or you're just curious about what's inside an ONNX model, maybe Sqwale is for you too. ✨

## Features 🎯

- **🎬 Frame interpolation** — Multiply video framerates (2×, 4×, 8×, …) using RIFE 4.25 with zero temporal artifacts
- **🔍 Smart scene detection** — Detect hard cuts and avoid blurry ghosting across scene changes (keeps audio in sync)
- **🖼️ Image & video upscaling** — Upscale photos, artwork, and video with any ONNX super-resolution model
- **🧩 Intelligent tiling** — Seamlessly handle images of any size with Hann-windowed blending (no visible seams)
- **⚡ Multi-GPU support** — Auto-detects TensorRT, CUDA, DirectML, CoreML, XNNPACK; falls back to CPU
- **🔀 Frequency blending** — Mix AI output with Lanczos upscaling via FFT Laplacian pyramid (best of both worlds)
- **🎛️ Ensemble mode** — Horizontal-flip averaging for even higher fidelity RIFE output
- **📦 Streaming pipelines** — Video data flows through OS pipes; zero temporary files written to disk
- **🚀 Prefetch threading** — Next image loads from disk while GPU processes the current one (no GPU idle time)
- **🎨 Beautiful output** — Per-tile timing, progress bars, colored status—actually enjoyable to watch
- **🛑 Graceful shutdown** — Ctrl+C finishes current item; second press exits immediately
- **🔬 Model inspection** — Analyze ONNX model metadata: scale factor, channel layout, precision, compatible tiling
- **🌍 Cross-platform** — Works on Windows, Linux, macOS
- **📚 Library support** — Use Sqwale as a Rust crate for integration into other projects

## Quick Start 🚀

### 1. Install

Requires Rust **1.85** or later. For video interpolation, requires [FFmpeg](https://ffmpeg.org/download.html) **≥ 5.1** on your `PATH`.

```bash
git clone https://github.com/Hyphonical/Sqwale.git
cd Sqwale
cargo build --release
```

Binary at `target/release/sqwale` (or `sqwale.exe` on Windows).

### 2. Multiply video framerate

```bash
# 2× frame rate (default)
sqwale interpolate myvideo.mp4

# 4× with high quality
sqwale interpolate myvideo.mp4 -x 4 --crf 16

# With smart scene detection (avoids blurry cuts)
sqwale interpolate myvideo.mp4 --scene-detect

# Explicit output path (mkv, mp4, or webm)
sqwale interpolate myvideo.mp4 -o output.mp4
```

### 3. Upscale images & video

```bash
# Single image (uses bundled 4xLSDIRCompactv2 model)
sqwale upscale photo.jpg

# Video upscaling (auto-detected, same command)
sqwale upscale myvideo.mp4

# Video with custom quality and fp16 inference
sqwale upscale myvideo.mp4 --crf 16 --fp16

# Batch processing with custom model
sqwale upscale "photos/*.jpg" -m mymodel.onnx -o upscaled/

# With frequency blending (smoother, less "plasticky")
sqwale upscale photo.jpg --blend 0.3
```

### 4. Inspect models

```bash
# See what's inside an ONNX model
sqwale inspect model.onnx
sqwale inspect "models/*.onnx"
sqwale inspect models/
```

## Usage

### `interpolate` - Multiply video frame rate

Make your video smoother by generating intermediate frames using RIFE 4.25. All processing happens in-memory through piped FFmpeg processes—nothing is written to disk between frames.

```bash
sqwale interpolate [OPTIONS] <INPUT>

Arguments:
  <INPUT>  Input video file path

Options:
  -x, --multiplier <N>         Frame rate multiplier (2, 4, 8, …)        [default: 2]
  -o, --output <PATH>          Output file path (mkv, mp4, or webm)        [default: auto]
  --crf <N>                    x264/NVENC quality (lower = better)        [default: 18]
  --ensemble                   Horizontal-flip averaging for higher quality
  --scene-detect               Detect hard cuts and duplicate instead of interpolating
  --scene-threshold <0.0–1.0>  Scene detection sensitivity               [default: 0.1]
```

**Examples:**

```bash
# Quick smooth—2× framerate, reasonable quality
sqwale interpolate movie.mp4

# Better quality output (slower, larger file)
sqwale interpolate movie.mp4 -x 4 --crf 16 --ensemble

# With scene detection (prevents ghosting across cuts)
sqwale interpolate action.mp4 --scene-detect --scene-threshold 0.15

# Ultra smooth (may look unnatural in some scenes)
sqwale interpolate footage.mp4 -x 8

# Increase scene detection sensitivity (lower = more sensitive to cuts)
sqwale interpolate interview.mp4 --scene-detect --scene-threshold 0.05
```

**How scene detection works:**
When `--scene-detect` is enabled, consecutive frames are scored using mean absolute difference (same algorithm as FFmpeg's `scdet`). If the score exceeds the threshold, the last pre-cut frame is duplicated instead of running RIFE inference. This keeps frame count correct for audio sync while avoiding blurry ghosting across cuts.

> [!TIP]
> **Start with default settings** — use `--scene-detect --scene-threshold 0.1` if your video has hard cuts (interviews, action scenes, screen recordings).

> [!TIP]
> **Ensemble mode is slower but smoother** — Use `--ensemble` when you have time and want maximum quality. It runs RIFE with horizontal flips and averages the results for less flickering.

**Example output:**
```
● interview.mp4
·  Config  2×  ·  CRF 18  ·  Standard  ·  Scene 0.10   RIFE 4.25 via CUDA
·  Input  1920×1080  24.00 fps  ~2400 frames
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  4799/4799 Interpolating…
·  Output  1920×1080  48.00 fps  4799 frames
✓ interview_2x.mkv  38s
```

> **See also:** [docs/architecture.md](docs/architecture.md) for pipeline internals and threading details.

---

### `upscale` - Enhance Images & Video

Upscale images and video using ONNX super-resolution models. Handles any image size through intelligent tiling. Comes bundled with [4xLSDIRCompactv2](https://openmodeldb.info/models/4x-LSDIRCompact-v2); works with any compatible ONNX model.

Single files are automatically probed: video files are upscaled frame-by-frame through FFmpeg pipes, while images go through the standard tiling pipeline. Glob patterns and directories always process images only.

```bash
sqwale upscale [OPTIONS] <INPUT>

Arguments:
  <INPUT>  Input image/video path or glob pattern

Options:
  -m, --model <PATH>           ONNX model file                            [default: bundled]
  -o, --output <PATH>          Output path or directory                    [default: auto]
  --tile-size <PX>             Tile size in pixels (0 = disable)          [default: 512]
  --tile-overlap <PX>          Overlap between tiles                      [default: 16]
  --blend <0.0–1.0>            Blend AI with Lanczos upscale via FFT      [default: 0.0]
  --grain <0–100>              Add monochrome luma noise post-upscale      [default: 0]
  --crf <N>                    Video encoding quality (lower = better)    [default: 18]
```

**Image examples:**

```bash
# Single image with bundled model (output: input_4x.jpg)
sqwale upscale photo.jpg

# Explicit model and output
sqwale upscale photo.jpg -m animesharp.onnx -o upscaled.jpg

# Batch processing into a folder
sqwale upscale "vacation/*.jpg" -m model.onnx -o vacation_upscaled/

# Frequency blending for smoother results (less "plasticky" AI look)
sqwale upscale photo.jpg --blend 0.5

# Custom tiling for high VRAM systems (faster, needs more memory)
sqwale upscale huge.jpg --tile-size 1024 --tile-overlap 64

# Disable tiling (loads full image into VRAM—use for small images or big GPUs)
sqwale upscale small.jpg --tile-size 0
```

**Video examples:**

```bash
# Upscale a video (auto-detected, output: input_4x.mkv)
sqwale upscale video.mp4

# Higher quality encoding
sqwale upscale video.mp4 --crf 16

# Explicit output format (mkv, mp4, or webm)
sqwale upscale video.mp4 -o upscaled.mp4

# With fp16 inference for lower VRAM usage
sqwale upscale video.mp4 --fp16
```

**Batch mode dual progress:**
```
● [1/5] vacation/img1.jpg
  ·  Input  5776×3856
  ━━━━━━━━━━━━━━━━━━━━╌╌╌╌╌╌╌╌╌  12/24 tiles  1m 02s  3.9s/tile
  ━━━━━━━━╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌   1/5  images  1m 02s  ~4m 08s remaining
```

**Video upscale example output:**
```
● clip.mp4
·  Model  4x  ·  RGB  ·  float32  ·  dynamic   embedded model via DirectML
·  Input  1280×720  23.98 fps  ~1200 frames
  ━━━━━━━━━━━━━━━━━━━━╌╌╌╌╌╌╌╌╌  6/12 tiles  2.1s/tile
  ━━━━━━━━╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌   120/1200  frames  8m 20s  ~1h 10m remaining
·  Output  5120×2880  23.98 fps  1200 frames
✓ clip_4x.mkv  1h 18m 42s
```

> [!TIP]
> **Frequency blending (`--blend`) tips:**
> - `0.0` (default): Pure AI output—sharpest but can look artificial
> - `0.3–0.5`: Sweet spot for photos—AI detail + natural look
> - `0.7–1.0`: AI provides fine detail only; Lanczos handles structure and color

> [!TIP]
> **Tile size tuning:** Larger tiles are faster but use more VRAM. Use `--tile-size 0` if your image fits in your GPU's memory; otherwise tune to your GPU.

> **See also:** [docs/tiling.md](docs/tiling.md) for VRAM budgeting and alignment requirements.

> **See also:** [docs/architecture.md](docs/architecture.md) for prefetch pipeline details.

---

### `inspect` - Analyze models

Examine ONNX model metadata without running inference. Perfect for understanding what a model does before committing to it.

```bash
sqwale inspect [OPTIONS] <PATTERN>

Arguments:
  <PATTERN>  File path, glob pattern, or directory containing .onnx files
```

**Examples:**

```bash
# Single model
sqwale inspect model.onnx

# All models in a directory
sqwale inspect models/

# Glob pattern
sqwale inspect "models/*.onnx"
```

**Example output:**
```
● 4x-LSDIRCompactv2.onnx
 ├─ Scale      4x  via DepthToSpace (PixelShuffle)
 ├─ Color      RGB  in:3 → out:3
 ├─ Precision  float32 → float32
 ├─ Opset      17
 ├─ Tiling     supported  dynamic spatial dims
 │   └─ Alignment  divisible by 8
 ╰─ Ops        566  total nodes
      ├─ 142  Constant
      ├─  89  Conv
      ├─  76  Add
      ├─  64  Unsqueeze
      ╰─       … more op types

● 2x-AnimeSharpV2.onnx
 ├─ Scale      2x  via ConvTranspose
 ├─ Color      RGB  in:3 → out:3
 ├─ Precision  float16 → float16
 ├─ Opset      14
 ├─ Tiling     not supported  fixed spatial dims
 ├─ Alignment  (N/A)
 ╰─ Ops        234  total nodes
      └─ …
```

**What you're looking at:**
- **Scale** — How much bigger the output is (2×, 4×, etc.) and how it's achieved
- **Color** — Color space and channel count (RGB or other)
- **Precision** — Input/output data types (float32, float16, etc.)
- **Opset** — ONNX operator set version
- **Tiling** — Whether the model supports tiling (for huge images)
- **Ops** — Breakdown of neural network layers

---

### Global Options

Options available for all commands:

```bash
--provider <TYPE>      Execution provider: auto, cpu, cuda, tensorrt, directml, coreml, xnnpack
--tile-size <PX>       Tile size in pixels (0 = disable tiling)
--tile-overlap <PX>    Overlap between tiles
--blend <0.0–1.0>      Blend AI with Lanczos upscale via FFT [default: 0.0]
--fp16                 Force half-precision (fp16) inference for lower VRAM usage
```

**Examples:**

```bash
# Force CPU mode
sqwale interpolate video.mp4 --provider cpu

# Use specific GPU provider
sqwale upscale photo.jpg --provider tensorrt

# Half-precision inference (lower VRAM, faster on some GPUs)
sqwale upscale video.mp4 --fp16
sqwale interpolate video.mp4 --fp16
```

---

## Hardware Support ⚡

Sqwale auto-detects your best available platform:

| Platform | Provider Priority | Performance |
|----------|------------------|-------------|
| **Windows** | TensorRT → CUDA → DirectML → CPU | Fastest with RTX GPUs |
| **Linux** | TensorRT → CUDA → XNNPACK → CPU | Fastest with any NVIDIA GPU |
| **macOS** | CoreML → CPU | Very fast on Apple Silicon (M1/M2/M3) |

> [!TIP]
> Override auto-detection with `--provider cuda` or `--provider cpu`. When a provider fails, Sqwale falls back automatically (with a warning).

| Provider | What You Need | Speed |
|----------|---------------|-------|
| **TensorRT** | NVIDIA GPU + CUDA Toolkit + TensorRT | Ultra-fast (50–100ms/image) |
| **CUDA** | NVIDIA GPU + CUDA Toolkit | Very fast (80–150ms/image) |
| **CoreML** | Apple Silicon (M1/M2/M3) | Very fast (80–150ms/image) |
| **DirectML** | Windows 10+ with GPU | Fast (150–300ms/image) |
| **XNNPACK** | ARM or x64 CPU | Moderate (200–400ms/image) |
| **CPU** | Any CPU | Slow (500–2000ms/image) |

> **See also:** [docs/providers.md](docs/providers.md) for detailed setup instructions per provider.

---

## Supported Media

| Format | Images | Video |
|--------|--------|-------|
| **Formats** | PNG, JPEG, WebP, GIF, TIFF, BMP, ICO, PNM, QOI, HDR | MP4, MKV, WebM, AVI, MOV (via FFmpeg) |
| **Output** | Any supported format | MKV, MP4, or WebM (defaults to input container) |

---

## Model Compatibility

Sqwale works with ONNX super-resolution models that follow these requirements:

- **Layout:** NCHW (batch, channels, height, width)
- **Color space:** RGB (3-channel input/output)
- **Precision:** float32 or float16
- **Normalization:** `[0, 1]` range
- **Scale detection:** Via DepthToSpace, ConvTranspose, Resize, or metadata

**Where to find models:**
- [OpenModelDB](https://openmodeldb.info) — Curated ONNX model repository
- [upscale.wiki](https://upscale.wiki) — Community upscaler database
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) — State-of-the-art upscalers

**Default bundled model:**
[4xLSDIRCompactv2](https://openmodeldb.info/models/4x-LSDIRCompact-v2) by Phhofm
- **Scale:** 4×
- **License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **Performance:** Great quality/speed tradeoff, ~100ms on CUDA, works on CPU

> **See also:** [docs/models.md](docs/models.md) for model recommendations and compatibility troubleshooting.

---

## Environment Variables

| Variable | Effect |
|----------|-----------|
| `NO_COLOR` | Disable colored output |
| `CI` | Disable colors and progress bars (auto-detected) |
| `RUST_LOG` | Log verbosity; e.g., `sqwale=debug` for detailed logs |
| `ORT_LOG_SEVERITY_LEVEL` | Control ONNX Runtime diagnostics (Sqwale suppresses by default) |

---

## Documentation 📚

| Doc | Purpose |
|-----|---------|
| [docs/architecture.md](docs/architecture.md) | Internals: threading, channel design, streaming pipelines |
| [docs/providers.md](docs/providers.md) | GPU provider setup, platform matrix, compatibility |
| [docs/tiling.md](docs/tiling.md) | Image tiling, VRAM budgeting, Hann blending math |
| [docs/models.md](docs/models.md) | Model selection, compatibility, troubleshooting |

---

## Contributing

PRs welcome! Please run `cargo fmt` and `cargo clippy` before submitting. Follow Rust idioms and the existing code style. Add tests for new behavior.

See [docs/architecture.md](docs/architecture.md) for codebase overview.

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

Made with ☕, neural networks, and Rust. If you've got videos that need smoothing or images that need sharpening, Sqwale's got you.
