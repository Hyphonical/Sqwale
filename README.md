# 🐋 Sqwale

A fast, cross-platform CLI for AI-powered video and image enhancement. Frame interpolation with RIFE, image upscaling with ONNX super-resolution models, GPU acceleration, and zero-disk pipeline architecture.

---

## ✨ Features

- 🎞️ **Frame interpolation** — multiply video frame rates using RIFE 4.25 (2×, 4×, 8×, …)
- 🔍 **Scene detection** — skip RIFE inference across hard cuts, duplicate the pre-cut frame to keep audio sync
- 🖼️ **Image upscaling** — batch upscale with any ONNX super-resolution model (4xLSDIRCompactv2 bundled)
- 🧩 **Tiling** — seamlessly process images of any size with Hann-windowed tile blending
- ⚡ **GPU acceleration** — TensorRT, CUDA, DirectML, CoreML, XNNPACK, with automatic CPU fallback
- 🔀 **Frequency blending** — mix AI output with Lanczos upscale via an FFT Laplacian pyramid
- 🎛️ **Ensemble mode** — horizontal-flip averaging for higher fidelity RIFE output
- 📦 **Zero-disk video pipeline** — all frame data flows through OS pipes; no temp files written
- 🎨 **Clean terminal output** — progress bars, per-tile timing, coloured status lines
- 🛑 **Graceful cancellation** — Ctrl+C finishes the current item; second press exits immediately

---

## 🗺️ Roadmap

- [ ] 🎬 **Video upscaling** — run super-resolution models frame-by-frame on video files
- [ ] 🔁 **Multi-model chaining** — compose upscale + interpolate in a single pass
- [ ] 🧪 **Benchmark mode** — measure throughput across providers and tile sizes
- [ ] 📊 **Verbose scene stats** — report scene cut count and timestamps after interpolation

---

## 📦 Installation

Requires Rust 1.85 or later. For video commands, [FFmpeg](https://ffmpeg.org/download.html) must be on your `PATH`.

```bash
git clone https://github.com/Hyphonical/Sqwale.git
cd Sqwale
cargo build --release
```

The binary lands at `target/release/sqwale` (or `sqwale.exe` on Windows).

---

## 🎞️ Interpolate

Multiply a video's frame rate using RIFE 4.25. All processing is done in-memory through piped FFmpeg processes — nothing is written to disk between frames.

```bash
# 2× frame rate (default)
sqwale interpolate input.mp4

# 4× with lower CRF for better quality output
sqwale interpolate input.mp4 -x 4 --crf 16

# Enable scene-aware detection so cuts don't get blended
sqwale interpolate input.mp4 --scene-detect

# Tune the cut sensitivity (default 0.4; lower = more sensitive)
sqwale interpolate input.mp4 --scene-detect --scene-threshold 0.3

# Ensemble mode for higher quality (slower)
sqwale interpolate input.mp4 --ensemble

# Explicit output path
sqwale interpolate input.mp4 -o output.mkv
```

```
● input.mp4
·  Config  2×  ·  CRF 18  ·  Standard  ·  Scene 0.40   RIFE 4.25 via auto
·  Input  1920×1080  24.00 fps  ~2400 frames
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  4799/4799 Interpolating…
·  Output  1920×1080  48.00 fps  4799 frames
✓ input_2x.mkv  38s
```

### How scene detection works

When `--scene-detect` is enabled, each pair of consecutive frames is scored using mean absolute difference (the same algorithm as FFmpeg's `scdet` filter). If the score exceeds the threshold, the last pre-cut frame is duplicated `(multiplier − 1)` times instead of running RIFE inference. This keeps the output frame count correct for audio sync while avoiding the blurry ghosting that RIFE produces across hard cuts.

### Interpolate options

| Option | Description | Default |
|---|---|---|
| `-x`, `--multiplier <N>` | Frame rate multiplier (power of two: 2, 4, 8, …) | `2` |
| `--crf <N>` | x264/NVENC quality (lower = better quality, larger file) | `18` |
| `--ensemble` | Horizontal-flip averaging for higher fidelity | off |
| `--scene-detect` | Detect cuts and duplicate instead of interpolating | off |
| `--scene-threshold <0–1>` | Cut sensitivity; lower = more sensitive | `0.4` |
| `-o`, `--output <path>` | Output path (always `.mkv`) | next to input |

---

## 🖼️ Upscale

Upscale images using any ONNX super-resolution model. The bundled default is [4xLSDIRCompactv2](https://openmodeldb.info/models/4x-LSDIRCompact-v2).

```bash
# Single image (output: input_4x.png)
sqwale upscale input.png -m model.onnx

# Explicit output path
sqwale upscale input.png -m model.onnx -o output.png

# Batch processing into a folder
sqwale upscale "photos/*.jpg" -m model.onnx -o upscaled/

# Custom tiling
sqwale upscale input.png -m model.onnx --tile-size 768 --tile-overlap 32

# Frequency-domain blending with Lanczos
sqwale upscale input.png -m model.onnx --blend 0.5
```

```
● input.jpg
·  Model  4x · RGB · float32 · dynamic  model.onnx via auto
·  Input  4000×6000
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  117/117 Upscaling…  27s  0.2s/tile
·  Output  16000×24000
✓ input_4x.jpg  27s
```

Batch mode shows dual progress bars — per-image tile progress on top, overall batch progress with ETA below:

```
● [1/5] photos/img1.jpg
  ·  Input  5776×3856
  ━━━━━━━━━━━━━━━━━━━━╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌  12/24 tiles  1m 02s  3.9s/tile
  ━━━━━━━━╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌   1/5  images  1m 02s  ~4m 08s remaining
```

### Upscale options

| Option | Description | Default |
|---|---|---|
| `-m`, `--model <path>` | ONNX model file (bundled 4xLSDIRCompactv2 if omitted) | bundled |
| `-o`, `--output <path>` | Output path or directory | next to input |
| `--tile-size <px>` | Tile size in pixels (`0` = no tiling) | `512` |
| `--tile-overlap <px>` | Overlap between adjacent tiles | `16` |
| `--blend <0.0–1.0>` | Blend AI output with Lanczos upscale via FFT Laplacian pyramid | `0.0` |

---

## 🔬 Inspect

Analyse ONNX model metadata without running inference:

```bash
sqwale inspect model.onnx
sqwale inspect "models/*.onnx"
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
      ╰─       … more op types
```

---

## ⚡ Provider selection

Sqwale picks the best available provider automatically:

| Platform | Priority |
|---|---|
| Windows | TensorRT → CUDA → DirectML → CPU |
| Linux | TensorRT → CUDA → XNNPACK → CPU |
| macOS | CoreML → CPU |

Override with `--provider cuda`. Falls back to CPU on failure.

```bash
sqwale interpolate input.mp4 --provider cuda
sqwale upscale input.png -m model.onnx --provider cpu
```

---

## 🧩 Tiling

Large images are split into overlapping tiles, each processed independently, then blended with cosine-weighted (Hann) windows for seamless output.

- **Dynamic models** use `--tile-size` (default 512 px), respecting model-alignment requirements.
- **Fixed-size models** automatically use the model's required tile dimensions.
- **Disable tiling** with `--tile-size 0` to process the full image at once.

---

## 🖼️ Supported image formats

PNG, JPEG, WebP, GIF, TIFF, BMP, ICO, PNM, QOI, HDR — via the [image](https://github.com/image-rs/image) crate. Output format always matches input.

---

## 🤖 Model compatibility

Works with ONNX super-resolution models that use:

- NCHW layout, RGB (3-channel) input/output
- float32 or float16 precision
- `[0, 1]` normalisation range
- Scale detection via DepthToSpace, ConvTranspose, Resize, metadata, or static shape ratio

**Where to find models:** [OpenModelDB](https://openmodeldb.info) · [upscale.wiki](https://upscale.wiki)

The bundled default is [4xLSDIRCompactv2](https://openmodeldb.info/models/4x-LSDIRCompact-v2) by Phhofm, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

---

## 🌍 Environment variables

| Variable | Effect |
|---|---|
| `NO_COLOR` | Disable coloured output |
| `CI` | Disable colours and progress bars |
| `RUST_LOG` | Log verbosity (e.g. `sqwale=debug`) |

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

Contributions welcome! Please run `cargo fmt` and `cargo clippy` before submitting, follow Rust idioms and the existing code style, and add tests for new behaviour.
