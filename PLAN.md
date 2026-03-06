# Sqwale v0.2 — Design Document

> **Version**: 0.2.0 (full rewrite)
> **MSRV**: 1.85
> **License**: MIT
> **Repository**: <https://github.com/Hyphonical/Sqwale>

---

## 1. Overview

Sqwale is a Rust **library and CLI** for running ONNX super-resolution models
on images. It provides two core capabilities:

1. **Model inspection** — analyse an ONNX model's properties (scale, color
   space, precision, tiling support, op histogram) without running inference.
2. **Image upscaling** — upscale single images or batches using tiled inference
   with seamless blending, progress bars, and GPU-accelerated execution
   providers.

Sqwale targets Windows, Linux, and macOS. It outputs beautiful, styled terminal
output inspired by the reference in `sqwale_style.txt`.

---

## 2. Goals

- **Rust library + CLI** in a single crate (`[[bin]]` + `[lib]`).
- **Cross-platform**: Windows, Linux, macOS.
- **Model inspection** without inference — detect scale, color space, precision,
  tiling support, opset, and op histogram directly from the ONNX graph.
- **Single-image and batch upscaling** with automatic tiling, overlap blending,
  and progress reporting.
- **Multiple execution providers**: CPU, CUDA, TensorRT, DirectML, CoreML,
  XNNPACK — with automatic selection and graceful fallback.
- **Float32 and float16** model support, including automatic detection and
  tensor conversion.
- **Grayscale, RGB, and RGBA** model support — match the model's channel count.
- **Beautiful, styled terminal output** with colored text, Unicode tree
  connectors, progress bars, and spinners.
- **Graceful Ctrl+C handling** — finish current work, skip remaining in batch.
- **Broad image format support** via the `image` crate.
- **High-quality output** — prefer quality over speed in all processing
  decisions (blending, conversion, rounding).

---

## 3. Non-Goals (v1)

| Not in scope                               | Rationale                           |
|--------------------------------------------|-------------------------------------|
| Format conversion (input ext ≠ output ext) | Use ImageMagick or similar          |
| Custom normalization profiles (mean/std)   | Future per-model config             |
| JSON or machine-readable CLI output        | CLI is for pretty human output only |
| Video processing                           | Different problem domain            |
| Model training or conversion               | Out of scope                        |
| Async / parallel image processing          | Sequential is simpler for v1        |
| GUI                                        | CLI and library only                |
| Docker support                             | Not shipping a container            |

---

## 4. Project Structure

```
Sqwale/
├── Cargo.toml
├── LICENSE
├── PLAN.md
├── README.md
├── rustfmt.toml
├── sqwale_style.txt
├── models/                    # ONNX models (gitignored, not shipped)
│
├── src/
│   ├── main.rs                # Binary entry: ORT init, tracing setup, CLI dispatch
│   ├── lib.rs                 # Library root: module declarations, convenience re-exports
│   ├── config.rs              # Library constants: tile defaults, inspect limits
│   ├── imageio.rs             # Image load/save, path resolution, extension checks
│   │
│   ├── cli/                   # CLI-only code (not part of the library)
│   │   ├── mod.rs             # Clap arg definitions, Commands enum
│   │   ├── inspect.rs         # `sqwale inspect` command handler
│   │   ├── upscale.rs         # `sqwale upscale` command handler
│   │   └── output.rs          # Shared formatting: symbols, colors, tree, progress, duration
│   │
│   ├── inspect/               # Model inspection subsystem (library)
│   │   ├── mod.rs             # Public types: ModelInfo, TileInfo, ColorSpace, ScaleSource
│   │   ├── detect.rs          # Detection algorithms: scale, tiling, channels
│   │   └── proto.rs           # Hand-rolled ONNX protobuf parser
│   │
│   ├── pipeline/              # Upscaling pipeline (library)
│   │   ├── mod.rs             # upscale_image(), UpscaleOptions, CancelToken
│   │   ├── tensor.rs          # Image ↔ NCHW tensor conversion, fp16 handling
│   │   └── tiling.rs          # Tile grid computation, padding, Hann-window blending
│   │
│   └── session/               # ORT session management (library)
│       ├── mod.rs             # SessionContext, load_model()
│       └── provider.rs        # ProviderSelection enum, EP construction, platform dispatch
│
└── tests/
    └── inspect.rs             # Structural integration tests for inspection
```

### Design rules

- **No folder has more than 5 files.**
- **`cli/` is binary-only** — `main.rs` declares `mod cli;`. The library
  (`lib.rs`) does not reference `cli/`.
- **Library modules are public** — `inspect`, `pipeline`, `session`, `imageio`,
  `config` are all declared in `lib.rs` and usable by external consumers.
- **Shared code lives in the library** — the CLI calls library functions and
  handles formatting/display. No business logic in `cli/`.

---

## 5. Cargo.toml

```toml
[package]
name         = "sqwale"
version      = "0.2.0"
edition      = "2024"
rust-version = "1.85.0"
authors      = ["Hyphonical"]
license      = "MIT"
repository   = "https://github.com/Hyphonical/Sqwale"
description  = "ONNX super-resolution inference library and CLI"

[[bin]]
name = "sqwale"
path = "src/main.rs"

[lib]
name = "sqwale"
path = "src/lib.rs"

[dependencies]
# ── Model Loading & Inference ──────────────────────────────────────────────
ort     = { version = "2.0.0-rc.12", features = ["api-24", "half"] }
ndarray = "0.17"
half    = "2"

# ── Image Processing ──────────────────────────────────────────────────────
image = { version = "0.25", default-features = false, features = [
    "png", "jpeg", "webp", "gif", "tiff", "bmp", "ico",
    "pnm", "qoi", "hdr",
] }

# ── CLI & Output ──────────────────────────────────────────────────────────
clap      = { version = "4", features = ["derive"] }
colored   = "3"
indicatif = "0.18"
ctrlc     = "3"

# ── Logging ───────────────────────────────────────────────────────────────
tracing            = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
tracing-indicatif  = "0.3"

# ── Utilities ─────────────────────────────────────────────────────────────
anyhow = "1"
glob   = "0.3"

[profile.release]
opt-level     = "z"
lto           = "fat"
codegen-units = 1
strip         = true
panic         = "abort"

[profile.fast]
inherits      = "release"
opt-level     = 1
debug         = 0
codegen-units = 32
incremental   = true
lto           = "off"
panic         = "abort"
strip         = false
```

### Dependency notes

| Crate               | Purpose                                              |
|----------------------|------------------------------------------------------|
| `ort`                | ONNX Runtime bindings for inference                  |
| `ndarray`            | N-dimensional arrays for tensor manipulation         |
| `half`               | IEEE 754 half-precision floats for fp16 models       |
| `image`              | Image decoding/encoding (broad format support)       |
| `clap`               | CLI argument parsing with derive macros              |
| `colored`            | Terminal color output (respects `NO_COLOR`)           |
| `indicatif`          | Progress bars and spinners                           |
| `ctrlc`              | Cross-platform Ctrl+C signal handling                |
| `tracing`            | Structured logging framework                         |
| `tracing-subscriber` | Log output formatting with env-filter                |
| `tracing-indicatif`  | Bridges tracing output with indicatif progress bars  |
| `anyhow`             | Ergonomic error handling with context                |
| `glob`               | Shell-style glob pattern expansion                   |

`is-terminal` was previously a dependency but is no longer needed —
`colored` 3.x already respects `NO_COLOR`, and `tracing-indicatif` handles
stderr TTY detection internally.

---

## 6. Configuration and Environment

### 6.1 Library constants (`config.rs`)

```rust
/// Default tile size in pixels for dynamic-spatial models.
pub const DEFAULT_TILE_SIZE: u32 = 512;

/// Default pixel overlap between adjacent tiles.
pub const DEFAULT_TILE_OVERLAP: u32 = 16;

/// Maximum op-types shown in the inspect tree before collapsing.
pub const INSPECT_MAX_OPS_SHOWN: usize = 8;
```

These are the **only** constants in the library. CLI-specific constants
(spinner timing, symbols, colors) live in `cli/output.rs`.

### 6.2 Environment variables

| Variable                  | Effect                                      | Default        |
|---------------------------|---------------------------------------------|----------------|
| `NO_COLOR`                | Disable colored output (any value)          | Unset          |
| `CI`                      | Disable colors and progress bars            | Unset          |
| `RUST_LOG`                | Control tracing log level filter            | `sqwale=warn`  |
| `ORT_LOG_SEVERITY_LEVEL`  | ONNX Runtime log verbosity (0–4)            | `3` (Error)    |

**ORT log suppression**: On startup (in `main.rs`), if `ORT_LOG_SEVERITY_LEVEL`
is **not** already set by the user, set it to `"3"` (Error) to suppress
verbose ORT diagnostics. If the user has set it explicitly, respect their
choice.

### 6.3 Color and progress detection

Implemented in `cli/output.rs`:

```rust
pub fn should_use_color() -> bool {
    // colored already checks NO_COLOR. We additionally check CI.
    std::env::var("CI").is_err()
}

pub fn should_show_progress() -> bool {
    // Progress bars only in interactive terminals without CI.
    std::io::stderr().is_terminal() && should_use_color()
}
```

When `should_use_color()` is false, call `colored::control::set_override(false)`
early in `main()` to globally disable color.

When `should_show_progress()` is false, progress bars are replaced by simple
one-line status messages on stdout (e.g. `"Processing image 1/5…"`).

---

## 7. CLI Specification

### 7.1 Top-level structure

```
sqwale [OPTIONS] <COMMAND>

Commands:
  inspect   Inspect ONNX model metadata
  upscale   Upscale images using an ONNX model

Options:
  --provider <PROVIDER>      Execution provider [default: auto]
                             Values: auto, cpu, cuda, tensorrt, directml, coreml, xnnpack
  --tile-size <PIXELS>       Tile size in pixels, 0 = disable tiling [default: 512]
  --tile-overlap <PIXELS>    Overlap between adjacent tiles [default: 16]
  -h, --help                 Print help
  -V, --version              Print version
```

### 7.2 Clap definitions (`cli/mod.rs`)

```rust
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "sqwale", author, version, about)]
#[command(propagate_version = true)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,

    /// Execution provider: auto, cpu, cuda, tensorrt, directml, coreml, xnnpack.
    #[arg(long, global = true, default_value = "auto")]
    pub provider: String,

    /// Tile size in pixels (0 = disable tiling).
    #[arg(long, global = true)]
    pub tile_size: Option<u32>,

    /// Overlap in pixels between adjacent tiles.
    #[arg(long, global = true)]
    pub tile_overlap: Option<u32>,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Inspect ONNX model metadata without running inference.
    Inspect {
        /// File path, glob pattern, or directory containing .onnx files.
        pattern: String,
    },

    /// Upscale images using an ONNX super-resolution model.
    Upscale {
        /// Input image path or glob pattern.
        input: String,

        /// Path to the ONNX model file.
        #[arg(short, long)]
        model: String,

        /// Output file path or directory.
        /// Omit to write next to the input as {stem}_{scale}x.{ext}.
        #[arg(short, long)]
        output: Option<String>,
    },
}
```

### 7.3 `sqwale inspect`

**Usage**:
```
sqwale inspect model.onnx
sqwale inspect "models/*.onnx"
sqwale inspect models/
```

**Behavior**:
1. Expand the pattern to a list of `.onnx` file paths:
   - Plain file path → single file.
   - Glob pattern (contains `*`, `?`, or `[`) → expand with `glob`.
   - Directory → recursively find all `*.onnx` files (`<dir>/**/*.onnx`).
2. If zero files matched → error with a hint.
3. **Single model** (exactly 1 file):
   - Call `sqwale::inspect_model(path)`.
   - Print styled inspection result to **stdout**.
   - No progress bar.
   - Exit 0 on success, non-zero on failure.
4. **Batch** (>1 file):
   - Show a single progress bar on **stderr** (`N/M models`).
   - For each model:
     - Run inspection.
     - On success: print styled result to **stdout**.
     - On failure: print error to **stdout**, log warning, continue.
   - After all models:
     - Exit 0 if all succeeded.
     - Exit non-zero if any failed.

**Error handling**: Individual model failures do not abort the batch. Errors are
displayed per-model in the styled output (red `✗` marker).

### 7.4 `sqwale upscale`

**Usage**:
```
sqwale upscale input.png -m model.onnx
sqwale upscale input.png -m model.onnx -o output.png
sqwale upscale input.png -m model.onnx -o output_dir/
sqwale upscale "images/*.png" -m model.onnx -o upscaled/
```

**Path semantics — single-file mode** (input is a single path or a glob
resolving to exactly 1 file):

| `--output` value    | Resolved output path                                  |
|---------------------|-------------------------------------------------------|
| Omitted             | `{input_dir}/{stem}_{scale}x.{input_ext}`             |
| Directory           | `{output_dir}/{stem}_{scale}x.{input_ext}`            |
| File with extension | Must match input extension; error otherwise            |

**Path semantics — batch mode** (glob resolving to >1 file):

| `--output` value    | Resolved output path                                  |
|---------------------|-------------------------------------------------------|
| Omitted             | Each file: `{own_dir}/{stem}_{scale}x.{input_ext}`    |
| Directory           | `{output_dir}/{stem}_{scale}x.{input_ext}`            |
| File with extension | **Error** — batch requires a directory                |

If the output directory does not exist, create it.

**Progress reporting**:

| Mode        | Tile bar (stderr)                    | Batch bar (stderr)               |
|-------------|--------------------------------------|----------------------------------|
| Single file | `━━━━━━━━ 12/24 tiles 1m02s 3.9s/t` | None                             |
| Batch       | Same (per current image)             | `━━━━ 2/5 images 5m46s elapsed`  |

**Summary output** (stdout, batch only):

On completion:
```
✓ 4 succeeded  ·  1 failed
  ✗  tests\Car 4.jpg  unexpected end of file at offset 48221
```

On interruption:
```
⚠ Interrupted — finishing current image, skipping remaining

✗ Cancelled after 2/5
✓ 2 succeeded  ·  3 skipped
```

No summary is printed for single-file mode (just the success/error line).

### 7.5 Exit codes

| Scenario                        | Exit code |
|---------------------------------|-----------|
| All operations succeeded        | `0`       |
| Any operation failed            | `1`       |
| Interrupted (Ctrl+C)            | `1`       |
| Invalid arguments               | `2`       |

---

## 8. Output and Formatting

### 8.1 Output streams

| Content                          | Stream   | Mechanism         |
|----------------------------------|----------|--------------------|
| Styled inspection results        | `stdout` | `println!`         |
| Styled upscale status/results    | `stdout` | `println!`         |
| Summary lines                    | `stdout` | `println!`         |
| Progress bars                    | `stderr` | `indicatif`        |
| Spinners                         | `stderr` | `indicatif`        |
| Tracing log messages             | `stderr` | `tracing-subscriber` via `tracing-indicatif` writer |
| Errors from library              | `stdout` | styled via `colored` |

**Key rule**: All user-facing information goes to stdout so it can be captured
or piped. Progress bars and logs are ephemeral and go to stderr.

### 8.2 Symbols and colors

Defined as constants in `cli/output.rs`:

```rust
// ── Symbols ────────────────────────────────────────────────────────────────
pub const SYM_BULLET: &str  = "●";   // Cyan bold — section header / major step
pub const SYM_CHECK: &str   = "✓";   // Green — success
pub const SYM_CROSS: &str   = "✗";   // Red bold — hard error
pub const SYM_WARN: &str    = "⚠";   // Yellow — warning
pub const SYM_DOT: &str     = "·";   // Dimmed — sub-detail
pub const SYM_ARROW: &str   = "→";   // Dimmed — directional separator
pub const SYM_TIMES: &str   = "×";   // Dimension separator (e.g. 1920×1080)
pub const SYM_ELLIPSIS: &str = "…";  // More items indicator

// ── Tree connectors ────────────────────────────────────────────────────────
pub const TREE_BRANCH: &str = "├─";  // U+251C U+2500
pub const TREE_LAST: &str   = "╰─";  // U+2570 U+2500
pub const TREE_PIPE: &str   = "│";   // U+2502
pub const TREE_SUB: &str    = "│   └─";  // Nested sub-item connector

// ── Truecolor values ──────────────────────────────────────────────────────
pub const CLR_PATH: (u8, u8, u8)  = (110, 148, 178);  // File paths
pub const CLR_VALUE: (u8, u8, u8) = (152, 195, 121);  // Model properties (green tint)

// ── Spinner ────────────────────────────────────────────────────────────────
pub const SPINNER_FRAMES: &[&str] = &["◐", "◓", "◑", "◒"];
pub const SPINNER_TICK_MS: u64 = 120;
```

### 8.3 Progress bar styles

Defined in `cli/output.rs`:

```rust
use indicatif::ProgressStyle;

/// Tile progress bar style (per-image).
pub fn tile_bar_style() -> ProgressStyle {
    ProgressStyle::default_bar()
        .template("  {bar:40.cyan/238}  {msg}")
        .unwrap()
        .progress_chars("━╌")
}

/// Batch progress bar style (overall images).
pub fn batch_bar_style() -> ProgressStyle {
    ProgressStyle::default_bar()
        .template("  {bar:40.cyan/238}  {msg}")
        .unwrap()
        .progress_chars("━╌")
}
```

Progress bar messages are formatted by the command handlers to include stats
like `12/24 tiles  1m 02s  3.9s/tile` or `2/5 images  5m 46s elapsed`.

### 8.4 Formatting helpers

```rust
/// Print a tree row: `prefix` `connector` `label` (padded) `value`.
///
/// Example output: ` ├─ Scale      2x  via DepthToSpace`
pub fn tree_row(
    prefix: &str,
    connector: &str,     // "├─" or "╰─"
    label: &str,         // "Scale", "Color", etc.
    label_width: usize,  // Column width for alignment (e.g. 9)
    value: &str,         // Pre-formatted value string
);

/// Format dimensions as "W×H" with bold bright-white numbers.
pub fn dims_str(w: u32, h: u32) -> String;

/// Format a Duration for display: "1m 02s", "3.2s", "1h 05m 12s".
pub fn format_duration(d: Duration) -> String;

/// Run a closure while displaying a spinner on stderr.
/// The spinner is cleared after `f` returns.
pub fn with_spinner<T>(label: &str, f: impl FnOnce() -> T) -> T;

/// Format a color space value with its color.
pub fn color_space_str(cs: &ColorSpace) -> ColoredString;

/// Format a dtype value with its color.
pub fn dtype_str(dtype: &str) -> ColoredString;
```

### 8.5 Inspect output format

Per model, printed to stdout:

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

**Color mapping** (referencing `colored` API):

| Element                    | Style                                    |
|----------------------------|------------------------------------------|
| `●` (bullet)               | `.cyan().bold()`                         |
| Model filename             | `.bold().bright_white()`                 |
| Tree connectors            | `.dimmed()`                              |
| Row labels (Scale, etc.)   | `.white()`                               |
| Numeric values (2x, 17)    | `.bold().bright_white()`                 |
| Detection source text      | `.dimmed()`                              |
| Color space (RGB)          | `.truecolor(152, 195, 121)`              |
| Dtype (float16)            | `.truecolor(152, 195, 121)`              |
| Tiling "supported"         | `.green()`                               |
| Tiling "fixed size"        | `.yellow()`                              |
| Op names (Constant, etc.)  | `.dimmed()`                              |
| Op counts                  | `.bold().bright_white()`                 |
| `… N more op types`        | `.dimmed()`                              |

### 8.6 Upscale output format

**Single-file mode** (stdout):
```
● tests\Car 1.jpg
  · Model   2x · RGB · float16 · dynamic  MoSR_Sharp_fp16.onnx via CUDA
  · Input   5776×3856
  · Output  11552×7712
✓ .\tests\upscaled\Car 1_2x.jpg  2m 34s
```

The tile progress bar appears on stderr between `Input` and `Output` lines and
is cleared once tiling completes.

**Batch mode** (stdout):
```
● Batch: 5 images → .\tests\upscaled\
  · Model   2x · RGB · float16 · dynamic
  · Loaded  MoSR_Sharp_fp16.onnx  via CUDA

● [1/5] tests\Car 1.jpg
  · Input   5776×3856
  · Output  11552×7712
  ✓ .\tests\upscaled\Car 1_2x.jpg  2m 34s

● [2/5] tests\Car 2.jpg
  · Input   4946×3307
  · Output  9892×6614
  ✓ .\tests\upscaled\Car 2_2x.jpg  3m 14s
```

With progress bars on stderr (see § 7.4 table).

---

## 9. Model Inspection Subsystem

### 9.1 Public types (`inspect/mod.rs`)

```rust
/// The color space inferred from the model's channel count.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ColorSpace {
    Grayscale,
    #[default]
    Rgb,
    Rgba,
    Unknown(u32),
}

impl ColorSpace {
    /// Number of channels for this color space.
    pub fn channels(&self) -> u32 {
        match self {
            Self::Grayscale => 1,
            Self::Rgb => 3,
            Self::Rgba => 4,
            Self::Unknown(n) => *n,
        }
    }
}

impl std::fmt::Display for ColorSpace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Grayscale => write!(f, "Grayscale"),
            Self::Rgb => write!(f, "RGB"),
            Self::Rgba => write!(f, "RGBA"),
            Self::Unknown(n) => write!(f, "Unknown ({n} channels)"),
        }
    }
}
```

```rust
/// How the upscale factor was detected.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ScaleSource {
    Metadata,
    StaticShapeRatio,
    DepthToSpace,
    ConvTransposeStride,
    #[default]
    Assumed,
}

impl std::fmt::Display for ScaleSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Metadata => write!(f, "metadata_props"),
            Self::StaticShapeRatio => write!(f, "static shape ratio"),
            Self::DepthToSpace => write!(f, "DepthToSpace (PixelShuffle)"),
            Self::ConvTransposeStride => write!(f, "ConvTranspose stride"),
            Self::Assumed => write!(f, "assumed (no upscale op found)"),
        }
    }
}
```

```rust
/// Tiling constraints extracted from the model graph.
#[derive(Debug, Clone, Default)]
pub struct TileInfo {
    /// Whether the model accepts variable spatial dimensions (dynamic H/W).
    pub supported: bool,
    /// Required alignment for spatial dimensions (e.g. 8, 16, 32).
    pub alignment: Option<u32>,
    /// Fully-static required input size (height, width).
    pub fixed_size: Option<(u64, u64)>,
}

impl TileInfo {
    /// Compute the effective tile size respecting model constraints.
    ///
    /// - Fixed-size model: returns min(h, w) from the fixed size, ignoring user preference.
    /// - Alignment required: rounds user_pref UP to the nearest multiple.
    /// - Otherwise: returns user_pref as-is.
    pub fn effective_tile_size(&self, user_pref: u32) -> u32 {
        if let Some((h, w)) = self.fixed_size {
            h.min(w) as u32
        } else if let Some(align) = self.alignment {
            let r = user_pref % align;
            if r == 0 { user_pref } else { user_pref + (align - r) }
        } else {
            user_pref
        }
    }
}
```

```rust
/// All metadata extracted from an ONNX model without running inference.
#[derive(Debug, Clone, Default)]
pub struct ModelInfo {
    /// Upscale factor (1 for restoration/denoising models).
    pub scale: u32,
    /// How the scale was determined.
    pub scale_source: ScaleSource,
    /// Input color space derived from channel count.
    pub color_space: ColorSpace,
    /// Number of input channels.
    pub input_channels: u32,
    /// Number of output channels.
    pub output_channels: u32,
    /// Tiling constraints.
    pub tile: TileInfo,
    /// Input element type (e.g. "float32", "float16").
    pub input_dtype: String,
    /// Output element type.
    pub output_dtype: String,
    /// Maximum opset version in use.
    pub opset: u64,
    /// Op-type histogram sorted by frequency descending.
    pub op_fingerprint: Vec<(String, usize)>,
}

impl ModelInfo {
    /// Returns true when the model expects half-precision input tensors.
    pub fn needs_fp16_input(&self) -> bool {
        self.input_dtype == "float16"
    }
}
```

### 9.2 Public API

```rust
/// Inspect an ONNX model file and extract all metadata without inference.
///
/// Reads the raw bytes, walks the protobuf structure, and infers
/// scale, color space, precision, tiling support, and op histogram.
pub fn inspect_model(path: &Path) -> Result<ModelInfo>;
```

This function is the **only** public entry point of the inspection subsystem.
It delegates to `detect.rs` for algorithm logic and `proto.rs` for parsing.

### 9.3 Detection algorithms (`inspect/detect.rs`)

**Scale detection** — tried in order, first match wins:

1. **Metadata**: look for `scale` key in ONNX `metadata_props`.
2. **Static shape ratio**: if both input and output have static spatial dims,
   compute `output_h / input_h`.
3. **DepthToSpace**: find a `DepthToSpace` node and read its `blocksize` attribute.
4. **ConvTranspose stride**: find a `ConvTranspose` node and read the max stride
   from its `strides` attribute.
5. **Assumed**: default to `1` (restoration/denoising model).

**Tiling detection**:

- `supported = true` when input spatial dims are dynamic (None values).
- `fixed_size` is set when both input H and W are static.
- `alignment` is detected by scanning `Reshape` nodes for power-of-2 shape
  values in the range `(1, 64]`. This is a heuristic for transformer-style
  window-partition patterns.

**Color space inference**: based on input channel count (1 → Grayscale,
3 → RGB, 4 → RGBA, other → Unknown).

**Op fingerprint**: count occurrences of each `op_type` across all graph nodes,
sort descending by count.

### 9.4 Protobuf parser (`inspect/proto.rs`)

A minimal hand-rolled protobuf reader that walks only the fields we need:

```
ModelProto
├── field 7  = graph (GraphProto)
│   ├── field 1  = node[] (NodeProto)
│   │   ├── field 4 = op_type
│   │   └── field 5 = attribute[] (AttributeProto)
│   │       ├── field 1 = name
│   │       ├── field 3 = i (int64)
│   │       └── field 7 = ints[] (repeated int64)
│   ├── field 11 = input[] (ValueInfoProto)
│   └── field 12 = output[] (ValueInfoProto)
├── field 8  = opset_import[] (OperatorSetIdProto → field 2 = version)
└── field 14 = metadata_props[] (StringStringEntryProto)
```

This avoids `prost` recursion limits and handles arbitrarily large ONNX files
with tensor initializer blobs. See § 16 for the full preserved source.

---

## 10. Upscaling Pipeline

### 10.1 Core API (`pipeline/mod.rs`)

```rust
/// A token for cooperative cancellation.
///
/// Clone-safe — all clones share the same underlying flag.
#[derive(Clone, Default)]
pub struct CancelToken(Arc<AtomicBool>);

impl CancelToken {
    pub fn new() -> Self {
        Self(Arc::new(AtomicBool::new(false)))
    }

    /// Signal cancellation.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::SeqCst);
    }

    /// Check if cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}
```

```rust
/// Options controlling the upscaling pipeline.
pub struct UpscaleOptions {
    /// Tile size in pixels. 0 = disable tiling (whole image at once).
    pub tile_size: u32,

    /// Overlap in pixels between adjacent tiles.
    pub tile_overlap: u32,

    /// Optional callback invoked after each tile completes.
    /// Arguments: (tiles_completed, total_tiles).
    pub on_tile_done: Option<Box<dyn Fn(usize, usize) + Send + Sync>>,

    /// Cancellation token — checked between tiles.
    pub cancel: CancelToken,
}

impl Default for UpscaleOptions {
    fn default() -> Self {
        Self {
            tile_size: crate::config::DEFAULT_TILE_SIZE,
            tile_overlap: crate::config::DEFAULT_TILE_OVERLAP,
            on_tile_done: None,
            cancel: CancelToken::new(),
        }
    }
}
```

```rust
/// Upscale a single image using the provided session and model info.
///
/// This is the primary library entry point for upscaling. It handles:
/// - Converting the input image to the appropriate tensor format.
/// - Computing the tile grid (or running whole-image if tiling disabled).
/// - Running inference on each tile.
/// - Blending overlapping tiles with Hann-window weights.
/// - Converting the output tensor back to a DynamicImage.
///
/// The caller is responsible for loading/saving images and displaying
/// progress — this function only reports via `options.on_tile_done`.
pub fn upscale_image(
    session: &SessionContext,
    model_info: &ModelInfo,
    input: &DynamicImage,
    options: &UpscaleOptions,
) -> Result<DynamicImage>;
```

**Internal flow of `upscale_image`**:

1. Get image dimensions `(W, H)`.
2. Convert `DynamicImage` to the appropriate channel count based on
   `model_info.input_channels` (see § 10.3).
3. Compute tile grid via `tiling::compute_tile_grid()`.
4. Allocate output canvas (`Array4<f32>`) and weight accumulator (`Array2<f32>`).
5. For each tile:
   a. Check `cancel.is_cancelled()` — bail if true.
   b. Crop the source region from the input.
   c. Convert to NCHW `f32` tensor.
   d. Apply padding if needed (mirror padding for quality).
   e. Convert to `f16` if `model_info.needs_fp16_input()`.
   f. Run inference via ORT session.
   g. Convert output back to `f32` if needed.
   h. Remove padding from output.
   i. Compute Hann-window blend weights for this tile.
   j. Accumulate tile into canvas with weights.
   k. Call `on_tile_done(completed, total)`.
6. Normalise canvas by dividing each pixel by its accumulated weight.
7. Clamp values to `[0, 1]`, convert to `u8`, build output `DynamicImage`.

### 10.2 Tensor conversion (`pipeline/tensor.rs`)

```rust
/// Convert a DynamicImage to an NCHW f32 tensor in [0, 1].
///
/// The image is converted to the appropriate channel count first
/// (RGB8, Luma8, or RGBA8), then laid out as (1, C, H, W).
pub fn image_to_tensor(image: &DynamicImage, channels: u32) -> Result<Array4<f32>>;

/// Convert an NCHW f32 tensor in [0, 1] back to a DynamicImage.
///
/// Values are clamped to [0, 1], scaled to [0, 255], and rounded.
/// The output DynamicImage variant matches the channel count:
/// - 1 channel → ImageLuma8
/// - 3 channels → ImageRgb8
/// - 4 channels → ImageRgba8
pub fn tensor_to_image(tensor: ArrayView4<f32>, channels: u32) -> Result<DynamicImage>;

/// Convert an f32 tensor to half::f16 for models requiring fp16 input.
pub fn tensor_f32_to_f16(tensor: &Array4<f32>) -> Result<Array4<half::f16>>;

/// Convert an f16 output tensor back to f32.
pub fn tensor_f16_to_f32(tensor: ArrayView4<half::f16>) -> Result<Array4<f32>>;
```

**Padding** — use **mirror (reflect) padding** rather than zero padding for
highest quality. Mirror padding extends the image by reflecting pixel values
at the boundaries, which produces far fewer artifacts at tile edges than
zero-filling.

```rust
/// Apply mirror padding to a tensor.
///
/// Extends each spatial dimension by reflecting pixel values.
/// Padding amounts: (left, top, right, bottom).
pub fn pad_tensor_mirror(
    tensor: &Array4<f32>,
    padding: Padding,
) -> Result<Array4<f32>>;

/// Remove padding from an output tensor (scale-adjusted).
///
/// Crops to the valid region after inference on a padded input.
pub fn crop_tensor(
    tensor: ArrayView4<f32>,
    padding: Padding,
    scale: u32,
) -> Result<Array4<f32>>;
```

### 10.3 Channel handling

The model's `input_channels` determines how the input image is prepared:

| Model channels | Input image conversion          | Notes                           |
|---------------|---------------------------------|---------------------------------|
| 1 (Grayscale) | `image.to_luma8()`              | Luminance extraction            |
| 3 (RGB)       | `image.to_rgb8()`               | Lossless if input is already RGB |
| 4 (RGBA)      | `image.to_rgba8()`              | Preserves alpha if present      |
| Other         | Error — unsupported             |                                 |

**Output reconstruction**: the output tensor's channel count
(`model_info.output_channels`) determines the `DynamicImage` variant:
- 1 → `DynamicImage::ImageLuma8`
- 3 → `DynamicImage::ImageRgb8`
- 4 → `DynamicImage::ImageRgba8`

**Input RGBA → Model RGB (3-channel)**: The alpha channel is **discarded**
during conversion to RGB8. The upscaled output will be RGB only. This is a
known v1 limitation — future versions may upscale alpha separately and
recombine.

### 10.4 Float16 handling

Detection via `ModelInfo::needs_fp16_input()` (checks `input_dtype == "float16"`).

**Workflow for fp16 models**:
1. Prepare the input tensor as `f32` (same pipeline as fp32 models).
2. Convert `f32 → f16` via `tensor_f32_to_f16()` right before inference.
3. Create ORT `Value` from the `f16` tensor.
4. Run inference.
5. Extract output — handle both `f16` and `f32` output tensors:
   - If output is `f16`: convert back to `f32` via `tensor_f16_to_f32()`.
   - If output is `f32`: use directly.
6. Continue with the `f32` output for blending and image reconstruction.

**No automatic fp32 promotion**: If a model requires fp16 input and the chosen
execution provider does not support fp16 tensors, the session creation will fail.
The provider fallback logic (§ 11) will try alternatives. If all providers fail
(including CPU), Sqwale surfaces a clear error. No silent promotion to fp32.

### 10.5 Tiling strategy (`pipeline/tiling.rs`)

#### Named types for clarity

```rust
/// A rectangle in pixel coordinates.
#[derive(Debug, Clone, Copy)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

/// Padding amounts for each side.
#[derive(Debug, Clone, Copy, Default)]
pub struct Padding {
    pub left: u32,
    pub top: u32,
    pub right: u32,
    pub bottom: u32,
}

impl Padding {
    pub fn is_zero(&self) -> bool {
        self.left == 0 && self.top == 0 && self.right == 0 && self.bottom == 0
    }
}

/// A single tile: where to crop from source, where to place in output,
/// and what padding to apply before inference.
#[derive(Debug, Clone)]
pub struct Tile {
    /// Region to crop from the source image.
    pub src: Rect,
    /// Region in the output image where this tile is placed (after scaling).
    pub dst: Rect,
    /// Padding to apply around the cropped region before inference.
    pub padding: Padding,
}
```

#### Tile grid computation

```rust
/// Compute the tile grid for the given image and model.
///
/// Handles three cases:
/// 1. Tiling disabled (tile_size == 0): single tile, whole image.
/// 2. Fixed-size model: sliding window with the model's required size.
/// 3. Dynamic model: sliding window with user tile size (aligned).
///
/// For dynamic models where the image is smaller than the tile size:
/// runs inference at the image's native size (no tiling, no padding).
pub fn compute_tile_grid(
    image_w: u32,
    image_h: u32,
    model_info: &ModelInfo,
    tile_size: u32,
    tile_overlap: u32,
) -> Result<Vec<Tile>>;
```

**Behavioral rules**:

| Condition                                          | Tile size used           | Padding     |
|----------------------------------------------------|--------------------------|-------------|
| `tile_size == 0` (tiling disabled)                 | Whole image              | None        |
| Fixed-size model                                   | `min(fixed_h, fixed_w)`  | Mirror pad  |
| Dynamic model, image ≤ tile size                   | Image native size        | None        |
| Dynamic model, image > tile size                   | `effective_tile_size()`  | Mirror pad  |
| `tile_size == 0` + fixed-size model                | Fixed size (overrides 0) | Mirror pad  |

**Step size**: `step = tile_size - 2 * overlap`. If `step <= 0`, return an
error (tile too small for the overlap).

**Edge tiles**: The last tile in each row/column may be smaller than the tile
size. For dynamic models, this is fine (model accepts any size). For fixed-size
models, the tile is padded to the required size.

### 10.6 Blending

Use **Hann window** (raised cosine) blending for maximum quality. Each tile is
assigned a 2D weight map where edge pixels taper smoothly to zero within the
overlap region.

```rust
/// Generate Hann-window blend weights for a tile.
///
/// Pixels within `overlap` distance from any edge taper smoothly
/// from 1.0 to 0.0 using the Hann function:
///   w(t) = 0.5 * (1 - cos(π * t))
/// where t ∈ [0, 1] is the normalized distance from the edge.
///
/// Interior pixels have weight 1.0.
pub fn blend_weights(tile_w: u32, tile_h: u32, overlap: u32) -> Array2<f32>;
```

The blending process:
1. For each tile, compute `blend_weights(tile_w, tile_h, overlap * scale)`.
2. For each pixel in the tile, multiply the pixel value by its blend weight.
3. Accumulate weighted values and weights into the output canvas.
4. After all tiles: divide each pixel by its accumulated weight.

This produces seamless, artifact-free blending across tile boundaries.

---

## 11. Session Management

### 11.1 Types (`session/mod.rs`)

```rust
/// An ORT session paired with model metadata and the provider that was used.
pub struct SessionContext {
    /// The ORT inference session. Private to the crate — only the pipeline
    /// module accesses it directly.
    pub(crate) session: Session,

    /// Metadata extracted from the model (scale, channels, tiling, etc.).
    pub model_info: ModelInfo,

    /// The execution provider that was actually used for this session.
    pub provider_used: ProviderSelection,
}
```

```rust
/// Load an ONNX model: inspect it and create an inference session.
///
/// 1. Calls `inspect_model(path)` to extract metadata.
/// 2. Creates an ORT session with the selected provider.
/// 3. Returns the session context with metadata and actual provider.
pub fn load_model(path: &Path, provider: ProviderSelection) -> Result<SessionContext>;
```

**Session reuse**: In batch mode, the CLI calls `load_model` once and reuses
the `SessionContext` for all images. No per-image session recreation.

### 11.2 Provider selection (`session/provider.rs`)

```rust
/// Execution provider selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ProviderSelection {
    /// Let ORT choose the best available device.
    #[default]
    Auto,
    Cpu,
    Cuda,
    TensorRT,
    DirectML,
    CoreML,
    XNNPack,
}

impl ProviderSelection {
    pub fn name(self) -> &'static str {
        match self {
            Self::Auto     => "auto",
            Self::Cpu      => "CPU",
            Self::Cuda     => "CUDA",
            Self::TensorRT => "TensorRT",
            Self::DirectML => "DirectML",
            Self::CoreML   => "CoreML",
            Self::XNNPack  => "XNNPACK",
        }
    }
}

impl FromStr for ProviderSelection {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "auto"              => Ok(Self::Auto),
            "cpu"               => Ok(Self::Cpu),
            "cuda"              => Ok(Self::Cuda),
            "tensorrt" | "trt"  => Ok(Self::TensorRT),
            "directml" | "dml"  => Ok(Self::DirectML),
            "coreml"            => Ok(Self::CoreML),
            "xnnpack"           => Ok(Self::XNNPack),
            other => anyhow::bail!(
                "Unknown provider '{other}'. Valid: auto, cpu, cuda, tensorrt, directml, coreml, xnnpack"
            ),
        }
    }
}
```

### 11.3 Provider support per platform

| Provider   | Windows | Linux | macOS |
|------------|---------|-------|-------|
| CPU        | ✓       | ✓     | ✓     |
| CUDA       | ✓       | ✓     |       |
| TensorRT   | ✓       | ✓     |       |
| DirectML   | ✓       |       |       |
| CoreML     |         |       | ✓     |
| XNNPACK    |         | ✓     |       |

Use `#[cfg(target_os = "...")]` to gate EP construction. If a user requests a
provider not available on their platform, surface a clear error message listing
valid providers for their OS.

### 11.4 Auto selection

For `--provider auto`, use ORT's built-in auto-device selection:

```rust
SessionBuilder::new()?
    .with_auto_device(AutoDevicePolicy::MaxPerformance)?
    .commit_from_file(path)?
```

This delegates provider selection entirely to ORT, which picks the best
available EP on the system.

### 11.5 Explicit provider with CPU fallback

When the user specifies a provider explicitly (e.g. `--provider cuda`):

1. Attempt to create a session with the requested EP.
2. If session creation **fails**:
   a. Print a styled warning: `⚠ CUDA provider failed (...), falling back to CPU`
   b. Attempt to create a session with CPU.
   c. If CPU also fails → surface error and exit.
3. Record the **actually used** provider in `SessionContext::provider_used`.

The warning is printed to **stdout** using the standard warning style (yellow
`⚠` symbol, dimmed details).

### 11.6 EP construction

Each EP is constructed with default settings:

```rust
fn make_ep(provider: ProviderSelection) -> Result<ExecutionProviderDispatch> {
    match provider {
        ProviderSelection::Cpu | ProviderSelection::Auto => {
            Ok(ort::ep::CPU::default().build())
        }
        ProviderSelection::Cuda => {
            #[cfg(any(target_os = "windows", target_os = "linux"))]
            { Ok(ort::ep::CUDA::default().build()) }
            #[cfg(not(any(target_os = "windows", target_os = "linux")))]
            { anyhow::bail!("CUDA is not available on this platform") }
        }
        ProviderSelection::TensorRT => {
            #[cfg(any(target_os = "windows", target_os = "linux"))]
            { Ok(ort::ep::TensorRT::default().build()) }
            #[cfg(not(any(target_os = "windows", target_os = "linux")))]
            { anyhow::bail!("TensorRT is not available on this platform") }
        }
        ProviderSelection::DirectML => {
            #[cfg(target_os = "windows")]
            { Ok(ort::ep::DirectML::default().build()) }
            #[cfg(not(target_os = "windows"))]
            { anyhow::bail!("DirectML is only available on Windows") }
        }
        ProviderSelection::CoreML => {
            #[cfg(target_os = "macos")]
            { Ok(ort::ep::CoreML::default().build()) }
            #[cfg(not(target_os = "macos"))]
            { anyhow::bail!("CoreML is only available on macOS") }
        }
        ProviderSelection::XNNPack => {
            #[cfg(target_os = "linux")]
            { Ok(ort::ep::XNNPACK::default().build()) }
            #[cfg(not(target_os = "linux"))]
            { anyhow::bail!("XNNPACK is only available on Linux") }
        }
    }
}
```

---

## 12. Image I/O (`imageio.rs`)

### 12.1 Public API

```rust
/// Load an image from disk.
pub fn load_image(path: &Path) -> Result<DynamicImage>;

/// Save an image to disk, inferring format from the file extension.
pub fn save_image(img: &DynamicImage, path: &Path) -> Result<()>;

/// Derive a default output path: {dir}/{stem}_{scale}x.{ext}.
pub fn default_output_path(input: &Path, scale: u32) -> PathBuf;

/// Validate that the output path's extension matches the input's.
///
/// Returns Ok if extensions match (case-insensitive) or if the output
/// has no extension (treated as a directory). Errors on mismatch because
/// format conversion is not supported.
pub fn check_extension_match(input: &Path, output: &Path) -> Result<()>;
```

### 12.2 Supported formats

Via the `image` crate features enabled in Cargo.toml:

| Format | Decode | Encode |
|--------|--------|--------|
| PNG    | ✓      | ✓      |
| JPEG   | ✓      | ✓      |
| WebP   | ✓      | ✓      |
| GIF    | ✓      | ✓      |
| TIFF   | ✓      | ✓      |
| BMP    | ✓      | ✓      |
| ICO    | ✓      | ✓      |
| PNM    | ✓      | ✓      |
| QOI    | ✓      | ✓      |
| HDR    | ✓      | ✓      |

Output format is **always** the same as input format. Mismatched extensions
are a hard error.

---

## 13. Error Handling

### 13.1 Strategy

Use `anyhow::Result` throughout both library and CLI for ergonomic error
handling with context. The `.context()` and `.with_context()` methods provide
meaningful error chains.

Custom error types (via `thiserror`) are a future enhancement for the library
API when stable, typed error matching is needed by downstream consumers.

### 13.2 Error context guidelines

Every fallible operation should include context describing **what** was being
attempted:

```rust
// Good: context explains the operation
fs::read(path).with_context(|| format!("Failed to read model: {}", path.display()))?;

// Bad: no context — raw IO error is confusing
fs::read(path)?;
```

### 13.3 Error display

Errors displayed to the user use the standard style:

```
✗ Failed to decode image: unexpected end of file at offset 48221
```

- `✗` in `.red().bold()`.
- Error summary in `.white()`.
- Detail/cause in `.dimmed()`.

### 13.4 Forbidden patterns

- **No `unwrap()` or `expect()` in library code** unless the operation
  is provably infallible and documented with a comment explaining why.
- **No `panic!()` for runtime conditions** — use `Result` and propagate.
- **No silent error swallowing** — all errors must be surfaced or logged.

---

## 14. Progress and Logging

### 14.1 Tracing setup

In `main.rs`, initialize the tracing subscriber with indicatif integration:

```rust
use tracing_indicatif::IndicatifLayer;
use tracing_subscriber::prelude::*;
use tracing_subscriber::EnvFilter;

fn init_tracing() {
    let indicatif_layer = IndicatifLayer::new();

    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("sqwale=warn"));

    tracing_subscriber::registry()
        .with(filter)
        .with(
            tracing_subscriber::fmt::layer()
                .with_writer(indicatif_layer.get_stderr_writer()),
        )
        .with(indicatif_layer)
        .init();
}
```

This ensures that any `tracing::warn!()`, `tracing::info!()`, etc. messages
are written through indicatif's stderr writer, which coordinates with active
progress bars. No visual corruption.

### 14.2 Progress bars

Progress bars are managed via `indicatif` directly (not through tracing spans).
The CLI command handlers create and manage progress bars explicitly:

**Inspect batch**: A single `ProgressBar` showing `N/M models` on stderr.

**Upscale single**: A single `ProgressBar` for tile progress on stderr.

**Upscale batch**: Two progress bars via `MultiProgress`:
- Top: tile progress for the current image.
- Bottom: overall image progress (`N/M images`).

Progress updates come from the `on_tile_done` callback in `UpscaleOptions`.

### 14.3 Spinners

For operations that take a noticeable amount of time but have no measurable
progress (e.g. model/session loading), show a spinner:

```rust
pub fn with_spinner<T>(label: &str, f: impl FnOnce() -> T) -> T {
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .tick_strings(SPINNER_FRAMES)
            .template("{spinner:.cyan} {msg}")
            .unwrap(),
    );
    pb.set_message(label.to_owned());
    pb.enable_steady_tick(Duration::from_millis(SPINNER_TICK_MS));
    let result = f();
    pb.finish_and_clear();
    result
}
```

### 14.4 Non-interactive mode

When `should_show_progress()` returns false (CI, non-TTY, NO_COLOR):

- No progress bars or spinners are displayed.
- Simple one-line status messages are printed to stdout instead:
  - `"Inspecting model 1/5: model_name.onnx"`
  - `"Processing image 1/5: input.png"`

---

## 15. Cancellation (Ctrl+C)

### 15.1 Setup

In `main.rs` or the upscale command handler, set up a `ctrlc` handler that
manages a `CancelToken` and an interrupt counter:

```rust
let cancel = CancelToken::new();
let interrupt_count = Arc::new(AtomicUsize::new(0));

{
    let cancel = cancel.clone();
    let count = interrupt_count.clone();
    ctrlc::set_handler(move || {
        let prev = count.fetch_add(1, Ordering::SeqCst);
        if prev == 0 {
            // First Ctrl+C: request graceful cancellation.
            cancel.cancel();
        } else {
            // Second Ctrl+C: force immediate exit.
            std::process::exit(1);
        }
    })?;
}
```

### 15.2 Single-item mode

For single-image upscale or single-model inspect:

- First Ctrl+C → immediate process exit (via the cancel token causing
  `upscale_image` to bail, or the handler calling `process::exit`).

### 15.3 Batch mode

For batch operations (multiple images or multiple models):

- **First Ctrl+C**:
  - Sets the cancel token.
  - Current tile/model finishes processing.
  - Remaining items are skipped.
  - A warning is displayed: `⚠ Interrupted — finishing current image, skipping remaining`
  - Summary shows cancelled status with skip count.

- **Second Ctrl+C**:
  - Forces immediate process exit, even mid-tile.

### 15.4 Cancel token integration

The `CancelToken` is checked:
- Between tiles in `upscale_image()` — if cancelled, return an error.
- Between images in the batch loop — if cancelled, break the loop.
- Between models in batch inspect — if cancelled, break the loop.

---

## 16. Testing

### 16.1 Strategy

Structural tests that verify the correctness of types, parsing, and detection
logic without running actual inference.

### 16.2 Test model

Use `models/2x_OpenProteus_Compact_i2_70K_fp32.onnx` as the primary test model:
- Small (70K parameters, compact architecture).
- fp32 (simple tensor handling).
- 2x scale (verifiable).
- Dynamic spatial dims (supports tiling).

### 16.3 Unit tests

Located in `#[cfg(test)] mod tests` within each source file.

**`inspect/proto.rs`**:
- Test `read_varint_at` with known byte sequences.
- Test `iter_fields` with a minimal hand-crafted protobuf message.
- Test `elem_type_name` for all known type IDs.
- Test `bytes_as_i64` and `bytes_as_packed_i64s`.

**`inspect/detect.rs`**:
- Test `infer_color_space` for all channel counts (1, 3, 4, other).
- Test `TileInfo::effective_tile_size` with various alignment values.
- Test `compute_op_fingerprint` with known node lists.

**`pipeline/tiling.rs`**:
- Test `compute_tile_grid` for:
  - Single tile (image smaller than tile size, dynamic model).
  - Multiple tiles with overlap.
  - Fixed-size model tiling.
  - Tiling disabled (`tile_size == 0`).
  - Alignment rounding.
- Test `blend_weights` produces correct dimensions and value range [0, 1].
- Test that blend weights at edges approach 0 and at center are 1.

**`imageio.rs`**:
- Test `default_output_path` for various inputs.
- Test `check_extension_match` for matching, mismatching, and no-extension cases.

**`session/provider.rs`**:
- Test `ProviderSelection::from_str` for all valid names and aliases.
- Test error on invalid provider name.

### 16.4 Integration tests (`tests/inspect.rs`)

```rust
use std::path::Path;
use sqwale::inspect::inspect_model;

#[test]
fn inspect_compact_model() {
    let path = Path::new("models/2x_OpenProteus_Compact_i2_70K_fp32.onnx");
    if !path.exists() {
        // Skip if models aren't available (e.g. CI without model artifacts).
        return;
    }
    let info = inspect_model(path).expect("inspection should succeed");

    assert_eq!(info.scale, 2);
    assert_eq!(info.input_channels, 3);
    assert_eq!(info.output_channels, 3);
    assert_eq!(info.input_dtype, "float32");
    assert!(info.opset > 0);
    assert!(!info.op_fingerprint.is_empty());
}
```

### 16.5 Running tests

```bash
cargo test              # All tests
cargo test --lib        # Unit tests only
cargo test --test '*'   # Integration tests only
```

---

## 17. Preserved Code — Inspection Backend

The following three files constitute the working inspection backend. They should
be placed in the new project **as-is** (with only import path adjustments if
the module structure changes).

### 17.1 `src/inspect/proto.rs`

```rust
//! Minimal hand-rolled protobuf reader for ONNX `ModelProto`.
//!
//! We walk only 4 levels deep to extract:
//!
//! ```text
//! ModelProto (field 7 = graph, field 8 = opset_import, field 14 = metadata_props)
//! └── GraphProto (field 1 = node[], field 11 = input[], field 12 = output[])
//!     └── NodeProto (field 4 = op_type, field 5 = attribute[])
//!         └── AttributeProto (field 1 = name, field 3 = i, field 7 = ints[])
//! ```
//!
//! This avoids `prost` recursion limits and works on any valid ONNX file
//! regardless of large tensor-initialiser blobs.

use std::collections::HashMap;

// ── Public Types ───────────────────────────────────────────────────────────

/// A single graph node with only the attributes we care about.
#[derive(Debug, Clone)]
pub struct NodeInfo {
    pub op_type: String,
    /// `(attr_name, single_int64)` for integer attributes.
    pub int_attrs: Vec<(String, i64)>,
    /// `(attr_name, values)` for packed repeated int64 attributes.
    pub ints_attrs: Vec<(String, Vec<i64>)>,
}

/// I/O tensor info `(channels, (opt_height, opt_width), dtype_string)`.
pub type IoInfo = (u32, (Option<u64>, Option<u64>), String);

// ── Top-level Extraction Helpers ───────────────────────────────────────────

/// Extract all graph nodes from raw model bytes.
pub fn extract_nodes(file_bytes: &[u8]) -> Vec<NodeInfo> {
    let graph = graph_bytes(file_bytes);
    iter_fields(&graph)
        .into_iter()
        .filter(|(f, w, _)| *f == 1 && *w == 2)
        .filter_map(|(_, _, b)| parse_node(&b))
        .collect()
}

/// Extract the maximum opset version from raw model bytes.
pub fn extract_opset(file_bytes: &[u8]) -> u64 {
    // ModelProto field 8 = opset_import[] → OperatorSetIdProto field 2 = version
    iter_fields(file_bytes)
        .into_iter()
        .filter(|(f, w, _)| *f == 8 && *w == 2)
        .filter_map(|(_, _, b)| {
            iter_fields(&b)
                .into_iter()
                .find(|(f, w, _)| *f == 2 && *w == 0)
                .map(|(_, _, vb)| bytes_as_i64(&vb) as u64)
        })
        .max()
        .unwrap_or(0)
}

/// Extract `metadata_props` key→value pairs.
pub fn extract_metadata(file_bytes: &[u8]) -> HashMap<String, String> {
    // ModelProto field 14 = metadata_props[] (StringStringEntryProto)
    iter_fields(file_bytes)
        .into_iter()
        .filter(|(f, w, _)| *f == 14 && *w == 2)
        .filter_map(|(_, _, b)| {
            let kv = iter_fields(&b);
            let key = kv
                .iter()
                .find(|(f, w, _)| *f == 1 && *w == 2)
                .and_then(|(_, _, b)| std::str::from_utf8(b).ok().map(str::to_owned))?;
            let val = kv
                .iter()
                .find(|(f, w, _)| *f == 2 && *w == 2)
                .and_then(|(_, _, b)| std::str::from_utf8(b).ok().map(str::to_owned))
                .unwrap_or_default();
            Some((key, val))
        })
        .collect()
}

/// Extract first input and first output `IoInfo` from raw model bytes.
pub fn extract_io_from_proto(file_bytes: &[u8]) -> (Option<IoInfo>, Option<IoInfo>) {
    let gb = graph_bytes(file_bytes);
    let graph_fields = iter_fields(&gb);

    // GraphProto field 11 = input[], field 12 = output[]
    let input = graph_fields
        .iter()
        .find(|(f, w, _)| *f == 11 && *w == 2)
        .and_then(|(_, _, b)| parse_value_info(b));

    let output = graph_fields
        .iter()
        .find(|(f, w, _)| *f == 12 && *w == 2)
        .and_then(|(_, _, b)| parse_value_info(b));

    (input, output)
}

// ── Internal Proto Walking ─────────────────────────────────────────────────

fn graph_bytes(file_bytes: &[u8]) -> Vec<u8> {
    // ModelProto field 7 = graph (GraphProto)
    iter_fields(file_bytes)
        .into_iter()
        .find(|(f, w, _)| *f == 7 && *w == 2)
        .map(|(_, _, b)| b)
        .unwrap_or_default()
}

fn iter_fields(buf: &[u8]) -> Vec<(u32, u8, Vec<u8>)> {
    let mut out = Vec::new();
    let mut pos = 0;

    while pos < buf.len() {
        let Some((tag, consumed)) = read_varint_at(buf, pos) else {
            break;
        };
        pos += consumed;

        let field_num = (tag >> 3) as u32;
        let wire_type = (tag & 0x7) as u8;

        match wire_type {
            0 => {
                let Some((val, consumed)) = read_varint_at(buf, pos) else {
                    break;
                };
                pos += consumed;
                out.push((field_num, wire_type, val.to_le_bytes().to_vec()));
            }
            1 => {
                if pos + 8 > buf.len() {
                    break;
                }
                out.push((field_num, wire_type, buf[pos..pos + 8].to_vec()));
                pos += 8;
            }
            2 => {
                let Some((len, consumed)) = read_varint_at(buf, pos) else {
                    break;
                };
                pos += consumed;
                let len = len as usize;
                if pos + len > buf.len() {
                    break;
                }
                out.push((field_num, wire_type, buf[pos..pos + len].to_vec()));
                pos += len;
            }
            5 => {
                if pos + 4 > buf.len() {
                    break;
                }
                out.push((field_num, wire_type, buf[pos..pos + 4].to_vec()));
                pos += 4;
            }
            _ => break,
        }
    }

    out
}

fn read_varint_at(buf: &[u8], mut pos: usize) -> Option<(u64, usize)> {
    let mut result: u64 = 0;
    let mut shift = 0u32;
    let start = pos;
    loop {
        if pos >= buf.len() || shift >= 64 {
            return None;
        }
        let b = buf[pos];
        pos += 1;
        result |= ((b & 0x7F) as u64) << shift;
        if b & 0x80 == 0 {
            break;
        }
        shift += 7;
    }
    Some((result, pos - start))
}

fn bytes_as_i64(bytes: &[u8]) -> i64 {
    let mut arr = [0u8; 8];
    let n = bytes.len().min(8);
    arr[..n].copy_from_slice(&bytes[..n]);
    i64::from_le_bytes(arr)
}

fn bytes_as_packed_i64s(bytes: &[u8]) -> Vec<i64> {
    let mut out = Vec::new();
    let mut pos = 0;
    while pos < bytes.len() {
        if let Some((val, consumed)) = read_varint_at(bytes, pos) {
            out.push(val as i64);
            pos += consumed;
        } else {
            break;
        }
    }
    out
}

fn parse_node(bytes: &[u8]) -> Option<NodeInfo> {
    let fields = iter_fields(bytes);

    let op_type = fields
        .iter()
        .find(|(f, w, _)| *f == 4 && *w == 2)
        .and_then(|(_, _, b)| std::str::from_utf8(b).ok().map(str::to_owned))?;

    let mut int_attrs: Vec<(String, i64)> = Vec::new();
    let mut ints_attrs: Vec<(String, Vec<i64>)> = Vec::new();

    for (field_num, wire_type, bytes) in &fields {
        if *field_num != 5 || *wire_type != 2 {
            continue;
        }
        let attr_fields = iter_fields(bytes);
        let name = attr_fields
            .iter()
            .find(|(f, w, _)| *f == 1 && *w == 2)
            .and_then(|(_, _, b)| std::str::from_utf8(b).ok().map(str::to_owned))
            .unwrap_or_default();

        for (af, aw, ab) in &attr_fields {
            match (af, aw) {
                (3, 0) => int_attrs.push((name.clone(), bytes_as_i64(ab))),
                (7, 2) => ints_attrs.push((name.clone(), bytes_as_packed_i64s(ab))),
                _ => {}
            }
        }
    }

    Some(NodeInfo {
        op_type,
        int_attrs,
        ints_attrs,
    })
}

fn parse_value_info(bytes: &[u8]) -> Option<IoInfo> {
    // ValueInfoProto field 2 = type (TypeProto)
    let type_bytes = iter_fields(bytes)
        .into_iter()
        .find(|(f, w, _)| *f == 2 && *w == 2)
        .map(|(_, _, b)| b)?;

    // TypeProto field 1 = tensor_type
    let tensor_bytes = iter_fields(&type_bytes)
        .into_iter()
        .find(|(f, w, _)| *f == 1 && *w == 2)
        .map(|(_, _, b)| b)?;

    let tensor_fields = iter_fields(&tensor_bytes);

    let elem_type = tensor_fields
        .iter()
        .find(|(f, w, _)| *f == 1 && *w == 0)
        .map(|(_, _, b)| bytes_as_i64(b) as i32)
        .unwrap_or(0);

    let dtype = elem_type_name(elem_type).to_owned();

    let shape_bytes = tensor_fields
        .iter()
        .find(|(f, w, _)| *f == 2 && *w == 2)
        .map(|(_, _, b)| b.clone())?;

    let dims: Vec<Option<u64>> = iter_fields(&shape_bytes)
        .into_iter()
        .filter(|(f, w, _)| *f == 1 && *w == 2)
        .map(|(_, _, b)| {
            iter_fields(&b)
                .into_iter()
                .find(|(f, w, _)| *f == 1 && *w == 0)
                .map(|(_, _, vb)| bytes_as_i64(&vb) as u64)
        })
        .collect();

    if dims.len() < 4 {
        return None;
    }

    let channels: u32 = dims[1].unwrap_or(0).try_into().ok()?;
    if channels == 0 {
        return None;
    }

    let h = dims[2].filter(|&v| v > 0);
    let w = dims[3].filter(|&v| v > 0);

    Some((channels, (h, w), dtype))
}

fn elem_type_name(t: i32) -> &'static str {
    match t {
        1 => "float32",
        2 => "uint8",
        3 => "int8",
        5 => "int32",
        6 => "int64",
        10 => "float16",
        11 => "float64",
        16 => "bfloat16",
        _ => "unknown",
    }
}
```

### 17.2 `src/inspect/mod.rs`

```rust
//! Model inspection: public types and inspection API.

mod detect;
pub(crate) mod proto;

pub use detect::inspect_model;

// ── Color Space ────────────────────────────────────────────────────────────

/// The color space inferred from the model's channel count.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ColorSpace {
    Grayscale,
    #[default]
    Rgb,
    Rgba,
    Unknown(u32),
}

impl std::fmt::Display for ColorSpace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Grayscale => write!(f, "Grayscale"),
            Self::Rgb => write!(f, "RGB"),
            Self::Rgba => write!(f, "RGBA"),
            Self::Unknown(n) => write!(f, "Unknown ({n} channels)"),
        }
    }
}

impl ColorSpace {
    /// Number of channels.
    pub fn channels(&self) -> u32 {
        match self {
            Self::Grayscale => 1,
            Self::Rgb => 3,
            Self::Rgba => 4,
            Self::Unknown(n) => *n,
        }
    }
}

// ── Scale Source ───────────────────────────────────────────────────────────

/// How the upscale factor was detected.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ScaleSource {
    Metadata,
    StaticShapeRatio,
    DepthToSpace,
    ConvTransposeStride,
    #[default]
    Assumed,
}

impl std::fmt::Display for ScaleSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Metadata => write!(f, "metadata_props"),
            Self::StaticShapeRatio => write!(f, "static shape ratio"),
            Self::DepthToSpace => write!(f, "DepthToSpace (PixelShuffle)"),
            Self::ConvTransposeStride => write!(f, "ConvTranspose stride"),
            Self::Assumed => write!(f, "assumed (no upscale op found)"),
        }
    }
}

// ── Tile Requirements ──────────────────────────────────────────────────────

/// Tiling constraints extracted from the model graph.
#[derive(Debug, Clone, Default)]
pub struct TileInfo {
    /// Whether the model supports tiling (dynamic spatial dims).
    pub supported: bool,
    /// Required alignment for spatial dims (e.g. 8, 16, 32).
    pub alignment: Option<u32>,
    /// Fully-static required input size (height, width).
    pub fixed_size: Option<(u64, u64)>,
}

impl TileInfo {
    /// Returns the tile size to use, given a user preference.
    ///
    /// * Fixed size → use min(h, w), ignore user preference.
    /// * Alignment → round user preference up to nearest multiple.
    /// * Otherwise → use user preference as-is.
    pub fn effective_tile_size(&self, user_pref: u32) -> u32 {
        if let Some((h, w)) = self.fixed_size {
            h.min(w) as u32
        } else if let Some(align) = self.alignment {
            let r = user_pref % align;
            if r == 0 { user_pref } else { user_pref + (align - r) }
        } else {
            user_pref
        }
    }
}

// ── Model Metadata ─────────────────────────────────────────────────────────

/// All metadata extracted from an ONNX model without running inference.
#[derive(Debug, Clone, Default)]
pub struct ModelInfo {
    /// Upscale factor (1 for restoration / denoising models).
    pub scale: u32,
    /// How the scale was determined.
    pub scale_source: ScaleSource,
    /// Input color space derived from channel count.
    pub color_space: ColorSpace,
    /// Number of input channels.
    pub input_channels: u32,
    /// Number of output channels.
    pub output_channels: u32,
    /// Tiling constraints.
    pub tile: TileInfo,
    /// Input element type (e.g. "float32", "float16").
    pub input_dtype: String,
    /// Output element type.
    pub output_dtype: String,
    /// Maximum opset version in use.
    pub opset: u64,
    /// Op-type histogram sorted by frequency descending.
    pub op_fingerprint: Vec<(String, usize)>,
}

impl ModelInfo {
    /// Returns true when the model expects half-precision input.
    pub fn needs_fp16_input(&self) -> bool {
        self.input_dtype == "float16"
    }
}
```

### 17.3 `src/inspect/detect.rs`

```rust
//! Model detection logic: scale, channels, tiling, and color space inference.

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

use super::proto::{self, IoInfo, NodeInfo};
use super::{ColorSpace, ModelInfo, ScaleSource, TileInfo};

type IoParseResult = (u32, (Option<u64>, Option<u64>), String);

/// Inspect an ONNX model and extract all metadata without creating
/// an inference session.
pub fn inspect_model(path: &Path) -> Result<ModelInfo> {
    let file_bytes =
        fs::read(path).with_context(|| format!("Failed to read model: {}", path.display()))?;

    let nodes = proto::extract_nodes(&file_bytes);
    let opset = proto::extract_opset(&file_bytes);
    let metadata = proto::extract_metadata(&file_bytes);
    let (input_io, output_io) = proto::extract_io_from_proto(&file_bytes);

    let (input_channels, input_spatial, input_dtype) = parse_io(input_io)?;
    let (output_channels, output_spatial, output_dtype) = parse_io(output_io)
        .unwrap_or_else(|_| (input_channels, (None, None), input_dtype.clone()));

    let (scale, scale_source) =
        detect_scale(&nodes, &metadata, input_spatial, output_spatial);
    let tile = detect_tiling(&nodes, input_spatial);
    let color_space = infer_color_space(input_channels);
    let op_fingerprint = compute_op_fingerprint(&nodes);

    Ok(ModelInfo {
        scale,
        scale_source,
        color_space,
        input_channels,
        output_channels,
        tile,
        input_dtype,
        output_dtype,
        opset,
        op_fingerprint,
    })
}

fn parse_io(io: Option<IoInfo>) -> Result<IoParseResult> {
    let (channels, spatial, dtype) =
        io.ok_or_else(|| anyhow::anyhow!("Missing I/O info"))?;
    Ok((channels, spatial, dtype))
}

fn infer_color_space(channels: u32) -> ColorSpace {
    match channels {
        1 => ColorSpace::Grayscale,
        3 => ColorSpace::Rgb,
        4 => ColorSpace::Rgba,
        n => ColorSpace::Unknown(n),
    }
}

fn detect_scale(
    nodes: &[NodeInfo],
    metadata: &HashMap<String, String>,
    input_spatial: (Option<u64>, Option<u64>),
    output_spatial: (Option<u64>, Option<u64>),
) -> (u32, ScaleSource) {
    if let Some(scale_str) = metadata.get("scale") {
        if let Ok(scale) = scale_str.parse::<u32>() {
            return (scale, ScaleSource::Metadata);
        }
    }

    if let (Some(in_h), Some(out_h)) = (input_spatial.0, output_spatial.0) {
        if in_h > 0 && out_h > 0 && out_h >= in_h {
            let ratio = (out_h / in_h) as u32;
            if ratio > 1 {
                return (ratio, ScaleSource::StaticShapeRatio);
            }
        }
    }

    for node in nodes {
        if node.op_type == "DepthToSpace" {
            if let Some((_, scale)) =
                node.int_attrs.iter().find(|(name, _)| name == "blocksize")
            {
                return (*scale as u32, ScaleSource::DepthToSpace);
            }
        }
        if node.op_type == "ConvTranspose" {
            if let Some((_, strides)) =
                node.ints_attrs.iter().find(|(name, _)| name == "strides")
            {
                if let Some(&stride) = strides.iter().max() {
                    if stride > 1 {
                        return (stride as u32, ScaleSource::ConvTransposeStride);
                    }
                }
            }
        }
    }

    (1, ScaleSource::Assumed)
}

fn detect_tiling(
    nodes: &[NodeInfo],
    input_spatial: (Option<u64>, Option<u64>),
) -> TileInfo {
    let has_dynamic_spatial =
        input_spatial.0.is_none() || input_spatial.1.is_none();

    let fixed_size = if let (Some(h), Some(w)) = input_spatial {
        Some((h, w))
    } else {
        None
    };

    let alignment = detect_alignment(nodes);

    TileInfo {
        supported: has_dynamic_spatial,
        alignment,
        fixed_size,
    }
}

fn detect_alignment(nodes: &[NodeInfo]) -> Option<u32> {
    for node in nodes {
        if node.op_type == "Reshape" {
            if let Some((_, values)) =
                node.ints_attrs.iter().find(|(name, _)| name == "shape")
            {
                for &val in values {
                    if val > 1 && val <= 64 && (val & (val - 1)) == 0 {
                        return Some(val as u32);
                    }
                }
            }
        }
    }
    None
}

fn compute_op_fingerprint(nodes: &[NodeInfo]) -> Vec<(String, usize)> {
    let mut counts: HashMap<String, usize> = HashMap::new();
    for node in nodes {
        *counts.entry(node.op_type.clone()).or_insert(0) += 1;
    }
    let mut pairs: Vec<_> = counts.into_iter().collect();
    pairs.sort_by(|a, b| b.1.cmp(&a.1));
    pairs
}
```

---

## 18. Module Dependency Graph

```
main.rs
  └── cli/
        ├── mod.rs
        ├── inspect.rs ──→ sqwale::inspect
        ├── upscale.rs ──→ sqwale::pipeline, sqwale::session, sqwale::imageio
        └── output.rs  ──→ sqwale::inspect (types for formatting)

lib.rs
  ├── config.rs        (no deps)
  ├── imageio.rs       (no deps)
  ├── inspect/
  │     ├── mod.rs     (no deps)
  │     ├── detect.rs  → proto, mod types
  │     └── proto.rs   (no deps)
  ├── pipeline/
  │     ├── mod.rs     → session, inspect, tiling, tensor
  │     ├── tensor.rs  → inspect (ModelInfo)
  │     └── tiling.rs  → inspect (ModelInfo, TileInfo)
  └── session/
        ├── mod.rs     → inspect, provider
        └── provider.rs (no deps)
```

**Key invariant**: The library modules (`inspect`, `pipeline`, `session`,
`imageio`, `config`) never reference `cli/`. The `cli/` module depends on
library modules but not vice versa.

---

## 19. `main.rs` Skeleton

```rust
//! Sqwale CLI entry point.

mod cli;

use anyhow::Result;
use clap::Parser;

fn main() -> Result<()> {
    // Suppress ORT diagnostic output unless the user has set a preference.
    if std::env::var("ORT_LOG_SEVERITY_LEVEL").is_err() {
        // SAFETY: Called before any threads are spawned and before ORT init.
        unsafe { std::env::set_var("ORT_LOG_SEVERITY_LEVEL", "3") };
    }

    // Disable colored output in CI environments.
    if !cli::output::should_use_color() {
        colored::control::set_override(false);
    }

    // Initialize tracing with indicatif integration.
    cli::output::init_tracing();

    // Parse CLI arguments and dispatch.
    let args = cli::Cli::parse();

    match &args.command {
        cli::Commands::Inspect { pattern } => {
            cli::inspect::run(pattern)?;
        }
        cli::Commands::Upscale { input, model, output } => {
            cli::upscale::run(input, model, output.as_deref(), &args)?;
        }
    }

    Ok(())
}
```

**Note**: `std::env::set_var` requires `unsafe` since Rust 1.66. The safety
justification is that it is called before any threads are spawned and before
ORT initializes. An alternative is to use the `ort::init()` builder API if it
exposes log-level configuration.

---

## 20. `lib.rs` Skeleton

```rust
//! sqwale — ONNX super-resolution inference library.
//!
//! Provides model inspection and image upscaling via ONNX Runtime.

// ── Public modules ──────────────────────────────────────────────────────────
pub mod config;
pub mod imageio;
pub mod inspect;
pub mod pipeline;
pub mod session;

// ── Convenience re-exports ──────────────────────────────────────────────────
pub use inspect::{inspect_model, ColorSpace, ModelInfo, ScaleSource, TileInfo};
pub use pipeline::{upscale_image, CancelToken, UpscaleOptions};
pub use session::{load_model, ProviderSelection, SessionContext};
```

---

## 21. Duration Formatting

The `format_duration` helper in `cli/output.rs` should format `Duration`
values for human display:

| Duration range     | Format           | Example  |
|--------------------|------------------|----------|
| < 1 second         | `{:.1}s`         | `0.3s`   |
| 1 s – 59 s         | `{:}s`           | `12s`    |
| 1 min – 59 min     | `{:}m {:02}s`    | `2m 34s` |
| ≥ 1 hour           | `{:}h {:02}m {:02}s` | `1h 05m 12s` |

---

## 22. Future Work

These items are explicitly out of scope for v0.2 but are worth tracking:

- **Per-model profiles** (TOML/YAML) for custom normalization (mean/std,
  `[-1, 1]` range, alternate color orders).
- **JSON output mode** (`--json`) for machine-readable CLI output.
- **Alpha channel preservation** — upscale alpha separately and recombine
  with RGB for RGBA inputs on RGB models.
- **Async / parallel batch processing** — process multiple images concurrently
  on different GPU streams.
- **Custom error types** (`thiserror`) for the library API.
- **Plugin system** for custom pre/post-processing pipelines.
- **Video frame processing** — extract frames, upscale, reassemble.
- **Progress callback trait** instead of closure for more structured reporting.
- **Model caching** — hash-based session caching to avoid redundant EP
  compilation (especially for TensorRT).

---

## 23. Style Reference

The definitive visual style reference is `sqwale_style.txt` in the repository
root. All CLI output formatting should match the patterns demonstrated in that
file, including:

- Symbol usage and color assignments.
- Tree connector characters and indentation.
- Progress bar template and characters (`━╌`).
- Spinner frames and tick rate.
- Summary format for batch operations.
- Error and warning presentation.

When in doubt about how output should look, consult `sqwale_style.txt`.
