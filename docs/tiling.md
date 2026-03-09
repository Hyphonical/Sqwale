# Image Tiling 🧩

**How Sqwale handles images that don't fit in your GPU's memory.**

This document explains how Sqwale splits huge images into tiny tiles, processes them independently, and blends them back together seamlessly. Also covers VRAM budgeting and how to tune `--tile-size` for your hardware.

---

## The Problem: What If The Image Doesn't Fit?

Neural networks have input size constraints. Your upscaling model might expect 512×512 inputs. But what if you have a 24 MP photo (6000×4000 pixels)?

You have three options:

1. **Resize the image down, upscale, then resize up** — lose quality in the resize
2. **Pad it to a massive size the GPU can handle** — run out of memory
3. **Tile it** — split into overlapping patches, upscale each, blend them back

Sqwale does option 3. It's smart.

---

## How Tiling Works

Given an image of size `(W, H)`:

1. Split into a grid of overlapping tiles, each of size `--tile-size × --tile-size` pixels (default 512)
2. Pad each tile with **mirror padding** if needed to satisfy the model's alignment requirements (e.g., "must be divisible by 16")
3. Upscale each tile independently on the GPU
4. Blend them back together using **Hann-window weighting** in the overlap zones
5. Reconstruct the full upscaled image on the output canvas

The overlap region prevents visible seams because each tile's edges (where boundary artifacts are worst) get weighted down, while the center (highest quality) gets weighted up.

> [!TIP]
> You don't need to understand the math to use tiling. Just know: larger `--tile-size` = faster but more VRAM. Smaller = slower but fits in less VRAM.

---

## Hann-Window Blending 🔊

In the overlap region where tiles meet, each tile's contribution is weighted by a smooth Hann (raised-cosine) window:

```
w(x) = 0.5 × (1 − cos(2π × x / (tile_size − 1)))
```

This creates a smooth fade: the window is ~0 at tile edges and peaks at the center.

**Why this matters:**
- Super-resolution models produce their best output in the center of the tile (far from boundary effects)
- The edges tend to have artifacts
- By upweighting the center and downweighting the edges, we hide the worst parts
- Adjacent tiles blend together smoothly with no visible seams

The final pixel value on the canvas is:

```
output[x, y] = sum(tile_contribution[x, y] × w[x, y]) / sum(w[x, y])
```

The denominator (`sum(w[x, y])`) normalizes to guarantee exact reconstruction outside overlaps and smooth blending inside.

---

## Model Alignment

Some neural networks, especially transformers, require input dimensions to be divisible by a specific value (e.g., 8, 16, or 32). This is called **alignment**.

Sqwale auto-detects alignment requirements when you inspect a model:

```bash
sqwale inspect model.onnx
```

Output:
```
├─ Tiling  supported  dynamic spatial dims
│   └─ Alignment  divisible by 16
```

When tiling, if your effective tile size doesn't satisfy the alignment, Sqwale rounds *up* to the nearest multiple. For example:

- User preference: `--tile-size 500`
- Model alignment: divisible by 16
- Effective size: 512 (rounded up)

**Fixed-size models** always use their fixed size regardless of `--tile-size`. If a model has a hardcoded 256×256 input, that's what gets used.

---

## VRAM Budget Calculation

VRAM usage scales with tile size and the model's scale factor. Rough formula:

```
VRAM ≈ tile_size² × channels × bytes_per_element × (1 + scale²)
      + model_weights
```

Breaking it down:
- `tile_size² × channels × bytes_per_element` = input tensor size
- `tile_size² × scale² × channels × bytes_per_element` = output tensor size
- `model_weights` = the model itself (usually 50–500 MB)

**Example: 4× upscaling, float32, 3 channels (RGB):**

| Tile size | Input tensor | Output tensor | Total inference | Model | Total |
|---|---|---|---|---|---|
| 256 px | 0.75 MB | 12 MB | ~13 MB | 200 MB | ~213 MB |
| 512 px | 3 MB | 48 MB | ~51 MB | 200 MB | ~251 MB |
| 768 px | 7 MB | 108 MB | ~115 MB | 200 MB | ~315 MB |
| 1024 px | 12 MB | 192 MB | ~204 MB | 200 MB | ~404 MB |

---

## Recommended Settings

| GPU VRAM | Recommended tile size | Why |
|---|---|---|
| **4 GB** | 256–384 | Conservative; avoids OOM |
| **6–8 GB** | 384–512 | Default (512) is safe; good speed |
| **10–12 GB** | 512–768 | Can push a bit; faster tiles |
| **16+ GB** | 768–1024 | Big tiles; maximum throughput |

**General strategy:**
1. Start with `--tile-size 512` (default)
2. If you get CUDA out-of-memory errors, *decrease* it to 384 or 256
3. If the GPU is idle and you have spare VRAM, *increase* it to 768 or 1024

> [!TIP]
> Monitor GPU memory with `nvidia-smi` (for NVIDIA) or `gpustat` (cross-platform). Watch the memory rise and fall as tiles are processed. If it consistently peaks below your GPU's total, you can safely increase `--tile-size`.

---

## When To Disable Tiling

Pass `--tile-size 0` to process the entire image in a single pass:

```bash
sqwale upscale image.jpg --tile-size 0
```

**Pros:**
- No blending artifacts (though they're rare with proper Hann windowing)
- Simpler pipeline; slightly faster per-image time
- Model sees the full context (good for architectural models)

**Cons:**
- Only works if the entire upscaled image fits in VRAM
- Will OOM on large images
- GPU utilization may be suboptimal for very small images

Use `--tile-size 0` only for small images (< 1000×1000) or if you have a massive GPU (24+ GB VRAM).

---

## Overlap Settings

`--tile-overlap` (default 16 pixels) controls the feathering width at tile boundaries:

| Overlap | Seam visibility | Tile count | When to use |
|---|---|---|---|
| **0** | Can see hard seams | Minimal | Never; overlapping is cheap |
| **16** | Invisible (default) | ~1.5% more | Most cases |
| **32–64** | Extremely smooth | ~2–3% more | Models with strong edge artifacts |

The overlap is at the *source* resolution. After upscaling by 4×, a 16-pixel source overlap becomes 64 pixels in the output (blended zone).

---

## Tile Count Estimation

Before running, estimate how many tiles you'll need:

```
tiles_x = ceil(width  / (tile_size - overlap))
tiles_y = ceil(height / (tile_size - overlap))
total   = tiles_x × tiles_y
```

**Example:** 5776 × 3856 image, tile size 512, overlap 16

```
tiles_x = ceil(5776 / 496) = 12
tiles_y = ceil(3856 / 496) = 8
total   = 96 tiles  (~13ms per tile on CUDA = ~1.2 seconds GPU time)
```

Each tile takes 10–50ms depending on your GPU. Estimate total time as `tiles × time_per_tile + overhead`.

---

## Troubleshooting Tiling

**"CUDA out of memory"** — Reduce `--tile-size`. Try 256 or 384.

**"Visible seams in the output"** — Rare. Try increasing `--tile-overlap` to 32.

**"Upscaling is slow"** — Multiple causes:
- Too many tiles → increase `--tile-size`
- GPU is busy elsewhere → close other GPU apps
- Provider is suboptimal → check `RUST_LOG=sqwale=debug`

**"Output looks slightly different in overlap zones"** — This is expected and usually imperceptible. The Hann blending creates smooth transitions, not perfect boundaries.

---

For more context on the full pipeline, see [docs/architecture.md](architecture.md).
