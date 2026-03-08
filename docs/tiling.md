# Tiling

This document covers how Sqwale splits large images into tiles, blends them back together, and how to choose tile parameters that balance VRAM usage against quality.

---

## Why tiling?

Super-resolution models have fixed or bounded input shapes. Running a 24 MP image through a model that expects 512 × 512 px inputs would require either resizing (losing quality) or padding the entire image to a size the GPU can handle (wasting VRAM and causing OOM for large images). Tiling avoids both problems by splitting the image into small, overlapping patches.

---

## Tile grid computation

Given an input image of size `(W, H)`:

1. **Effective tile size** is determined from the user's `--tile-size` preference and the model's requirements (see [Model alignment](#model-alignment) below).
2. The image is divided into a grid of tiles. Each tile's source region in the original image is `tile_size × tile_size` pixels, except at the right and bottom edges where it may be smaller.
3. Each tile is padded with **mirror padding** before inference if its dimensions are not divisible by the model's required alignment. The padding is removed from the output after inference using the actual scale factor.
4. Tile outputs are accumulated onto a canvas using a Hann (cosine-squared) weight window, then normalised.

---

## Hann-window blending

Adjacent tiles overlap by `--tile-overlap` pixels (default 16). In the overlap region, each tile's contribution is weighted by a two-dimensional Hann window:

```
w(x) = 0.5 × (1 − cos(2π × x / (tile_size − 1)))
```

This is a smooth raised-cosine taper that goes to zero at the tile edges and peaks at the centre. The result is that the centre of each tile (where the model produces the most reliable output, far from boundary artifacts) contributes most heavily, while the edges taper smoothly into the neighbouring tile.

The final canvas is normalised by dividing each pixel by the sum of all weights accumulated at that position, which guarantees exact reconstruction in non-overlapping areas and smooth blending where tiles meet.

---

## Model alignment

Some models require input spatial dimensions to be divisible by a certain value (e.g. 8, 16, or 32). This is inferred at model inspection time from Reshape or window-partition patterns in transformer-based models.

When a model reports an alignment requirement, the effective tile size is **rounded up** to the nearest multiple of that value. For example, a user preference of `--tile-size 500` with an alignment requirement of 16 becomes an effective size of 512.

Fixed-size models (those with fully static spatial input dimensions) always use their fixed size regardless of `--tile-size`.

---

## VRAM budgeting

VRAM usage scales with tile size and model scale factor. The rough formula for a single tile inference:

```
VRAM ≈ tile_size² × channels × bytes_per_element × scale²  (output tensor)
      + tile_size² × channels × bytes_per_element            (input tensor)
      + model weights
```

For a 4× float32 model:
- 512 px tile: ~25 MB tensors + model weights
- 768 px tile: ~57 MB tensors + model weights
- 1024 px tile: ~100 MB tensors + model weights

**Practical guidance:**

| GPU VRAM | Recommended `--tile-size` |
|---|---|
| 4 GB | 256–384 |
| 6–8 GB | 384–512 (default) |
| 10–12 GB | 512–768 |
| 16 GB+ | 768–1024 |

These are approximate. If you see CUDA OOM errors, reduce `--tile-size`. If inference is fast but you have idle VRAM, increase it.

**Disabling tiling** (`--tile-size 0`) processes the entire image in one pass. This is only practical for small images or high-VRAM GPUs, but it maxes out GPU utilisation for a single image and eliminates any risk of blending seams.

---

## Overlap guidance

`--tile-overlap` controls the width of the feathering zone at tile boundaries. Larger overlap reduces visible seams at the cost of more inference calls:

- **0** — no overlap; tiles are stitched hard. Seams may be visible on models that produce edge artifacts.
- **16** (default) — gentle feathering; correct for most models.
- **32–64** — smoother seams for models with strong edge artifacts; significantly increases tile count on large images.

The overlap is applied at the *source* resolution. The blended overlap in the output is `overlap × scale` pixels wide.

---

## Tile count estimation

To estimate how many tiles will be needed before running:

```
tiles_x = ceil(W / (tile_size - overlap))
tiles_y = ceil(H / (tile_size - overlap))
total   = tiles_x × tiles_y
```

Example: 5776 × 3856 image, tile size 512, overlap 16:
- `tiles_x = ceil(5776 / 496) = 12`
- `tiles_y = ceil(3856 / 496) = 8`
- `total = 96 tiles`
