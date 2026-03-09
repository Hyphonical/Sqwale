# Model Compatibility 🤖

**What makes a model work with Sqwale?**

This document explains the ONNX requirements, how Sqwale detects model properties, where to find compatible models, and what to do when things don't work.

---

## What Sqwale Needs

A compatible super-resolution model must satisfy these requirements:

| Property | Requirement | Why? |
|---|---|---|
| **Format** | ONNX (`.onnx` file) | ONNX Runtime standard |
| **Layout** | NCHW (`[batch, channels, height, width]`) | GPU-optimized memory layout |
| **Channels** | 1 (grayscale), 3 (RGB), or 4 (RGBA) | Standard image channel counts |
| **Precision** | float32 or float16 | Both supported natively |
| **Value range** | `[0.0, 1.0]` normalized | Sqwale's input standard |
| **Batch size** | 1 (static or dynamic) | Single-image processing |

> [!TIP]
> NHWC layouts (last-channel convention) are detected automatically and transposed to NCHW. So if your model uses NHWC, Sqwale will figure it out.

---

## Scale Detection 🔍

Sqwale doesn't require you to manually specify the upscale factor. It inspects the model graph and tries to figure it out automatically:

| Strategy | How it works | Priority |
|---|---|---|
| `metadata_props` | Reads the `scale` field from ONNX metadata | 1st (most reliable) |
| `DepthToSpace` | Detects PixelShuffle and reads `blocksize` | 2nd |
| `ConvTranspose` | Reads stride values | 3rd |
| `Resize` | Inspects the scales tensor | 4th |
| `static shape ratio` | Computes `output_height / input_height` | 5th |
| `assumed` | Falls back to scale = 4 | Last resort |

When you run `sqwale inspect model.onnx`, it shows which strategy worked:

```
├─ Scale  4x  via DepthToSpace (PixelShuffle)
```

or

```
├─ Scale  2x  via ConvTranspose stride=2
```

Usually Sqwale gets it right on the first or second strategy. If it guesses wrong, you can override it by adding a `scale` field to the model's ONNX metadata.

---

## Tiling Support 🧩

Whether a model can be tiled depends on its spatial dimension constraints:

| What | Tiling | Example |
|---|---|---|
| **Dynamic spatial dims** | ✓ Freely tileable | Dims are `?` or named like `img_h` |
| **Fixed spatial dims** | ✗ Not tileable | Input is hardcoded as `64×64` |
| **Alignment requirement** | ✓ Works with alignment | Transformer; requires divisibility by 16 |

Sqwale auto-detects these at inspection time:

```bash
sqwale inspect model.onnx
```

**Output for a tiling-compatible model:**
```
├─ Tiling  supported  dynamic spatial dims
│   └─ Alignment  divisible by 16
```

**Output for a fixed-size model:**
```
├─ Tiling  not supported  fixed 64×64 input
```

If a model is fixed-size, Sqwale will always use that size for tiles, regardless of `--tile-size`.

---

## Precision Handling

Both float32 and float16 models are fully supported:

- **float32:** Standard precision; lossless  
- **float16:** Half precision; uses less VRAM, slightly faster, minor precision loss

Sqwale automatically converts to the model's expected precision before inference and converts back to float32 for blending and saving. No manual configuration needed.

The `sqwale inspect` output shows the declared types:

```
├─ Precision  float16 → float16
```

or

```
├─ Precision  float32 → float32
```

---

## Color Space Detection

Sqwale infers the color space from input channel count:

| Channels | Color space | Behavior |
|---|---|---|
| **1** | Grayscale | Upscales grayscale; saves as grayscale |
| **3** | RGB | Standard RGB upscaling |
| **4** | RGBA | Upscales including alpha channel |
| **Other** | Unknown | Error; open an issue |

The model **must** match your image channel count. You can't upscale RGB images with a grayscale model.

---

## The Bundled Default: 4xLSDIRCompactv2

Sqwale comes with a solid general-purpose model packed inside the binary:

- **Author:** [Phhofm](https://github.com/Phhofm)
- **Scale:** 4× (outputs 4× the resolution)
- **Channels:** 3 (RGB)
- **Precision:** float32
- **Training:** LSDIR dataset
- **License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **Tiling:** Fully supported; dynamic spatial dims
- **Alignment:** None (no restrictions)
- **Performance:** ~100ms per 512px tile on CUDA

It's modern, fast, and produces good results on photographs and artwork. Fair warning: it's a model trained on diverse content, so don't expect perfection on specialized domains (anime, extreme close-ups, etc.).

---

## Finding Better Models

**[OpenModelDB](https://openmodeldb.info)** — The gold standard. Curated database with hundreds of super-resolution and other image models. Filterable by:
- Scale factor (2×, 4×, etc.)
- Target domain (photography, anime, art, etc.)
- Architecture (Real-ESRGAN, SRVGGv2, etc.)
- License

**[upscale.wiki](https://upscale.wiki)** — Community wiki with:
- Detailed model comparisons and benchmarks
- Use-case guides ("best for anime," "best for photos," etc.)
- Links to model repositories
- User reviews

**Downloading tips:**
1. Look for `.onnx` files (not `.pth`, `.safetensors`, `.pt`, etc.)
2. Check the license matches your use case
3. Verify the stated scale factor matches your need
4. Test on a small image first before batch processing

---

## Color Space Basics

If your model expects a different color space, you have options:

| Scenario | Solution |
|---|---|
| Model expects `[0, 255]` range but Sqwale sends `[0, 1]` | Re-export the ONNX with corrected normalization (advanced) |
| Model expects BGR but you have RGB | Use ffmpeg to convert: `ffmpeg -i input.jpg -vf "format=bgr24" output.jpg` then upscale |
| Model has wrong channel count | Downscale to grayscale or convert color space upstream |

Most modern models expect `[0, 1]` range and RGB order, so these issues are rare.

---

## Troubleshooting

### "Inference produced zero output"

**Symptoms:** Output image is black or transparent.

**Likely cause:** Model input dtype or value range mismatch.

**Fix:** 
- Verify the model was trained on `[0, 1]` inputs, not `[0, 255]`
- Check channel count (1, 3, or 4)
- Re-export the model if needed

### "Unsupported output tensor rank N"

**Symptoms:** Error message referencing an unexpected tensor shape.

**Likely cause:** Model outputs an unusual tensor layout.

**Fix:** Open an issue with the output of `sqwale inspect model.onnx` and we can debug it.

### "Inference failed: <error from ONNX Runtime>"

**Symptoms:** Runtime error during processing.

**Common causes:**
- Opset version mismatch (model is too new or too old for installed ONNX Runtime)
- Model uses custom operators not in standard ONNX Runtime
- Unsupported layer (e.g., some GPU-specific extensions)

**Fix:**
- Try `--provider cpu` to see if it's provider-specific
- Check model opset version: `sqwale inspect model.onnx | grep Opset`
- Report the full error with model details

### "Output is washed out or inverted"

**Symptoms:** Image colors look completely wrong.

**Likely cause:** The model expects `[0, 1]` but your images are in a different range, or vice versa.

**Fix:**
- Double-check the model's training documentation
- Try a different model to isolate the issue
- Use `ffmpeg` to normalize images before upscaling

---

## Advanced: Custom Model Inference

You can use Sqwale as a Rust library for custom inference workflows:

```rust
use sqwale::session::Session;

let session = Session::new("my_model.onnx", "auto")?;
let // ... load image, run inference
```

See the [architecture documentation](architecture.md) for pipeline details.
