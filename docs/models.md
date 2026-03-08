# Model Compatibility

This document describes the ONNX model requirements for use with Sqwale, how scale factors and other properties are detected, and where to find compatible models.

---

## Requirements

A compatible super-resolution model must satisfy:

| Property | Requirement |
|---|---|
| Format | ONNX (`.onnx`) |
| Layout | NCHW (`[batch, channels, height, width]`) |
| Channels | 1 (grayscale), 3 (RGB), or 4 (RGBA) |
| Precision | float32 or float16 |
| Value range | `[0.0, 1.0]` normalised |
| Batch size | 1 (static or dynamic) |

NHWC layouts (last-channel convention) are detected automatically from the output shape heuristic and transposed to NCHW before further processing.

---

## Scale detection

Sqwale infers the upscale factor by inspecting the model graph. It tries the following detection strategies in priority order:

| Strategy | How it works |
|---|---|
| `metadata_props` | Reads the `scale` field from ONNX model metadata if present |
| `DepthToSpace (PixelShuffle)` | Detects a `DepthToSpace` node and reads its `blocksize` attribute |
| `ConvTranspose stride` | Detects transposed convolutions and reads stride values |
| `Resize scales initializer` | Inspects the scales tensor of a `Resize` node |
| `static shape ratio` | Computes `output_height / input_height` from static shapes |
| `assumed` | Falls back to scale = 4 when no other evidence is found |

The detected scale and its source are shown by `sqwale inspect`:

```
├─ Scale  4x  via DepthToSpace (PixelShuffle)
```

---

## Tiling compatibility

Whether a model can be tiled depends on its spatial dimension constraints:

- **Dynamic spatial dims** — `height` and `width` input dimensions are symbolic (e.g. `?` or named). These models can be tiled freely, subject to alignment.
- **Fixed spatial dims** — input dimensions are hardcoded integers. Sqwale uses those exact dimensions as the tile size regardless of `--tile-size`.
- **Alignment requirement** — transformer-based models with window-partition patterns require input dimensions to be divisible by a specific value (typically 8, 16, or 32). Sqwale auto-detects this and rounds up the tile size accordingly.

Run `sqwale inspect` to see what a model reports:

```
├─ Tiling  supported  dynamic spatial dims
│   └─ Alignment  divisible by 16
```

or

```
├─ Tiling  not supported  fixed 64×64 input
```

---

## Precision

Both float32 and float16 models are supported. For float16 models, input tensors are automatically cast to `f16` before inference and outputs are cast back to `f32` for blending and saving. No manual configuration is required.

The precision reported by `sqwale inspect` reflects the model's declared input and output element types:

```
├─ Precision  float16 → float16
```

---

## Color space

Sqwale infers the color space from the number of input channels:

| Channels | Color space |
|---|---|
| 1 | Grayscale |
| 3 | RGB |
| 4 | RGBA |
| other | Unknown (N channels) |

Images are loaded and saved in the same format as the input file. No color space conversion is performed.

---

## The bundled model

The bundled default is **4xLSDIRCompactv2** by [Phhofm](https://github.com/Phhofm), a compact 4× RGB super-resolution model trained on the LSDIR dataset.

- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- Scale: 4×
- Precision: float32
- Tiling: supported, dynamic spatial dims
- Alignment: 1 (no restriction)
- Source: [OpenModelDB](https://openmodeldb.info/models/4x-LSDIRCompact-v2)

---

## Finding models

**[OpenModelDB](https://openmodeldb.info)** — the largest curated database of super-resolution models. Filterable by scale, architecture, license, and training dataset.

**[upscale.wiki](https://upscale.wiki)** — community wiki with model comparisons, use-case guides, and links to model repositories.

When downloading models, check that:
- The file extension is `.onnx` (not `.pth`, `.safetensors`, etc.).
- The listed scale matches your intended use.
- The license permits your use case.

---

## Troubleshooting

**"No tiles were processed"** — the model produced no output. This usually means the model's input dtype is not float32 or float16, or the value range is not `[0, 1]`.

**"Unsupported output tensor rank N"** — the model outputs a tensor with an unexpected shape. Open an issue with the output of `sqwale inspect model.onnx`.

**"Inference failed: …"** — an ORT runtime error. Common causes: the model requires an opset version not supported by the installed ORT version, or the model uses a custom operator not included in the standard ORT build.

**Black or washed-out output** — the model likely expects input in the `[0, 255]` range rather than `[0, 1]`. Sqwale normalises to `[0, 1]`. If you have a model like this, it cannot be used with Sqwale without reexporting the ONNX graph with corrected normalisation.
