# Pipeline Architecture

This document describes the internal thread and channel layout for both video interpolation and batch image upscaling.

---

## Interpolation pipeline (`sqwale interpolate`)

Interpolation is a five-actor system. Each actor runs on its own OS thread and communicates exclusively via bounded `crossbeam` channels, so no actor can stall another beyond its buffer capacity.

```
FFmpeg reader process
        │  stdout (raw RGB24 bytes)
        ▼
  Reader thread  ──[read_tx/read_rx  cap 8]──►  Prep thread
                                                     │  bytes → f32 tensor (CPU)
                                                     ▼
                                             [prep_tx/prep_rx  cap 4]
                                                     │
                                                     ▼
                                           Main thread (GPU inference)
                                                     │  tensor → interpolated frames
                                                     ▼
                                             [write_tx/write_rx  cap 16]
                                                     │
                                                     ▼
                                             Writer thread
                                                     │  tensor → bytes (CPU) + write
                                                     ▼
                                         FFmpeg writer process
                                              stdin (raw RGB24 bytes)
```

### Channel sizing rationale

| Channel | Capacity | Reason |
|---|---|---|
| `read_rx` | 8 | FFmpeg is fast; a larger buffer prevents it from stalling when the prep thread is busy |
| `prep_rx` | 4 | Each entry holds a full decoded frame tensor (~25 MB for 1080p); keeping it small bounds peak RAM |
| `write_rx` | 16 | Output frames arrive in short bursts from recursive mid-frame generation; headroom avoids GPU idle |

### Thread roles

**Reader thread** — wraps FFmpeg's stdout in a `BufReader` and reads exactly `width × height × 3` bytes per frame. On EOF it exits cleanly, closing `read_tx` which signals the prep thread.

**Prep thread** — calls `bytes_to_tensor` to convert raw `u8` RGB24 into a normalised `[0, 1]` f32 NCHW `Array4`. This is pure CPU work; overlapping it with GPU inference is the main purpose of the three-stage split.

**Main thread (GPU)** — runs RIFE inference. Receives `(raw_bytes, tensor_a)` pairs from the prep channel, calls `generate_midframes`, and sends the results to the write channel. Owns frame ordering and scene-cut logic.

**Writer thread** — receives `WriteTask` variants:
- `Raw(Vec<u8>)` — written directly to FFmpeg stdin (passthrough for the first frame and scene-cut duplicates).
- `Tensor(Array4<f32>)` — converted to `u8` RGB24 via `tensor_to_bytes` before writing. This conversion runs here so it overlaps with the next GPU inference call.

### Recursive mid-frame generation

For a multiplier of `N`, `generate_midframes` produces `N − 1` evenly-spaced intermediate frames using recursive binary subdivision:

- `2×` → t = 0.5 (one RIFE call)
- `4×` → t = 0.25, 0.5, 0.75 (three calls: first compute t = 0.5, then recurse each half)
- `8×` → seven calls using the same subdivision

This preserves temporal consistency because each pair of bounding frames is always used directly, rather than blending across larger time gaps at higher multipliers.

### Scene detection

When `--scene-detect` is active, consecutive raw frame buffers are scored with a mean absolute difference formula identical to FFmpeg's `scdet` filter:

```
score = sum(|a[i] - b[i]|) / (N × 255)
```

Scores above the threshold indicate a cut. Instead of running RIFE, the last pre-cut frame is duplicated `(multiplier − 1)` times, keeping total output frame count and audio sync intact.

Scene scoring uses Rayon parallel iterators, splitting the ~6 MB 1080p buffer across all CPU cores to avoid a single-threaded bottleneck on the main inference thread.

---

## Batch upscale pipeline (`sqwale upscale` — batch mode)

Single-file upscale is fully synchronous. Batch mode introduces a two-actor split:

```
  Prefetch thread  ──[prefetch_tx/prefetch_rx  cap 2]──►  GPU loop (main thread)
  (disk I/O)                                               (inference + save)
```

### Prefetch thread

Iterates the resolved input list in order. For each file:
1. Calls `resolve_batch_output` to build the output path.
2. Calls `imageio::load_image` to decode the image into a `DynamicImage`.
3. Sends a `PrefetchedImage { index, input, output, result }` down the channel.

Load and path-resolution errors are wrapped in `result: Err(…)` rather than aborting the thread — the GPU loop handles them as per-image failures and continues. The thread exits cleanly when the channel receiver is dropped (cancellation) or when the input list is exhausted.

Channel capacity of 2 means at most two full decoded images live in RAM ahead of the GPU. For large images this is intentionally conservative — a 24 MP JPEG decoded to RGB24 is ~72 MB uncompressed.

### GPU loop (main thread)

Drains the prefetch channel via `for item in &prefetch_rx`. For each item:
1. Checks the cancellation token and breaks if set, dropping the receiver to unblock the prefetch thread.
2. Prints the per-image header at this point (not in the prefetch thread), so terminal output reflects when the GPU actually starts on each image.
3. Handles load errors by recording them in the `failed` list and continuing.
4. Calls `process_single_image` with the already-decoded `DynamicImage`.
5. Updates progress bars.

After the loop, joins the prefetch thread and reports the batch summary.

### Error collection

Both load errors (from the prefetch thread) and inference/save errors (from `process_single_image`) are appended to the same `failed: Vec<(PathBuf, String)>` list. The summary at the end of the batch reports all failures together, consistent with the pre-existing single-threaded behaviour.
