# Pipeline Architecture 🏗️

**How Sqwale actually works under the hood.**

This document describes the thread and channel layout for both video interpolation and batch image upscaling. If you're curious about how Sqwale keeps the GPU busy while files load, or why scene detection doesn't make clips smooth out, you've come to the right place.

---

## Interpolation Pipeline (`sqwale interpolate`)

Video interpolation is a five-actor system. Each actor runs on its own OS thread and communicates exclusively via bounded `crossbeam` channels. No actor can stall another beyond its buffer capacity—this is deliberate, allowing us to hide latency and keep the GPU working while frames load.

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

### Why Split It Up?

FFmpeg I/O, CPU normalization, and GPU inference all have different speeds and bottleneck points. By splitting them into separate threads with bounded channels, each stage can proceed at its own pace:

- **Reader** can fill its buffer without waiting for the GPU
- **Prep** converts bytes to tensors while the GPU is busy with previous frames  
- **GPU** does the actual RIFE inference (the expensive part)
- **Writer** encodes and writes while the GPU is already working on the next frame

This `3+4+16` channel capacity arrangement (9 frames of potential buffering) means the GPU is almost never idle, and your video encodes in minimal wall-clock time.

### Channel Sizing Explained

| Channel | Capacity | Why? |
|---|---|---|
| `read_rx` | 8 | FFmpeg is fast; larger buffer prevents stalling when Prep is busy |
| `prep_rx` | 4 | Each tensor is ~25 MB for 1080p; small buffer keeps peak RAM bounded |
| `write_rx` | 16 | Output frames arrive in short bursts during recursive generation; headroom avoids GPU idle |

### What Each Thread Does

**Reader thread** — Wraps FFmpeg's stdout in a `BufReader` and reads exactly `width × height × 3` bytes per frame. Runs continuously until EOF, then closes `read_tx` to signal the prep thread it's done.

**Prep thread** — Calls `bytes_to_tensor` to convert raw `u8` RGB24 into normalized `[0, 1]` f32 NCHW tensors. Pure CPU work, overlapped with GPU inference—that's the whole point of this split.

**Main thread (GPU)** — Runs RIFE inference. Receives `(raw_bytes, tensor_a)` pairs from prep, calls `generate_midframes`, sends results to write. Owns frame ordering and scene-cut logic. This is where RIFE does its magic.

**Writer thread** — Receives two types of tasks:
- `Raw(Vec<u8>)` — written directly to FFmpeg stdin (used for the very first frame and scene-cut duplicates).  
- `Tensor(Array4<f32>)` — converted to `u8` RGB24 via `tensor_to_bytes` before writing. This conversion runs async, overlapping with the next GPU inference call.

---

### Recursive Mid-frame Generation

When you ask for `N×` frame multiplication, RIFE generates `N − 1` intermediate frames using recursive binary subdivision. This is key to temporal consistency:

- **2×** → `t = 0.5` (one RIFE call)
- **4×** → `t = 0.25, 0.5, 0.75` (three calls: first compute `t = 0.5`, then recurse each half)  
- **8×** → seven calls, same principle

Why recursive? Because each pair of bounding frames stays "anchored" directly by RIFE, rather than blending across larger time gaps. This prevents temporal smearing at higher multipliers and keeps motion coherent throughout the framerate multiplication.

---

### Scene Detection 🎬

When `--scene-detect` is active, consecutive raw frame buffers are scored with a mean absolute difference formula identical to FFmpeg's `scdet`:

```
score = sum(|a[i] - b[i]|) / (N × 255)
```

Scores above the threshold indicate a hard cut. Instead of running RIFE (which would produce ghosting), the last pre-cut frame is duplicated `(multiplier − 1)` times. Output frame count stays the same, audio sync is preserved, and no blur across the cut.

Frame scoring uses Rayon parallel iterators to split the ~6 MB 1080p buffer across all CPU cores. This keeps the main GPU thread unblocked while scoring happens.

> [!TIP]
> Scene detection is disabled by default because it adds per-frame overhead. Enable it only if your video has hard cuts (interviews, sports, screen recordings).

---

## Batch Upscale Pipeline (`sqwale upscale` — batch mode)

Single-image upscaling is synchronous and simple. But batch mode is where things get fun—we use two actors to hide disk I/O latency:

```
  Prefetch thread  ──[prefetch_tx/prefetch_rx  cap 2]──►  GPU loop (main thread)
  (disk I/O)                                               (inference + save)
```

### Prefetch Thread (Disk Loading)

Runs ahead of the GPU. For each image in the batch:

1. Resolves the output path
2. Calls `imageio::load_image` to decode the image into memory (this is slow—JPEG decompression, etc.)  
3. Sends `PrefetchedImage { index, input, output, result }` down the channel

If a load fails, the error is wrapped in `result: Err(…)` instead of crashing the thread. The GPU loop handles per-image failures and continues.

Channel capacity is intentionally small (2). For a 24 MP JPEG decoded to RGB24, that's ~72 MB uncompressed. Two images in the buffer is conservative but safe—no runaway memory usage even with huge images.

> [!TIP]
> The prefetch thread exits cleanly when you press Ctrl+C (the channel receiver is dropped), unblocking it to terminate gracefully.

### GPU Loop (Main Thread)

Drains the prefetch channel continuously. For each image:

1. Checks the cancellation token; breaks if Ctrl+C was pressed
2. Prints the per-image header now (not in the prefetch thread), so terminal output reflects *when the GPU actually starts*, not when it finished loading
3. Handles load errors by appending to the `failed` list and continuing  
4. Calls `process_single_image` with the already-decoded image
5. Updates progress bars

After the channel empties, joins the prefetch thread and prints the summary (files processed, failed, etc.).

### Error Handling

Both load errors (prefetch thread) and inference/save errors (GPU loop) go into the same `failed: Vec<(PathBuf, String)>` list. At the end, the batch summary reports all failures together, consistent with single-file mode behavior.

---

## Why This Matters

These thread designs exist to solve real bottlenecks:

- **Video interpolation:** Unbundled reader/prep/GPU/writer keeps every component fed, minimizing wall-clock time.
- **Batch upscaling:** Prefetch loading hides disk latency, so while image `N` loads from disk, image `N-1` is already done and saved.

Both allow you to feed Sqwale work faster than it can consume it, which is the sign of good pipelining.

For more details on tiling and blending, see [docs/tiling.md](tiling.md).
