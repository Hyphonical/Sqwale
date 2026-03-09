# Plan: `--continue` Flag, Scene Detection Fix & Video Upscale Notes

> Status: **Prerequisites confirmed**. Ready for implementation.

### Confirmed Answers (Round 2)

| # | Decision |
|---|---|
| Q1 | `.ts` for **all** runs, not just `--continue` |
| Q2 | Auto-detect output path (same derivation); warn if output already appears complete |
| Q3 | Fast-seek first, then `select=gte(n,delta)` for sub-GOP accuracy |
| Q4a | Always defer audio; final FFmpeg audio-mux pass at the end of every successful run |
| Q4b | Remove audio from `spawn_writer`; simplifies it |
| Q5 | Direct append to `.ts` IS viable with `-output_ts_offset`. One small temp only when trim is needed (see §1.4) |
| Q6 | `floor((P-1)/M)` is correct |
| Q7 | `-bf 0` on all runs |
| Q8 | Scene detection has never been tested with debug logging; the 0.4 default was silently ignoring all cuts |
| Q9 | Default `scene_threshold` → `0.1`; add `trace!` per-frame logging |
| Q10 | 0.1 confirmed to work in manual testing |
| Q11–13 | Video upscale reuses same piped-FFmpeg architecture; no cross-feature `--continue` interaction |

---

## 1. `--continue` Flag

### 1.1 Why MPEG-TS (`.ts`) Over MKV

MKV is a Matroska container. It stores an index at the end of the file (or in a
seek-table scattered through the body). An interrupted write leaves an
incomplete index, making the file appear corrupt or zero-duration in most
players. Frame-accurate resumption requires re-encoding the entire file to
rebuild the index.

MPEG-TS is a _packet-by-packet_ container (fixed 188-byte cells) with no end
index. Individual packets are self-describing. You can:
- Read the frame count of an incomplete file with `ffprobe`.
- Concatenate two `.ts` files with `cat` or FFmpeg's concat protocol and
  correct timestamps in a single pass.
- Stop writing at any point; the written portion is always valid.

### 1.2 B-frames and Seekability

B-frames (Bidirectional Predicted frames) reference both past **and future**
frames as prediction sources. An encoder cannot flush them immediately; it
must buffer and re-order them. If the process is killed, the last buffered
B-frames are never flushed and the GOP is incomplete. On resume, re-alignment
with the last clean I- or P-frame boundary is needed.

**Fix:** Disable B-frames with `-bf 0` on the NVENC encoder. The stream then
only has I- and P-frames. Every P-frame can be decoded from any earlier
I-frame — the GOP boundary is always clean.

```
hevc_nvenc + -bf 0  →  stream is recoverable at every I-frame
```

For `libx264` (CPU fallback), `-bf 0` also works and is already an option.
The trade-off is ~5–10% larger file size vs. the same CRF setting.

### 1.3 Frame Boundary Maths

With multiplier `M` and partial output frame count `P`:

```
input_frames_per_output_group = M          (1 source + (M-1) interp)
last_complete_source_frame     = (P - 1) / M   (integer division)
clean_output_frames            = last_complete_source_frame * M + 1
```

Example (2×, partial = 5 frames written):
```
frames in partial: 0  0.5  1  1.5  2
P = 5
last_complete_source = (5-1)/2 = 2
clean_output_frames  = 2*2+1 = 5   ← already clean, no trim needed
```

Example (2×, partial = 4 frames written — interrupted mid-pair):
```
frames in partial: 0  0.5  1  1.5
P = 4
last_complete_source = (4-1)/2 = 1   (floor)
clean_output_frames  = 1*2+1 = 3
→ trim partial to 3 frames, resume from source frame 1
```

Trimming: `ffmpeg -i partial.ts -frames:v {clean_output_frames} -c copy trimmed.ts`

### 1.4 Q5 — Direct Append vs. Temp File (Resolved)

**Your instinct is correct.** MPEG-TS supports direct append. The reason my
original plan mentioned a "temp continuation file" was an over-complication.
Here is what is actually needed and why:

#### The PTS problem

Every FFmpeg encoder session starts its Presentation TimeStamps (PTS) from 0.
If you simply open the output file and let FFmpeg write to it again, the new
packets carry timestamps `0, 1/fps, 2/fps …` while the existing packets end at
`P/output_fps`. Players treat this as a timestamp discontinuity and either loop
or fail to seek past it.

**Fix:** pass `-output_ts_offset {clean_time}` to the FFmpeg writer. It shifts
all output PTS by that offset, so the new packets seamlessly continue from
where the partial left off. Combined with `pipe:1` (writing to stdout) and a
Rust thread that appends those bytes to the existing `.ts` file, the continuation
is clean with **zero additional files**.

#### The trim edge case

The output sequence for multiplier M is:

```
src_0 | mid(0→1)×(M-1) | src_1 | mid(1→2)×(M-1) | src_2 | …
```

A "clean boundary" is when the last written frame is a source frame:
`(P − 1) % M == 0`.

If the process was killed while mid-frames were being written for a pair,
P is not at a clean boundary. Those orphaned mid-frames must be removed before
appending the continuation, otherwise the resumed source frame appears twice
and the timeline is corrupted.

Trimming is done by remuxing: `ffmpeg -i output.ts -frames:v {clean_P} -c copy
output.ts.tmp` then atomically replacing `output.ts`. This temp file:
- Is at most the same size as the partial (usually slightly smaller)
- Is created and deleted immediately before the continuation starts
- Is only needed when `(P − 1) % M != 0` — i.e., killed mid-pair

If Ctrl+C fires between pairs (the common case), P is already clean and **no
temp file is created at all**.

#### Summary of the accurate flow

```
1. ffprobe output.ts → P (complete frames)
2. clean_P = floor((P-1)/M) * M + 1
3. IF P != clean_P:
     ffmpeg -i output.ts -frames:v {clean_P} -c copy output.ts.tmp
     rename output.ts.tmp → output.ts
4. source_start = (clean_P - 1) / M
5. Spawn FFmpeg reader with fast-seek to source_start (select filter for sub-GOP accuracy)
6. Spawn FFmpeg writer:
     -output_ts_offset {clean_P / output_fps}
     -f mpegts pipe:1      (stdout)
     NO audio (-map 0:v only, -bf 0)
   Writer stdout → Rust thread that opens output.ts in APPEND mode
7. Run inference pipeline normally from source_start
8. On successful completion:
     ffmpeg -i output.ts -i input.mp4 -map 0:v -map 1:a? -c copy output.final.ts
     rename output.final.ts → output.ts
```

No large temp segment. No merge step. One optional small trim-temp.



### 1.5 Full Implementation Flow

```
sqwale interpolate input.mp4                   # normal, always .ts
sqwale interpolate --continue input.mp4        # resume
```

**Normal run:**
1. Resolve output path as `{stem}_{M}x.ts`.
2. If output already exists AND `--continue` not given → overwrite (existing behaviour).
3. Spawn FFmpeg writer: video-only (`-map 0:v -bf 0 -f mpegts pipe:1`), write to file.
4. Run full pipeline.
5. On success: `ffmpeg -i output.ts -i input.mp4 -map 0:v -map 1:a? -c copy output.final.ts` → rename.

**Continue run (`--continue`):**
1. Resolve output path identically.
2. If output does **not** exist → proceed as normal run (emit note: "Starting fresh, no partial found").
3. If output exists AND appears fully complete (frame count ≈ expected total):
   - Print friendly warning: "Output already looks complete. Use --output to write elsewhere."; exit.
4. Probe `output.ts` → `P` complete frames.
5. `clean_P = floor((P-1)/M) * M + 1` ; `source_start = (clean_P - 1) / M`.
6. If `P != clean_P` (interrupted mid-pair):
   - `ffmpeg -i output.ts -frames:v {clean_P} -c copy output.ts.tmp`
   - Atomic rename: `output.ts.tmp → output.ts`
7. Seek FFmpeg reader to `source_start` (fast-seek + `select` sub-GOP correction).
8. Spawn FFmpeg writer:
   - Video-only (`-map 0:v -bf 0 -f mpegts pipe:1`) 
   - `-output_ts_offset {clean_P / output_fps}`
   - stdout → Rust append-writer thread → opens `output.ts` in `OpenOptions::append(true)`
9. Run pipeline from `source_start`.
10. On success: same audio-mux final pass as normal run (step 5 above).

### 1.6 Source Seeking Strategy (Confirmed: Fast-seek + select)

```
ffmpeg [-hwaccel cuda] -ss {source_start / fps} -i input.mp4
  -vf "select=gte(n\,{source_start - keyframe_n}),setpts=PTS-STARTPTS"
  -f rawvideo -pix_fmt rgb24 -
```

- `-ss` before `-i` lands on the last keyframe ≤ target (fast, O(1) seek for most formats).
- `select=gte(n,delta)` burns through the remaining frames within that GOP (at most ~250 frames at 24fps / 10s GOP).
- `setpts=PTS-STARTPTS` resets timestamps to 0, as the pipeline expects.
- `keyframe_n` is the frame index FFmpeg actually landed on after the fast seek;
  we derive it as `floor(source_start / fps / keyframe_interval) * keyframe_interval`
  or simply probe the keyframe timestamps from `ffprobe -select_streams v -show_packets -skip_frame nokey`.

For sources with small GOPs (≤ 5s) or when `source_start` is small (< 1000 frames), full-decode with select-only is also acceptable.

### 1.7 Audio — Always Deferred (Confirmed)

The writer **never** maps audio. After every successful run (normal or continue), one final remux pass adds the source audio:

```
ffmpeg -i output.ts -i input.mp4 -map 0:v -map 1:a? -c copy output.final.ts
```

Then `output.final.ts` is renamed to `output.ts`. If the source has no audio stream (`-map 1:a?` produces nothing) the merge is still clean. This is the same logic for every run.

### 1.8 NVENC & Container Changes (Confirmed)

**All runs:**
- Output container: `-f mpegts` / `.ts` extension
- B-frames disabled: `-bf 0`
- NVENC: `hevc_nvenc -preset p4 -rc vbr -cq {crf} -b:v 0 -bf 0`
- CPU fallback: `libx264 -preset fast -crf {crf} -bf 0`
- Audio: **none** in writer (see §1.7)

**`resolve_output`** helper updated to force `.ts` instead of `.mkv`.

---

## 2. Scene Detection Bug

### 2.1 Diagnosis

The `scene_score` function computes **mean absolute difference** normalised to
`[0.0, 1.0]`:

```rust
sad as f64 / (a.len() as f64 * 255.0)
```

This is identical to FFmpeg's `scdet` filter formula, but `scdet` expresses
the result as a percentage (multiplied by 100). Therefore:

| Our code | FFmpeg `scdet` |
|---|---|
| `0.0` | `0.0%` |
| **`0.4` (our default)** | **`40%` (current effective threshold)** |
| `0.1` | `10%` |

FFmpeg `scdet` documentation states:
> "Good values are in the `[8.0, 14.0]` range. Default value is `10.`"

Translated to our scale: **good values are `[0.08, 0.14]`**, default `0.10`.

**Our default `0.4` is ~4× too high.** Hard scene cuts in typical live-action
content score roughly `0.08–0.20`. Animated content can score lower (~`0.05`).
A threshold of `0.4` will catch almost nothing.

### 2.2 Fix

Change the default from `0.4` to `0.1` in [src/cli/mod.rs](../src/cli/mod.rs):
```rust
#[arg(long, default_value_t = 0.1, value_parser = parse_scene_threshold)]
scene_threshold: f64,
```

And update the doc comment to reflect the corrected comparison to `scdet`.

### 2.3 Additional Debugging Aid

Currently `debug!` is only emitted on detected cuts:
```rust
debug!("Scene cut at frame {} (score={:.3}, threshold={:.3})", ...);
```

We should add a trace-level log for every frame score to help tune the
threshold:
```rust
trace!("Scene score frame {}: {:.4} (threshold={:.3})", frames_read, score, threshold);
```

This avoids flooding normal debug output while allowing `RUST_LOG=sqwale=trace`
for investigation.

---

## 3. Notes on Future Video Upscaling

### 3.1 Structural Similarity to Interpolation

The interpolation pipeline:
```
FFmpeg reader → prep thread (bytes→tensor) → GPU inference → writer thread (tensor→bytes → FFmpeg stdin)
```

A video upscale pipeline would be structurally similar:
```
FFmpeg reader → prep thread (bytes→DynamicImage) → GPU inference (tiled) → writer thread (image→bytes → FFmpeg stdin)
```

The main differences:
- **Output dimensions change**: `width*scale × height*scale` — the FFmpeg writer
  needs to know the target dimensions.
- **Frame independence**: Each frame upscales independently (vs. RIFE needing
  pairs). This potentially allows the prep and inference to overlap more easily.
- **Tiling needed for large frames**: The existing `tiling.rs` and
  `pipeline/mod.rs` already handles this for images. Video upscale can reuse
  it directly.
- **Model loading**: Same `session/` infrastructure applies.

### 3.2 Key Gotchas

- The `image_to_tensor` / `tensor_to_image` functions in `pipeline/tensor.rs`
  operate on `DynamicImage`, not on raw RGB24 bytes. A bridge:
  ```rust
  fn rgb24_to_dynamic_image(bytes: &[u8], w: usize, h: usize) -> DynamicImage { ... }
  fn dynamic_image_to_rgb24(img: &DynamicImage) -> Vec<u8> { ... }
  ```
  would be the only new conversion needed.

- **VRAM budget**: At 4K input, a 4× upscale → 16K output. Even with tiling,
  each tile can be large. The tile-overlap-blending pass is CPU-only in the
  current pipeline and should be manageable.

- **Audio**: Identical to interpolation — copy from source unchanged.

- **Frame rate**: Output FPS equals input FPS (no multiplier). Simpler than
  interpolation.

- **Progress** reporting: `total_frames = frame_count`, increment by 1 per
  frame. Simpler than interpolation math.

### 3.3 `--continue` Applicability

Video upscaling is even more amenable to `--continue` than interpolation:
- Each input frame maps to exactly one output frame.
- `source_start = P` (partial frame count equals the source frame to resume from).
- No boundary-alignment edge cases.

If MPEG-TS and `--continue` land for interpolation first, video upscaling can
reuse the exact same mechanism with `M=1`.
