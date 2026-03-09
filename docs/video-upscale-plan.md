# Video Upscaling Feature — Planning Questions

Answer these questions so a detailed implementation plan can be written.

---

## 1. Input detection — how to distinguish a video from an image?

The `upscale` command currently accepts a file path or glob pattern.
For video support, when a single file is given, should the tool:

- a) Detect by file extension (`.mp4`, `.mkv`, `.mov`, `.avi`, `.webm`, …)?
- b) Always probe with `ffprobe` and branch on whether a video stream is present?
- c) Require an explicit flag like `--video` to opt into video mode?

Glob patterns (e.g. `*.png`) should always stay image mode. Does that work for you?

Glob patterns should stay image mode, yes. For single-file input, I think probing with `ffprobe` is the most robust approach, since file extensions can be misleading. So option (b) sounds best to me. When glob patterns are used, we automatically do not support video, only images. When a single file is given, we check if it's a video by probing with `ffprobe`. If it has a video stream, we switch to video mode; otherwise, we treat it as an image. This way we can handle cases where the extension might not be standard or is missing.

---

## 2. Output container and codec

For the upscaled video:

- Should the output always be MKV + libx264/hevc_nvenc (same as interpolation), or should it mirror the input container (e.g. MP4 in → MP4 out)?
- Should there be a `--crf` option, or a fixed default quality?  
  Current interpolation default is 18; does that work for upscaling too?
- Should B-frames be enabled (they are in the interpolation writer)?

MKV + libx264/hevc_nvenc is a good default for video output, as it offers wide compatibility and good quality. Default to 18 for CRF, but allow changing it through a `--crf` option. Enabling B-frames is fine, as it can improve compression efficiency without significantly impacting quality for most content. So the defaults would be MKV container, libx264 or hevc_nvenc codec depending on hardware support, CRF 18, and B-frames enabled.

---

## 3. Frame transfer — pipe or temp files?

Two architectures are possible:

**A — In-memory pipe (like interpolation):**  
`ffmpeg` reader → raw RGB24 pipe → upscale each frame → raw RGB24 pipe → `ffmpeg` writer.  
No disk usage, but the entire pipeline must stay alive concurrently. The upscaler is single-frame sequential (no batching benefit across frames).

**B — Temp frames on disk:**  
Extract all frames to a temp directory → run the upscaler over them frame-by-frame → reassemble with `ffmpeg`.  
Simpler to implement and resume-friendly, but needs potentially tens of GB of temp space for long videos.

Which do you prefer? (A is cleaner and reuses the existing pipe infrastructure directly.)

I prefer option A.

---

## 4. Progress display

Currently, single-image upscaling shows a **tile progress bar** (e.g. `12/48 tiles`).  
For video, each frame is tiled internally. Two options:

- a) **Frame-level bar only** — `frame 42/3000` with elapsed time and ETA. Tile progress within each frame is hidden.
- b) **Two-level display** — outer bar for frames, inner `MultiProgress` bar for tiles within the current frame (like batch image mode already does).

Which do you prefer?

I prefer option B, as it provides more granular feedback on the progress of each frame, which can be helpful for longer videos where individual frames might take a significant amount of time to process. The outer bar will show the overall progress through the video, while the inner bar will give insight into the processing of the current frame.

---

## 5. Blend and grain — keep or drop for video?

Both `--blend` (frequency-domain Lanczos blend) and `--grain` (post-upscale noise) currently work for images.

- Should `--blend` apply per-frame during video upscaling? (It adds significant compute per frame.)
- Should `--grain` apply per-frame? (Constant grain seed per frame would look artificial — should it be randomised per frame, or disabled for video?)

These arguments should be disabled for video upscaling, at least for the initial implementation. The additional compute from `--blend` could significantly increase processing time for videos, and the visual benefits may not be worth it in a video context. For `--grain`, applying a constant grain seed per frame would indeed look artificial, and randomizing it per frame could lead to inconsistent results. Disabling both options for video upscaling simplifies the implementation and avoids potential issues with performance and visual quality.

---

## 6. Audio handling

Same approach as interpolation: upscale the video stream only, then a final `mux_audio_into` pass copies the original audio losslessly.

- Does that match your expectation?
- What should happen if the source has no audio? (Silently skip the mux pass — same as current interpolation behaviour.)

Same approach as interpolation sounds good. If the source has no audio, silently skipping the mux pass is fine and keeps the behavior consistent with interpolation. The focus of this tool is on video upscaling, so handling audio in a straightforward manner by copying it when present and ignoring it when absent makes sense.

---

## 7. Output path naming

For an input `my_video.mp4` with a 4× model:

- Default output name: `my_video_4x.mkv` — OK?
- If `--output` is given as a file path: use it directly (with `.mkv` forced)?
- If `--output` is given as a directory: write `<dir>/my_video_4x.mkv`?

I would suggest the following naming conventions:
- For an input `my_video.mp4` with a 4× model, the default output name should be `my_video_4x.mkv`. This clearly indicates the scale factor and the output format.
- If `--output` is given as a file path, we should use it directly, but we can enforce the `.mkv` extension to maintain consistency with the default output format. If the user specifies a different extension, we can either override it to `.mkv` or raise a warning and proceed with the specified name. For simplicity, enforcing the `.mkv` extension when a file path is provided seems reasonable.
- If `--output` is given as a directory, we should write the output file as `<dir>/my_video_4x.mkv`, following the same naming convention as the default output name. This allows users to specify an output directory while still maintaining a clear and consistent naming scheme for the upscaled video.

---

## 8. Batch video upscaling

You said batch video upscaling is **not needed**. Confirming: a glob like `*.mp4` should error out rather than silently processing each file, right? Or should it just process the first match?

It should error out rather than silently processing each file. This way, users are made aware that batch video upscaling is not supported and can adjust their input accordingly. Processing only the first match could lead to confusion if users expect all matched files to be processed, so it's better to explicitly require a single file input for video upscaling and provide clear feedback when a glob pattern is used with video files.

Eventually we could consider adding batch video upscaling support in the future, which would give a triple progress bar display (overall video progress, per-video progress, per-frame tile progress), but for the initial implementation, it's best to keep it simple and require single-file input for videos.

---

## 9. Tiling for video frames

The upscaler already tiles large images. For a 1920×1080 frame with default tile size 512:

- Should the same global `--tile-size` and `--tile-overlap` CLI flags control tiling for video frames?
- Any desire to expose a per-video-upscale tile override, or is inheriting the global flags fine?

No, keep the same global `--tile-size` and `--tile-overlap` CLI flags for controlling tiling for video frames. This keeps the interface consistent and avoids adding unnecessary complexity with per-video-upscale tile overrides. Users can adjust the tile size and overlap globally, and those settings will apply to both image and video upscaling. If there is a need for more granular control in the future, we can consider adding specific flags for video tiling, but for now, inheriting the global flags should be sufficient.

---

## 10. Model selection

Same as image upscaling — user can pass `--model path/to/model.onnx` or omit it to use the bundled model.

- Is the bundled 4× model (currently `4xLSDIRCompactv2`) appropriate as the default for video too, or should video default to a different model?
- Should the scale factor still be auto-detected from the ONNX model metadata (as it is for images)?

Same as image upscaling, the bundled 4× model (currently `4xLSDIRCompactv2`) is appropriate as the default for video upscaling as well. This provides a consistent experience across both image and video upscaling. The scale factor should still be auto-detected from the ONNX model metadata, just like it is for images. This allows users to easily switch between different models with varying scale factors without needing to specify additional parameters, and it keeps the user interface simple and intuitive.

---

## 11. Error handling mid-video

If the upscaler fails on frame N (OOM, inference error, etc.):

- Should the whole run abort and delete the partial output?
- Or should it write whatever was encoded and leave the partial file with a warning?

Write whatever was encoded and leave the partial file with a warning. This way, users can still access the successfully processed portion of the video, which can be useful for diagnosing issues or recovering some content. Aborting and deleting the partial output would result in losing all progress made up to that point, which could be frustrating for users, especially if they have processed a significant portion of the video before encountering an error. Providing a warning about the failure allows users to understand what happened and take appropriate action without losing all their work.

---

## 12. Large-video memory pressure

For a 4× upscale of 1920×1080, each output frame is 7680×4320 ≈ 96 MB (raw RGB). At the default 512-tile size the GPU processes ~480 tiles per frame.

- Is there any concern about VRAM pressure that should drive a different default tile size for video vs image?
- Any desire for a `--max-frames-in-flight` or similar back-pressure control, or trust the existing pipe/channel sizing?

The existing pipe/channel sizing should be sufficient for managing memory pressure during video upscaling. The tile size can be adjusted globally with the `--tile-size` flag, and users can modify it if they encounter VRAM issues. However, I don't think we need a separate default tile size for video vs image upscaling, as the current defaults should work for most cases. If users find that they are running into memory issues with larger videos, they can simply reduce the tile size to alleviate the pressure. Adding a `--max-frames-in-flight` option could add unnecessary complexity at this stage, and the current pipeline design should be able to handle the processing without needing additional back-pressure controls.

---

## Decision Summary

| Topic | Decision |
|-------|----------|
| **Input detection** | Probe with `ffprobe`; glob patterns stay image-only |
| **Output container** | MKV default, libx264/hevc_nvenc, CRF 18, B-frames enabled |
| **Architecture** | In-memory pipe (option A) — same as interpolation |
| **Progress display** | Two-level: outer frame bar + inner tile bar |
| **Blend / Grain** | Disabled for video upscaling |
| **Audio** | Copy losslessly via `mux_audio_into`; skip silently if absent |
| **Output naming** | `{stem}_{scale}x.mkv`; enforce `.mkv` when `--output` given |
| **Batch video** | Not supported — error out on glob with video files |
| **Tiling** | Reuse global `--tile-size` / `--tile-overlap` flags |
| **Model** | Same as images — bundled 4xLSDIRCompactv2, auto-detect scale |
| **Error mid-video** | Keep partial output with a warning |
| **Memory pressure** | Trust existing pipe sizing; no extra back-pressure controls |