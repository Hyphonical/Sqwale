//! sqwale — ONNX super-resolution inference library.
//!
//! Provides model inspection and image upscaling via ONNX Runtime.

// ── Bundled default model ───────────────────────────────────────────────────
// 4xLSDIRCompactv2 — https://openmodeldb.info/models/4x-LSDIRCompact-v2
// License: CC-BY-4.0  (https://creativecommons.org/licenses/by/4.0/)
// Author: Phhofm — https://github.com/Phhofm
pub const DEFAULT_MODEL_BYTES: &[u8] = include_bytes!("../models/4xLSDIRCompactv2.onnx");

// ── Public modules ──────────────────────────────────────────────────────────
pub mod config;
pub mod ffmpeg;
pub mod imageio;
pub mod inspect;
pub mod interpolate;
pub mod pipeline;
pub mod session;

// ── Convenience re-exports ──────────────────────────────────────────────────
pub use inspect::{
	ColorSpace, ModelInfo, ScaleSource, TileInfo, inspect_model, inspect_model_bytes,
};
pub use pipeline::blend::{frequency_blend, frequency_blend_with_original};
pub use pipeline::{CancelToken, UpscaleOptions, upscale_image, upscale_raw};
pub use session::{ProviderSelection, SessionContext, load_model, load_model_bytes};
