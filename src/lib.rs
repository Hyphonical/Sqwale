//! sqwale — ONNX super-resolution inference library.
//!
//! Provides model inspection and image upscaling via ONNX Runtime.

// ── Bundled default model ───────────────────────────────────────────────────
// 4xSPANkendata — https://openmodeldb.info/models/4x-SPANkendata
// License: CC-BY-SA-4.0  (https://creativecommons.org/licenses/by-sa/4.0/)
// Author: terrainer — https://github.com/terrainer/AI-Upscaling-Models
pub const DEFAULT_MODEL_BYTES: &[u8] = include_bytes!("../model.onnx");

// ── Public modules ──────────────────────────────────────────────────────────
pub mod config;
pub mod imageio;
pub mod inspect;
pub mod pipeline;
pub mod session;

// ── Convenience re-exports ──────────────────────────────────────────────────
pub use inspect::{
	ColorSpace, ModelInfo, ScaleSource, TileInfo, inspect_model, inspect_model_bytes,
};
pub use pipeline::blend::{frequency_blend, frequency_blend_with_original};
pub use pipeline::{CancelToken, UpscaleOptions, upscale_image};
pub use session::{ProviderSelection, SessionContext, load_model, load_model_bytes};
