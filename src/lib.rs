//! sqwale — ONNX super-resolution inference library.
//!
//! Provides model inspection and image upscaling via ONNX Runtime.

// ── Public modules ──────────────────────────────────────────────────────────
pub mod config;
pub mod imageio;
pub mod inspect;
pub mod pipeline;
pub mod session;

// ── Convenience re-exports ──────────────────────────────────────────────────
pub use inspect::{ColorSpace, ModelInfo, ScaleSource, TileInfo, inspect_model};
pub use pipeline::{CancelToken, UpscaleOptions, upscale_image};
pub use session::{ProviderSelection, SessionContext, load_model};
