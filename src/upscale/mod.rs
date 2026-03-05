//! Upscaling module for super-resolution inference.

mod provider;
mod session;

pub use provider::Provider;
pub use session::{UpscaleOptions, UpscaleSession};
