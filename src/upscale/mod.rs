//! Upscaling module for super-resolution inference.

mod provider;
mod session;
pub mod tile_size;
mod tiling;

pub use provider::Provider;
pub use session::{UpscaleOptions, UpscaleSession};
pub use tiling::{Tile, TileConfig};
