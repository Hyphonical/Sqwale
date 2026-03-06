//! Library constants for tile defaults and inspection limits.

/// Default tile size in pixels for dynamic-spatial models.
pub const DEFAULT_TILE_SIZE: u32 = 512;

/// Default pixel overlap between adjacent tiles.
pub const DEFAULT_TILE_OVERLAP: u32 = 16;

/// Maximum op-types shown in the inspect tree before collapsing.
pub const INSPECT_MAX_OPS_SHOWN: usize = 8;
