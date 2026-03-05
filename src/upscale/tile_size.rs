//! Static tile-size selection.
//!
//! We use a conservative fixed tile size instead of trying to estimate available
//! VRAM at runtime.  The chosen default (512 px) works well across a wide range
//! of hardware — it keeps peak memory low on integrated GPUs while still being
//! large enough to minimise tiling overhead on discrete cards.

use crate::inspect::ModelInfo;

/// Default tile size used when the model has dynamic spatial dims.
const DEFAULT_TILE_SIZE: u32 = 512;

/// Choose the tile size for a given model.
///
/// * **Fixed-size models** → use the model's declared spatial size.
/// * **Dynamic models** → use [`DEFAULT_TILE_SIZE`], rounded up to the model's
///   alignment requirement (if any).
pub fn effective_tile_size(info: &ModelInfo) -> u32 {
	if let Some((h, w)) = info.tile.fixed_size {
		return h.min(w) as u32;
	}

	let base = DEFAULT_TILE_SIZE;
	match info.tile.alignment {
		Some(align) if base % align != 0 => base + (align - base % align),
		_ => base,
	}
}
