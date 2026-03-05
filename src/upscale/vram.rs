//! VRAM and memory estimation for automatic tile size calculation.

use crate::inspect::ModelInfo;

/// Estimate optimal tile size based on available memory.
///
/// This is a conservative estimate that aims to use ~80-85% of available memory
/// to leave room for system operations and avoid OOM errors.
pub fn estimate_tile_size(model_info: &ModelInfo, _provider: &crate::upscale::Provider) -> u32 {
	// For now, use conservative fixed size
	// TODO: Query actual VRAM/RAM based on provider
	let default_tile_size = 512u32;

	// Apply model alignment if needed
	if let Some(align) = model_info.tile.alignment {
		let r = default_tile_size % align;
		if r != 0 {
			default_tile_size + (align - r)
		} else {
			default_tile_size
		}
	} else {
		default_tile_size
	}
}
