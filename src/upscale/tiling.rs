//! Tiling and blending for large image inference.

use anyhow::Result;
use ndarray::Array2;

use crate::inspect::ModelInfo;

/// Configuration for tiling behavior.
#[derive(Debug, Clone)]
pub struct TileConfig {
	/// Tile size in pixels (will be adjusted for alignment).
	pub tile_size: u32,
	/// Overlap between tiles in pixels.
	pub overlap: u32,
}

impl Default for TileConfig {
	fn default() -> Self {
		Self {
			tile_size: 1024,
			overlap: 16,
		}
	}
}

/// A single tile with its position and padding information.
#[derive(Debug, Clone)]
pub struct Tile {
	/// Source rectangle in input image (x, y, width, height).
	pub src_rect: (u32, u32, u32, u32),
	/// Destination rectangle in output image (x, y, width, height).
	pub dst_rect: (u32, u32, u32, u32),
	/// Padding needed (left, top, right, bottom) to reach tile_size.
	pub padding: (u32, u32, u32, u32),
}

/// Compute tile grid for an image.
pub fn compute_tile_grid(
	image_width: u32,
	image_height: u32,
	model_info: &ModelInfo,
	config: &TileConfig,
) -> Result<Vec<Tile>> {
	let scale = model_info.scale;

	// Determine effective tile size
	let mut tile_size = config.tile_size;

	// For fixed-size models, use the model's required size
	if let Some((h, w)) = model_info.tile.fixed_size {
		tile_size = h.min(w) as u32;
		// Fixed models don't use overlap (padding-based approach instead)
		return compute_fixed_tiles(image_width, image_height, tile_size, scale);
	}

	// Apply alignment requirements
	if let Some(align) = model_info.tile.alignment {
		let r = tile_size % align;
		if r != 0 {
			tile_size += align - r;
		}
	}

	// Check if image fits in a single tile (no tiling needed)
	if image_width <= tile_size && image_height <= tile_size {
		let padding = (
			0,
			0,
			tile_size.saturating_sub(image_width),
			tile_size.saturating_sub(image_height),
		);
		return Ok(vec![Tile {
			src_rect: (0, 0, image_width, image_height),
			dst_rect: (0, 0, image_width * scale, image_height * scale),
			padding,
		}]);
	}

	// Compute tile grid
	let overlap = config.overlap;
	let step = tile_size.saturating_sub(2 * overlap);

	anyhow::ensure!(step > 0, "Tile size too small for overlap");

	let cols = image_width.div_ceil(step).max(1);
	let rows = image_height.div_ceil(step).max(1);

	let mut tiles = Vec::new();

	for row in 0..rows {
		for col in 0..cols {
			let x = col * step;
			let y = row * step;

			// Calculate actual tile dimensions with overlap
			let tile_w = if col == cols - 1 {
				image_width.saturating_sub(x)
			} else {
				tile_size.min(image_width.saturating_sub(x))
			};

			let tile_h = if row == rows - 1 {
				image_height.saturating_sub(y)
			} else {
				tile_size.min(image_height.saturating_sub(y))
			};

			// Padding to reach tile_size (or aligned size)
			let pad_right = tile_size.saturating_sub(tile_w);
			let pad_bottom = tile_size.saturating_sub(tile_h);

			// Apply alignment to padded dimensions
			let (final_w, final_h) = if let Some(align) = model_info.tile.alignment {
				let w = tile_w + pad_right;
				let h = tile_h + pad_bottom;
				let aligned_w = w.div_ceil(align) * align;
				let aligned_h = h.div_ceil(align) * align;
				(aligned_w, aligned_h)
			} else {
				(tile_w + pad_right, tile_h + pad_bottom)
			};

			let pad_right = final_w.saturating_sub(tile_w);
			let pad_bottom = final_h.saturating_sub(tile_h);

			tiles.push(Tile {
				src_rect: (x, y, tile_w, tile_h),
				dst_rect: (x * scale, y * scale, tile_w * scale, tile_h * scale),
				padding: (0, 0, pad_right, pad_bottom),
			});
		}
	}

	Ok(tiles)
}

/// Compute tiles for fixed-size models (no overlap, padding-based).
fn compute_fixed_tiles(
	image_width: u32,
	image_height: u32,
	tile_size: u32,
	scale: u32,
) -> Result<Vec<Tile>> {
	let cols = image_width.div_ceil(tile_size);
	let rows = image_height.div_ceil(tile_size);

	let mut tiles = Vec::new();

	for row in 0..rows {
		for col in 0..cols {
			let x = col * tile_size;
			let y = row * tile_size;

			let tile_w = tile_size.min(image_width.saturating_sub(x));
			let tile_h = tile_size.min(image_height.saturating_sub(y));

			let pad_right = tile_size.saturating_sub(tile_w);
			let pad_bottom = tile_size.saturating_sub(tile_h);

			tiles.push(Tile {
				src_rect: (x, y, tile_w, tile_h),
				dst_rect: (x * scale, y * scale, tile_w * scale, tile_h * scale),
				padding: (0, 0, pad_right, pad_bottom),
			});
		}
	}

	Ok(tiles)
}

/// Generate cosine blending weights for a tile.
///
/// Returns a 2D weight map where edges fade using cosine window.
pub fn generate_blend_weights(
	tile_width: u32,
	tile_height: u32,
	overlap: u32,
) -> Result<Array2<f32>> {
	let h = tile_height as usize;
	let w = tile_width as usize;
	let overlap = overlap as usize;

	let mut weights = Array2::<f32>::ones((h, w));

	// Apply cosine window on all edges
	for y in 0..h {
		for x in 0..w {
			let mut weight_x: f32 = 1.0;
			let mut weight_y: f32 = 1.0;

			// Left edge
			if x < overlap {
				let t = x as f32 / overlap as f32;
				weight_x = weight_x.min(0.5 * (1.0 - (std::f32::consts::PI * t).cos()));
			}
			// Right edge
			if x >= w - overlap {
				let t = (w - 1 - x) as f32 / overlap as f32;
				weight_x = weight_x.min(0.5 * (1.0 - (std::f32::consts::PI * t).cos()));
			}

			// Top edge
			if y < overlap {
				let t = y as f32 / overlap as f32;
				weight_y = weight_y.min(0.5 * (1.0 - (std::f32::consts::PI * t).cos()));
			}
			// Bottom edge
			if y >= h - overlap {
				let t = (h - 1 - y) as f32 / overlap as f32;
				weight_y = weight_y.min(0.5 * (1.0 - (std::f32::consts::PI * t).cos()));
			}

			weights[[y, x]] = weight_x * weight_y;
		}
	}

	Ok(weights)
}
