//! Tile grid computation, padding, and Hann-window blending.

use anyhow::{Result, bail};
use ndarray::Array2;

use crate::inspect::ModelInfo;

/// A rectangle in pixel coordinates.
#[derive(Debug, Clone, Copy)]
pub struct Rect {
	pub x: u32,
	pub y: u32,
	pub width: u32,
	pub height: u32,
}

/// Padding amounts for each side.
#[derive(Debug, Clone, Copy, Default)]
pub struct Padding {
	pub left: u32,
	pub top: u32,
	pub right: u32,
	pub bottom: u32,
}

impl Padding {
	pub fn is_zero(&self) -> bool {
		self.left == 0 && self.top == 0 && self.right == 0 && self.bottom == 0
	}
}

/// A single tile: where to crop from source, where to place in output,
/// and what padding to apply before inference.
#[derive(Debug, Clone)]
pub struct Tile {
	/// Region to crop from the source image.
	pub src: Rect,
	/// Region in the output image where this tile is placed (after scaling).
	pub dst: Rect,
	/// Padding to apply around the cropped region before inference.
	pub padding: Padding,
}

/// Compute the tile grid for the given image and model.
///
/// Handles three cases:
/// 1. Tiling disabled (`tile_size == 0`): single tile, whole image.
/// 2. Fixed-size model: sliding window with the model's required size.
/// 3. Dynamic model: sliding window with user tile size (aligned).
///
/// For dynamic models where the image is smaller than the tile size:
/// runs inference at the image's native size (no tiling, no padding).
pub fn compute_tile_grid(
	image_w: u32,
	image_h: u32,
	model_info: &ModelInfo,
	tile_size: u32,
	tile_overlap: u32,
) -> Result<Vec<Tile>> {
	let scale = model_info.scale;

	// Fixed-size model always uses its required size, regardless of tile_size.
	if let Some((fixed_h, fixed_w)) = model_info.tile.fixed_size {
		let ts = fixed_h.min(fixed_w) as u32;
		return compute_sliding_window(image_w, image_h, ts, tile_overlap, scale, true);
	}

	if tile_size == 0 {
		// Tiling disabled: single tile covering the whole image.
		return Ok(vec![Tile {
			src: Rect {
				x: 0,
				y: 0,
				width: image_w,
				height: image_h,
			},
			dst: Rect {
				x: 0,
				y: 0,
				width: image_w * scale,
				height: image_h * scale,
			},
			padding: Padding::default(),
		}]);
	}

	// Dynamic model: use effective tile size (respects alignment).
	let effective = model_info.tile.effective_tile_size(tile_size);

	// If the image fits in a single tile, skip tiling entirely.
	if image_w <= effective && image_h <= effective {
		return Ok(vec![Tile {
			src: Rect {
				x: 0,
				y: 0,
				width: image_w,
				height: image_h,
			},
			dst: Rect {
				x: 0,
				y: 0,
				width: image_w * scale,
				height: image_h * scale,
			},
			padding: Padding::default(),
		}]);
	}

	compute_sliding_window(image_w, image_h, effective, tile_overlap, scale, false)
}

/// Compute sliding-window tiles with overlap.
fn compute_sliding_window(
	image_w: u32,
	image_h: u32,
	tile_size: u32,
	overlap: u32,
	scale: u32,
	is_fixed: bool,
) -> Result<Vec<Tile>> {
	let step = tile_size.saturating_sub(2 * overlap);
	if step == 0 {
		bail!("Tile size {tile_size} is too small for overlap {overlap} (step would be 0)");
	}

	let mut tiles = Vec::new();

	let mut y = 0u32;
	while y < image_h {
		let mut x = 0u32;
		while x < image_w {
			// Crop region from source (clamped to image bounds).
			let src_w = tile_size.min(image_w - x);
			let src_h = tile_size.min(image_h - y);

			let src = Rect {
				x,
				y,
				width: src_w,
				height: src_h,
			};

			// Compute padding needed to reach tile_size for fixed-size models.
			let padding = if is_fixed && (src_w < tile_size || src_h < tile_size) {
				Padding {
					left: 0,
					top: 0,
					right: tile_size - src_w,
					bottom: tile_size - src_h,
				}
			} else {
				Padding::default()
			};

			let dst = Rect {
				x: x * scale,
				y: y * scale,
				width: src_w * scale,
				height: src_h * scale,
			};

			tiles.push(Tile { src, dst, padding });

			if x + tile_size >= image_w {
				break;
			}
			x += step;
		}

		if y + tile_size >= image_h {
			break;
		}
		y += step;
	}

	Ok(tiles)
}

/// Generate Hann-window blend weights for a tile.
///
/// Pixels within `overlap` distance from any edge taper smoothly
/// from 1.0 to 0.0 using the Hann function:
///   `w(t) = 0.5 * (1 - cos(π * t))`
/// where `t ∈ [0, 1]` is the normalized distance from the edge.
///
/// Interior pixels have weight 1.0.
pub fn blend_weights(tile_w: u32, tile_h: u32, overlap: u32) -> Array2<f32> {
	let w = tile_w as usize;
	let h = tile_h as usize;
	let o = overlap as usize;

	let mut weights = Array2::ones((h, w));

	if o == 0 {
		return weights;
	}

	for y in 0..h {
		for x in 0..w {
			let mut factor = 1.0f32;

			// Left edge taper.
			if x < o {
				let t = (x as f32 + 0.5) / o as f32;
				factor *= 0.5 * (1.0 - (std::f32::consts::PI * t).cos());
			}
			// Right edge taper.
			if x >= w - o {
				let t = ((w - x) as f32 - 0.5) / o as f32;
				factor *= 0.5 * (1.0 - (std::f32::consts::PI * t).cos());
			}
			// Top edge taper.
			if y < o {
				let t = (y as f32 + 0.5) / o as f32;
				factor *= 0.5 * (1.0 - (std::f32::consts::PI * t).cos());
			}
			// Bottom edge taper.
			if y >= h - o {
				let t = ((h - y) as f32 - 0.5) / o as f32;
				factor *= 0.5 * (1.0 - (std::f32::consts::PI * t).cos());
			}

			weights[[y, x]] = factor;
		}
	}

	weights
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::inspect::{ColorSpace, ScaleSource, TileInfo};

	fn make_model_info(scale: u32, tile: TileInfo) -> ModelInfo {
		ModelInfo {
			scale,
			scale_source: ScaleSource::Assumed,
			color_space: ColorSpace::Rgb,
			input_channels: 3,
			output_channels: 3,
			tile,
			input_dtype: "float32".into(),
			output_dtype: "float32".into(),
			opset: 17,
			op_fingerprint: vec![],
		}
	}

	#[test]
	fn single_tile_small_image() {
		let info = make_model_info(
			2,
			TileInfo {
				supported: true,
				alignment: None,
				fixed_size: None,
			},
		);
		let tiles = compute_tile_grid(100, 100, &info, 512, 16).unwrap();
		assert_eq!(tiles.len(), 1);
		assert_eq!(tiles[0].src.width, 100);
		assert_eq!(tiles[0].dst.width, 200);
	}

	#[test]
	fn multiple_tiles_with_overlap() {
		let info = make_model_info(
			2,
			TileInfo {
				supported: true,
				alignment: None,
				fixed_size: None,
			},
		);
		let tiles = compute_tile_grid(1024, 1024, &info, 512, 16).unwrap();
		assert!(tiles.len() > 1);
	}

	#[test]
	fn tiling_disabled() {
		let info = make_model_info(
			2,
			TileInfo {
				supported: true,
				alignment: None,
				fixed_size: None,
			},
		);
		let tiles = compute_tile_grid(2000, 2000, &info, 0, 16).unwrap();
		assert_eq!(tiles.len(), 1);
		assert_eq!(tiles[0].src.width, 2000);
	}

	#[test]
	fn fixed_size_model() {
		let info = make_model_info(
			2,
			TileInfo {
				supported: false,
				alignment: None,
				fixed_size: Some((256, 256)),
			},
		);
		let tiles = compute_tile_grid(512, 512, &info, 512, 16).unwrap();
		assert!(tiles.len() > 1);
		// Each tile src should be at most 256x256.
		for tile in &tiles {
			assert!(tile.src.width <= 256);
			assert!(tile.src.height <= 256);
		}
	}

	#[test]
	fn alignment_rounding() {
		let info = make_model_info(
			2,
			TileInfo {
				supported: true,
				alignment: Some(16),
				fixed_size: None,
			},
		);
		// tile_size 500 should round up to 512 (nearest multiple of 16).
		let tiles = compute_tile_grid(600, 600, &info, 500, 16).unwrap();
		// Image 600x600 with effective tile 512 → should be 1 or more tiles.
		assert!(!tiles.is_empty());
	}

	#[test]
	fn blend_weights_dimensions() {
		let w = blend_weights(64, 64, 8);
		assert_eq!(w.shape(), &[64, 64]);
	}

	#[test]
	fn blend_weights_range() {
		let w = blend_weights(64, 64, 8);
		for &val in w.iter() {
			assert!((0.0..=1.0).contains(&val));
		}
	}

	#[test]
	fn blend_weights_center_is_one() {
		let w = blend_weights(64, 64, 8);
		// Center pixel should have weight 1.0.
		assert!((w[[32, 32]] - 1.0).abs() < 1e-6);
	}

	#[test]
	fn blend_weights_edge_approaches_zero() {
		let w = blend_weights(64, 64, 8);
		// Corner pixel should be close to zero.
		assert!(w[[0, 0]] < 0.1);
	}
}
