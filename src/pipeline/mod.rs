//! Upscaling pipeline: tiling, tensor conversion, and inference.

pub mod tensor;
pub mod tiling;

use anyhow::{Result, bail};
use image::DynamicImage;
use ndarray::{Array2, Array4};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::inspect::ModelInfo;
use crate::session::SessionContext;

use tensor::{crop_tensor, image_to_tensor, pad_tensor_mirror, tensor_f32_to_f16, tensor_to_image};
use tiling::{Tile, blend_weights, compute_tile_grid};

/// A token for cooperative cancellation.
///
/// Clone-safe — all clones share the same underlying flag.
#[derive(Clone, Default)]
pub struct CancelToken(Arc<AtomicBool>);

impl CancelToken {
	pub fn new() -> Self {
		Self(Arc::new(AtomicBool::new(false)))
	}

	/// Signal cancellation.
	pub fn cancel(&self) {
		self.0.store(true, Ordering::SeqCst);
	}

	/// Check if cancellation has been requested.
	pub fn is_cancelled(&self) -> bool {
		self.0.load(Ordering::SeqCst)
	}
}

/// Options controlling the upscaling pipeline.
pub struct UpscaleOptions {
	/// Tile size in pixels. 0 = disable tiling (whole image at once).
	pub tile_size: u32,

	/// Overlap in pixels between adjacent tiles.
	pub tile_overlap: u32,

	/// Optional callback invoked after each tile completes.
	/// Arguments: (tiles_completed, total_tiles).
	pub on_tile_done: Option<Box<dyn Fn(usize, usize) + Send + Sync>>,

	/// Cancellation token — checked between tiles.
	pub cancel: CancelToken,
}

impl Default for UpscaleOptions {
	fn default() -> Self {
		Self {
			tile_size: crate::config::DEFAULT_TILE_SIZE,
			tile_overlap: crate::config::DEFAULT_TILE_OVERLAP,
			on_tile_done: None,
			cancel: CancelToken::new(),
		}
	}
}

/// Upscale a single image using the provided session and model info.
///
/// This is the primary library entry point for upscaling. It handles:
/// - Converting the input image to the appropriate tensor format.
/// - Computing the tile grid (or running whole-image if tiling disabled).
/// - Running inference on each tile.
/// - Blending overlapping tiles with Hann-window weights.
/// - Converting the output tensor back to a `DynamicImage`.
///
/// The caller is responsible for loading/saving images and displaying
/// progress — this function only reports via `options.on_tile_done`.
pub fn upscale_image(
	ctx: &mut SessionContext,
	input: &DynamicImage,
	options: &UpscaleOptions,
) -> Result<DynamicImage> {
	// Clone model_info to avoid holding an immutable borrow on ctx
	// while passing &mut ctx to run_tile.
	let model_info = ctx.model_info.clone();
	let scale = model_info.scale;
	let out_channels = model_info.output_channels;
	let (img_w, img_h) = (input.width(), input.height());
	let out_w = img_w * scale;
	let out_h = img_h * scale;

	// Compute tile grid.
	let tiles = compute_tile_grid(
		img_w,
		img_h,
		&model_info,
		options.tile_size,
		options.tile_overlap,
	)?;
	let total_tiles = tiles.len();

	// Allocate output accumulators.
	let mut canvas =
		Array4::<f32>::zeros((1, out_channels as usize, out_h as usize, out_w as usize));
	let mut weight_map = Array2::<f32>::zeros((out_h as usize, out_w as usize));

	for (idx, tile) in tiles.iter().enumerate() {
		// Check cancellation between tiles.
		if options.cancel.is_cancelled() {
			bail!("Cancelled");
		}

		let tile_output = run_tile(ctx, input, tile, &model_info)?;

		// Blend tile into canvas.
		let scaled_overlap = options.tile_overlap * scale;
		accumulate_tile(
			&mut canvas,
			&mut weight_map,
			&tile_output,
			tile,
			scaled_overlap,
		);

		if let Some(ref cb) = options.on_tile_done {
			cb(idx + 1, total_tiles);
		}
	}

	// Normalise canvas by accumulated weights.
	normalise_canvas(&mut canvas, &weight_map);

	tensor_to_image(canvas.view(), out_channels)
}

/// Run inference on a single tile.
fn run_tile(
	ctx: &mut SessionContext,
	input: &DynamicImage,
	tile: &Tile,
	model_info: &ModelInfo,
) -> Result<Array4<f32>> {
	let in_channels = model_info.input_channels;
	let out_channels = model_info.output_channels as usize;
	let scale = model_info.scale as usize;

	// Crop the source region from the input image.
	let cropped = input.crop_imm(tile.src.x, tile.src.y, tile.src.width, tile.src.height);

	// Convert to NCHW f32 tensor.
	let mut tensor = image_to_tensor(&cropped, in_channels)?;

	// Apply mirror padding if needed.
	if !tile.padding.is_zero() {
		tensor = pad_tensor_mirror(&tensor, tile.padding);
	}

	// Compute expected output shape (input spatial dims × scale).
	let in_h = tensor.shape()[2];
	let in_w = tensor.shape()[3];
	let expected_shape = (1, out_channels, in_h * scale, in_w * scale);

	// Run inference.
	let output_tensor = if model_info.needs_fp16_input() {
		let f16_tensor = tensor_f32_to_f16(&tensor);
		let input_value = ort::value::Value::from_array(f16_tensor)
			.map_err(|e| anyhow::anyhow!("Failed to create ORT value from fp16 tensor: {e}"))?;
		let outputs = ctx
			.session
			.run(ort::inputs![input_value])
			.map_err(|e| anyhow::anyhow!("Inference failed: {e}"))?;
		extract_output_f32(&outputs, expected_shape)?
	} else {
		let input_value = ort::value::Value::from_array(tensor)
			.map_err(|e| anyhow::anyhow!("Failed to create ORT value from f32 tensor: {e}"))?;
		let outputs = ctx
			.session
			.run(ort::inputs![input_value])
			.map_err(|e| anyhow::anyhow!("Inference failed: {e}"))?;
		extract_output_f32(&outputs, expected_shape)?
	};

	// Remove padding from output (scale-adjusted).
	Ok(crop_tensor(
		output_tensor.view(),
		tile.padding,
		model_info.scale,
	))
}

/// Extract the output tensor as f32, handling both f32 and f16 outputs.
fn extract_output_f32(
	outputs: &ort::session::SessionOutputs<'_>,
	expected_shape: (usize, usize, usize, usize),
) -> Result<Array4<f32>> {
	let (_, output) = outputs
		.iter()
		.next()
		.ok_or_else(|| anyhow::anyhow!("Model produced no outputs"))?;

	// Try f32 first.
	if let Ok((_shape, data)) = output.try_extract_tensor::<f32>() {
		return Array4::from_shape_vec(expected_shape, data.to_vec())
			.map_err(|e| anyhow::anyhow!("Failed to reshape f32 output: {e}"));
	}

	// Try f16.
	if let Ok((_shape, data)) = output.try_extract_tensor::<half::f16>() {
		let f32_data: Vec<f32> = data.iter().map(|v| half::f16::to_f32(*v)).collect();
		return Array4::from_shape_vec(expected_shape, f32_data)
			.map_err(|e| anyhow::anyhow!("Failed to reshape f16 output: {e}"));
	}

	bail!("Unsupported output tensor dtype (expected float32 or float16)")
}

/// Accumulate a tile's output into the canvas with Hann-window blending.
fn accumulate_tile(
	canvas: &mut Array4<f32>,
	weight_map: &mut Array2<f32>,
	tile_output: &Array4<f32>,
	tile: &Tile,
	scaled_overlap: u32,
) {
	let tile_h = tile_output.shape()[2] as u32;
	let tile_w = tile_output.shape()[3] as u32;
	let channels = tile_output.shape()[1];

	let weights = blend_weights(tile_w, tile_h, scaled_overlap);

	let dst_y = tile.dst.y as usize;
	let dst_x = tile.dst.x as usize;

	for y in 0..tile_h as usize {
		for x in 0..tile_w as usize {
			let wy = dst_y + y;
			let wx = dst_x + x;
			if wy >= canvas.shape()[2] || wx >= canvas.shape()[3] {
				continue;
			}

			let w = weights[[y, x]];
			weight_map[[wy, wx]] += w;

			for c in 0..channels {
				canvas[[0, c, wy, wx]] += tile_output[[0, c, y, x]] * w;
			}
		}
	}
}

/// Normalise canvas by dividing each pixel by its accumulated weight.
fn normalise_canvas(canvas: &mut Array4<f32>, weight_map: &Array2<f32>) {
	let h = canvas.shape()[2];
	let w = canvas.shape()[3];
	let c = canvas.shape()[1];

	for y in 0..h {
		for x in 0..w {
			let wt = weight_map[[y, x]];
			if wt > 0.0 {
				for ch in 0..c {
					canvas[[0, ch, y, x]] /= wt;
				}
			}
		}
	}
}
