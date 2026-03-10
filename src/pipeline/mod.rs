//! Upscaling pipeline: tiling, tensor conversion, and inference.

pub mod blend;
pub mod tensor;
pub mod tiling;

use anyhow::{Result, bail};
use image::DynamicImage;
use ndarray::{Array2, Array4};
use ort::session::Session;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::inspect::ModelInfo;
use crate::session::SessionContext;

use tensor::{crop_tensor, extract_output_f32, image_to_tensor, pad_tensor_mirror, tensor_f32_to_f16, tensor_to_image};
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

	/// Optional callback invoked after each major blend step completes.
	/// Arguments: (steps_completed, total_steps).
	/// Only called when `blend > 0.0`.
	pub on_blend_step: Option<Box<dyn Fn(usize, usize) + Send + Sync>>,

	/// Cancellation token — checked between tiles.
	pub cancel: CancelToken,

	/// Frequency-domain blend strength \[0.0, 1.0\].
	///
	/// `0.0` (default) — pure AI output, blending disabled.
	/// `1.0` — AI supplies fine detail; Lanczos upscale of the original
	///         supplies global structure (colour, tone, large shapes).
	pub blend: f32,

	/// Force half-precision (fp16) input regardless of model metadata.
	pub force_fp16: bool,
}

impl Default for UpscaleOptions {
	fn default() -> Self {
		Self {
			tile_size: crate::config::DEFAULT_TILE_SIZE,
			tile_overlap: crate::config::DEFAULT_TILE_OVERLAP,
			on_tile_done: None,
			on_blend_step: None,
			cancel: CancelToken::new(),
			blend: 0.0,
			force_fp16: false,
		}
	}
}

pub fn upscale_raw(
	ctx: &mut SessionContext,
	input: &DynamicImage,
	options: &UpscaleOptions,
) -> Result<DynamicImage> {
	let model_info = &ctx.model_info;
	let (img_w, img_h) = (input.width(), input.height());

	let tiles = compute_tile_grid(
		img_w,
		img_h,
		model_info,
		options.tile_size,
		options.tile_overlap,
	)?;
	let total_tiles = tiles.len();

	let use_fp16 = options.force_fp16 || model_info.needs_fp16_input();

	let mut canvas: Option<Array4<f32>> = None;
	let mut weight_map: Option<Array2<f32>> = None;
	let mut actual_scale = model_info.scale;
	let mut weight_cache: HashMap<(u32, u32), Array2<f32>> = HashMap::new();

	for (idx, tile) in tiles.iter().enumerate() {
		if options.cancel.is_cancelled() {
			bail!("Cancelled");
		}

		let (tile_output, tile_scale) = run_tile(&mut ctx.session, input, tile, model_info, use_fp16)?;

		if canvas.is_none() {
			actual_scale = tile_scale;
			let out_channels = tile_output.shape()[1];
			let out_w = img_w * actual_scale;
			let out_h = img_h * actual_scale;
			canvas = Some(Array4::<f32>::zeros((
				1,
				out_channels,
				out_h as usize,
				out_w as usize,
			)));
			weight_map = Some(Array2::<f32>::zeros((out_h as usize, out_w as usize)));
		}

		// Both are initialised on the first iteration above.
		let canvas = canvas.as_mut().unwrap();
		let weight_map = weight_map.as_mut().unwrap();

		let actual_dst_x = tile.src.x * actual_scale;
		let actual_dst_y = tile.src.y * actual_scale;
		let scaled_overlap = options.tile_overlap * actual_scale;

		let tile_h = tile_output.shape()[2] as u32;
		let tile_w = tile_output.shape()[3] as u32;
		let weights = weight_cache
			.entry((tile_h, tile_w))
			.or_insert_with(|| blend_weights(tile_w, tile_h, scaled_overlap));

		accumulate_tile(
			canvas,
			weight_map,
			&tile_output,
			actual_dst_x,
			actual_dst_y,
			weights,
		);

		if let Some(ref cb) = options.on_tile_done {
			cb(idx + 1, total_tiles);
		}
	}

	let (mut canvas, weight_map) = canvas
		.zip(weight_map)
		.ok_or_else(|| anyhow::anyhow!("No tiles were processed"))?;
	let out_channels = canvas.shape()[1] as u32;

	normalise_canvas(&mut canvas, &weight_map);

	tensor_to_image(canvas.view(), out_channels)
}

/// Upscale a single image with optional frequency-domain blending.
///
/// Convenience wrapper around [`upscale_raw`] that also applies blending
/// when `options.blend > 0.0`.
pub fn upscale_image(
	ctx: &mut SessionContext,
	input: &DynamicImage,
	options: &UpscaleOptions,
) -> Result<DynamicImage> {
	let ai_result = upscale_raw(ctx, input, options)?;

	if options.blend > 0.0 {
		let blend_cb = options
			.on_blend_step
			.as_ref()
			.map(|cb| cb.as_ref() as &dyn Fn(usize, usize));
		blend::frequency_blend_with_original(&ai_result, input, options.blend, blend_cb)
	} else {
		Ok(ai_result)
	}
}

/// Run inference on a single tile.
///
/// Returns the cropped output tensor together with the scale factor the model
/// actually applied (derived from the ratio of output to input spatial dims).
/// This may differ from `model_info.scale` when static scale detection fails
/// (e.g. Resize-based upscalers that have no `DepthToSpace` or `ConvTranspose`).
fn run_tile(
	session: &mut Session,
	input: &DynamicImage,
	tile: &Tile,
	model_info: &ModelInfo,
	use_fp16: bool,
) -> Result<(Array4<f32>, u32)> {
	let in_channels = model_info.input_channels;

	// Crop the source region from the input image.
	let cropped = input.crop_imm(tile.src.x, tile.src.y, tile.src.width, tile.src.height);

	// Convert to NCHW f32 tensor.
	let mut tensor = image_to_tensor(&cropped, in_channels)?;

	// Apply mirror padding if needed.
	if !tile.padding.is_zero() {
		tensor = pad_tensor_mirror(&tensor, tile.padding);
	}

	// Record padded input height to derive the actual scale after inference.
	let in_h = tensor.shape()[2];

	// Run inference — output shape is determined by the model, not by us.
	let raw_output = if use_fp16 {
		let f16_tensor = tensor_f32_to_f16(&tensor);
		let input_value = ort::value::Value::from_array(f16_tensor)
			.map_err(|e| anyhow::anyhow!("Failed to create ORT value from fp16 tensor: {e}"))?;
		let outputs = session
			.run(ort::inputs![input_value])
			.map_err(|e| anyhow::anyhow!("Inference failed: {e}"))?;
		extract_output_f32(&outputs)?
	} else {
		let input_value = ort::value::Value::from_array(tensor)
			.map_err(|e| anyhow::anyhow!("Failed to create ORT value from f32 tensor: {e}"))?;
		let outputs = session
			.run(ort::inputs![input_value])
			.map_err(|e| anyhow::anyhow!("Inference failed: {e}"))?;
		extract_output_f32(&outputs)?
	};

	// Derive the actual scale from the ratio of output to (padded) input height.
	// Use 1 as the minimum so restoration-only models (scale = 1) still work.
	let actual_scale = (raw_output.shape()[2] / in_h).max(1) as u32;

	// Remove padding from output using the *actual* scale, not the declared one.
	let tile_output = crop_tensor(raw_output.view(), tile.padding, actual_scale);

	Ok((tile_output, actual_scale))
}

/// Accumulate a tile's output into the canvas with Hann-window blending.
fn accumulate_tile(
	canvas: &mut Array4<f32>,
	weight_map: &mut Array2<f32>,
	tile_output: &Array4<f32>,
	dst_x: u32,
	dst_y: u32,
	weights: &Array2<f32>,
) {
	let tile_h = tile_output.shape()[2] as u32;
	let tile_w = tile_output.shape()[3] as u32;
	let channels = tile_output.shape()[1];

	let dst_y = dst_y as usize;
	let dst_x = dst_x as usize;

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
