//! Upscale session management with fp16 support.

use anyhow::{Context, Result};
use image::{DynamicImage, GenericImageView};
use ndarray::{s, Array2, Array4, ArrayView4, ArrayViewD};
use ort::session::builder::{AutoDevicePolicy, SessionBuilder};
use ort::session::Session;
use ort::value::Value;
use std::path::Path;

use crate::inspect::{inspect_model, ModelInfo};

use super::provider::Provider;
use super::tile_size::effective_tile_size;
use super::tiling::{compute_tile_grid, generate_blend_weights, TileConfig};

// ── Options ────────────────────────────────────────────────────────────────

/// Configuration options for upscaling.
pub struct UpscaleOptions {
	pub provider: Provider,
}

impl Default for UpscaleOptions {
	fn default() -> Self {
		Self {
			provider: Provider::Auto,
		}
	}
}

// ── Session ────────────────────────────────────────────────────────────────

/// Session for upscaling images with ONNX models.
pub struct UpscaleSession {
	session: Session,
	model_info: ModelInfo,
	actual_provider: Provider,
	tile_config: TileConfig,
}

impl UpscaleSession {
	/// Create a new upscale session from a model file.
	pub fn new(model_path: &Path, options: &UpscaleOptions) -> Result<Self> {
		let model_info = inspect_model(model_path)
			.with_context(|| format!("Failed to inspect model '{}'", model_path.display()))?;

		let (ep_dispatch, actual_provider, warning) = options.provider.build();

		if let Some(warn_msg) = warning {
			eprintln!("{warn_msg}");
		}

		let mut builder =
			SessionBuilder::new().context("Failed to create ONNX Runtime session builder")?;

		if ep_dispatch.is_empty() {
			builder = builder
				.with_auto_device(AutoDevicePolicy::MaxPerformance)
				.map_err(|e| anyhow::anyhow!("Auto-device selection failed: {e}"))?;
		} else {
			for ep in ep_dispatch {
				builder = builder
					.with_execution_providers([ep])
					.map_err(|e| anyhow::anyhow!("Failed to configure execution provider: {e}"))?;
			}
		}

		let session = builder
			.commit_from_file(model_path)
			.with_context(|| format!("Failed to load ONNX model '{}'", model_path.display()))?;

		let tile_size = effective_tile_size(&model_info);
		let overlap: u32 = 16;

		let tile_config = TileConfig { tile_size, overlap };

		Ok(Self {
			session,
			model_info,
			actual_provider,
			tile_config,
		})
	}

	/// Get model information.
	pub fn model_info(&self) -> &ModelInfo {
		&self.model_info
	}

	/// Get the actual provider being used.
	pub fn provider(&self) -> Provider {
		self.actual_provider
	}

	/// Get the tile configuration.
	pub fn tile_config(&self) -> &TileConfig {
		&self.tile_config
	}

	/// Upscale an image using tiled inference.
	///
	/// `on_tile` is called after each tile completes with `(completed, total)`.
	pub fn upscale<F>(&mut self, image: DynamicImage, mut on_tile: F) -> Result<DynamicImage>
	where
		F: FnMut(usize, usize),
	{
		let (width, height) = image.dimensions();
		let scale = self.model_info.scale;

		let tiles = compute_tile_grid(width, height, &self.model_info, &self.tile_config)?;
		let total_tiles = tiles.len();

		let output_width = width * scale;
		let output_height = height * scale;

		let mut output_canvas =
			Array4::<f32>::zeros((1, 3, output_height as usize, output_width as usize));
		let mut weight_canvas =
			Array2::<f32>::zeros((output_height as usize, output_width as usize));

		for (tile_idx, tile) in tiles.iter().enumerate() {
			let tile_img = image.crop_imm(
				tile.src_rect.0,
				tile.src_rect.1,
				tile.src_rect.2,
				tile.src_rect.3,
			);

			let mut tile_tensor = image_to_tensor(&tile_img);

			if tile.padding != (0, 0, 0, 0) {
				tile_tensor = pad_tensor(tile_tensor, tile.padding);
			}

			let output_tile = self.infer_tile(tile_tensor)?;

			let valid_output = crop_tensor(
				output_tile.view(),
				tile.padding.0 * scale,
				tile.padding.1 * scale,
				tile.padding.2 * scale,
				tile.padding.3 * scale,
			);

			let tile_out_w = tile.src_rect.2 * scale;
			let tile_out_h = tile.src_rect.3 * scale;
			let overlap_scaled = self.tile_config.overlap * scale;

			let weights = if self.model_info.tile.fixed_size.is_some() {
				Array2::<f32>::ones((tile_out_h as usize, tile_out_w as usize))
			} else {
				generate_blend_weights(tile_out_w, tile_out_h, overlap_scaled)
			};

			accumulate_tile(
				&mut output_canvas,
				&mut weight_canvas,
				&valid_output,
				&weights,
				tile.dst_rect.0 as usize,
				tile.dst_rect.1 as usize,
				tile_out_w as usize,
				tile_out_h as usize,
				output_width as usize,
				output_height as usize,
			);

			on_tile(tile_idx + 1, total_tiles);
		}

		normalize_canvas(&mut output_canvas, &weight_canvas);

		tensor_to_image(output_canvas.view())
	}

	/// Run inference on a single tile tensor.
	fn infer_tile(&mut self, input_tensor: Array4<f32>) -> Result<Array4<f32>> {
		let is_fp16 = self.model_info.needs_fp16_input();

		let outputs = if is_fp16 {
			let shape = input_tensor.shape();
			let f16_data: Vec<half::f16> = input_tensor
				.iter()
				.map(|&v| half::f16::from_f32(v))
				.collect();
			let f16_tensor = Array4::<half::f16>::from_shape_vec(
				(shape[0], shape[1], shape[2], shape[3]),
				f16_data,
			)
			.context("Failed to create fp16 input tensor")?;

			let input_value = Value::from_array(f16_tensor)
				.context("Failed to create ONNX input value (fp16)")?;
			self.session
				.run(ort::inputs![input_value])
				.context("Inference failed on tile (fp16 model)")?
		} else {
			let input_value = Value::from_array(input_tensor)
				.context("Failed to create ONNX input value (fp32)")?;
			self.session
				.run(ort::inputs![input_value])
				.context("Inference failed on tile (fp32 model)")?
		};

		if is_fp16 {
			let view: ArrayViewD<half::f16> = outputs[0]
				.try_extract_array()
				.context("Failed to extract fp16 output tensor")?;
			let view4: ArrayView4<half::f16> = view
				.into_dimensionality()
				.context("Output tensor is not 4-dimensional")?;
			let shape = view4.shape();
			let f32_data: Vec<f32> = view4.iter().map(|&v| v.to_f32()).collect();
			Array4::<f32>::from_shape_vec((shape[0], shape[1], shape[2], shape[3]), f32_data)
				.context("Failed to reshape fp16→fp32 output tensor")
		} else {
			let view: ArrayViewD<f32> = outputs[0]
				.try_extract_array()
				.context("Failed to extract fp32 output tensor")?;
			let view4: ArrayView4<f32> = view
				.into_dimensionality()
				.context("Output tensor is not 4-dimensional")?;
			Ok(view4.to_owned())
		}
	}
}

// ── Tensor Helpers (free functions) ────────────────────────────────────────

/// Convert an image to a normalised [0, 1] NCHW tensor.
fn image_to_tensor(image: &DynamicImage) -> Array4<f32> {
	let rgb = image.to_rgb8();
	let (w, h) = rgb.dimensions();
	let mut tensor = Array4::<f32>::zeros((1, 3, h as usize, w as usize));

	for (x, y, pixel) in rgb.enumerate_pixels() {
		let (xu, yu) = (x as usize, y as usize);
		tensor[[0, 0, yu, xu]] = pixel[0] as f32 / 255.0;
		tensor[[0, 1, yu, xu]] = pixel[1] as f32 / 255.0;
		tensor[[0, 2, yu, xu]] = pixel[2] as f32 / 255.0;
	}

	tensor
}

/// Zero-pad a tensor on all four sides.
fn pad_tensor(tensor: Array4<f32>, padding: (u32, u32, u32, u32)) -> Array4<f32> {
	let (pl, pt, pr, pb) = padding;
	let [batch, ch, h, w] = *tensor.shape() else {
		unreachable!()
	};
	let nh = h + pt as usize + pb as usize;
	let nw = w + pl as usize + pr as usize;
	let mut out = Array4::<f32>::zeros((batch, ch, nh, nw));

	out.slice_mut(s![
		..,
		..,
		pt as usize..pt as usize + h,
		pl as usize..pl as usize + w
	])
	.assign(&tensor);

	out
}

/// Crop padding from a tensor.
fn crop_tensor(tensor: ArrayView4<f32>, pl: u32, pt: u32, pr: u32, pb: u32) -> Array4<f32> {
	let [_b, _c, h, w] = *tensor.shape() else {
		unreachable!()
	};
	let ch = h - pt as usize - pb as usize;
	let cw = w - pl as usize - pr as usize;

	tensor
		.slice(s![
			..,
			..,
			pt as usize..pt as usize + ch,
			pl as usize..pl as usize + cw
		])
		.to_owned()
}

/// Accumulate a processed tile into the output canvas using blend weights.
fn accumulate_tile(
	canvas: &mut Array4<f32>,
	weights_map: &mut Array2<f32>,
	tile: &Array4<f32>,
	weights: &Array2<f32>,
	dst_x: usize,
	dst_y: usize,
	tw: usize,
	th: usize,
	canvas_w: usize,
	canvas_h: usize,
) {
	for c in 0..3 {
		for y in 0..th {
			for x in 0..tw {
				let cy = dst_y + y;
				let cx = dst_x + x;
				if cy < canvas_h && cx < canvas_w {
					let w = weights[[y, x]];
					canvas[[0, c, cy, cx]] += tile[[0, c, y, x]] * w;
					if c == 0 {
						weights_map[[cy, cx]] += w;
					}
				}
			}
		}
	}
}

/// Divide accumulated pixel values by their total weight.
fn normalize_canvas(canvas: &mut Array4<f32>, weights: &Array2<f32>) {
	let [_, _, h, w] = *canvas.shape() else {
		unreachable!()
	};
	for c in 0..3 {
		for y in 0..h {
			for x in 0..w {
				let wt = weights[[y, x]];
				if wt > 0.0 {
					canvas[[0, c, y, x]] /= wt;
				}
			}
		}
	}
}

/// Convert a [0, 1] NCHW tensor back to an RGB image.
fn tensor_to_image(tensor: ArrayView4<f32>) -> Result<DynamicImage> {
	let shape = tensor.shape();
	anyhow::ensure!(shape[0] == 1, "Expected batch size 1 but got {}", shape[0]);

	let (h, w) = (shape[2], shape[3]);
	let mut buf = image::RgbImage::new(w as u32, h as u32);

	for y in 0..h {
		for x in 0..w {
			let r = (tensor[[0, 0, y, x]].clamp(0.0, 1.0) * 255.0) as u8;
			let g = (tensor[[0, 1, y, x]].clamp(0.0, 1.0) * 255.0) as u8;
			let b = (tensor[[0, 2, y, x]].clamp(0.0, 1.0) * 255.0) as u8;
			buf.put_pixel(x as u32, y as u32, image::Rgb([r, g, b]));
		}
	}

	Ok(DynamicImage::ImageRgb8(buf))
}
