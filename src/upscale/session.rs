//! Upscale session management with fp16 support.

use anyhow::{Context, Result};
use colored::Colorize;
use image::{DynamicImage, GenericImageView};
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::{s, Array2, Array4, ArrayView4, ArrayViewD};
use ort::session::builder::{AutoDevicePolicy, SessionBuilder};
use ort::session::Session;
use ort::value::Value;
use std::path::Path;

use crate::inspect::{inspect_model, ModelInfo};

use super::provider::Provider;
use super::tiling::{compute_tile_grid, generate_blend_weights, TileConfig};
use super::vram::estimate_tile_size;

/// Configuration options for upscaling.
pub struct UpscaleOptions {
	pub provider: Provider,
	pub tile_size: Option<u32>,
	pub overlap: u32,
}

impl Default for UpscaleOptions {
	fn default() -> Self {
		Self {
			provider: Provider::Auto,
			tile_size: None,
			overlap: 16,
		}
	}
}

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
			.with_context(|| format!("Failed to inspect model: {}", model_path.display()))?;

		let (ep_dispatch, actual_provider, warning) = options.provider.build();

		if let Some(warn_msg) = warning {
			eprintln!("{}", warn_msg);
		}

		let mut builder = SessionBuilder::new().context("Failed to create session builder")?;

		// Use auto_device if no providers specified (Auto variant)
		if ep_dispatch.is_empty() {
			builder = builder
				.with_auto_device(AutoDevicePolicy::MaxPerformance)
				.map_err(|e| anyhow::anyhow!("Failed to enable auto device selection: {}", e))?;
		} else {
			for ep in ep_dispatch {
				builder = builder.with_execution_providers([ep]).map_err(|e| {
					anyhow::anyhow!("Failed to configure execution provider: {}", e)
				})?;
			}
		}

		let session = builder
			.commit_from_file(model_path)
			.with_context(|| format!("Failed to load model: {}", model_path.display()))?;

		if !options.provider.name().eq_ignore_ascii_case("cpu") {
			eprintln!(
				"{} Using {} for inference",
				"✓".green(),
				actual_provider.name().bright_green()
			);
		}

		// Determine tile configuration
		let tile_size = options
			.tile_size
			.unwrap_or_else(|| estimate_tile_size(&model_info, &actual_provider));

		let tile_config = TileConfig {
			tile_size,
			overlap: options.overlap,
		};

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

	/// Upscale an image using tiled inference.
	pub fn upscale(&mut self, image: DynamicImage) -> Result<DynamicImage> {
		let (width, height) = image.dimensions();
		let scale = self.model_info.scale;

		// Compute tile grid
		let tiles = compute_tile_grid(width, height, &self.model_info, &self.tile_config)?;

		// Allocate output and weight canvases
		let output_width = width * scale;
		let output_height = height * scale;

		// Setup progress bar
		let pb = ProgressBar::new(tiles.len() as u64);
		pb.set_style(
			ProgressStyle::default_bar()
				.template(&format!(
					"  {} {{bar:40.cyan/blue}} {{pos}}/{{len}} tiles | {{msg}}",
					"→".bright_blue()
				))
				.unwrap()
				.progress_chars("━━╸"),
		);
		pb.set_message(format!(
			"{}×{} → {}×{}",
			width, height, output_width, output_height
		));
		let mut output_canvas =
			Array4::<f32>::zeros((1, 3, output_height as usize, output_width as usize));
		let mut weight_canvas =
			Array2::<f32>::zeros((output_height as usize, output_width as usize));

		// Process each tile
		let start_time = std::time::Instant::now();
		for (tile_idx, tile) in tiles.iter().enumerate() {
			// Extract tile from input image
			let tile_img = image.crop_imm(
				tile.src_rect.0,
				tile.src_rect.1,
				tile.src_rect.2,
				tile.src_rect.3,
			);

			// Convert to tensor and add padding
			let mut tile_tensor = self.image_to_tensor(&tile_img)?;

			// Apply padding if needed
			if tile.padding != (0, 0, 0, 0) {
				tile_tensor = self.pad_tensor(tile_tensor, tile.padding)?;
			}

			// Run inference on tile
			let output_tile = self.infer_tile(tile_tensor)?;

			// Remove padding from output
			let valid_output = self.crop_tensor(
				output_tile.view(),
				tile.padding.0 * scale,
				tile.padding.1 * scale,
				tile.padding.2 * scale,
				tile.padding.3 * scale,
			)?;

			// Generate blend weights for this tile
			let tile_out_w = tile.src_rect.2 * scale;
			let tile_out_h = tile.src_rect.3 * scale;
			let overlap_scaled = self.tile_config.overlap * scale;

			let weights = if self.model_info.tile.fixed_size.is_some() {
				// Fixed-size models: no blending (no overlap)
				Array2::<f32>::ones((tile_out_h as usize, tile_out_w as usize))
			} else {
				generate_blend_weights(tile_out_w, tile_out_h, overlap_scaled)?
			};

			// Accumulate into output canvas
			let dst_x = tile.dst_rect.0 as usize;
			let dst_y = tile.dst_rect.1 as usize;
			let dst_w = tile_out_w as usize;
			let dst_h = tile_out_h as usize;

			for c in 0..3 {
				for y in 0..dst_h {
					for x in 0..dst_w {
						let canvas_y = dst_y + y;
						let canvas_x = dst_x + x;
						if canvas_y < output_height as usize && canvas_x < output_width as usize {
							let weight = weights[[y, x]];
							output_canvas[[0, c, canvas_y, canvas_x]] +=
								valid_output[[0, c, y, x]] * weight;
							if c == 0 {
								weight_canvas[[canvas_y, canvas_x]] += weight;
							}
						}
					}
				}
			}

			pb.inc(1);

			// Update ETA after each tile
			let elapsed = start_time.elapsed().as_secs_f32();
			let tiles_done = tile_idx + 1;
			let avg_time = elapsed / tiles_done as f32;
			let remaining_tiles = tiles.len() - tiles_done;
			let eta_secs = remaining_tiles as f32 * avg_time;

			if remaining_tiles > 0 {
				let eta_str = if eta_secs < 60.0 {
					format!("{:.0}s", eta_secs)
				} else {
					format!("{:.0}m {:.0}s", eta_secs / 60.0, eta_secs % 60.0)
				};
				pb.set_message(format!(
					"{}×{} → {}×{} | ETA: {} | {:.1}s/tile",
					width, height, output_width, output_height, eta_str, avg_time
				));
			}
		}

		let total_time = start_time.elapsed().as_secs_f32();
		let time_str = if total_time < 60.0 {
			format!("{:.1}s", total_time)
		} else {
			format!("{:.0}m {:.0}s", total_time / 60.0, total_time % 60.0)
		};
		pb.finish_with_message(format!(
			"✓ {} tiles completed in {} ({:.1}s/tile)",
			tiles.len(),
			time_str,
			total_time / tiles.len() as f32
		));

		// Normalize by weights
		for c in 0..3 {
			for y in 0..(output_height as usize) {
				for x in 0..(output_width as usize) {
					let weight = weight_canvas[[y, x]];
					if weight > 0.0 {
						output_canvas[[0, c, y, x]] /= weight;
					}
				}
			}
		}

		self.tensor_to_image(output_canvas.view())
	}

	/// Run inference on a single tile tensor.
	fn infer_tile(&mut self, input_tensor: Array4<f32>) -> Result<Array4<f32>> {
		let is_fp16 = self.model_info.input_dtype.eq_ignore_ascii_case("float16");

		let outputs = if is_fp16 {
			// Convert f32 tensor to f16
			let shape = input_tensor.shape();
			let f16_data: Vec<half::f16> = input_tensor
				.iter()
				.map(|&v| half::f16::from_f32(v))
				.collect();
			let f16_tensor = Array4::<half::f16>::from_shape_vec(
				(shape[0], shape[1], shape[2], shape[3]),
				f16_data,
			)
			.context("Failed to create fp16 tensor")?;

			let input_value = Value::from_array(f16_tensor)?;
			self.session
				.run(ort::inputs![input_value])
				.context("Inference failed")?
		} else {
			// Use f32 tensor directly
			let input_value = Value::from_array(input_tensor)?;
			self.session
				.run(ort::inputs![input_value])
				.context("Inference failed")?
		};

		// Extract output
		let output_tensor = if is_fp16 {
			let output_view: ArrayViewD<half::f16> = outputs[0]
				.try_extract_array()
				.context("Failed to extract fp16 output tensor")?;
			let output_4d: ArrayView4<half::f16> = output_view
				.into_dimensionality()
				.context("Failed to reshape output tensor")?;

			// Convert f16 back to f32
			let shape = output_4d.shape();
			let f32_data: Vec<f32> = output_4d.iter().map(|&v| v.to_f32()).collect();
			Array4::<f32>::from_shape_vec((shape[0], shape[1], shape[2], shape[3]), f32_data)
				.context("Failed to convert fp16 output to fp32")?
		} else {
			let output_view: ArrayViewD<f32> = outputs[0]
				.try_extract_array()
				.context("Failed to extract output tensor")?;
			let output_4d: ArrayView4<f32> = output_view
				.into_dimensionality()
				.context("Failed to reshape output tensor")?;
			output_4d.to_owned()
		};

		Ok(output_tensor)
	}

	/// Convert image to normalized tensor.
	fn image_to_tensor(&self, image: &DynamicImage) -> Result<Array4<f32>> {
		let rgb = image.to_rgb8();
		let (width, height) = rgb.dimensions();

		let mut tensor = Array4::<f32>::zeros((1, 3, height as usize, width as usize));

		for y in 0..height {
			for x in 0..width {
				let pixel = rgb.get_pixel(x, y);
				tensor[[0, 0, y as usize, x as usize]] = pixel[0] as f32 / 255.0;
				tensor[[0, 1, y as usize, x as usize]] = pixel[1] as f32 / 255.0;
				tensor[[0, 2, y as usize, x as usize]] = pixel[2] as f32 / 255.0;
			}
		}

		Ok(tensor)
	}

	/// Add zero padding to a tensor.
	fn pad_tensor(
		&self,
		tensor: Array4<f32>,
		padding: (u32, u32, u32, u32),
	) -> Result<Array4<f32>> {
		let (pad_left, pad_top, pad_right, pad_bottom) = padding;
		let shape = tensor.shape();
		let (batch, channels, height, width) = (shape[0], shape[1], shape[2], shape[3]);

		let new_height = height + pad_top as usize + pad_bottom as usize;
		let new_width = width + pad_left as usize + pad_right as usize;

		let mut padded = Array4::<f32>::zeros((batch, channels, new_height, new_width));

		// Copy original data into center
		for b in 0..batch {
			for c in 0..channels {
				for y in 0..height {
					for x in 0..width {
						padded[[b, c, y + pad_top as usize, x + pad_left as usize]] =
							tensor[[b, c, y, x]];
					}
				}
			}
		}

		Ok(padded)
	}

	/// Crop a tensor by removing padding.
	fn crop_tensor(
		&self,
		tensor: ArrayView4<f32>,
		pad_left: u32,
		pad_top: u32,
		pad_right: u32,
		pad_bottom: u32,
	) -> Result<Array4<f32>> {
		let shape = tensor.shape();
		let (_batch, _channels, height, width) = (shape[0], shape[1], shape[2], shape[3]);

		let crop_height = height - pad_top as usize - pad_bottom as usize;
		let crop_width = width - pad_left as usize - pad_right as usize;

		let cropped = tensor
			.slice(s![
				..,
				..,
				pad_top as usize..pad_top as usize + crop_height,
				pad_left as usize..pad_left as usize + crop_width
			])
			.to_owned();

		Ok(cropped)
	}

	/// Convert output tensor to image.
	fn tensor_to_image(&self, tensor: ArrayView4<f32>) -> Result<DynamicImage> {
		let shape = tensor.shape();
		anyhow::ensure!(shape[0] == 1, "Expected batch size 1, got {}", shape[0]);

		let _channels = shape[1];
		let height = shape[2];
		let width = shape[3];

		let mut imgbuf = image::RgbImage::new(width as u32, height as u32);

		for y in 0..height {
			for x in 0..width {
				let r = (tensor[[0, 0, y, x]].clamp(0.0, 1.0) * 255.0) as u8;
				let g = (tensor[[0, 1, y, x]].clamp(0.0, 1.0) * 255.0) as u8;
				let b = (tensor[[0, 2, y, x]].clamp(0.0, 1.0) * 255.0) as u8;
				imgbuf.put_pixel(x as u32, y as u32, image::Rgb([r, g, b]));
			}
		}

		Ok(DynamicImage::ImageRgb8(imgbuf))
	}
}
