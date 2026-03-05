//! Upscale session management with fp16 support.

use anyhow::{Context, Result};
use colored::Colorize;
use image::DynamicImage;
use ndarray::{Array4, ArrayView4, ArrayViewD};
use ort::session::builder::{AutoDevicePolicy, SessionBuilder};
use ort::session::Session;
use ort::value::Value;
use std::path::Path;

use crate::inspect::{inspect_model, ModelInfo};

use super::provider::Provider;

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

/// Session for upscaling images with ONNX models.
pub struct UpscaleSession {
	session: Session,
	model_info: ModelInfo,
	actual_provider: Provider,
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

		Ok(Self {
			session,
			model_info,
			actual_provider,
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

	/// Upscale an image.
	pub fn upscale(&mut self, image: DynamicImage) -> Result<DynamicImage> {
		let input_tensor = self.prepare_input(image)?;

		// Check if model expects fp16
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

		// Extract output (model output dtype matches input dtype)
		let output_tensor = if is_fp16 {
			let output_view: ArrayViewD<half::f16> = outputs[0]
				.try_extract_array()
				.context("Failed to extract fp16 output tensor")?;
			let output_4d: ArrayView4<half::f16> = output_view
				.into_dimensionality()
				.context("Failed to reshape output tensor")?;

			// Convert f16 back to f32 for image processing
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

		drop(outputs);

		self.tensor_to_image(output_tensor.view())
	}

	/// Prepare input tensor from image.
	fn prepare_input(&self, image: DynamicImage) -> Result<Array4<f32>> {
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
