//! RIFE 4.25 frame interpolation: ORT session wrapper, tensor conversion,
//! and ensemble inference.
//!
//! Two model variants are embedded:
//!
//! | Flag      | Weights | Input dtype | Output dtype |
//! |-----------|---------|-------------|---------------|
//! | (default) | fp32    | `f32`       | `f32`         |
//! | `--fp16`  | fp16    | `f16`       | `f16`         |
//!
//! The fp16 variant was exported with `keep_io_types=False`, so the ORT
//! session strictly expects `f16` input and returns `f16` output. Conversion
//! between the pipeline's `f32` storage and `f16` wire tensors is done inside
//! [`RifeSession::run_once`].
//!
//! Input tensor layout: `(1, 7, H, W)` — channels 0–2 = frame0 RGB,
//! 3–5 = frame1 RGB, 6 = timestep broadcast. H and W must be multiples of 32.
//! Output: interpolated frame `(1, 3, H, W)`, always returned as `f32`.

use anyhow::{Result, bail};
use ndarray::{Array4, Axis, concatenate, s};
use ort::session::Session;
use rayon::prelude::*;

use crate::pipeline::tensor::{crop_tensor, pad_tensor_mirror};
use crate::pipeline::tiling::Padding;
use crate::session::{ProviderSelection, make_ep};

/// RIFE 4.25 embedded model bytes — standard float32 weights.
const RIFE_FP32_MODEL_BYTES: &[u8] = include_bytes!("../../models/rife425_fp32_op21_slim.onnx");

/// RIFE 4.25 embedded model bytes — fully fp16-quantized weights.
///
/// Exported with `keep_io_types=False`: the model strictly expects `f16`
/// input tensors and returns `f16` output tensors. Conversion is performed
/// by [`RifeSession::run_once`] so all other pipeline code stays `f32`.
const RIFE_FP16_MODEL_BYTES: &[u8] = include_bytes!("../../models/rife425_fp16_op21_slim.onnx");

/// Alignment requirement for RIFE spatial dimensions.
const RIFE_ALIGNMENT: usize = 32;

/// An ORT session loaded with the embedded RIFE 4.25 model.
pub struct RifeSession {
	session: Session,
	/// Whether this session expects and produces `f16` tensors.
	fp16: bool,
}

impl RifeSession {
	/// Create a new RIFE session using the given execution provider.
	///
	/// When `fp16` is `true`, the fully-quantized model is loaded. Input tensors
	/// are converted from `f32` to `f16` before each ORT call, and outputs are
	/// converted back to `f32` afterwards. Ensemble mode works identically.
	///
	/// Reuses the same provider-fallback logic as the upscale pipeline.
	pub fn new(provider: ProviderSelection, fp16: bool) -> Result<Self> {
		let session = create_rife_session(provider, fp16)?;
		Ok(Self { session, fp16 })
	}

	/// Run a single interpolation between two frames at the given timestep.
	///
	/// Both frames must be `(1, 3, H, W)` f32 tensors normalised to `[0, 1]`.
	/// The timestep is typically `0.5` for simple 2× interpolation.
	///
	/// Padding to a multiple of 32 is handled automatically and cropped after
	/// inference.
	pub fn run_once(
		&mut self,
		frame0: &Array4<f32>,
		frame1: &Array4<f32>,
		timestep: f32,
	) -> Result<Array4<f32>> {
		let h = frame0.shape()[2];
		let w = frame0.shape()[3];

		// Compute padding to align to RIFE_ALIGNMENT.
		let pad_h = (RIFE_ALIGNMENT - (h % RIFE_ALIGNMENT)) % RIFE_ALIGNMENT;
		let pad_w = (RIFE_ALIGNMENT - (w % RIFE_ALIGNMENT)) % RIFE_ALIGNMENT;

		let padding = Padding {
			top: 0,
			left: 0,
			bottom: pad_h as u32,
			right: pad_w as u32,
		};

		// Pad both input frames.
		let f0 = pad_tensor_mirror(frame0, padding);
		let f1 = pad_tensor_mirror(frame1, padding);

		let padded_h = h + pad_h;
		let padded_w = w + pad_w;

		// Build the combined (1,7,H,W) tensor in-place instead of
		// allocating three separate arrays and concatenating.
		let spatial = padded_h * padded_w;
		let mut combined_data = vec![0.0f32; 7 * spatial];
		if let (Some(s0), Some(s1)) = (f0.as_slice(), f1.as_slice()) {
			// Channels 0-2: frame0, channels 3-5: frame1.
			combined_data[..3 * spatial].copy_from_slice(s0);
			combined_data[3 * spatial..6 * spatial].copy_from_slice(s1);
		} else {
			// Fallback for non-contiguous tensors.
			let combined_arr = concatenate(Axis(1), &[f0.view(), f1.view()])
				.map_err(|e| anyhow::anyhow!("Failed to concatenate input: {e}"))?;
			if let Some(s) = combined_arr.as_slice() {
				combined_data[..6 * spatial].copy_from_slice(s);
			}
		}
		// Channel 6: timestep broadcast.
		combined_data[6 * spatial..].fill(timestep);

		let combined = Array4::from_shape_vec((1, 7, padded_h, padded_w), combined_data)
			.map_err(|e| anyhow::anyhow!("Failed to build combined tensor: {e}"))?;

		// Run inference — convert to f16 wire type for the quantized model.
		let outputs = if self.fp16 {
			let f16_combined = combined.mapv(half::f16::from_f32);
			let input_value = ort::value::Value::from_array(f16_combined)
				.map_err(|e| anyhow::anyhow!("Failed to create ORT f16 value: {e}"))?;
			self.session
				.run(ort::inputs![input_value])
				.map_err(|e| anyhow::anyhow!("RIFE inference failed: {e}"))?
		} else {
			let input_value = ort::value::Value::from_array(combined)
				.map_err(|e| anyhow::anyhow!("Failed to create ORT value: {e}"))?;
			self.session
				.run(ort::inputs![input_value])
				.map_err(|e| anyhow::anyhow!("RIFE inference failed: {e}"))?
		};

		// Extract output tensor.
		let (_, output) = outputs
			.iter()
			.next()
			.ok_or_else(|| anyhow::anyhow!("RIFE model produced no outputs"))?;

		let raw = extract_output_f32(&output)?;

		// Crop padding (scale = 1 for interpolation).
		Ok(crop_tensor(raw.view(), padding, 1))
	}

	/// Interpolate between two frames, optionally using ensemble averaging.
	///
	/// When `ensemble` is true, a second inference is run with horizontally
	/// flipped inputs. The two results are averaged for improved quality at
	/// the cost of roughly 2× inference time.
	pub fn interpolate(
		&mut self,
		f0: &Array4<f32>,
		f1: &Array4<f32>,
		timestep: f32,
		ensemble: bool,
	) -> Result<Array4<f32>> {
		let mut normal = self.run_once(f0, f1, timestep)?;

		if !ensemble {
			return Ok(normal);
		}

		// Flip both frames horizontally, interpolate, then flip back.
		let f0_flip = flip_horizontal(f0);
		let f1_flip = flip_horizontal(f1);
		let flipped_mid = self.run_once(&f0_flip, &f1_flip, timestep)?;
		let flipped_back = flip_horizontal(&flipped_mid);

		// Average in-place to avoid an extra allocation.
		normal += &flipped_back;
		normal /= 2.0;
		Ok(normal)
	}
}

// ── Tensor ↔ raw byte conversion ──────────────────────────────────────────

/// Convert raw RGB24 bytes to an NCHW f32 tensor normalised to `[0, 1]`.
///
/// Layout: `(1, 3, H, W)`.
pub fn bytes_to_tensor(bytes: &[u8], w: usize, h: usize) -> Array4<f32> {
	let num_pixels = w * h;
	let mut planar = vec![0.0f32; num_pixels * 3];
	let (r_plane, rest) = planar.split_at_mut(num_pixels);
	let (g_plane, b_plane) = rest.split_at_mut(num_pixels);

	r_plane
		.par_iter_mut()
		.zip(g_plane.par_iter_mut())
		.zip(b_plane.par_iter_mut())
		.enumerate()
		.for_each(|(i, ((r, g), b))| {
			let idx = i * 3;
			*r = bytes[idx] as f32 / 255.0;
			*g = bytes[idx + 1] as f32 / 255.0;
			*b = bytes[idx + 2] as f32 / 255.0;
		});

	Array4::from_shape_vec((1, 3, h, w), planar).expect("buffer size matches tensor shape")
}

/// Convert an NCHW f32 tensor `(1, 3, H, W)` back to raw RGB24 bytes.
///
/// Values are clamped to `[0, 1]` before scaling to `[0, 255]`.
pub fn tensor_to_bytes(tensor: &Array4<f32>) -> Vec<u8> {
	let shape = tensor.shape();
	let (h, w) = (shape[2], shape[3]);
	let num_pixels = w * h;
	let slice = tensor.as_slice().expect("tensor must be contiguous");
	let r_plane = &slice[0..num_pixels];
	let g_plane = &slice[num_pixels..num_pixels * 2];
	let b_plane = &slice[num_pixels * 2..];

	let mut bytes = vec![0u8; num_pixels * 3];
	bytes
		.par_chunks_exact_mut(3)
		.enumerate()
		.for_each(|(i, pixel)| {
			pixel[0] = (r_plane[i].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
			pixel[1] = (g_plane[i].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
			pixel[2] = (b_plane[i].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
		});
	bytes
}

// ── Helpers ────────────────────────────────────────────────────────────────

/// Flip a `(1, C, H, W)` tensor horizontally (reverse the W axis).
fn flip_horizontal(tensor: &Array4<f32>) -> Array4<f32> {
	tensor.slice(s![.., .., .., ..;-1]).to_owned()
}

/// Extract the first output as an f32 `Array4` with shape `(1, C, H, W)`.
///
/// Handles both `f32` (standard model) and `f16` (quantized model) outputs.
fn extract_output_f32(output: &ort::value::ValueRef<'_>) -> Result<Array4<f32>> {
	if let Ok((shape, data)) = output.try_extract_tensor::<f32>() {
		let dims: Vec<usize> = (**shape).iter().map(|&d| d as usize).collect();
		return reshape_to_nchw(data.to_vec(), &dims);
	}

	if let Ok((shape, data)) = output.try_extract_tensor::<half::f16>() {
		let dims: Vec<usize> = (**shape).iter().map(|&d| d as usize).collect();
		let f32_data: Vec<f32> = data.iter().map(|v| v.to_f32()).collect();
		return reshape_to_nchw(f32_data, &dims);
	}

	bail!("RIFE output is neither float32 nor float16")
}

/// Reshape a flat `f32` buffer into an NCHW `Array4`, accepting 3-D or 4-D shapes.
fn reshape_to_nchw(data: Vec<f32>, dims: &[usize]) -> Result<Array4<f32>> {
	match dims {
		[n, c, h, w] => Array4::from_shape_vec((*n, *c, *h, *w), data)
			.map_err(|e| anyhow::anyhow!("Failed to reshape RIFE output: {e}")),
		[c, h, w] => Array4::from_shape_vec((1, *c, *h, *w), data)
			.map_err(|e| anyhow::anyhow!("Failed to reshape RIFE output: {e}")),
		_ => bail!("Unexpected RIFE output shape: {dims:?} (expected 3-D or 4-D)"),
	}
}

/// Create an ORT session for the embedded RIFE model with provider fallback.
fn create_rife_session(provider: ProviderSelection, fp16: bool) -> Result<Session> {
	use ort::session::builder::AutoDevicePolicy;
	use tracing::warn;

	let model_bytes: &[u8] = if fp16 {
		RIFE_FP16_MODEL_BYTES
	} else {
		RIFE_FP32_MODEL_BYTES
	};
	let commit = |b: &mut ort::session::builder::SessionBuilder| b.commit_from_memory(model_bytes);

	// CPU inference: use all available cores for intra-op parallelism.
	let configure_cpu =
		|b: ort::session::builder::SessionBuilder| -> Result<ort::session::builder::SessionBuilder> {
			b.with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
				.map_err(|e| anyhow::anyhow!("Failed to set optimization level: {e}"))?
				.with_intra_threads(
					std::thread::available_parallelism().map_or(4, |n| n.get()),
				)
				.map_err(|e| anyhow::anyhow!("Failed to set thread count: {e}"))
		};

	// GPU inference: do not override ORT's default CPU thread count (1).
	// Forcing many intra-op threads on a GPU session competes with Rayon's
	// pool on the CPU side without benefiting GPU throughput.
	let configure_gpu =
		|b: ort::session::builder::SessionBuilder| -> Result<ort::session::builder::SessionBuilder> {
			b.with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
				.map_err(|e| anyhow::anyhow!("Failed to set optimization level: {e}"))
		};

	match provider {
		ProviderSelection::Auto => {
			let mut builder = configure_gpu(
				Session::builder()
					.map_err(|e| anyhow::anyhow!("Failed to create session builder: {e}"))?
					.with_auto_device(AutoDevicePolicy::MaxPerformance)
					.map_err(|e| anyhow::anyhow!("Failed to configure auto device: {e}"))?,
			)?;
			Ok(commit(&mut builder)
				.map_err(|e| anyhow::anyhow!("Failed to load RIFE model: {e}"))?)
		}
		ProviderSelection::Cpu => {
			let ep = make_ep(ProviderSelection::Cpu)?;
			let mut builder = configure_cpu(
				Session::builder()
					.map_err(|e| anyhow::anyhow!("Failed to create session builder: {e}"))?
					.with_execution_providers([ep])
					.map_err(|e| anyhow::anyhow!("Failed to configure CPU provider: {e}"))?,
			)?;
			Ok(commit(&mut builder)
				.map_err(|e| anyhow::anyhow!("Failed to load RIFE model: {e}"))?)
		}
		requested => {
			let ep = make_ep(requested)?;
			let try_result = Session::builder()
				.map_err(|e| e.to_string())
				.and_then(|b| b.with_execution_providers([ep]).map_err(|e| e.to_string()))
				.and_then(|b| configure_gpu(b).map_err(|e| e.to_string()))
				.and_then(|mut b| commit(&mut b).map_err(|e| e.to_string()));

			match try_result {
				Ok(session) => Ok(session),
				Err(e) => {
					warn!(
						"{} provider failed ({}), falling back to CPU",
						requested.name(),
						e
					);
					let ep = make_ep(ProviderSelection::Cpu)?;
					let mut builder = configure_cpu(
						Session::builder()
							.map_err(|e| anyhow::anyhow!("Failed to create session builder: {e}"))?
							.with_execution_providers([ep])
							.map_err(|e| {
								anyhow::anyhow!("Failed to configure CPU fallback: {e}")
							})?,
					)?;
					Ok(commit(&mut builder).map_err(|e| {
						anyhow::anyhow!("Failed to load RIFE model with CPU fallback: {e}")
					})?)
				}
			}
		}
	}
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn bytes_to_tensor_normalises_correctly() {
		// Create a 2×2 RGB image: red, green, blue, white.
		let bytes: Vec<u8> = vec![
			255, 0, 0, // red
			0, 255, 0, // green
			0, 0, 255, // blue
			255, 255, 255, // white
		];
		let t = bytes_to_tensor(&bytes, 2, 2);
		assert_eq!(t.shape(), &[1, 3, 2, 2]);

		// Red channel.
		assert!((t[[0, 0, 0, 0]] - 1.0).abs() < 1e-6); // red pixel
		assert!(t[[0, 0, 0, 1]].abs() < 1e-6); // green pixel
	}

	#[test]
	fn tensor_to_bytes_roundtrip() {
		let bytes: Vec<u8> = vec![128, 64, 200, 0, 255, 100, 50, 150, 250, 10, 20, 30];
		let t = bytes_to_tensor(&bytes, 2, 2);
		let result = tensor_to_bytes(&t);
		assert_eq!(bytes, result);
	}

	#[test]
	fn flip_horizontal_works() {
		let mut t = Array4::<f32>::zeros((1, 1, 2, 3));
		t[[0, 0, 0, 0]] = 1.0;
		t[[0, 0, 0, 1]] = 2.0;
		t[[0, 0, 0, 2]] = 3.0;
		let flipped = flip_horizontal(&t);
		assert!((flipped[[0, 0, 0, 0]] - 3.0).abs() < 1e-6);
		assert!((flipped[[0, 0, 0, 1]] - 2.0).abs() < 1e-6);
		assert!((flipped[[0, 0, 0, 2]] - 1.0).abs() < 1e-6);
	}
}
