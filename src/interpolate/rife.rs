//! RIFE 4.25 frame interpolation: ORT session wrapper, tensor conversion,
//! and ensemble inference.
//!
//! The embedded model expects a single input tensor:
//! - **Input** — `(1, 7, H, W)` where channels 0–2 are frame0 RGB, 3–5 are
//!   frame1 RGB, and channel 6 is the timestep broadcast to spatial dims.
//!   H and W must be multiples of 32.
//! - **Output** — interpolated frame: `(1, 3, H, W)`.

use anyhow::{Result, bail};
use ndarray::{Array4, Axis, concatenate, s};
use ort::session::Session;
use tracing::debug;

use crate::pipeline::tensor::{crop_tensor, pad_tensor_mirror};
use crate::pipeline::tiling::Padding;
use crate::session::{ProviderSelection, make_ep};

/// RIFE 4.25 embedded model bytes.
const RIFE_MODEL_BYTES: &[u8] = include_bytes!("../../models/rife425_fp32_op21_slim.onnx");

/// Alignment requirement for RIFE spatial dimensions.
const RIFE_ALIGNMENT: usize = 32;

/// An ORT session loaded with the embedded RIFE 4.25 model.
pub struct RifeSession {
	session: Session,
}

impl RifeSession {
	/// Create a new RIFE session using the given execution provider.
	///
	/// Reuses the same provider-fallback logic as the upscale pipeline.
	pub fn new(provider: ProviderSelection) -> Result<Self> {
		let session = create_rife_session(provider)?;
		Ok(Self { session })
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

		debug!(
			"RIFE padding: {}×{} → {}×{} (pad_h={}, pad_w={})",
			w,
			h,
			w + pad_w,
			h + pad_h,
			pad_h,
			pad_w
		);

		// Pad both input frames.
		let f0 = pad_tensor_mirror(frame0, padding);
		let f1 = pad_tensor_mirror(frame1, padding);

		let padded_h = h + pad_h;
		let padded_w = w + pad_w;

		// Concatenate along the channel axis:
		// (1,3,H,W) + (1,3,H,W) + (1,1,H,W) → (1,7,H,W).
		let ts_channel = Array4::<f32>::from_elem((1, 1, padded_h, padded_w), timestep);
		let combined = concatenate(Axis(1), &[f0.view(), f1.view(), ts_channel.view()])
			.map_err(|e| anyhow::anyhow!("Failed to concatenate input: {e}"))?;

		// Create ORT input value.
		let input_value = ort::value::Value::from_array(combined)
			.map_err(|e| anyhow::anyhow!("Failed to create ORT value: {e}"))?;

		// Run inference.
		let outputs = self
			.session
			.run(ort::inputs![input_value])
			.map_err(|e| anyhow::anyhow!("RIFE inference failed: {e}"))?;

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
		let normal = self.run_once(f0, f1, timestep)?;

		if !ensemble {
			return Ok(normal);
		}

		// Flip both frames horizontally, interpolate, then flip back.
		let f0_flip = flip_horizontal(f0);
		let f1_flip = flip_horizontal(f1);
		let flipped_mid = self.run_once(&f0_flip, &f1_flip, timestep)?;
		let flipped_back = flip_horizontal(&flipped_mid);

		// Average the two results.
		Ok((&normal + &flipped_back) / 2.0)
	}
}

// ── Tensor ↔ raw byte conversion ──────────────────────────────────────────

/// Convert raw RGB24 bytes to an NCHW f32 tensor normalised to `[0, 1]`.
///
/// Layout: `(1, 3, H, W)`.
pub fn bytes_to_tensor(bytes: &[u8], w: usize, h: usize) -> Array4<f32> {
	let mut tensor = Array4::<f32>::zeros((1, 3, h, w));
	for y in 0..h {
		for x in 0..w {
			let idx = (y * w + x) * 3;
			tensor[[0, 0, y, x]] = bytes[idx] as f32 / 255.0;
			tensor[[0, 1, y, x]] = bytes[idx + 1] as f32 / 255.0;
			tensor[[0, 2, y, x]] = bytes[idx + 2] as f32 / 255.0;
		}
	}
	tensor
}

/// Convert an NCHW f32 tensor `(1, 3, H, W)` back to raw RGB24 bytes.
///
/// Values are clamped to `[0, 1]` before scaling to `[0, 255]`.
pub fn tensor_to_bytes(tensor: &Array4<f32>, w: usize, h: usize) -> Vec<u8> {
	let mut bytes = vec![0u8; h * w * 3];
	for y in 0..h {
		for x in 0..w {
			let idx = (y * w + x) * 3;
			bytes[idx] = (tensor[[0, 0, y, x]].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
			bytes[idx + 1] = (tensor[[0, 1, y, x]].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
			bytes[idx + 2] = (tensor[[0, 2, y, x]].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
		}
	}
	bytes
}

// ── Helpers ────────────────────────────────────────────────────────────────

/// Flip a `(1, C, H, W)` tensor horizontally (reverse the W axis).
fn flip_horizontal(tensor: &Array4<f32>) -> Array4<f32> {
	tensor.slice(s![.., .., .., ..;-1]).to_owned()
}

/// Extract the first output as an f32 `Array4` with shape `(1, C, H, W)`.
fn extract_output_f32(output: &ort::value::ValueRef<'_>) -> Result<Array4<f32>> {
	if let Ok((shape, data)) = output.try_extract_tensor::<f32>() {
		let dims: Vec<usize> = (**shape).iter().map(|&d| d as usize).collect();
		return match dims.as_slice() {
			[n, c, h, w] => Array4::from_shape_vec((*n, *c, *h, *w), data.to_vec())
				.map_err(|e| anyhow::anyhow!("Failed to reshape RIFE output: {e}")),
			[c, h, w] => Array4::from_shape_vec((1, *c, *h, *w), data.to_vec())
				.map_err(|e| anyhow::anyhow!("Failed to reshape RIFE output: {e}")),
			_ => bail!(
				"Unexpected RIFE output shape: {:?} (expected 3-D or 4-D)",
				dims
			),
		};
	}
	bail!("RIFE output is not float32")
}

/// Create an ORT session for the embedded RIFE model with provider fallback.
fn create_rife_session(provider: ProviderSelection) -> Result<Session> {
	use ort::session::builder::AutoDevicePolicy;
	use tracing::warn;

	let commit =
		|b: &mut ort::session::builder::SessionBuilder| b.commit_from_memory(RIFE_MODEL_BYTES);

	match provider {
		ProviderSelection::Auto => {
			let mut builder = Session::builder()
				.map_err(|e| anyhow::anyhow!("Failed to create session builder: {e}"))?
				.with_auto_device(AutoDevicePolicy::MaxPerformance)
				.map_err(|e| anyhow::anyhow!("Failed to configure auto device: {e}"))?;
			Ok(commit(&mut builder)
				.map_err(|e| anyhow::anyhow!("Failed to load RIFE model: {e}"))?)
		}
		ProviderSelection::Cpu => {
			let ep = make_ep(ProviderSelection::Cpu)?;
			let mut builder = Session::builder()
				.map_err(|e| anyhow::anyhow!("Failed to create session builder: {e}"))?
				.with_execution_providers([ep])
				.map_err(|e| anyhow::anyhow!("Failed to configure CPU provider: {e}"))?;
			Ok(commit(&mut builder)
				.map_err(|e| anyhow::anyhow!("Failed to load RIFE model: {e}"))?)
		}
		requested => {
			let ep = make_ep(requested)?;
			let try_result = Session::builder()
				.map_err(|e| e.to_string())
				.and_then(|b| b.with_execution_providers([ep]).map_err(|e| e.to_string()))
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
					let mut builder = Session::builder()
						.map_err(|e| anyhow::anyhow!("Failed to create session builder: {e}"))?
						.with_execution_providers([ep])
						.map_err(|e| anyhow::anyhow!("Failed to configure CPU fallback: {e}"))?;
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
		let result = tensor_to_bytes(&t, 2, 2);
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
