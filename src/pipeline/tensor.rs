//! Image ↔ NCHW tensor conversion and fp16 handling.

use anyhow::{Result, bail};
use image::DynamicImage;
use ndarray::{Array4, ArrayView4, s};
use ort::session::SessionOutputs;

use super::tiling::Padding;

/// Convert a `DynamicImage` to an NCHW f32 tensor in [0, 1].
///
/// The image is converted to the appropriate channel count first
/// (Luma8, Rgb8, or Rgba8), then laid out as `(1, C, H, W)`.
pub fn image_to_tensor(image: &DynamicImage, channels: u32) -> Result<Array4<f32>> {
	let (w, h) = (image.width(), image.height());

	match channels {
		1 => {
			let gray = image.to_luma8();
			let w = w as usize;
			let h = h as usize;
			let mut tensor = Array4::<f32>::zeros((1, 1, h, w));
			for (i, &px) in gray.as_raw().iter().enumerate() {
				tensor[[0, 0, i / w, i % w]] = px as f32 / 255.0;
			}
			Ok(tensor)
		}
		3 => {
			let rgb = image.to_rgb8();
			let w = w as usize;
			let h = h as usize;
			let mut tensor = Array4::<f32>::zeros((1, 3, h, w));
			for (i, chunk) in rgb.as_raw().chunks_exact(3).enumerate() {
				let y = i / w;
				let x = i % w;
				tensor[[0, 0, y, x]] = chunk[0] as f32 / 255.0;
				tensor[[0, 1, y, x]] = chunk[1] as f32 / 255.0;
				tensor[[0, 2, y, x]] = chunk[2] as f32 / 255.0;
			}
			Ok(tensor)
		}
		4 => {
			let rgba = image.to_rgba8();
			let w = w as usize;
			let h = h as usize;
			let mut tensor = Array4::<f32>::zeros((1, 4, h, w));
			for (i, chunk) in rgba.as_raw().chunks_exact(4).enumerate() {
				let y = i / w;
				let x = i % w;
				tensor[[0, 0, y, x]] = chunk[0] as f32 / 255.0;
				tensor[[0, 1, y, x]] = chunk[1] as f32 / 255.0;
				tensor[[0, 2, y, x]] = chunk[2] as f32 / 255.0;
				tensor[[0, 3, y, x]] = chunk[3] as f32 / 255.0;
			}
			Ok(tensor)
		}
		n => bail!("Unsupported channel count: {n}"),
	}
}

/// Convert an NCHW f32 tensor in [0, 1] back to a `DynamicImage`.
///
/// Values are clamped to [0, 1], scaled to [0, 255], and rounded.
pub fn tensor_to_image(tensor: ArrayView4<f32>, channels: u32) -> Result<DynamicImage> {
	let h = tensor.shape()[2];
	let w = tensor.shape()[3];

	match channels {
		1 => {
			let mut img_raw = vec![0u8; h * w];
			for y in 0..h {
				let row = y * w;
				for x in 0..w {
					img_raw[row + x] =
						(tensor[[0, 0, y, x]].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
				}
			}
			let luma_image = image::GrayImage::from_raw(w as u32, h as u32, img_raw)
				.ok_or_else(|| anyhow::anyhow!("Failed to create image from raw buffer"))?;
			Ok(DynamicImage::ImageLuma8(luma_image))
		}
		3 => {
			let mut img_raw = vec![0u8; h * w * 3];
			for y in 0..h {
				let row = y * w * 3;
				for x in 0..w {
					let idx = row + x * 3;
					img_raw[idx] =
						(tensor[[0, 0, y, x]].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
					img_raw[idx + 1] =
						(tensor[[0, 1, y, x]].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
					img_raw[idx + 2] =
						(tensor[[0, 2, y, x]].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
				}
			}
			let rgb_image = image::RgbImage::from_raw(w as u32, h as u32, img_raw)
				.ok_or_else(|| anyhow::anyhow!("Failed to create image from raw buffer"))?;
			Ok(DynamicImage::ImageRgb8(rgb_image))
		}
		4 => {
			let mut img_raw = vec![0u8; h * w * 4];
			for y in 0..h {
				let row = y * w * 4;
				for x in 0..w {
					let idx = row + x * 4;
					img_raw[idx] =
						(tensor[[0, 0, y, x]].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
					img_raw[idx + 1] =
						(tensor[[0, 1, y, x]].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
					img_raw[idx + 2] =
						(tensor[[0, 2, y, x]].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
					img_raw[idx + 3] =
						(tensor[[0, 3, y, x]].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
				}
			}
			let rgba_image = image::RgbaImage::from_raw(w as u32, h as u32, img_raw)
				.ok_or_else(|| anyhow::anyhow!("Failed to create image from raw buffer"))?;
			Ok(DynamicImage::ImageRgba8(rgba_image))
		}
		n => bail!("Unsupported channel count for output: {n}"),
	}
}

/// Convert an f32 tensor to `half::f16` for models requiring fp16 input.
pub fn tensor_f32_to_f16(tensor: &Array4<f32>) -> Array4<half::f16> {
	tensor.mapv(half::f16::from_f32)
}

/// Apply mirror padding to a tensor.
///
/// Extends each spatial dimension by reflecting pixel values.
pub fn pad_tensor_mirror(tensor: &Array4<f32>, padding: Padding) -> Array4<f32> {
	if padding.is_zero() {
		return tensor.clone();
	}

	let (_, c, h, w) = (
		tensor.shape()[0],
		tensor.shape()[1],
		tensor.shape()[2],
		tensor.shape()[3],
	);

	let new_h = h + padding.top as usize + padding.bottom as usize;
	let new_w = w + padding.left as usize + padding.right as usize;
	let mut padded = Array4::<f32>::zeros((1, c, new_h, new_w));

	let top = padding.top as usize;
	let left = padding.left as usize;

	// Copy original data.
	padded
		.slice_mut(s![.., .., top..top + h, left..left + w])
		.assign(tensor);

	// Mirror padding: reflect for each edge.
	for ch in 0..c {
		// Top rows.
		for y in 0..top {
			let src_y = top - y; // reflect index
			let src_y = src_y.min(h - 1);
			for x in 0..new_w {
				let src_x = mirror_index(x, left, w);
				padded[[0, ch, y, x]] = tensor[[0, ch, src_y, src_x]];
			}
		}
		// Bottom rows.
		for y in (top + h)..new_h {
			let src_y = h - 1 - (y - top - h).min(h - 1);
			for x in 0..new_w {
				let src_x = mirror_index(x, left, w);
				padded[[0, ch, y, x]] = tensor[[0, ch, src_y, src_x]];
			}
		}
		// Left/right columns (for original rows only).
		for y in top..(top + h) {
			let src_y = y - top;
			for x in 0..left {
				let src_x = left - x;
				let src_x = src_x.min(w - 1);
				padded[[0, ch, y, x]] = tensor[[0, ch, src_y, src_x]];
			}
			for x in (left + w)..new_w {
				let src_x = w - 1 - (x - left - w).min(w - 1);
				padded[[0, ch, y, x]] = tensor[[0, ch, src_y, src_x]];
			}
		}
	}

	padded
}

/// Helper: mirror-reflect an index in the padded space back to the source.
fn mirror_index(padded_x: usize, offset: usize, len: usize) -> usize {
	if padded_x < offset {
		let dist = offset - padded_x;
		dist.min(len - 1)
	} else if padded_x >= offset + len {
		let dist = padded_x - offset - len;
		len.saturating_sub(1).saturating_sub(dist.min(len - 1))
	} else {
		padded_x - offset
	}
}

/// Remove padding from an output tensor (scale-adjusted).
///
/// Crops to the valid region after inference on a padded input.
pub fn crop_tensor(tensor: ArrayView4<f32>, padding: Padding, scale: u32) -> Array4<f32> {
	if padding.is_zero() {
		return tensor.to_owned();
	}

	let h = tensor.shape()[2];
	let w = tensor.shape()[3];

	let top = padding.top as usize * scale as usize;
	let left = padding.left as usize * scale as usize;
	let bottom = padding.bottom as usize * scale as usize;
	let right = padding.right as usize * scale as usize;

	let crop_h = h - top - bottom;
	let crop_w = w - left - right;

	tensor
		.slice(s![.., .., top..top + crop_h, left..left + crop_w])
		.to_owned()
}

/// Extract the first output tensor from ORT session results as an f32 NCHW `Array4`.
///
/// Handles f32 and f16 output dtypes, and coerces NHWC/CHW layouts to NCHW.
pub fn extract_output_f32(outputs: &SessionOutputs<'_>) -> Result<Array4<f32>> {
	let (_, output) = outputs
		.iter()
		.next()
		.ok_or_else(|| anyhow::anyhow!("Model produced no outputs"))?;

	// Try f32 first.
	if let Ok((shape, data)) = output.try_extract_tensor::<f32>() {
		let dims: Vec<usize> = (**shape).iter().map(|&d| d as usize).collect();
		return coerce_to_nchw(data.to_vec(), &dims);
	}

	// Try f16.
	if let Ok((shape, data)) = output.try_extract_tensor::<half::f16>() {
		let dims: Vec<usize> = (**shape).iter().map(|&d| d as usize).collect();
		let f32_data: Vec<f32> = data.iter().map(|v| v.to_f32()).collect();
		return coerce_to_nchw(f32_data, &dims);
	}

	bail!("Unsupported output tensor dtype (expected float32 or float16)")
}

/// Reshape a flat data buffer into an NCHW `Array4<f32>`.
///
/// Handles:
/// - 4-D NCHW `[N, C, H, W]` — used as-is.
/// - 4-D NHWC `[N, H, W, C]` — detected when the last dim is ≤ 4 and the
///   second dim is clearly spatial (> 4), then transposed to `[N, C, H, W]`.
/// - 3-D CHW `[C, H, W]` — a batch dimension of 1 is prepended.
fn coerce_to_nchw(data: Vec<f32>, dims: &[usize]) -> Result<Array4<f32>> {
	match dims {
		[d0, d1, d2, d3] => {
			// Heuristic: if the last dim is a plausible channel count (≤ 4) and
			// the second dim is clearly spatial (> 4), the layout is NHWC.
			if *d3 <= 4 && *d1 > 4 {
				let nhwc = Array4::from_shape_vec((*d0, *d1, *d2, *d3), data)
					.map_err(|e| anyhow::anyhow!("Failed to create NHWC tensor: {e}"))?;
				// Permute [N, H, W, C] → [N, C, H, W]
				Ok(nhwc.permuted_axes([0, 3, 1, 2]).into_owned())
			} else {
				Array4::from_shape_vec((*d0, *d1, *d2, *d3), data)
					.map_err(|e| anyhow::anyhow!("Failed to create NCHW tensor: {e}"))
			}
		}
		[c, h, w] => Array4::from_shape_vec((1, *c, *h, *w), data)
			.map_err(|e| anyhow::anyhow!("Failed to create CHW tensor: {e}")),
		_ => bail!(
			"Unsupported output tensor rank {} (expected 3-D or 4-D, shape: {:?})",
			dims.len(),
			dims
		),
	}
}
