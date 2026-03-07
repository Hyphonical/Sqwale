//! Image ↔ NCHW tensor conversion and fp16 handling.

use anyhow::{Result, bail};
use image::DynamicImage;
use ndarray::{Array4, ArrayView4, s};

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
			let mut tensor = Array4::<f32>::zeros((1, 1, h as usize, w as usize));
			for y in 0..h as usize {
				for x in 0..w as usize {
					tensor[[0, 0, y, x]] = gray.get_pixel(x as u32, y as u32)[0] as f32 / 255.0;
				}
			}
			Ok(tensor)
		}
		3 => {
			let rgb = image.to_rgb8();
			let mut tensor = Array4::<f32>::zeros((1, 3, h as usize, w as usize));
			for y in 0..h as usize {
				for x in 0..w as usize {
					let p = rgb.get_pixel(x as u32, y as u32);
					tensor[[0, 0, y, x]] = p[0] as f32 / 255.0;
					tensor[[0, 1, y, x]] = p[1] as f32 / 255.0;
					tensor[[0, 2, y, x]] = p[2] as f32 / 255.0;
				}
			}
			Ok(tensor)
		}
		4 => {
			let rgba = image.to_rgba8();
			let mut tensor = Array4::<f32>::zeros((1, 4, h as usize, w as usize));
			for y in 0..h as usize {
				for x in 0..w as usize {
					let p = rgba.get_pixel(x as u32, y as u32);
					tensor[[0, 0, y, x]] = p[0] as f32 / 255.0;
					tensor[[0, 1, y, x]] = p[1] as f32 / 255.0;
					tensor[[0, 2, y, x]] = p[2] as f32 / 255.0;
					tensor[[0, 3, y, x]] = p[3] as f32 / 255.0;
				}
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
			let mut img = image::GrayImage::new(w as u32, h as u32);
			for y in 0..h {
				for x in 0..w {
					let v = tensor[[0, 0, y, x]].clamp(0.0, 1.0);
					img.put_pixel(x as u32, y as u32, image::Luma([(v * 255.0 + 0.5) as u8]));
				}
			}
			Ok(DynamicImage::ImageLuma8(img))
		}
		3 => {
			let mut img = image::RgbImage::new(w as u32, h as u32);
			for y in 0..h {
				for x in 0..w {
					let r = tensor[[0, 0, y, x]].clamp(0.0, 1.0);
					let g = tensor[[0, 1, y, x]].clamp(0.0, 1.0);
					let b = tensor[[0, 2, y, x]].clamp(0.0, 1.0);
					img.put_pixel(
						x as u32,
						y as u32,
						image::Rgb([
							(r * 255.0 + 0.5) as u8,
							(g * 255.0 + 0.5) as u8,
							(b * 255.0 + 0.5) as u8,
						]),
					);
				}
			}
			Ok(DynamicImage::ImageRgb8(img))
		}
		4 => {
			let mut img = image::RgbaImage::new(w as u32, h as u32);
			for y in 0..h {
				for x in 0..w {
					let r = tensor[[0, 0, y, x]].clamp(0.0, 1.0);
					let g = tensor[[0, 1, y, x]].clamp(0.0, 1.0);
					let b = tensor[[0, 2, y, x]].clamp(0.0, 1.0);
					let a = tensor[[0, 3, y, x]].clamp(0.0, 1.0);
					img.put_pixel(
						x as u32,
						y as u32,
						image::Rgba([
							(r * 255.0 + 0.5) as u8,
							(g * 255.0 + 0.5) as u8,
							(b * 255.0 + 0.5) as u8,
							(a * 255.0 + 0.5) as u8,
						]),
					);
				}
			}
			Ok(DynamicImage::ImageRgba8(img))
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
