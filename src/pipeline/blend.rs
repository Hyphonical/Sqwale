//! FFT-based Laplacian pyramid blending — fully optimized.
//!
//! All pyramid bands are collapsed into a single real-valued per-bin transfer
//! function H_ai(u,v).  Each colour channel needs only 2 forward R2C FFTs and
//! 1 inverse C2R FFT instead of 2×LEVELS full complex round-trips.
//!
//! Key wins over the naïve version:
//!  • \~12× fewer 1-D FFT operations (R2C/C2R are \~2× faster than complex too)
//!  • \~3.5× lower peak RAM  (half-width spectra + streaming channels)
//!  • Zero per-iteration heap allocations (all buffers pre-allocated and reused)

use anyhow::Result;
use image::DynamicImage;
use realfft::RealFftPlanner;
use rustfft::{Fft, FftPlanner, num_complex::Complex};
use std::f32::consts::PI;
use std::sync::Arc;

const PYRAMID_LEVELS: usize = 6;
const SIGMA_BASE_FACTOR: f32 = 1.0 / 512.0;

// ── Public API ───────────────────────────────────────────────────────────────

/// Lanczos-upscale `original` to the dimensions of `ai`, then frequency-blend.
pub fn frequency_blend_with_original(
	ai: &DynamicImage,
	original: &DynamicImage,
	blend: f32,
	on_step: Option<&dyn Fn(usize, usize)>,
) -> Result<DynamicImage> {
	if blend <= 0.0 {
		return Ok(ai.clone());
	}
	let lanczos = original.resize_exact(
		ai.width(),
		ai.height(),
		image::imageops::FilterType::Lanczos3,
	);
	frequency_blend(ai, &lanczos, blend, on_step)
}

/// Blend `ai` and `lanczos` (same dimensions) via a single-pass frequency-domain
/// transfer function derived from a 6-level Laplacian pyramid decomposition.
pub fn frequency_blend(
	ai: &DynamicImage,
	lanczos: &DynamicImage,
	blend: f32,
	on_step: Option<&dyn Fn(usize, usize)>,
) -> Result<DynamicImage> {
	let blend = blend.clamp(0.0, 1.0);
	if blend <= 0.0 {
		return Ok(ai.clone());
	}

	let w = ai.width() as usize;
	let h = ai.height() as usize;
	let pixels = w * h;
	let half_w = w / 2 + 1;
	let spec_len = h * half_w;

	// ── Raw pixel bytes (interleaved, u8) ────────────────────────────────
	let (ai_raw, lz_raw, n_ch) = extract_raw_pair(ai, lanczos);

	// ── FFT plans (allocated once, shared across all channels) ───────────
	let mut rp = RealFftPlanner::<f32>::new();
	let r2c = rp.plan_fft_forward(w);
	let c2r = rp.plan_fft_inverse(w);

	let mut cp = FftPlanner::<f32>::new();
	let fft_col = cp.plan_fft_forward(h);
	let ifft_col = cp.plan_fft_inverse(h);

	// ── Pre-allocated scratch buffers (tiny) ─────────────────────────────
	let mut r2c_scratch = r2c.make_scratch_vec();
	let mut c2r_scratch = c2r.make_scratch_vec();
	let col_scratch_len = fft_col
		.get_inplace_scratch_len()
		.max(ifft_col.get_inplace_scratch_len());
	let mut col_scratch = vec![Complex::new(0.0f32, 0.0); col_scratch_len];

	// ── Reusable working buffers ─────────────────────────────────────────
	let mut spec_a = vec![Complex::new(0.0f32, 0.0); spec_len]; // AI half-spectrum
	let mut spec_l = vec![Complex::new(0.0f32, 0.0); spec_len]; // LZ half-spectrum
	let mut row_r = vec![0.0f32; w]; // real row (R2C in / C2R out)
	let mut row_c = vec![Complex::new(0.0f32, 0.0); half_w]; // complex row (R2C out / C2R in)
	let mut col = vec![Complex::new(0.0f32, 0.0); h]; // column gather/scatter

	// ── Transfer function (half-spectrum, f32) ───────────────────────────
	let sigma_base = ((w as f32 * h as f32).sqrt() * SIGMA_BASE_FACTOR).max(1.0);
	let sigmas: Vec<f32> = (0..PYRAMID_LEVELS)
		.map(|k| sigma_base * 2.0_f32.powi(k as i32))
		.collect();
	let h_ai = build_transfer_function(w, h, &sigmas, blend);

	// ── Output byte buffer ───────────────────────────────────────────────
	let mut out_raw = vec![0u8; pixels * n_ch];
	let norm = 1.0 / pixels as f32;

	// ── Per-channel: 2 forward FFTs + blend + 1 inverse FFT ─────────────
	let total_steps = n_ch * 3;
	for ch in 0..n_ch {
		// Forward 2-D R2C FFT of AI channel
		forward_r2c_2d(
			&ai_raw,
			n_ch,
			ch,
			w,
			h,
			&r2c,
			&fft_col,
			&mut row_r,
			&mut row_c,
			&mut col,
			&mut r2c_scratch,
			&mut col_scratch,
			&mut spec_a,
		);
		if let Some(cb) = on_step {
			cb(ch * 3 + 1, total_steps);
		}

		// Forward 2-D R2C FFT of Lanczos channel
		forward_r2c_2d(
			&lz_raw,
			n_ch,
			ch,
			w,
			h,
			&r2c,
			&fft_col,
			&mut row_r,
			&mut row_c,
			&mut col,
			&mut r2c_scratch,
			&mut col_scratch,
			&mut spec_l,
		);
		if let Some(cb) = on_step {
			cb(ch * 3 + 2, total_steps);
		}

		// Blend in the frequency domain:  out = LZ + (AI − LZ) · H_ai
		for i in 0..spec_len {
			spec_a[i] = spec_l[i] + (spec_a[i] - spec_l[i]) * h_ai[i];
		}

		// Inverse 2-D C2R FFT → quantise → write into output bytes
		inverse_c2r_2d(
			&mut spec_a,
			w,
			h,
			n_ch,
			ch,
			norm,
			&c2r,
			&ifft_col,
			&mut row_c,
			&mut col,
			&mut row_r,
			&mut c2r_scratch,
			&mut col_scratch,
			&mut out_raw,
		);
		if let Some(cb) = on_step {
			cb(ch * 3 + 3, total_steps);
		}
	}

	Ok(raw_to_image(out_raw, w as u32, h as u32, n_ch))
}

// ── Transfer function ────────────────────────────────────────────────────────

/// Precompute H_ai for the half-spectrum (h × half_w floats).
///
/// Derivation (all linear, so it factors per-bin):
///
///   output = Σ_k [(1−w_k)·DoG_k(AI) + w_k·DoG_k(LZ)] + (1−b)·G_n(AI) + b·G_n(LZ)
///          = LZ  +  (AI − LZ) · H_ai          (in the frequency domain)
///
///   H_ai(u,v) = Σ_{k=0}^{n-1} (1 − b·k/(n−1)) · (G_k − G_{k+1})  +  (1−b)·G_n
///
/// where G_0 = 1 and G_k = exp(−2π²σ_k²·(f_v² + f_u²)).
fn build_transfer_function(w: usize, h: usize, sigmas: &[f32], blend: f32) -> Vec<f32> {
	let n = sigmas.len();
	let half_w = w / 2 + 1;
	let two_pi_sq = 2.0 * PI * PI;
	let neg_c: Vec<f32> = sigmas.iter().map(|s| -two_pi_sq * s * s).collect();
	let inv_n_minus_1 = 1.0 / (n - 1) as f32;
	let mut tf = vec![0.0f32; h * half_w];

	for v in 0..h {
		let fv = if v <= h / 2 {
			v as f32
		} else {
			v as f32 - h as f32
		};
		let fv_sq = (fv / h as f32) * (fv / h as f32);
		let row = v * half_w;

		for u in 0..half_w {
			let fu = u as f32 / w as f32;
			let freq_sq = fv_sq + fu * fu;

			let mut g_prev = 1.0f32;
			let mut val = 0.0f32;

			for (k, &neg_ck) in neg_c.iter().enumerate() {
				let g_curr = (neg_ck * freq_sq).exp();
				let ai_weight = 1.0 - blend * k as f32 * inv_n_minus_1;
				val += ai_weight * (g_prev - g_curr);
				g_prev = g_curr;
			}
			val += (1.0 - blend) * g_prev; // DC residual

			tf[row + u] = val;
		}
	}

	tf
}

// ── 2-D R2C forward FFT ─────────────────────────────────────────────────────

/// Row-wise R2C then column-wise complex FFT.
/// Reads one channel directly from interleaved u8 raw bytes (zero extra copies).
// All buffer arguments are pre-allocated by the caller and reused across channels
// to avoid per-channel heap allocations in this hot path.
#[allow(clippy::too_many_arguments)]
fn forward_r2c_2d(
	raw: &[u8],
	stride: usize,
	ch: usize,
	w: usize,
	h: usize,
	r2c: &Arc<dyn realfft::RealToComplex<f32>>,
	fft_col: &Arc<dyn Fft<f32>>,
	row_r: &mut [f32],
	row_c: &mut [Complex<f32>],
	col: &mut [Complex<f32>],
	r2c_scratch: &mut [Complex<f32>],
	col_scratch: &mut [Complex<f32>],
	spectrum: &mut [Complex<f32>],
) {
	let half_w = w / 2 + 1;

	// R2C each row: u8 → f32 → half-spectrum complex.
	for y in 0..h {
		let base = y * w * stride;
		for x in 0..w {
			row_r[x] = raw[base + x * stride + ch] as f32 * (1.0 / 255.0);
		}
		r2c.process_with_scratch(row_r, row_c, r2c_scratch)
			.expect("R2C failed");
		spectrum[y * half_w..(y + 1) * half_w].copy_from_slice(row_c);
	}

	// Complex FFT each column of the half-spectrum.
	for u in 0..half_w {
		for y in 0..h {
			col[y] = spectrum[y * half_w + u];
		}
		fft_col.process_with_scratch(col, col_scratch);
		for y in 0..h {
			spectrum[y * half_w + u] = col[y];
		}
	}
}

// ── 2-D C2R inverse FFT ─────────────────────────────────────────────────────

/// Column-wise complex IFFT then row-wise C2R.
/// Normalises, clamps, and writes directly into interleaved u8 output bytes.
// All buffer arguments are pre-allocated by the caller and reused across channels
// to avoid per-channel heap allocations in this hot path.
#[allow(clippy::too_many_arguments)]
fn inverse_c2r_2d(
	spectrum: &mut [Complex<f32>],
	w: usize,
	h: usize,
	stride: usize,
	ch: usize,
	norm: f32,
	c2r: &Arc<dyn realfft::ComplexToReal<f32>>,
	ifft_col: &Arc<dyn Fft<f32>>,
	row_c: &mut [Complex<f32>],
	col: &mut [Complex<f32>],
	row_r: &mut [f32],
	c2r_scratch: &mut [Complex<f32>],
	col_scratch: &mut [Complex<f32>],
	out_raw: &mut [u8],
) {
	let half_w = w / 2 + 1;

	// Inverse complex FFT each column.
	for u in 0..half_w {
		for y in 0..h {
			col[y] = spectrum[y * half_w + u];
		}
		ifft_col.process_with_scratch(col, col_scratch);
		for y in 0..h {
			spectrum[y * half_w + u] = col[y];
		}
	}

	// C2R each row → normalise → clamp → write u8.
	for y in 0..h {
		row_c.copy_from_slice(&spectrum[y * half_w..(y + 1) * half_w]);

		// ── Enforce Hermitian symmetry constraint ───────────────
		// The DC bin (index 0) must be real-valued.
		// The Nyquist bin (last index) must also be real when w is even.
		// Column FFTs + blending introduce floating-point drift into
		// these imaginary parts; realfft's C2R strictly validates this.
		row_c[0].im = 0.0;
		if w % 2 == 0 {
			row_c[half_w - 1].im = 0.0;
		}

		c2r.process_with_scratch(row_c, row_r, c2r_scratch)
			.expect("C2R failed");
		let base = y * w * stride;
		for x in 0..w {
			let v = (row_r[x] * norm).clamp(0.0, 1.0);
			out_raw[base + x * stride + ch] = (v * 255.0 + 0.5) as u8;
		}
	}
}

// ── Image helpers ────────────────────────────────────────────────────────────

/// Convert both images to matching interleaved u8 buffers.
fn extract_raw_pair(ai: &DynamicImage, lz: &DynamicImage) -> (Vec<u8>, Vec<u8>, usize) {
	match ai.color().channel_count() {
		1 => (ai.to_luma8().into_raw(), lz.to_luma8().into_raw(), 1),
		3 => (ai.to_rgb8().into_raw(), lz.to_rgb8().into_raw(), 3),
		_ => (ai.to_rgba8().into_raw(), lz.to_rgba8().into_raw(), 4),
	}
}

/// Reconstruct a `DynamicImage` from interleaved raw bytes (takes ownership, no copy).
fn raw_to_image(raw: Vec<u8>, w: u32, h: u32, channels: usize) -> DynamicImage {
	match channels {
		1 => DynamicImage::ImageLuma8(
			image::GrayImage::from_raw(w, h, raw).expect("buffer size mismatch"),
		),
		3 => DynamicImage::ImageRgb8(
			image::RgbImage::from_raw(w, h, raw).expect("buffer size mismatch"),
		),
		4 => DynamicImage::ImageRgba8(
			image::RgbaImage::from_raw(w, h, raw).expect("buffer size mismatch"),
		),
		_ => unreachable!(),
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn blend_zero_is_identity() {
		let ai = DynamicImage::ImageRgb8(image::RgbImage::from_fn(8, 8, |x, y| {
			image::Rgb([(x * 30) as u8, (y * 30) as u8, 128])
		}));
		let original = DynamicImage::ImageRgb8(image::RgbImage::from_fn(4, 4, |x, y| {
			image::Rgb([(x * 60) as u8, (y * 60) as u8, 64])
		}));
		let result = frequency_blend_with_original(&ai, &original, 0.0, None).unwrap();
		assert_eq!(result.to_rgb8().as_raw(), ai.to_rgb8().as_raw());
	}

	#[test]
	fn blend_identical_images_is_stable() {
		let img = DynamicImage::ImageRgb8(image::RgbImage::from_fn(16, 16, |x, y| {
			image::Rgb([(x * 15) as u8, (y * 15) as u8, 100])
		}));
		let result = frequency_blend(&img, &img, 1.0, None).unwrap();
		let a = img.to_rgb8();
		let b = result.to_rgb8();
		for (pa, pb) in a.pixels().zip(b.pixels()) {
			for c in 0..3 {
				let diff = (pa[c] as i16 - pb[c] as i16).unsigned_abs();
				assert!(diff <= 1, "pixel diff {diff} exceeds 1 lsb");
			}
		}
	}
}
