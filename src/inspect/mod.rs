//! Model inspection: public types and inspection API.

mod detect;
pub(crate) mod proto;

pub use detect::inspect_model;

// ── Color Space ────────────────────────────────────────────────────────────

/// The color space inferred from the model's channel count.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ColorSpace {
	Grayscale,
	Rgb,
	Rgba,
	Unknown(u32),
}

impl std::fmt::Display for ColorSpace {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		match self {
			Self::Grayscale => write!(f, "Grayscale"),
			Self::Rgb => write!(f, "RGB"),
			Self::Rgba => write!(f, "RGBA"),
			Self::Unknown(n) => write!(f, "Unknown ({n} channels)"),
		}
	}
}

impl ColorSpace {
	/// Number of channels.
	pub fn channels(&self) -> u32 {
		match self {
			Self::Grayscale => 1,
			Self::Rgb => 3,
			Self::Rgba => 4,
			Self::Unknown(n) => *n,
		}
	}
}

// ── Scale Source ───────────────────────────────────────────────────────────

/// How the upscale factor was detected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScaleSource {
	Metadata,
	StaticShapeRatio,
	DepthToSpace,
	ConvTransposeStride,
	Assumed,
}

impl std::fmt::Display for ScaleSource {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		match self {
			Self::Metadata => write!(f, "metadata_props"),
			Self::StaticShapeRatio => write!(f, "static shape ratio"),
			Self::DepthToSpace => write!(f, "DepthToSpace (PixelShuffle)"),
			Self::ConvTransposeStride => write!(f, "ConvTranspose stride"),
			Self::Assumed => write!(f, "assumed (no upscale op found)"),
		}
	}
}

// ── Tile Requirements ──────────────────────────────────────────────────────

/// Tiling constraints extracted from the model graph.
#[derive(Debug, Clone)]
pub struct TileInfo {
	/// Whether the model supports tiling (dynamic spatial dims).
	pub supported: bool,
	/// Required alignment for spatial dimensions (e.g. 8, 16, 32).
	/// Detected from Reshape / window-partition patterns in transformers.
	pub alignment: Option<u32>,
	/// Fully-static required input size `(height, width)`.
	pub fixed_size: Option<(u64, u64)>,
}

impl TileInfo {
	/// Returns the tile size to use, given a user preference.
	///
	/// * If the model has a fixed input size, that size is returned and the
	///   user preference is ignored.
	/// * Otherwise the user preference is rounded up to the nearest multiple of
	///   the model's required alignment (if any).
	pub fn effective_tile_size(&self, user_pref: u32) -> u32 {
		if let Some((h, w)) = self.fixed_size {
			// Both dims must be equal for a square tile; use height as proxy.
			// Non-square fixed models are unusual; this covers the common case.
			h.min(w) as u32
		} else if let Some(align) = self.alignment {
			// Round up to alignment boundary.
			let r = user_pref % align;
			if r == 0 {
				user_pref
			} else {
				user_pref + (align - r)
			}
		} else {
			user_pref
		}
	}
}

// ── Model Metadata ─────────────────────────────────────────────────────────

/// All metadata extracted from an ONNX model without running inference.
#[derive(Debug, Clone)]
pub struct ModelInfo {
	/// Upscale factor (1 for restoration / denoising models).
	pub scale: u32,
	/// How the scale was determined.
	pub scale_source: ScaleSource,
	/// Input color space derived from channel count.
	pub color_space: ColorSpace,
	/// Number of input channels.
	pub input_channels: u32,
	/// Number of output channels.
	pub output_channels: u32,
	/// Tiling constraints.
	pub tile: TileInfo,
	/// Input element type (e.g. `"float32"`, `"float16"`).
	pub input_dtype: String,
	/// Output element type.
	pub output_dtype: String,
	/// Maximum opset version in use.
	pub opset: u64,
	/// Op-type histogram sorted by frequency descending.
	pub op_fingerprint: Vec<(String, usize)>,
}

impl ModelInfo {
	/// Returns `true` when the model expects half-precision input.
	pub fn needs_fp16_input(&self) -> bool {
		self.input_dtype == "float16"
	}
}
