//! CLI argument definitions and command dispatch.

pub mod inspect;
pub mod interpolate;
pub mod output;
pub mod upscale;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "sqwale", author, version, about)]
#[command(propagate_version = true)]
pub struct Cli {
	#[command(subcommand)]
	pub command: Commands,

	/// Execution provider: auto, cpu, cuda, tensorrt, directml, coreml, xnnpack.
	#[arg(long, global = true, default_value = "auto")]
	pub provider: String,

	/// Tile size in pixels (0 = disable tiling).
	#[arg(long, global = true)]
	pub tile_size: Option<u32>,

	/// Overlap in pixels between adjacent tiles.
	#[arg(long, global = true)]
	pub tile_overlap: Option<u32>,

	/// Blend AI upscale with a Lanczos upscale via an FFT Laplacian pyramid [0.0–1.0].
	///
	/// 0.0 (default) disables blending. 1.0 makes the AI supply fine detail while
	/// the Lanczos upscale of the original supplies global structure and colour.
	#[arg(long, global = true, default_value_t = 0.0, value_parser = parse_blend)]
	pub blend: f32,

	/// Force half-precision (fp16) inference for reduced VRAM and faster GPU execution.
	///
	/// For upscale models, the input tensor is converted to float16 regardless of
	/// what the model declares. For RIFE interpolation, the embedded fp16 model
	/// variant is used. May not work on CPU.
	#[arg(long, global = true)]
	pub fp16: bool,
}

fn parse_blend(s: &str) -> Result<f32, String> {
	let v: f32 = s
		.parse()
		.map_err(|_| format!("'{s}' is not a valid number"))?;
	if (0.0..=1.0).contains(&v) {
		Ok(v)
	} else {
		Err(format!("blend value {v} is out of range, must be 0.0–1.0"))
	}
}

fn parse_grain(s: &str) -> Result<u8, String> {
	let v: u8 = s
		.parse()
		.map_err(|_| format!("'{s}' is not a valid number"))?;
	if v <= 100 {
		Ok(v)
	} else {
		Err(format!("grain strength {v} is out of range, must be 0–100"))
	}
}

fn parse_scene_threshold(s: &str) -> Result<f64, String> {
	let v: f64 = s
		.parse()
		.map_err(|_| format!("'{s}' is not a valid number"))?;
	if (0.0..=1.0).contains(&v) {
		Ok(v)
	} else {
		Err(format!(
			"scene threshold {v} is out of range, must be 0.0–1.0"
		))
	}
}

#[derive(Subcommand)]
pub enum Commands {
	/// Inspect ONNX model metadata without running inference.
	Inspect {
		/// File path, glob pattern, or directory containing .onnx files.
		pattern: String,
	},

	/// Upscale images using an ONNX super-resolution model.
	Upscale {
		/// Input image path or glob pattern.
		input: String,

		/// Path to the ONNX model file.
		/// If omitted, the bundled 4xSPANkendata model is used.
		#[arg(short, long)]
		model: Option<String>,

		/// Output file path or directory.
		/// Omit to write next to the input as `{stem}_{scale}x.{ext}`.
		#[arg(short, long)]
		output: Option<String>,

		/// Add monochrome luma noise post-upscale to reduce the "plastic" AI look.
		/// Scale is 0 to 100. (Recommended: 5 for subtle texture, 20 for heavy film grain).
		#[arg(long, default_value_t = 0, value_parser = parse_grain)]
		grain: u8,

		/// Video output quality (lower = better quality, larger file).
		/// Only used when the input is a video file.
		#[arg(long, default_value_t = 18)]
		crf: u32,
	},

	/// Interpolate video frames using RIFE 4.25 (requires FFmpeg).
	Interpolate {
		/// Input video file path.
		input: String,

		/// Output video file path (mkv, mp4, or webm).
		/// Omit to write next to the input with the same container format.
		#[arg(short, long)]
		output: Option<String>,

		/// Frame rate multiplier (must be a power of two: 2, 4, 8, …).
		#[arg(short = 'x', long, default_value_t = 2)]
		multiplier: u32,

		/// x264 CRF quality value (lower = better quality, larger file).
		#[arg(long, default_value_t = 18)]
		crf: u32,

		/// Use ensemble mode (horizontal-flip averaging) for higher quality.
		#[arg(long)]
		ensemble: bool,

		/// Detect scene changes and duplicate the last pre-cut frame instead of
		/// interpolating across cuts. Prevents blurry or ghosted transitions.
		#[arg(long)]
		scene_detect: bool,

		/// Scene change detection sensitivity [0.0–1.0].
		/// Lower values detect subtler cuts; higher values only catch hard cuts.
		/// Has no effect unless `--scene-detect` is also set.
		#[arg(long, default_value_t = 0.1, value_parser = parse_scene_threshold)]
		scene_threshold: f64,
	},
}
