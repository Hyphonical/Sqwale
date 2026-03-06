//! CLI argument definitions and command dispatch.

pub mod inspect;
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
	},
}
