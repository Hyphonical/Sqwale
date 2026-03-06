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
		#[arg(short, long)]
		model: String,

		/// Output file path or directory.
		/// Omit to write next to the input as `{stem}_{scale}x.{ext}`.
		#[arg(short, long)]
		output: Option<String>,
	},
}
