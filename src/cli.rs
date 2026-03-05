//! CLI argument definitions using clap.

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "sqwale")]
#[command(author, version, about, long_about = None)]
#[command(propagate_version = true)]
pub struct Cli {
	#[command(subcommand)]
	pub command: Commands,

	/// Verbose output
	#[arg(short, long, global = true)]
	pub verbose: bool,

	/// Suppress output
	#[arg(short, long, global = true)]
	pub quiet: bool,
}

#[derive(Subcommand)]
pub enum Commands {
	/// Inspect ONNX model metadata
	Inspect {
		/// Path or glob pattern to .onnx file(s)
		pattern: String,
	},
}
