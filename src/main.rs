//! CLI entry point for sqwale.

mod cli;
mod commands;

use anyhow::Result;
use clap::Parser;
use cli::{Cli, Commands};

fn main() -> Result<()> {
	let cli = Cli::parse();

	match &cli.command {
		Commands::Inspect { pattern } => commands::inspect::run(pattern, cli.verbose, cli.quiet),
		Commands::Upscale {
			input,
			model,
			output,
			provider,
		} => commands::upscale::run(input, model, output.as_deref(), provider, cli.quiet),
	}
}
