//! CLI entry point for sqwale.

mod cli;
mod commands;

use anyhow::Result;
use clap::Parser;
use cli::{Cli, Commands};
use tracing_indicatif::IndicatifLayer;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

fn main() -> Result<()> {
	let cli = Cli::parse();

	// Set up tracing with indicatif integration so that log lines and
	// progress bars never clobber each other.
	let indicatif_layer = IndicatifLayer::new();

	tracing_subscriber::registry()
		.with(
			tracing_subscriber::fmt::layer()
				.without_time()
				.with_target(false)
				.with_level(false)
				.with_writer(indicatif_layer.get_stderr_writer()),
		)
		.with(indicatif_layer)
		.init();

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
