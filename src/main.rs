//! Sqwale CLI entry point.

mod cli;

use anyhow::Result;
use clap::Parser;

fn main() -> Result<()> {
	// Suppress ORT diagnostic output unless the user has set a preference.
	if std::env::var("ORT_LOG_SEVERITY_LEVEL").is_err() {
		// SAFETY: Called before any threads are spawned and before ORT init.
		unsafe { std::env::set_var("ORT_LOG_SEVERITY_LEVEL", "3") };
	}

	// Disable colored output in CI environments.
	if !cli::output::should_use_color() {
		colored::control::set_override(false);
	}

	// Initialize tracing with indicatif integration.
	cli::output::init_tracing();

	// Parse CLI arguments and dispatch.
	let args = cli::Cli::parse();

	match &args.command {
		cli::Commands::Inspect { pattern } => {
			cli::inspect::run(pattern)?;
		}
		cli::Commands::Upscale {
			input,
			model,
			output,
		} => {
			cli::upscale::run(input, model.as_deref(), output.as_deref(), &args)?;
		}
	}

	Ok(())
}
