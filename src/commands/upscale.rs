//! Upscale command implementation.

use anyhow::{Context, Result};
use colored::Colorize;
use image::GenericImageView;
use std::path::{Path, PathBuf};

use sqwale::upscale::{Provider, UpscaleOptions, UpscaleSession};

/// Run the upscale command.
pub fn run(
	input: &str,
	model: &str,
	output: Option<&str>,
	provider_str: &str,
	quiet: bool,
) -> Result<()> {
	let input_path = Path::new(input);
	if !input_path.exists() {
		anyhow::bail!(
			"{} '{}'\n  {} Check the file path and ensure it exists",
			"Input file not found:".red().bold(),
			input.bright_white(),
			"Hint:".yellow()
		);
	}

	let model_path = Path::new(model);
	if !model_path.exists() {
		anyhow::bail!(
			"{} '{}'\n  {} Check the model path and ensure it exists",
			"Model file not found:".red().bold(),
			model.bright_white(),
			"Hint:".yellow()
		);
	}

	let provider = provider_str
		.parse::<Provider>()
		.context("Invalid provider specified")?;

	if !quiet {
		eprintln!(
			"{} Loading model {}",
			"→".bright_blue(),
			model_path.display().to_string().bright_white()
		);
	}

	let options = UpscaleOptions { provider };
	let mut session = UpscaleSession::new(model_path, &options)?;

	if !quiet {
		let info = session.model_info();
		eprintln!(
			"{} Model: {}x upscale, {} channels, {} input",
			"✓".green(),
			info.scale.to_string().green(),
			info.input_channels.to_string().cyan(),
			info.input_dtype.bright_green()
		);
		eprintln!(
			"{} Loading image {}",
			"→".bright_blue(),
			input_path.display().to_string().bright_white()
		);
	}

	let input_image = image::open(input_path)
		.with_context(|| format!("Failed to load image: {}", input_path.display()))?;

	let (orig_width, orig_height) = input_image.dimensions();
	if !quiet {
		eprintln!(
			"{} Input: {}×{} pixels",
			"✓".green(),
			orig_width.to_string().cyan(),
			orig_height.to_string().cyan()
		);
		eprintln!("{} Upscaling...", "→".bright_blue());
	}

	let upscaled = session.upscale(input_image)?;
	let (new_width, new_height) = upscaled.dimensions();

	if !quiet {
		eprintln!(
			"{} Output: {}×{} pixels",
			"✓".green(),
			new_width.to_string().cyan(),
			new_height.to_string().cyan()
		);
	}

	let output_path = determine_output_path(input_path, output, &upscaled)?;

	if !quiet {
		eprintln!(
			"{} Saving to {}",
			"→".bright_blue(),
			output_path.display().to_string().bright_white()
		);
	}

	upscaled
		.save(&output_path)
		.with_context(|| format!("Failed to save image: {}", output_path.display()))?;

	if !quiet {
		eprintln!("{} Done!", "✓".green().bold());
	}

	Ok(())
}

/// Determine output path based on user input and source format.
fn determine_output_path(
	input_path: &Path,
	output: Option<&str>,
	_upscaled: &image::DynamicImage,
) -> Result<PathBuf> {
	let output_path = match output {
		Some(out) => {
			let out_path = Path::new(out);
			if out_path.extension().is_some() {
				// Has extension, use as-is
				out_path.to_path_buf()
			} else {
				// No extension, use source format
				let source_ext = input_path
					.extension()
					.and_then(|e| e.to_str())
					.unwrap_or("png");
				PathBuf::from(format!("{}.{}", out, source_ext))
			}
		}
		None => {
			// Default: input_upscaled.ext in same directory
			let stem = input_path
				.file_stem()
				.and_then(|s| s.to_str())
				.unwrap_or("output");
			let ext = input_path
				.extension()
				.and_then(|e| e.to_str())
				.unwrap_or("png");
			let parent = input_path.parent().unwrap_or(Path::new("."));
			parent.join(format!("{}_upscaled.{}", stem, ext))
		}
	};

	Ok(output_path)
}
