//! Upscale command implementation.

use anyhow::{Context, Result};
use colored::Colorize;
use glob::glob;
use image::GenericImageView;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use sqwale::upscale::{Provider, UpscaleOptions, UpscaleSession};

/// Run the upscale command.
pub fn run(
	input: &str,
	model: &str,
	output: Option<&str>,
	provider_str: &str,
	quiet: bool,
) -> Result<()> {
	// Setup Ctrl+C handler
	let interrupted = Arc::new(AtomicBool::new(false));
	let interrupted_clone = interrupted.clone();
	ctrlc::set_handler(move || {
		interrupted_clone.store(true, Ordering::SeqCst);
		eprintln!(
			"\n{} Interrupted by user. Cleaning up...",
			"⚠".yellow().bold()
		);
	})
	.context("Failed to set Ctrl+C handler")?;

	// Check if input is a glob pattern
	let has_glob = input.contains('*') || input.contains('?') || input.contains('[');

	if has_glob {
		run_batch(input, model, output, provider_str, quiet, interrupted)
	} else {
		run_single(input, model, output, provider_str, quiet, interrupted)
	}
}

/// Run upscale on a single image.
fn run_single(
	input: &str,
	model: &str,
	output: Option<&str>,
	provider_str: &str,
	quiet: bool,
	interrupted: Arc<AtomicBool>,
) -> Result<()> {
	let input_path = PathBuf::from(input);
	let output_path = output.map(PathBuf::from);
	run_single_internal(
		&input_path,
		model,
		output_path.as_deref(),
		provider_str,
		quiet,
		interrupted,
		None,
	)
}

/// Internal implementation of single image upscale.
fn run_single_internal(
	input_path: &Path,
	model: &str,
	output_path: Option<&Path>,
	provider_str: &str,
	quiet: bool,
	interrupted: Arc<AtomicBool>,
	session: Option<&mut UpscaleSession>,
) -> Result<()> {
	if !input_path.exists() {
		anyhow::bail!(
			"{} '{}'\n  {} Check the file path and ensure it exists",
			"Input file not found:".red().bold(),
			input_path.display().to_string().bright_white(),
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

	let options = UpscaleOptions {
		provider,
		tile_size: None, // Auto-calculate
		overlap: 16,     // Default overlap
	};

	let mut owned_session;
	let session_ref = if let Some(s) = session {
		s
	} else {
		if !quiet {
			eprintln!(
				"{} Loading model {}",
				"→".bright_blue(),
				model_path.display().to_string().bright_white()
			);
		}
		owned_session = UpscaleSession::new(model_path, &options)?;
		&mut owned_session
	};

	if !quiet {
		let info = session_ref.model_info();
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

	// Check for interruption before starting
	if interrupted.load(Ordering::SeqCst) {
		anyhow::bail!("{} Operation cancelled", "✗".red());
	}

	let upscaled = session_ref.upscale(input_image)?;
	let (new_width, new_height) = upscaled.dimensions();

	if !quiet {
		eprintln!(
			"{} Output: {}×{} pixels",
			"✓".green(),
			new_width.to_string().cyan(),
			new_height.to_string().cyan()
		);
	}

	let final_output_path = match output_path {
		Some(p) => p.to_path_buf(),
		None => determine_output_path(input_path, None, &upscaled)?,
	};

	if !quiet {
		eprintln!(
			"{} Saving to {}",
			"→".bright_blue(),
			final_output_path.display().to_string().bright_white()
		);
	}

	upscaled
		.save(&final_output_path)
		.with_context(|| format!("Failed to save image: {}", final_output_path.display()))?;

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

/// Run upscale on a batch of images matching a glob pattern.
fn run_batch(
	pattern: &str,
	model: &str,
	output_dir: Option<&str>,
	provider_str: &str,
	quiet: bool,
	interrupted: Arc<AtomicBool>,
) -> Result<()> {
	// Find matching files
	let matches: Vec<PathBuf> = glob(pattern)
		.context("Invalid glob pattern")?
		.filter_map(Result::ok)
		.filter(|p| p.is_file())
		.collect();

	if matches.is_empty() {
		anyhow::bail!(
			"{} No files matched pattern: {}\n  {} Try a different pattern like *.jpg or images/*.png",
			"No matches:".red().bold(),
			pattern.bright_white(),
			"Hint:".yellow()
		);
	}

	if !quiet {
		eprintln!(
			"{} Found {} image(s) matching pattern",
			"✓".green(),
			matches.len().to_string().cyan()
		);
	}

	// Determine output directory
	let out_dir = output_dir.map(PathBuf::from).unwrap_or_else(|| {
		Path::new(pattern)
			.parent()
			.unwrap_or_else(|| Path::new("."))
			.join("upscaled")
	});

	if !out_dir.exists() {
		std::fs::create_dir_all(&out_dir)
			.with_context(|| format!("Failed to create output directory: {}", out_dir.display()))?;
		if !quiet {
			eprintln!(
				"{} Created output directory: {}",
				"✓".green(),
				out_dir.display().to_string().bright_white()
			);
		}
	}

	// Load model once
	let model_path = Path::new(model);
	let provider = provider_str.parse::<Provider>()?;
	let options = UpscaleOptions {
		provider,
		tile_size: None,
		overlap: 16,
	};

	if !quiet {
		eprintln!(
			"{} Loading model {}",
			"→".bright_blue(),
			model_path.display().to_string().bright_white()
		);
	}

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
	}

	// Process each image
	let mut success_count = 0;
	let mut error_count = 0;

	for (idx, input_path) in matches.iter().enumerate() {
		if interrupted.load(Ordering::SeqCst) {
			eprintln!(
				"\n{} Batch processing cancelled after {}/{} images",
				"⚠".yellow().bold(),
				idx,
				matches.len()
			);
			break;
		}

		if !quiet {
			eprintln!(
				"\n{} [{}/{}] Processing {}",
				"→".bright_blue(),
				idx + 1,
				matches.len(),
				input_path.display().to_string().bright_white()
			);
		}

		// Determine output path
		let filename = input_path
			.file_name()
			.and_then(|n| n.to_str())
			.unwrap_or("output.png");
		let output_path = out_dir.join(filename);

		match run_single_internal(
			input_path,
			model,
			Some(&output_path),
			provider_str,
			quiet,
			interrupted.clone(),
			Some(&mut session),
		) {
			Ok(_) => success_count += 1,
			Err(e) => {
				error_count += 1;
				eprintln!(
					"{}  Failed to process {}: {}",
					"✗".red(),
					input_path.display(),
					e
				);
			}
		}
	}

	// Summary
	if !quiet {
		eprintln!("\n{} Batch complete:", "✓".green().bold());
		eprintln!("  {} {} succeeded", "✓".green(), success_count);
		if error_count > 0 {
			eprintln!("  {} {} failed", "✗".red(), error_count);
		}
	}

	Ok(())
}
