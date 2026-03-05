//! Upscale command implementation.
//!
//! Styled output following `sqwale_style.txt`:
//! - `●` cyan bold   — section header / major step
//! - `✓` green       — success
//! - `✗` red bold    — hard error
//! - `⚠` yellow      — warning
//! - `·` dimmed      — sub-detail
//!
//! Progress bars (via `tracing-indicatif`):
//! - Tile bar   (child span)  — top
//! - Batch bar  (parent span) — bottom

use anyhow::{Context, Result};
use colored::Colorize;
use glob::glob;
use image::GenericImageView;
use indicatif::ProgressStyle;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tracing::info_span;
use tracing_indicatif::span_ext::IndicatifSpanExt;

use sqwale::inspect::ModelInfo;
use sqwale::upscale::{Provider, UpscaleOptions, UpscaleSession};

// ── Progress bar style ────────────────────────────────────────────────────

fn tile_bar_style() -> ProgressStyle {
	ProgressStyle::with_template("  {bar:40.cyan/black} {pos:>3}/{len} tiles  {elapsed}  {per_sec}")
		.expect("valid tile progress bar template")
		.progress_chars("━━╌")
}

fn batch_bar_style() -> ProgressStyle {
	ProgressStyle::with_template("  {bar:40.cyan/black} {pos:>3}/{len}  images  {elapsed} elapsed")
		.expect("valid batch progress bar template")
		.progress_chars("━━╌")
}

// ── Helpers ────────────────────────────────────────────────────────────────

/// Format a model summary line like: `2x · RGB · float16 · dynamic`
fn model_summary(info: &ModelInfo) -> String {
	let tiling = if info.tile.supported {
		"dynamic"
	} else {
		"fixed"
	};
	format!(
		"{}x · {} · {} · {}",
		info.scale, info.color_space, info.input_dtype, tiling
	)
}

/// Format duration as human-readable string.
fn fmt_duration(secs: f64) -> String {
	if secs < 60.0 {
		format!("{secs:.1}s")
	} else {
		format!("{:.0}m {:.0}s", secs / 60.0, secs % 60.0)
	}
}

/// Determine output path for a single image.
fn output_path_for(input: &Path, output: Option<&Path>, scale: u32) -> PathBuf {
	if let Some(out) = output {
		if out.extension().is_some() {
			return out.to_path_buf();
		}
		let ext = input.extension().and_then(|e| e.to_str()).unwrap_or("png");
		return PathBuf::from(format!("{}.{ext}", out.display()));
	}

	let stem = input
		.file_stem()
		.and_then(|s| s.to_str())
		.unwrap_or("output");
	let ext = input.extension().and_then(|e| e.to_str()).unwrap_or("png");
	let parent = input.parent().unwrap_or(Path::new("."));
	parent.join(format!("{stem}_{scale}x.{ext}"))
}

// ── Entry point ────────────────────────────────────────────────────────────

/// Run the upscale command.
pub fn run(
	input: &str,
	model: &str,
	output: Option<&str>,
	provider_str: &str,
	quiet: bool,
) -> Result<()> {
	let interrupted = Arc::new(AtomicBool::new(false));
	{
		let flag = interrupted.clone();
		ctrlc::set_handler(move || {
			flag.store(true, Ordering::SeqCst);
		})
		.context("Failed to install Ctrl+C handler")?;
	}

	let has_glob = input.contains('*') || input.contains('?') || input.contains('[');

	if has_glob {
		run_batch(input, model, output, provider_str, quiet, &interrupted)
	} else {
		run_single(input, model, output, provider_str, quiet, &interrupted)
	}
}

// ── Single Image ───────────────────────────────────────────────────────────

fn run_single(
	input: &str,
	model: &str,
	output: Option<&str>,
	provider_str: &str,
	quiet: bool,
	interrupted: &Arc<AtomicBool>,
) -> Result<()> {
	let input_path = PathBuf::from(input);
	let model_path = Path::new(model);

	validate_paths(&input_path, model_path)?;

	let provider = provider_str
		.parse::<Provider>()
		.context("Invalid execution provider")?;

	let options = UpscaleOptions { provider };
	let mut session = UpscaleSession::new(model_path, &options)?;
	let info = session.model_info().clone();

	if !quiet {
		eprintln!(
			"{}",
			format!("── upscale · single image ──{:─<30}", "").dimmed()
		);
		eprintln!();
		eprintln!(
			"{} {}",
			"●".cyan().bold(),
			input_path.display().to_string().bright_white()
		);
		eprintln!(
			"{}",
			format!(
				"  · Model   {}  {} via {}",
				model_summary(&info).cyan(),
				model_path
					.file_name()
					.and_then(|n| n.to_str())
					.unwrap_or("unknown"),
				session.provider()
			)
			.dimmed()
		);
	}

	let input_image = image::open(&input_path)
		.with_context(|| format!("Failed to decode image '{}'", input_path.display()))?;

	let (w, h) = input_image.dimensions();
	if !quiet {
		eprintln!(
			"{}",
			format!(
				"  · Input   {}×{}",
				w.to_string().bright_white().bold(),
				h.to_string().bright_white().bold()
			)
			.dimmed()
		);
	}

	if interrupted.load(Ordering::SeqCst) {
		anyhow::bail!("Cancelled before upscaling started");
	}

	let start = std::time::Instant::now();

	// Create a tracing span for the tile progress bar
	let tile_span = info_span!("tiles");
	tile_span.pb_set_style(&tile_bar_style());
	let _tile_guard = tile_span.enter();

	let upscaled = session.upscale(input_image, |done, total| {
		if done == 1 {
			tile_span.pb_set_length(total as u64);
		}
		tile_span.pb_set_position(done as u64);
	})?;

	drop(_tile_guard);

	let elapsed = start.elapsed().as_secs_f64();
	let (ow, oh) = upscaled.dimensions();

	if !quiet {
		eprintln!(
			"{}",
			format!(
				"  · Output  {}×{}",
				ow.to_string().bright_white().bold(),
				oh.to_string().bright_white().bold()
			)
			.dimmed()
		);
	}

	let out = output_path_for(&input_path, output.map(Path::new), info.scale);
	if let Some(parent) = out.parent() {
		if !parent.exists() {
			std::fs::create_dir_all(parent).with_context(|| {
				format!("Failed to create output directory '{}'", parent.display())
			})?;
		}
	}

	upscaled
		.save(&out)
		.with_context(|| format!("Failed to save output image '{}'", out.display()))?;

	if !quiet {
		eprintln!(
			"{} {}  {}",
			"✓".green(),
			out.display().to_string().bright_white(),
			fmt_duration(elapsed).dimmed()
		);
	}

	Ok(())
}

// ── Batch ──────────────────────────────────────────────────────────────────

fn run_batch(
	pattern: &str,
	model: &str,
	output_dir: Option<&str>,
	provider_str: &str,
	quiet: bool,
	interrupted: &Arc<AtomicBool>,
) -> Result<()> {
	let model_path = Path::new(model);
	if !model_path.exists() {
		anyhow::bail!("Model file not found: '{model}'");
	}

	let matches: Vec<PathBuf> = glob(pattern)
		.with_context(|| format!("Invalid glob pattern '{pattern}'"))?
		.filter_map(Result::ok)
		.filter(|p| p.is_file())
		.collect();

	if matches.is_empty() {
		anyhow::bail!("No files matched pattern '{pattern}'");
	}

	let provider = provider_str
		.parse::<Provider>()
		.context("Invalid execution provider")?;

	let options = UpscaleOptions { provider };
	let mut session = UpscaleSession::new(model_path, &options)?;
	let info = session.model_info().clone();

	let out_dir = output_dir.map(PathBuf::from).unwrap_or_else(|| {
		Path::new(pattern)
			.parent()
			.unwrap_or(Path::new("."))
			.join("upscaled")
	});

	if !out_dir.exists() {
		std::fs::create_dir_all(&out_dir).with_context(|| {
			format!("Failed to create output directory '{}'", out_dir.display())
		})?;
	}

	let total = matches.len();

	if !quiet {
		eprintln!("{}", format!("── upscale · batch ──{:─<36}", "").dimmed());
		eprintln!();
		eprintln!(
			"{} {}  {}",
			"●".cyan().bold(),
			format!("Batch: {} images", total.to_string().bright_white().bold()).white(),
			format!("→ {}", out_dir.display()).dimmed()
		);
		eprintln!(
			"{}",
			format!("  · Model   {}", model_summary(&info).cyan()).dimmed()
		);
		eprintln!(
			"{}",
			format!(
				"  · Loaded  {}  via {}",
				model_path
					.file_name()
					.and_then(|n| n.to_str())
					.unwrap_or("unknown"),
				session.provider()
			)
			.dimmed()
		);
		eprintln!();
	}

	// Parent span → batch progress bar (bottom)
	let batch_span = info_span!("batch");
	batch_span.pb_set_style(&batch_bar_style());
	batch_span.pb_set_length(total as u64);
	let _batch_guard = batch_span.enter();

	let mut ok = 0usize;
	let mut failed: Vec<(PathBuf, String)> = Vec::new();
	let mut skipped = 0usize;

	for (idx, input_path) in matches.iter().enumerate() {
		if interrupted.load(Ordering::SeqCst) {
			skipped = total - idx;
			if !quiet {
				eprintln!();
				eprintln!(
					"{} {}",
					"⚠".yellow().bold(),
					"Interrupted — finishing current image, skipping remaining".white()
				);
			}
			break;
		}

		let img_start = std::time::Instant::now();

		if !quiet {
			eprintln!(
				"{} {} {}",
				"●".cyan().bold(),
				format!("[{}/{}]", idx + 1, total).dimmed(),
				input_path.display().to_string().bright_white()
			);
		}

		let input_image = match image::open(input_path) {
			Ok(img) => img,
			Err(e) => {
				let msg = format!("{e:#}");
				if !quiet {
					eprintln!(
						"  {} {}",
						"✗".red(),
						format!("Failed to decode image: {msg}").white()
					);
				}
				failed.push((input_path.clone(), msg));
				batch_span.pb_inc(1);
				continue;
			}
		};

		let (w, h) = input_image.dimensions();
		if !quiet {
			eprintln!(
				"{}",
				format!(
					"  · Input   {}×{}",
					w.to_string().bright_white().bold(),
					h.to_string().bright_white().bold()
				)
				.dimmed()
			);
		}

		// Child span → tile progress bar (top)
		let tile_span = info_span!("tiles");
		tile_span.pb_set_style(&tile_bar_style());
		let tile_span_entered = tile_span.enter();

		let result = session.upscale(input_image, |done, total_tiles| {
			if done == 1 {
				tile_span.pb_set_length(total_tiles as u64);
			}
			tile_span.pb_set_position(done as u64);
		});

		drop(tile_span_entered);

		match result {
			Ok(upscaled) => {
				let (ow, oh) = upscaled.dimensions();
				if !quiet {
					eprintln!(
						"{}",
						format!(
							"  · Output  {}×{}",
							ow.to_string().bright_white().bold(),
							oh.to_string().bright_white().bold()
						)
						.dimmed()
					);
				}

				let out_name = input_path
					.file_stem()
					.and_then(|s| s.to_str())
					.unwrap_or("output");
				let out_ext = input_path
					.extension()
					.and_then(|e| e.to_str())
					.unwrap_or("png");
				let out_path = out_dir.join(format!("{out_name}_{s}x.{out_ext}", s = info.scale));

				match upscaled.save(&out_path) {
					Ok(()) => {
						let elapsed = img_start.elapsed().as_secs_f64();
						ok += 1;
						if !quiet {
							eprintln!(
								"  {} {}  {}",
								"✓".green(),
								out_path.display().to_string().bright_white(),
								fmt_duration(elapsed).dimmed()
							);
							eprintln!();
						}
					}
					Err(e) => {
						let msg = format!("Failed to save: {e:#}");
						if !quiet {
							eprintln!("  {} {}", "✗".red(), msg.white());
						}
						failed.push((input_path.clone(), msg));
					}
				}
			}
			Err(e) => {
				let msg = format!("{e:#}");
				if !quiet {
					eprintln!(
						"  {} {}",
						"✗".red(),
						format!("Upscale failed: {msg}").white()
					);
				}
				failed.push((input_path.clone(), msg));
			}
		}

		batch_span.pb_inc(1);
	}

	drop(_batch_guard);

	// ── Summary ────────────────────────────────────────────────────────────

	if !quiet {
		if interrupted.load(Ordering::SeqCst) {
			eprintln!();
			eprintln!(
				"{} {}",
				"✗".red().bold(),
				format!("Cancelled after {}/{total}", ok + failed.len()).white()
			);
		}

		if ok > 0 {
			eprint!(
				"{} {} {}",
				"✓".green(),
				ok.to_string().bright_white().bold(),
				"succeeded".white()
			);
		}
		if !failed.is_empty() {
			eprint!(
				"  {}  {} {}",
				"·".dimmed(),
				failed.len().to_string().red(),
				"failed".red()
			);
		}
		if skipped > 0 {
			eprint!("  {}  {} skipped", "·".dimmed(), skipped);
		}
		eprintln!();

		// List failures
		for (path, msg) in &failed {
			eprintln!("  {} {}  {}", "✗".dimmed(), path.display(), msg.dimmed());
		}
	}

	Ok(())
}

// ── Validation ─────────────────────────────────────────────────────────────

fn validate_paths(input: &Path, model: &Path) -> Result<()> {
	if !input.exists() {
		anyhow::bail!("Input file not found: '{}'", input.display());
	}
	if !model.exists() {
		anyhow::bail!("Model file not found: '{}'", model.display());
	}
	Ok(())
}
