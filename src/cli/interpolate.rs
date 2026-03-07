//! `sqwale interpolate` command handler.

use anyhow::{Context, Result, bail};
use colored::Colorize;
use indicatif::ProgressBar;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use sqwale::interpolate::{self, InterpolateOptions, RifeSession};
use sqwale::pipeline::CancelToken;
use sqwale::session::ProviderSelection;

use super::Cli;
use super::output::*;

/// Run the interpolate command.
pub fn run(
	input: &str,
	output_arg: Option<&str>,
	multiplier: u32,
	crf: u32,
	ensemble: bool,
	args: &Cli,
) -> Result<()> {
	// Validate input file exists.
	let input_path = Path::new(input);
	if !input_path.exists() {
		bail!("Input file not found: {input}");
	}

	// Validate multiplier is a power of two ≥ 2.
	if multiplier < 2 || !multiplier.is_power_of_two() {
		bail!("Multiplier must be a power of two (2, 4, 8, …), got {multiplier}");
	}

	// Resolve output path (always .mkv).
	let output_path = resolve_output(input_path, output_arg, multiplier)?;

	// Parse provider.
	let provider: ProviderSelection = args.provider.parse().context("Invalid --provider value")?;

	// Set up cancellation.
	let cancel = CancelToken::new();
	let interrupt_count = Arc::new(AtomicUsize::new(0));

	{
		let cancel = cancel.clone();
		let count = interrupt_count.clone();
		ctrlc::set_handler(move || {
			let prev = count.fetch_add(1, Ordering::SeqCst);
			if prev == 0 {
				cancel.cancel();
			} else {
				std::process::exit(1);
			}
		})
		.context("Failed to set Ctrl+C handler")?;
	}

	// Header.
	println!(
		"{} {}",
		SYM_BULLET.cyan().bold(),
		path_str(&input_path.display().to_string())
	);

	// Configuration line.
	let config_summary = format!(
		"{}{}{}{}{}",
		format!("{multiplier}×").truecolor(CLR_VALUE.0, CLR_VALUE.1, CLR_VALUE.2),
		format!(" {SYM_DOT} ").dimmed(),
		format!("CRF {crf}").truecolor(CLR_VALUE.0, CLR_VALUE.1, CLR_VALUE.2),
		format!(" {SYM_DOT} ").dimmed(),
		if ensemble {
			"Ensemble"
				.truecolor(CLR_VALUE.0, CLR_VALUE.1, CLR_VALUE.2)
				.to_string()
		} else {
			"Standard"
				.truecolor(CLR_VALUE.0, CLR_VALUE.1, CLR_VALUE.2)
				.to_string()
		},
	);
	println!(
		"{}  {} {}  {}",
		SYM_DOT.dimmed(),
		"Config".dimmed(),
		config_summary,
		format!("RIFE 4.25 via {}", provider.name()).dimmed(),
	);

	// Probe input for metadata.
	let info = sqwale::ffmpeg::probe(input_path)?;

	println!(
		"{}  {} {}  {}  {}",
		SYM_DOT.dimmed(),
		"Input".dimmed(),
		dims_str(info.width as u32, info.height as u32),
		format_args!(
			"{}{}",
			format!("{:.2}", info.fps).truecolor(CLR_VALUE.0, CLR_VALUE.1, CLR_VALUE.2),
			" fps".dimmed()
		),
		format_args!(
			"~{} frames",
			info.frame_count
				.to_string()
				.truecolor(CLR_VALUE.0, CLR_VALUE.1, CLR_VALUE.2)
		),
	);

	// Load RIFE model with spinner.
	let mut rife = with_spinner("Loading RIFE model…", || RifeSession::new(provider))
		.context("Failed to load RIFE model")?;

	let start = Instant::now();
	let show_progress = should_show_progress();

	// Setup progress bar.
	let total_output_frames = if info.frame_count > 0 {
		(info.frame_count - 1) * multiplier as usize + 1
	} else {
		0
	};

	let pb = if show_progress {
		let pb = ProgressBar::new(total_output_frames as u64).with_style(interp_bar_style());
		pb.enable_steady_tick(Duration::from_millis(SPINNER_TICK_MS));
		Some(pb)
	} else {
		None
	};

	let pb_clone = pb.clone();
	let options = InterpolateOptions {
		multiplier,
		ensemble,
		crf,
		cancel: cancel.clone(),
		on_progress: Some(Box::new(move |done, total| {
			if let Some(ref pb) = pb_clone {
				pb.set_length(total as u64);
				pb.set_position(done as u64);
			}
		})),
	};

	let result = interpolate::run(input_path, &output_path, &mut rife, &options);

	if let Some(ref pb) = pb {
		pb.finish_and_clear();
	}

	// Free model memory.
	drop(rife);

	let result = result?;
	let elapsed = start.elapsed();

	let out_fps = info.fps * multiplier as f64;

	println!(
		"{}  {} {}  {}  {}",
		SYM_DOT.dimmed(),
		"Output".dimmed(),
		dims_str(info.width as u32, info.height as u32),
		format_args!(
			"{}{}",
			format!("{:.2}", out_fps).truecolor(CLR_VALUE.0, CLR_VALUE.1, CLR_VALUE.2),
			" fps".dimmed()
		),
		format_args!(
			"{} Interpolating…",
			result
				.frames_written
				.to_string()
				.truecolor(CLR_VALUE.0, CLR_VALUE.1, CLR_VALUE.2)
		),
	);

	println!(
		"{} {}  {}",
		SYM_CHECK.green(),
		path_str(&output_path.display().to_string()),
		format_duration(elapsed).dimmed()
	);

	Ok(())
}

// ── Helpers ────────────────────────────────────────────────────────────────

/// Resolve the output path, forcing `.mkv` extension.
fn resolve_output(input: &Path, output_arg: Option<&str>, multiplier: u32) -> Result<PathBuf> {
	let path = if let Some(out) = output_arg {
		let mut p = PathBuf::from(out);
		// Force .mkv extension.
		if p.extension().is_some_and(|ext| ext != "mkv") {
			tracing::warn!(
				"Output extension changed to .mkv (was .{})",
				p.extension().unwrap().to_string_lossy()
			);
		}
		p.set_extension("mkv");
		p
	} else {
		// Default: {stem}_{multiplier}x.mkv
		let stem = input
			.file_stem()
			.context("Input has no file stem")?
			.to_string_lossy();
		let parent = input.parent().unwrap_or(Path::new("."));
		parent.join(format!("{stem}_{multiplier}x.mkv"))
	};

	Ok(path)
}
