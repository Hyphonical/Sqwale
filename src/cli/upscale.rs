//! `sqwale upscale` command handler.

use anyhow::{Context, Result, bail};
use colored::Colorize;
use crossbeam_channel::bounded;
use image::DynamicImage;
use indicatif::{MultiProgress, ProgressBar};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use sqwale::imageio;
use sqwale::pipeline::{CancelToken, UpscaleOptions};
use sqwale::session;

use super::Cli;
use super::output::*;

/// Run the upscale command.
pub fn run(
	input_pattern: &str,
	model_path: Option<&str>,
	output_arg: Option<&str>,
	args: &Cli,
) -> Result<()> {
	// Resolve input files.
	let inputs = expand_input_pattern(input_pattern)?;
	if inputs.is_empty() {
		bail!(
			"No image files found matching '{input_pattern}'. \
			 Use a file path or glob pattern (e.g. \"images/*.png\")."
		);
	}

	let is_batch = inputs.len() > 1;

	// Validate output arg for batch mode.
	if is_batch {
		if let Some(out) = output_arg {
			let out_path = Path::new(out);
			if out_path.extension().is_some() && !out_path.is_dir() {
				bail!(
					"Batch mode requires a directory for --output, not a file. \
					 Got: {out}"
				);
			}
		}
	}

	// Parse provider.
	let provider: sqwale::ProviderSelection =
		args.provider.parse().context("Invalid --provider value")?;

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

	// Load model with spinner.
	let (mut ctx, model_filename) = if let Some(path) = model_path {
		let model_p = Path::new(path);
		let ctx = with_spinner("Loading model…", || {
			session::load_model(model_p, provider)
		})
		.with_context(|| format!("Failed to load model: {path}"))?;
		let name = model_p
			.file_name()
			.unwrap_or_default()
			.to_string_lossy()
			.into_owned();
		(ctx, name)
	} else {
		let ctx = with_spinner("Loading model…", || {
			session::load_model_bytes(sqwale::DEFAULT_MODEL_BYTES, provider)
		})
		.context("Failed to load built-in model")?;
		(ctx, "embedded model".to_owned())
	};

	let scale = ctx.model_info.scale;
	let tile_size = args.tile_size.unwrap_or(sqwale::config::DEFAULT_TILE_SIZE);
	let tile_overlap = args
		.tile_overlap
		.unwrap_or(sqwale::config::DEFAULT_TILE_OVERLAP);
	let blend = args.blend;

	let show_progress = should_show_progress();

	if is_batch {
		run_batch(
			&inputs,
			output_arg,
			&mut ctx,
			scale,
			&model_filename,
			tile_size,
			tile_overlap,
			blend,
			show_progress,
			&cancel,
		)
	} else {
		run_single(
			&inputs[0],
			output_arg,
			ctx,
			scale,
			&model_filename,
			tile_size,
			tile_overlap,
			blend,
			show_progress,
			&cancel,
		)
	}
}

/// Run upscale for a single image.
#[allow(clippy::too_many_arguments)]
fn run_single(
	input: &Path,
	output_arg: Option<&str>,
	mut ctx: sqwale::SessionContext,
	scale: u32,
	model_filename: &str,
	tile_size: u32,
	tile_overlap: u32,
	blend: f32,
	show_progress: bool,
	cancel: &CancelToken,
) -> Result<()> {
	let output_path = resolve_output_path(input, output_arg, scale)?;

	// Header.
	println!(
		"{} {}",
		SYM_BULLET.cyan().bold(),
		path_str(&input.display().to_string())
	);

	// Model info line.
	let model_summary = format!(
		"{}{}{}{}{}{}{}",
		format!("{scale}x").truecolor(CLR_VALUE.0, CLR_VALUE.1, CLR_VALUE.2),
		format!(" {SYM_DOT} ").dimmed(),
		color_space_str(&ctx.model_info.color_space),
		format!(" {SYM_DOT} ").dimmed(),
		dtype_str(&ctx.model_info.input_dtype),
		format!(" {SYM_DOT} ").dimmed(),
		if ctx.model_info.tile.supported {
			"dynamic"
				.truecolor(CLR_VALUE.0, CLR_VALUE.1, CLR_VALUE.2)
				.to_string()
		} else {
			"fixed".yellow().to_string()
		},
	);
	println!(
		"{}  {} {}  {}",
		SYM_DOT.dimmed(),
		"Model".dimmed(),
		model_summary,
		format!("{model_filename} via {}", ctx.provider_used.name()).dimmed()
	);

	// Load image.
	let img = imageio::load_image(input)?;
	let (img_w, img_h) = (img.width(), img.height());

	println!(
		"{}  {} {}",
		SYM_DOT.dimmed(),
		"Input".dimmed(),
		dims_str(img_w, img_h),
	);

	let start = Instant::now();

	// Setup progress bar.
	let pb = if show_progress {
		let pb = ProgressBar::new(0).with_style(tile_bar_style());
		pb.enable_steady_tick(Duration::from_millis(SPINNER_TICK_MS));
		Some(pb)
	} else {
		None
	};

	let pb_clone = pb.clone();
	let options = UpscaleOptions {
		tile_size,
		tile_overlap,
		blend: 0.0,
		cancel: cancel.clone(),
		on_tile_done: Some(Box::new(move |done, total| {
			if let Some(ref pb) = pb_clone {
				pb.set_length(total as u64);
				pb.set_position(done as u64);
				let elapsed = start.elapsed();
				let per_tile = if done > 0 {
					elapsed / done as u32
				} else {
					Duration::ZERO
				};
				pb.set_message(format!(
					"{}/{} Upscaling…  {}  {}/tile",
					done.to_string().bold().bright_white(),
					total,
					format_duration(elapsed).dimmed(),
					format_duration(per_tile).dimmed(),
				));
			}
		})),
		on_blend_step: None,
	};

	let result = sqwale::pipeline::upscale_raw(&mut ctx, &img, &options);

	if let Some(ref pb) = pb {
		pb.finish_and_clear();
	}

	// Free the model session to reclaim GPU memory before blending and saving.
	drop(ctx);

	let ai_img = result?;

	// Apply frequency-domain blending if requested.
	let output_img = if blend > 0.0 {
		let pb = if show_progress {
			let pb = ProgressBar::new(0).with_style(tile_bar_style());
			pb.enable_steady_tick(Duration::from_millis(SPINNER_TICK_MS));
			Some(pb)
		} else {
			None
		};
		let pb_clone = pb.clone();
		let on_step = move |done: usize, total: usize| {
			if let Some(ref pb) = pb_clone {
				pb.set_length(total as u64);
				pb.set_position(done as u64);
				pb.set_message(format!(
					"{}/{} Blending…  {}",
					done.to_string().bold().bright_white(),
					total,
					format_duration(start.elapsed()).dimmed(),
				));
			}
		};
		let blended = sqwale::frequency_blend_with_original(&ai_img, &img, blend, Some(&on_step))?;
		if let Some(ref pb) = pb {
			pb.finish_and_clear();
		}
		blended
	} else {
		ai_img
	};

	let elapsed = start.elapsed();

	let (out_w, out_h) = (output_img.width(), output_img.height());
	println!(
		"{}  {} {}",
		SYM_DOT.dimmed(),
		"Output".dimmed(),
		dims_str(out_w, out_h),
	);

	imageio::save_image(&output_img, &output_path)?;

	println!(
		"{} {}  {}",
		SYM_CHECK.green(),
		path_str(&output_path.display().to_string()),
		format_duration(elapsed).dimmed()
	);

	Ok(())
}

/// A prefetched image ready for GPU processing.
struct PrefetchedImage {
	/// Original index in the input list (0-based).
	index: usize,
	/// Resolved input path.
	input: PathBuf,
	/// Resolved output path.
	output: PathBuf,
	/// Load result — an error here is collected, not fatal to the whole batch.
	result: Result<DynamicImage>,
}

/// Run upscale for a batch of images.
///
/// A background prefetch thread loads images from disk through a `bounded(2)`
/// channel so disk I/O overlaps GPU inference, keeping the GPU fed.
#[allow(clippy::too_many_arguments)]
fn run_batch(
	inputs: &[PathBuf],
	output_arg: Option<&str>,
	ctx: &mut sqwale::SessionContext,
	scale: u32,
	model_filename: &str,
	tile_size: u32,
	tile_overlap: u32,
	blend: f32,
	show_progress: bool,
	cancel: &CancelToken,
) -> Result<()> {
	let output_dir = output_arg.map(Path::new);

	// Header.
	let dest_str = output_dir
		.map(|p| p.display().to_string())
		.unwrap_or_else(|| "alongside inputs".to_string());

	println!(
		"{} {}{} {} {} {}",
		SYM_BULLET.cyan().bold(),
		"Batch: ".white(),
		inputs.len().to_string().bold().bright_white(),
		"images".white(),
		SYM_ARROW.dimmed(),
		path_str(&dest_str)
	);

	let model_summary = format!(
		"{}{}{}{}{}{}{}",
		format!("{scale}x").truecolor(CLR_VALUE.0, CLR_VALUE.1, CLR_VALUE.2),
		format!(" {SYM_DOT} ").dimmed(),
		color_space_str(&ctx.model_info.color_space),
		format!(" {SYM_DOT} ").dimmed(),
		dtype_str(&ctx.model_info.input_dtype),
		format!(" {SYM_DOT} ").dimmed(),
		if ctx.model_info.tile.supported {
			"dynamic"
				.truecolor(CLR_VALUE.0, CLR_VALUE.1, CLR_VALUE.2)
				.to_string()
		} else {
			"fixed".yellow().to_string()
		},
	);
	println!(
		"{}  {} {}",
		SYM_DOT.dimmed(),
		"Model".dimmed(),
		model_summary
	);
	println!(
		"{}  {} {}  {}",
		SYM_DOT.dimmed(),
		"Loaded".dimmed(),
		model_filename.truecolor(CLR_VALUE.0, CLR_VALUE.1, CLR_VALUE.2),
		format!("via {}", ctx.provider_used.name()).dimmed()
	);
	println!();

	let multi = if show_progress {
		Some(MultiProgress::new())
	} else {
		None
	};

	// Pre-create both bars in the MultiProgress so they coexist:
	// tile bar on top, batch bar on bottom.
	let tile_pb = multi.as_ref().map(|mp| {
		let pb = mp.add(ProgressBar::new(0));
		pb.set_style(tile_bar_style());
		pb.enable_steady_tick(Duration::from_millis(SPINNER_TICK_MS));
		pb
	});

	let batch_pb = multi.as_ref().map(|mp| {
		let pb = mp.add(ProgressBar::new(inputs.len() as u64));
		pb.set_style(batch_bar_style());
		pb.enable_steady_tick(Duration::from_millis(SPINNER_TICK_MS));
		pb
	});

	// Prefetch thread: resolves output paths and loads images from disk,
	// keeping up to 2 images buffered ahead of the GPU loop.
	let (prefetch_tx, prefetch_rx) = bounded::<PrefetchedImage>(2);
	let prefetch_inputs: Vec<PathBuf> = inputs.to_vec();
	let prefetch_cancel = cancel.clone();
	// Convert to owned so the closure can be `'static`.
	let prefetch_output_dir: Option<PathBuf> = output_dir.map(Path::to_path_buf);
	let prefetch_handle = thread::spawn(move || {
		for (index, input) in prefetch_inputs.into_iter().enumerate() {
			if prefetch_cancel.is_cancelled() {
				break;
			}
			let output_result = resolve_batch_output(&input, prefetch_output_dir.as_deref(), scale);
			let item = match output_result {
				Err(e) => PrefetchedImage {
					index,
					output: PathBuf::new(),
					result: Err(e),
					input,
				},
				Ok(output) => {
					let result = imageio::load_image(&input);
					PrefetchedImage {
						index,
						input,
						output,
						result,
					}
				}
			};
			if prefetch_tx.send(item).is_err() {
				break;
			}
		}
	});

	let total = inputs.len();
	let batch_start = Instant::now();
	let mut succeeded = 0usize;
	let mut failed: Vec<(PathBuf, String)> = Vec::new();
	let mut skipped = 0usize;

	let print_line = |multi: &Option<MultiProgress>, line: String| {
		if let Some(mp) = multi {
			let _ = mp.println(&line);
		} else {
			println!("{line}");
		}
	};

	// GPU loop: drains the prefetch channel.
	for item in &prefetch_rx {
		let i = item.index;

		if cancel.is_cancelled() {
			// Drain remaining items so the prefetch thread can exit cleanly.
			drop(prefetch_rx);
			skipped = total.saturating_sub(i);
			print_line(
				&multi,
				format!(
					"\n{} {} {}",
					SYM_WARN.yellow(),
					"Interrupted".white(),
					"— finishing current image, skipping remaining".dimmed()
				),
			);
			break;
		}

		// Update batch bar.
		if let Some(ref pb) = batch_pb {
			pb.set_position(i as u64);
			update_batch_message(pb, i, total, batch_start);
		}

		// Reset tile bar for this image.
		if let Some(ref pb) = tile_pb {
			pb.reset();
			pb.set_length(0);
			pb.set_message("".to_string());
		}

		// Per-image header — printed here (GPU side) not during prefetch.
		print_line(
			&multi,
			format!(
				"{} {}{}{} {}",
				SYM_BULLET.cyan().bold(),
				"[".dimmed(),
				(i + 1).to_string().bold().bright_white(),
				format!("/{total}]").dimmed(),
				path_str(&item.input.display().to_string())
			),
		);

		let img = match item.result {
			Err(e) => {
				let msg = format!("{e:#}");
				failed.push((item.input.clone(), msg.clone()));
				print_line(
					&multi,
					format!(
						"  {} {} {}",
						SYM_CROSS.red().bold(),
						"Failed to load image".white(),
						msg.dimmed()
					),
				);
				print_line(&multi, String::new());
				if let Some(ref pb) = batch_pb {
					pb.set_position((i + 1) as u64);
					update_batch_message(pb, i + 1, total, batch_start);
				}
				continue;
			}
			Ok(img) => img,
		};

		match process_single_image(
			&img,
			&item.output,
			ctx,
			tile_size,
			tile_overlap,
			blend,
			cancel,
			tile_pb.as_ref(),
			&multi,
		) {
			Ok(()) => {
				succeeded += 1;
			}
			Err(e) => {
				let msg = format!("{e:#}");
				if msg == "Cancelled" {
					skipped = total.saturating_sub(i);
					break;
				}
				failed.push((item.input.clone(), msg.clone()));
				print_line(
					&multi,
					format!(
						"  {} {} {}",
						SYM_CROSS.red().bold(),
						"Failed".white(),
						format!(": {msg}").dimmed()
					),
				);
			}
		}

		// Update batch bar after completing this image.
		if let Some(ref pb) = batch_pb {
			pb.set_position((i + 1) as u64);
			update_batch_message(pb, i + 1, total, batch_start);
		}

		print_line(&multi, String::new());
	}

	prefetch_handle.join().expect("prefetch thread panicked");

	// Clear both bars.
	if let Some(ref pb) = tile_pb {
		pb.finish_and_clear();
	}
	if let Some(ref pb) = batch_pb {
		pb.finish_and_clear();
	}

	// Summary.
	if cancel.is_cancelled() {
		println!(
			"\n{} {}{}{}",
			SYM_CROSS.red().bold(),
			"Cancelled after ".white(),
			format!("{succeeded}/{total}").bold().bright_white(),
			"".normal()
		);
	}

	print!(
		"{} {} {}",
		SYM_CHECK.green(),
		succeeded.to_string().bold().bright_white(),
		"succeeded".white()
	);

	if !failed.is_empty() {
		print!(
			"  {}  {} {}",
			SYM_DOT.dimmed(),
			failed.len().to_string().bold().bright_white(),
			"failed".red()
		);
	}

	if skipped > 0 {
		print!(
			"  {}  {} {}",
			SYM_DOT.dimmed(),
			skipped.to_string().bold().bright_white(),
			"skipped".dimmed()
		);
	}

	println!();

	// Failed file details.
	for (path, reason) in &failed {
		println!(
			"  {} {} {}",
			SYM_CROSS.dimmed(),
			path.display().to_string().dimmed(),
			reason.dimmed()
		);
	}

	if !failed.is_empty() {
		bail!("{} image(s) failed", failed.len());
	}

	Ok(())
}

/// Update the batch progress bar message with elapsed time and ETA.
fn update_batch_message(pb: &ProgressBar, done: usize, total: usize, start: Instant) {
	let elapsed = start.elapsed();
	let eta = if done > 0 {
		let avg = elapsed / done as u32;
		let remaining = total.saturating_sub(done);
		format!("  ~{} remaining", format_duration(avg * remaining as u32)).dimmed()
	} else {
		"".dimmed()
	};
	pb.set_message(format!(
		"{}/{}  images  {}{}",
		done.to_string().bold().bright_white(),
		total,
		format_duration(elapsed).dimmed(),
		eta,
	));
}

/// Process a single already-loaded image within a batch.
///
/// Receives the decoded image rather than a path so that loading has already
/// happened on the prefetch thread, overlapping disk I/O with GPU inference.
/// The tile progress bar is owned by the caller and reused across images.
#[allow(clippy::too_many_arguments)]
fn process_single_image(
	img: &DynamicImage,
	output_path: &Path,
	ctx: &mut sqwale::SessionContext,
	tile_size: u32,
	tile_overlap: u32,
	blend: f32,
	cancel: &CancelToken,
	tile_pb: Option<&ProgressBar>,
	multi: &Option<MultiProgress>,
) -> Result<()> {
	let print_line = |line: String| {
		if let Some(mp) = multi {
			let _ = mp.println(&line);
		} else {
			println!("{line}");
		}
	};

	let (img_w, img_h) = (img.width(), img.height());

	print_line(format!(
		"  {} {} {}",
		SYM_DOT.dimmed(),
		"Input".dimmed(),
		dims_str(img_w, img_h)
	));

	let start = Instant::now();

	let pb_clone = tile_pb.cloned();
	let pb_blend_clone = tile_pb.cloned();
	let options = UpscaleOptions {
		tile_size,
		tile_overlap,
		blend,
		cancel: cancel.clone(),
		on_tile_done: Some(Box::new(move |done, total| {
			if let Some(ref pb) = pb_clone {
				pb.set_length(total as u64);
				pb.set_position(done as u64);
				let elapsed = start.elapsed();
				let per_tile = if done > 0 {
					elapsed / done as u32
				} else {
					Duration::ZERO
				};
				pb.set_message(format!(
					"{}/{} Upscaling…  {}  {}/tile",
					done.to_string().bold().bright_white(),
					total,
					format_duration(elapsed).dimmed(),
					format_duration(per_tile).dimmed(),
				));
			}
		})),
		on_blend_step: if blend > 0.0 {
			Some(Box::new(move |done, total| {
				if let Some(ref pb) = pb_blend_clone {
					pb.set_length(total as u64);
					pb.set_position(done as u64);
					pb.set_message(format!(
						"{}/{} Blending…  {}",
						done.to_string().bold().bright_white(),
						total,
						format_duration(start.elapsed()).dimmed(),
					));
				}
			}))
		} else {
			None
		},
	};

	let result = sqwale::pipeline::upscale_image(ctx, img, &options);

	// Clear the tile bar but don't remove it from MultiProgress — it's reused.
	if let Some(pb) = tile_pb {
		pb.set_position(pb.length().unwrap_or(0));
		pb.set_message("".to_string());
	}

	let output_img = result?;
	let elapsed = start.elapsed();

	let (out_w, out_h) = (output_img.width(), output_img.height());
	print_line(format!(
		"  {} {} {}",
		SYM_DOT.dimmed(),
		"Output".dimmed(),
		dims_str(out_w, out_h)
	));

	imageio::save_image(&output_img, output_path)?;

	print_line(format!(
		"  {} {}  {}",
		SYM_CHECK.green(),
		path_str(&output_path.display().to_string()),
		format_duration(elapsed).dimmed()
	));

	Ok(())
}

/// Expand input pattern to a list of image file paths.
fn expand_input_pattern(pattern: &str) -> Result<Vec<PathBuf>> {
	let path = Path::new(pattern);

	if path.is_dir() {
		let glob_pattern = format!("{}/**/*", path.display());
		return collect_image_glob(&glob_pattern);
	}

	if pattern.contains('*') || pattern.contains('?') || pattern.contains('[') {
		return collect_image_glob(pattern);
	}

	if path.exists() {
		Ok(vec![path.to_path_buf()])
	} else {
		bail!("File not found: {pattern}");
	}
}

/// Collect glob matches for image files.
fn collect_image_glob(pattern: &str) -> Result<Vec<PathBuf>> {
	let image_extensions = [
		"png", "jpg", "jpeg", "webp", "gif", "tiff", "tif", "bmp", "ico", "pnm", "qoi", "hdr",
	];

	let mut paths: Vec<PathBuf> = glob::glob(pattern)
		.with_context(|| format!("Invalid glob pattern: {pattern}"))?
		.filter_map(Result::ok)
		.filter(|p| {
			p.extension().is_some_and(|e| {
				let e = e.to_string_lossy().to_lowercase();
				image_extensions.contains(&e.as_str())
			})
		})
		.collect();
	paths.sort();
	Ok(paths)
}

/// Resolve output path for a single-file upscale.
fn resolve_output_path(input: &Path, output_arg: Option<&str>, scale: u32) -> Result<PathBuf> {
	match output_arg {
		None => Ok(imageio::default_output_path(input, scale)),
		Some(out) => {
			let out_path = Path::new(out);
			if out_path.is_dir() || out_path.extension().is_none() {
				// Treat as directory.
				let filename = imageio::default_output_path(input, scale);
				let name = filename.file_name().unwrap_or_default();
				Ok(out_path.join(name))
			} else {
				imageio::check_extension_match(input, out_path)?;
				Ok(out_path.to_path_buf())
			}
		}
	}
}

/// Resolve output path for a batch image.
fn resolve_batch_output(input: &Path, output_dir: Option<&Path>, scale: u32) -> Result<PathBuf> {
	match output_dir {
		None => Ok(imageio::default_output_path(input, scale)),
		Some(dir) => {
			let filename = imageio::default_output_path(input, scale);
			let name = filename.file_name().unwrap_or_default();
			Ok(dir.join(name))
		}
	}
}
