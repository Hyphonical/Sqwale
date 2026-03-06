//! `sqwale inspect` command handler.

use anyhow::{Context, Result, bail};
use colored::Colorize;
use indicatif::ProgressBar;
use std::path::{Path, PathBuf};
use std::time::Duration;

use sqwale::ModelInfo;
use sqwale::inspect::inspect_model;

use super::output::*;

/// Run the inspect command for a given pattern (file, glob, or directory).
pub fn run(pattern: &str) -> Result<()> {
	let paths = expand_pattern(pattern)?;

	if paths.is_empty() {
		bail!(
			"No .onnx files found matching '{pattern}'. \
			 Use a file path, glob pattern (e.g. \"models/*.onnx\"), or directory."
		);
	}

	let is_batch = paths.len() > 1;
	let show_progress = is_batch && should_show_progress();

	let pb = if show_progress {
		let pb = ProgressBar::new(paths.len() as u64);
		pb.set_style(batch_bar_style());
		pb.enable_steady_tick(Duration::from_millis(SPINNER_TICK_MS));
		Some(pb)
	} else {
		None
	};

	let mut failed = 0usize;

	for (i, path) in paths.iter().enumerate() {
		if let Some(ref pb) = pb {
			pb.set_message(format!(
				"{}/{} models",
				(i + 1).to_string().bold().bright_white(),
				paths.len()
			));
			pb.set_position(i as u64);
		}

		let is_last = i + 1 == paths.len();

		match inspect_model(path) {
			Ok(info) => {
				if let Some(ref pb) = pb {
					print_inspect_result(path, &info, |s| pb.println(s));
					if !is_last {
						pb.println("");
					}
				} else {
					print_inspect_result(path, &info, |s| println!("{s}"));
					if is_batch && !is_last {
						println!();
					}
				}
			}
			Err(e) => {
				failed += 1;
				let err_line = format!(
					"{} {} {}",
					SYM_CROSS.red().bold(),
					path.display().to_string().bold().bright_white(),
					format!("{e:#}").dimmed()
				);
				if let Some(ref pb) = pb {
					pb.println(&err_line);
					if !is_last {
						pb.println("");
					}
				} else {
					println!("{err_line}");
					if is_batch && !is_last {
						println!();
					}
				}
			}
		}
	}

	if let Some(pb) = pb {
		pb.finish_and_clear();
	}

	if failed > 0 {
		bail!("{failed} model(s) failed to inspect");
	}

	Ok(())
}

/// Expand a pattern to a list of .onnx file paths.
fn expand_pattern(pattern: &str) -> Result<Vec<PathBuf>> {
	let path = Path::new(pattern);

	// Directory: find all .onnx files recursively.
	if path.is_dir() {
		let glob_pattern = format!("{}/**/*.onnx", path.display());
		return collect_glob(&glob_pattern);
	}

	// If it looks like a glob pattern, expand it.
	if pattern.contains('*') || pattern.contains('?') || pattern.contains('[') {
		return collect_glob(pattern);
	}

	// Plain file path.
	if path.exists() {
		Ok(vec![path.to_path_buf()])
	} else {
		bail!("File not found: {pattern}");
	}
}

/// Collect glob matches into a sorted list.
fn collect_glob(pattern: &str) -> Result<Vec<PathBuf>> {
	let mut paths: Vec<PathBuf> = glob::glob(pattern)
		.with_context(|| format!("Invalid glob pattern: {pattern}"))?
		.filter_map(Result::ok)
		.filter(|p| {
			p.extension()
				.is_some_and(|e| e.eq_ignore_ascii_case("onnx"))
		})
		.collect();
	paths.sort();
	Ok(paths)
}

/// Print a styled inspection result for a single model.
///
/// `println_fn` is called once per output line so callers can route through a
/// progress bar (`pb.println`) or plain stdout (`println!`) without suspending.
fn print_inspect_result(path: &Path, info: &ModelInfo, mut println_fn: impl FnMut(&str)) {
	let filename = path.file_name().unwrap_or_default().to_string_lossy();

	// Header.
	println_fn(&format!(
		"{} {}",
		SYM_BULLET.cyan().bold(),
		filename.bold().bright_white()
	));

	let label_w = 11;
	let prefix = " ";

	// Scale.
	let scale_val = format!(
		"{}{}",
		info.scale.to_string().bold().bright_white(),
		"x".bold().bright_white()
	);
	let scale_source = format!("  {}", format!("via {}", info.scale_source).dimmed());
	println_fn(&fmt_tree_row(
		prefix,
		TREE_BRANCH,
		"Scale",
		label_w,
		&format!("{scale_val}{scale_source}"),
	));

	// Color.
	let cs = color_space_str(&info.color_space);
	let channels = format!(
		"  {}{}{}{}{}{}",
		"in:".dimmed(),
		info.input_channels.to_string().bold().bright_white(),
		" ".dimmed(),
		SYM_ARROW.dimmed(),
		" out:".dimmed(),
		info.output_channels.to_string().bold().bright_white(),
	);
	println_fn(&fmt_tree_row(
		prefix,
		TREE_BRANCH,
		"Color",
		label_w,
		&format!("{cs}{channels}"),
	));

	// Precision.
	let prec = format!(
		"{} {} {}",
		dtype_str(&info.input_dtype),
		SYM_ARROW.dimmed(),
		dtype_str(&info.output_dtype),
	);
	println_fn(&fmt_tree_row(
		prefix,
		TREE_BRANCH,
		"Precision",
		label_w,
		&prec,
	));

	// Opset.
	println_fn(&fmt_tree_row(
		prefix,
		TREE_BRANCH,
		"Opset",
		label_w,
		&info.opset.to_string().bold().bright_white().to_string(),
	));

	// Tiling.
	let tiling_str = if info.tile.supported {
		format!(
			"{}  {}",
			"supported".green(),
			"dynamic spatial dims".dimmed()
		)
	} else if let Some((h, w)) = info.tile.fixed_size {
		format!(
			"{}  {}",
			"fixed size".yellow(),
			format!("{h}{SYM_TIMES}{w}").dimmed()
		)
	} else {
		"unknown".dimmed().to_string()
	};

	// Determine if there's an alignment sub-row.
	let has_alignment = info.tile.alignment.is_some();
	let tiling_connector = if has_alignment || !info.op_fingerprint.is_empty() {
		TREE_BRANCH
	} else {
		TREE_LAST
	};
	println_fn(&fmt_tree_row(
		prefix,
		tiling_connector,
		"Tiling",
		label_w,
		&tiling_str,
	));

	// Alignment sub-row.
	if let Some(align) = info.tile.alignment {
		println_fn(&format!(
			" {} {} {} {}",
			TREE_SUB.dimmed(),
			"Alignment".dimmed(),
			"divisible by".dimmed(),
			align.to_string().bold().bright_white()
		));
	}

	// Ops.
	let total_nodes: usize = info.op_fingerprint.iter().map(|(_, c)| c).sum();
	let ops_val = format!(
		"{} {}",
		total_nodes.to_string().bold().bright_white(),
		"total nodes".dimmed()
	);
	println_fn(&fmt_tree_row(prefix, TREE_LAST, "Ops", label_w, &ops_val));

	// Op list.
	let max_ops = sqwale::config::INSPECT_MAX_OPS_SHOWN;
	let show_count = info.op_fingerprint.len().min(max_ops);
	let max_count_width = info
		.op_fingerprint
		.iter()
		.take(show_count)
		.map(|(_, c)| c.to_string().len())
		.max()
		.unwrap_or(1);

	for (i, (op, count)) in info.op_fingerprint.iter().take(show_count).enumerate() {
		let connector = if i + 1 < show_count || info.op_fingerprint.len() > max_ops {
			TREE_BRANCH
		} else {
			TREE_LAST
		};
		println_fn(&format!(
			"     {} {:>width$}  {}",
			connector.dimmed(),
			count.to_string().bold().bright_white(),
			op.dimmed(),
			width = max_count_width,
		));
	}

	if info.op_fingerprint.len() > max_ops {
		let remaining = info.op_fingerprint.len() - max_ops;
		println_fn(&format!(
			"     {}       {}",
			TREE_LAST.dimmed(),
			format!("{SYM_ELLIPSIS} {remaining} more op types").dimmed()
		));
	}
}
