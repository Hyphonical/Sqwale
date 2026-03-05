//! Inspect command implementation.

use anyhow::{Context, Result};
use colored::Colorize;
use glob::glob;
use std::path::{Path, PathBuf};
use treelog::{render_to_string, Tree};

use sqwale::inspect::{inspect_model, ModelInfo};

/// Run the inspect command on a path or glob pattern.
pub fn run(pattern: &str, _verbose: bool, quiet: bool) -> Result<()> {
	let paths = collect_paths(pattern)?;

	if paths.is_empty() {
		anyhow::bail!(
			"{}\n  {} Try: sqwale inspect path/to/models/*.onnx",
			"No ONNX models found matching pattern".red().bold(),
			"Hint:".yellow()
		);
	}

	let mut results: Vec<(PathBuf, Result<ModelInfo>)> = Vec::new();
	let mut success_count = 0;
	let mut error_count = 0;

	for path in &paths {
		let result = inspect_model(path);
		if result.is_ok() {
			success_count += 1;
		} else {
			error_count += 1;
		}
		results.push((path.clone(), result));
	}

	for (path, result) in &results {
		match result {
			Ok(info) => {
				if !quiet {
					print_model_info(path, info);
				}
			}
			Err(e) => {
				eprintln!(
					"{} {}",
					"✗".red().bold(),
					path.display().to_string().bright_white()
				);
				eprintln!("  {}: {}", "Error".red(), e);
			}
		}
	}

	if error_count > 0 && !quiet {
		eprintln!();
		eprintln!(
			"{} {} {} model(s) inspected, {} {} failed",
			"Summary:".bright_white().bold(),
			success_count.to_string().green(),
			"✓".green(),
			error_count.to_string().red(),
			"✗".red()
		);
	}

	if error_count > 0 {
		anyhow::bail!("{} model(s) failed inspection", error_count);
	}

	Ok(())
}

/// Collect all .onnx file paths matching the pattern.
fn collect_paths(pattern: &str) -> Result<Vec<PathBuf>> {
	let is_glob = pattern.contains('*') || pattern.contains('?') || pattern.contains('[');

	if is_glob {
		let paths: Vec<_> = glob(pattern)
			.with_context(|| {
				format!(
					"{} '{}'\n  {} Ensure the glob pattern is valid",
					"Invalid glob pattern:".red().bold(),
					pattern.bright_white(),
					"Hint:".yellow()
				)
			})?
			.filter_map(|e| e.ok())
			.filter(|p| p.extension().map(|e| e == "onnx").unwrap_or(false))
			.collect();
		Ok(paths)
	} else {
		let p = Path::new(pattern);
		if p.exists() {
			Ok(vec![p.to_path_buf()])
		} else {
			anyhow::bail!(
				"{} '{}'\n  {} Check the file path and ensure it exists",
				"File not found:".red().bold(),
				pattern.bright_white(),
				"Hint:".yellow()
			);
		}
	}
}

/// Print model info using treelog with proper formatting and colors.
fn print_model_info(path: &Path, info: &ModelInfo) {
	let filename = path
		.file_name()
		.and_then(|n| n.to_str())
		.unwrap_or("unknown")
		.cyan()
		.bold();

	let mut children = Vec::new();

	children.push(Tree::Node(
		"Scale".bright_white().to_string(),
		vec![Tree::Leaf(vec![format!(
			"{}x ({})",
			info.scale.to_string().green().to_string(),
			info.scale_source.to_string().dimmed().to_string()
		)])],
	));

	children.push(Tree::Node(
		"Color Space".bright_white().to_string(),
		vec![Tree::Leaf(vec![info
			.color_space
			.to_string()
			.yellow()
			.to_string()])],
	));

	children.push(Tree::Node(
		"Channels".bright_white().to_string(),
		vec![Tree::Leaf(vec![format!(
			"Input: {}, Output: {}",
			info.input_channels.to_string().bright_cyan().to_string(),
			info.output_channels.to_string().bright_cyan().to_string()
		)])],
	));

	children.push(Tree::Node(
		"Data Types".bright_white().to_string(),
		vec![Tree::Leaf(vec![format!(
			"Input: {}, Output: {}",
			format_dtype(&info.input_dtype),
			format_dtype(&info.output_dtype)
		)])],
	));

	let mut tiling_children = Vec::new();
	tiling_children.push(Tree::Leaf(vec![format!(
		"Supported: {}",
		if info.tile.supported {
			"yes".green().to_string()
		} else {
			"no (fixed size)".yellow().to_string()
		}
	)]));

	if let Some((h, w)) = info.tile.fixed_size {
		tiling_children.push(Tree::Leaf(vec![format!(
			"Fixed Size: {}×{}",
			h.to_string().magenta().to_string(),
			w.to_string().magenta().to_string()
		)]));
	}

	if let Some(align) = info.tile.alignment {
		tiling_children.push(Tree::Leaf(vec![format!(
			"Alignment: {} pixels",
			align.to_string().magenta().to_string()
		)]));
	}

	children.push(Tree::Node(
		"Tiling".bright_white().to_string(),
		tiling_children,
	));

	children.push(Tree::Node(
		"ONNX Metadata".bright_white().to_string(),
		vec![Tree::Leaf(vec![format!(
			"Opset: {}",
			info.opset.to_string().blue().to_string()
		)])],
	));

	let tree = Tree::Node(filename.to_string(), children);
	println!("{}", render_to_string(&tree));
}

fn format_dtype(dtype: &str) -> colored::ColoredString {
	match dtype {
		"float32" => dtype.green(),
		"float16" => dtype.bright_green(),
		"int8" | "uint8" => dtype.blue(),
		"int32" | "int64" => dtype.bright_blue(),
		_ => dtype.white(),
	}
}
