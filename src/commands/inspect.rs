//! Inspect command implementation.
//!
//! Matches the visual style from `sqwale_style.txt`:
//!
//! ```text
//! ● 2x-Model_fp16.onnx
//!  ├─ Scale      2x  via DepthToSpace (PixelShuffle)
//!  ├─ Color      RGB  in:3 → out:3
//!  ├─ Precision  float16 → float16
//!  ├─ Opset      17
//!  ├─ Tiling     supported  dynamic spatial dims
//!  │   └─ Alignment  divisible by 16
//!  ╰─ Ops        866 total nodes
//!      ├─ 180  Constant
//!      ╰─      … 10 more op types
//! ```

use anyhow::{Context, Result};
use colored::Colorize;
use glob::glob;
use std::path::{Path, PathBuf};
use treelog::{render_to_string, Tree};

use sqwale::inspect::{inspect_model, ModelInfo};

/// Run the inspect command on a path or glob pattern.
pub fn run(pattern: &str, verbose: bool, quiet: bool) -> Result<()> {
	let paths = collect_paths(pattern)?;

	if paths.is_empty() {
		anyhow::bail!("No ONNX models found matching '{pattern}'");
	}

	if !quiet {
		eprintln!("{}", format!("── inspect ──{:─<44}", "").dimmed());
		eprintln!();
	}

	let mut ok = 0usize;
	let mut fail = 0usize;

	for path in &paths {
		match inspect_model(path) {
			Ok(info) => {
				ok += 1;
				if !quiet {
					print_model_info(path, &info, verbose);
				}
			}
			Err(e) => {
				fail += 1;
				eprintln!(
					"{} {}",
					"✗".red().bold(),
					path.display().to_string().bright_white()
				);
				eprintln!("  {}", format!("{e:#}").dimmed());
			}
		}
	}

	if !quiet && paths.len() > 1 {
		eprintln!();
		if ok > 0 {
			eprint!(
				"{} {} inspected",
				"✓".green(),
				ok.to_string().bright_white().bold()
			);
		}
		if fail > 0 {
			eprint!(
				"  {}  {} {} failed",
				"·".dimmed(),
				"✗".red(),
				fail.to_string().bright_white().bold()
			);
		}
		eprintln!();
	}

	if fail > 0 {
		anyhow::bail!("{fail} model(s) failed inspection");
	}

	Ok(())
}

// ── Path Collection ────────────────────────────────────────────────────────

fn collect_paths(pattern: &str) -> Result<Vec<PathBuf>> {
	let is_glob = pattern.contains('*') || pattern.contains('?') || pattern.contains('[');

	if is_glob {
		let paths: Vec<_> = glob(pattern)
			.with_context(|| format!("Invalid glob pattern '{pattern}'"))?
			.filter_map(|e| e.ok())
			.filter(|p| p.extension().is_some_and(|e| e == "onnx"))
			.collect();
		Ok(paths)
	} else {
		let p = Path::new(pattern);
		if p.exists() {
			Ok(vec![p.to_path_buf()])
		} else {
			anyhow::bail!("File not found: '{pattern}'");
		}
	}
}

// ── Pretty Printing ───────────────────────────────────────────────────────

fn print_model_info(path: &Path, info: &ModelInfo, verbose: bool) {
	let filename = path
		.file_name()
		.and_then(|n| n.to_str())
		.unwrap_or("unknown");

	// Header
	eprintln!("{} {}", "●".cyan().bold(), filename.bright_white().bold(),);

	// Build tree children
	let mut children = Vec::new();

	// Scale
	children.push(Tree::Node(
		format!(
			"{:<11}{}",
			"Scale".white(),
			format!(
				"{}x  {}",
				info.scale.to_string().bright_white().bold(),
				format!("via {}", info.scale_source).dimmed()
			)
		),
		vec![],
	));

	// Color
	children.push(Tree::Node(
		format!(
			"{:<11}{}  {}",
			"Color".white(),
			info.color_space.to_string().cyan(),
			format!(
				"in:{} → out:{}",
				info.input_channels.to_string().bright_white().bold(),
				info.output_channels.to_string().bright_white().bold()
			)
			.dimmed()
		),
		vec![],
	));

	// Precision
	children.push(Tree::Node(
		format!(
			"{:<11}{} {} {}",
			"Precision".white(),
			format_dtype(&info.input_dtype),
			"→".dimmed(),
			format_dtype(&info.output_dtype),
		),
		vec![],
	));

	// Opset
	children.push(Tree::Node(
		format!(
			"{:<11}{}",
			"Opset".white(),
			info.opset.to_string().bright_white().bold(),
		),
		vec![],
	));

	// Tiling
	let tiling_label = if info.tile.supported {
		format!(
			"{}  {}",
			"supported".green(),
			"dynamic spatial dims".dimmed()
		)
	} else {
		let dims = info
			.tile
			.fixed_size
			.map(|(h, w)| format!("{h}×{w}"))
			.unwrap_or_default();
		format!(
			"{}{}",
			"fixed".yellow(),
			if dims.is_empty() {
				String::new()
			} else {
				format!("  {dims}")
			}
		)
	};

	let tiling_children = if let Some(align) = info.tile.alignment {
		vec![Tree::Leaf(vec![format!(
			"{}  {}",
			"Alignment".dimmed(),
			format!("divisible by {}", align.to_string().bright_white().bold()).dimmed()
		)])]
	} else {
		vec![]
	};

	children.push(Tree::Node(
		format!("{:<11}{}", "Tiling".white(), tiling_label),
		tiling_children,
	));

	// Ops (only in verbose mode or always show top-level count)
	let total_ops: usize = info.op_fingerprint.iter().map(|(_, c)| c).sum();
	let max_display = if verbose { 20 } else { 8 };

	let mut op_children: Vec<Tree> = info
		.op_fingerprint
		.iter()
		.take(max_display)
		.map(|(name, count)| {
			Tree::Leaf(vec![format!(
				"{:>4}  {}",
				count.to_string().bright_white().bold(),
				name.dimmed()
			)])
		})
		.collect();

	let remaining = info.op_fingerprint.len().saturating_sub(max_display);
	if remaining > 0 {
		op_children.push(Tree::Leaf(vec![format!(
			"      {}",
			format!("… {remaining} more op types").dimmed()
		)]));
	}

	children.push(Tree::Node(
		format!(
			"{:<11}{}  {}",
			"Ops".white(),
			total_ops.to_string().bright_white().bold(),
			"total nodes".dimmed()
		),
		op_children,
	));

	// Render tree — treelog handles the tree characters (├─, ╰─, │, etc.)
	let tree = Tree::Node(String::new(), children);
	let rendered = render_to_string(&tree);

	// Print each line with leading space for proper indentation
	for line in rendered.lines() {
		if !line.trim().is_empty() {
			eprintln!("{line}");
		}
	}
	eprintln!();
}

fn format_dtype(dtype: &str) -> colored::ColoredString {
	match dtype {
		"float32" => dtype.green(),
		"float16" => dtype.cyan(),
		"int8" | "uint8" => dtype.blue(),
		"int32" | "int64" => dtype.bright_blue(),
		_ => dtype.white(),
	}
}
