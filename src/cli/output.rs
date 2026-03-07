//! Output formatting: symbols, colors, tree helpers, progress, and tracing.

use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};
use std::time::Duration;
use tracing_indicatif::IndicatifLayer;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::prelude::*;

// ── Symbols ────────────────────────────────────────────────────────────────
pub const SYM_BULLET: &str = "●";
pub const SYM_CHECK: &str = "✓";
pub const SYM_CROSS: &str = "✗";
pub const SYM_WARN: &str = "⚠";
pub const SYM_DOT: &str = "·";
pub const SYM_ARROW: &str = "→";
pub const SYM_TIMES: &str = "×";
pub const SYM_ELLIPSIS: &str = "…";

// ── Tree connectors ────────────────────────────────────────────────────────
pub const TREE_BRANCH: &str = "├─";
pub const TREE_LAST: &str = "╰─";
pub const TREE_SUB: &str = "│   └─";

// ── Truecolor values ──────────────────────────────────────────────────────
pub const CLR_PATH: (u8, u8, u8) = (110, 148, 178);
pub const CLR_VALUE: (u8, u8, u8) = (152, 195, 121);

// ── Spinner ────────────────────────────────────────────────────────────────
pub const SPINNER_FRAMES: &[&str] = &["◐", "◓", "◑", "◒"];
pub const SPINNER_TICK_MS: u64 = 120;

/// Initialize tracing with indicatif integration for log/progress coordination.
pub fn init_tracing() {
	let indicatif_layer = IndicatifLayer::new();

	let filter =
		EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("sqwale=warn"));

	tracing_subscriber::registry()
		.with(filter)
		.with(tracing_subscriber::fmt::layer().with_writer(indicatif_layer.get_stderr_writer()))
		.with(indicatif_layer)
		.init();
}

/// Whether to use colored output.
///
/// Returns false if `NO_COLOR` is set or stdout is not a terminal.
pub fn should_use_color() -> bool {
	if std::env::var_os("NO_COLOR").is_some() {
		return false;
	}
	std::io::IsTerminal::is_terminal(&std::io::stdout())
}

/// Whether to show progress bars and spinners.
///
/// Returns false if stdout is not a terminal or `CI` is set.
pub fn should_show_progress() -> bool {
	if std::env::var_os("CI").is_some() {
		return false;
	}
	std::io::IsTerminal::is_terminal(&std::io::stderr())
}

/// Tile progress bar style (per-image).
pub fn tile_bar_style() -> ProgressStyle {
	ProgressStyle::default_bar()
		.tick_strings(SPINNER_FRAMES)
		.template("  {spinner:.cyan} {bar:40.cyan/238}  {msg}")
		.unwrap()
		.progress_chars("━╌")
}

/// Batch progress bar style (overall images).
pub fn batch_bar_style() -> ProgressStyle {
	ProgressStyle::default_bar()
		.tick_strings(SPINNER_FRAMES)
		.template("  {spinner:.cyan} {bar:40.cyan/238}  {msg}")
		.unwrap()
		.progress_chars("━╌")
}

/// Interpolation progress bar style with built-in ETA.
pub fn interp_bar_style() -> ProgressStyle {
	ProgressStyle::default_bar()
		.tick_strings(SPINNER_FRAMES)
		.template(
			"  {spinner:.cyan} {bar:40.cyan/238}  {pos}/{len} Interpolating… · {elapsed_precise} · ~{eta} left",
		)
		.unwrap()
		.progress_chars("━╌")
}

/// Run a closure while displaying a spinner on stderr.
pub fn with_spinner<T>(label: &str, f: impl FnOnce() -> T) -> T {
	if !should_show_progress() {
		return f();
	}

	let pb = ProgressBar::new_spinner();
	pb.set_style(
		ProgressStyle::default_spinner()
			.tick_strings(SPINNER_FRAMES)
			.template("{spinner:.cyan} {msg}")
			.unwrap(),
	);
	pb.set_message(label.to_owned());
	pb.enable_steady_tick(Duration::from_millis(SPINNER_TICK_MS));
	let result = f();
	pb.finish_and_clear();
	result
}

/// Format a tree row as a String (without printing).
///
/// Example: ` ├─ Scale      2x  via DepthToSpace`
pub fn fmt_tree_row(
	prefix: &str,
	connector: &str,
	label: &str,
	label_width: usize,
	value: &str,
) -> String {
	let padded = format!("{:width$}", label, width = label_width);
	format!(
		"{}{} {}{}",
		prefix,
		connector.dimmed(),
		padded.white(),
		value
	)
}

/// Format dimensions as "W×H" with bold bright-white numbers.
pub fn dims_str(w: u32, h: u32) -> String {
	format!(
		"{}{}{}",
		w.to_string().bold().bright_white(),
		SYM_TIMES,
		h.to_string().bold().bright_white()
	)
}

/// Format a `Duration` for human display.
pub fn format_duration(d: Duration) -> String {
	let secs = d.as_secs();
	if secs == 0 {
		format!("{:.1}s", d.as_secs_f64())
	} else if secs < 60 {
		format!("{secs}s")
	} else if secs < 3600 {
		let m = secs / 60;
		let s = secs % 60;
		format!("{m}m {s:02}s")
	} else {
		let h = secs / 3600;
		let m = (secs % 3600) / 60;
		let s = secs % 60;
		format!("{h}h {m:02}m {s:02}s")
	}
}

/// Format a color space value with styling.
pub fn color_space_str(cs: &sqwale::ColorSpace) -> String {
	cs.to_string()
		.truecolor(CLR_VALUE.0, CLR_VALUE.1, CLR_VALUE.2)
		.to_string()
}

/// Format a dtype value with styling.
pub fn dtype_str(dtype: &str) -> String {
	dtype
		.truecolor(CLR_VALUE.0, CLR_VALUE.1, CLR_VALUE.2)
		.to_string()
}

/// Format a file path with the standard path color.
pub fn path_str(p: &str) -> String {
	p.truecolor(CLR_PATH.0, CLR_PATH.1, CLR_PATH.2).to_string()
}
