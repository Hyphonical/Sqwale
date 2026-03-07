//! Frame interpolation pipeline: piped FFmpeg reader → RIFE inference → piped FFmpeg writer.
//!
//! Orchestrates the zero-disk architecture — all frame data flows through
//! in-memory buffers between two FFmpeg child processes, with RIFE inference
//! producing the interpolated frames in between.

pub mod rife;

use anyhow::{Result, bail};
use ndarray::Array4;
use std::io::Write;
use std::path::Path;
use tracing::{debug, info};

use crate::ffmpeg;
use crate::pipeline::CancelToken;

pub use rife::RifeSession;

/// Options controlling the interpolation pipeline.
pub struct InterpolateOptions {
	/// Frame rate multiplier (must be a power of two ≥ 2).
	pub multiplier: u32,

	/// Use ensemble (horizontal-flip averaging) for higher quality.
	pub ensemble: bool,

	/// x264 CRF value for the output encoder.
	pub crf: u32,

	/// Cooperative cancellation token.
	pub cancel: CancelToken,

	/// Progress callback: `(frames_written, total_output_frames)`.
	pub on_progress: Option<Box<dyn Fn(usize, usize) + Send + Sync>>,
}

/// Result of a completed interpolation run.
pub struct InterpolateResult {
	/// Total number of output frames written.
	pub frames_written: usize,
	/// Total number of input frames consumed.
	pub frames_read: usize,
}

/// Run the full interpolation pipeline.
///
/// 1. Probes the input video for metadata.
/// 2. Spawns piped FFmpeg reader and writer processes.
/// 3. Reads raw frames, runs recursive RIFE inference, and writes the result.
pub fn run(
	input: &Path,
	output: &Path,
	rife: &mut RifeSession,
	options: &InterpolateOptions,
) -> Result<InterpolateResult> {
	let info = ffmpeg::probe(input)?;

	let out_fps_str = ffmpeg::multiply_fps(&info.fps_str, options.multiplier)?;

	info!(
		"Interpolation: {}×{} @ {} fps → {} fps ({}× multiplier)",
		info.width, info.height, info.fps_str, out_fps_str, options.multiplier
	);

	let frame_size = info.width * info.height * 3;
	let total_output_frames = if info.frame_count > 0 {
		(info.frame_count - 1) * options.multiplier as usize + 1
	} else {
		0
	};

	debug!(
		"Frame size: {} bytes, estimated output frames: {}",
		frame_size, total_output_frames
	);

	// Spawn FFmpeg processes.
	let (mut reader_child, mut reader_stdout) = ffmpeg::spawn_reader(input)?;
	let (mut writer_child, mut writer_stdin) = ffmpeg::spawn_writer(
		input,
		output,
		info.width,
		info.height,
		&out_fps_str,
		options.crf,
	)?;

	// Pre-allocate frame buffers.
	let mut buf_a = vec![0u8; frame_size];
	let mut buf_b = vec![0u8; frame_size];

	// Read the first frame.
	if !ffmpeg::read_frame(&mut reader_stdout, &mut buf_a)? {
		// Close pipes and wait for child processes.
		drop(writer_stdin);
		let _ = reader_child.wait();
		let _ = writer_child.wait();
		bail!("Video contains no frames");
	}

	let mut tensor_a = rife::bytes_to_tensor(&buf_a, info.width, info.height);
	let mut frames_written: usize = 0;
	let mut frames_read: usize = 1;

	// Write the first frame as-is.
	write_bytes(&mut writer_stdin, &buf_a)?;
	frames_written += 1;
	report_progress(&options.on_progress, frames_written, total_output_frames);

	// Main loop: read subsequent frames, interpolate, write.
	loop {
		if options.cancel.is_cancelled() {
			info!("Interpolation cancelled — finalising output");
			break;
		}

		if !ffmpeg::read_frame(&mut reader_stdout, &mut buf_b)? {
			break; // No more input frames.
		}
		frames_read += 1;

		let tensor_b = rife::bytes_to_tensor(&buf_b, info.width, info.height);

		// Generate intermediate frames recursively for the given multiplier.
		let mids = generate_midframes(
			rife,
			&tensor_a,
			&tensor_b,
			options.multiplier,
			options.ensemble,
		)?;

		// Write all intermediate frames.
		for mid in &mids {
			let bytes = rife::tensor_to_bytes(mid, info.width, info.height);
			write_bytes(&mut writer_stdin, &bytes)?;
			frames_written += 1;
			report_progress(&options.on_progress, frames_written, total_output_frames);
		}

		// Write frame B.
		write_bytes(&mut writer_stdin, &buf_b)?;
		frames_written += 1;
		report_progress(&options.on_progress, frames_written, total_output_frames);

		// Slide window: B becomes the new A.
		tensor_a = tensor_b;
		std::mem::swap(&mut buf_a, &mut buf_b);
	}

	// Close writer stdin to signal EOF, then wait for FFmpeg to finalise.
	drop(writer_stdin);
	// Drop reader stdout so the reader process can exit cleanly.
	drop(reader_stdout);

	let reader_status = reader_child.wait().ok();
	let writer_status = writer_child
		.wait()
		.map_err(|e| anyhow::anyhow!("FFmpeg writer did not exit cleanly: {e}"))?;

	if !writer_status.success() {
		bail!(
			"FFmpeg writer exited with status {}",
			writer_status.code().unwrap_or(-1)
		);
	}

	debug!(
		"Reader exit: {:?}, Writer exit: {:?}",
		reader_status.map(|s| s.code()),
		writer_status.code()
	);

	info!(
		"Interpolation complete: {} input frames → {} output frames",
		frames_read, frames_written
	);

	Ok(InterpolateResult {
		frames_written,
		frames_read,
	})
}

/// Recursively generate intermediate frames between `f0` and `f1`.
///
/// For a multiplier of `N`, this produces `N - 1` evenly-spaced intermediate
/// frames by recursively halving the time interval.
///
/// Example for 4×: generates frames at t = 0.25, 0.5, 0.75.
fn generate_midframes(
	rife: &mut RifeSession,
	f0: &Array4<f32>,
	f1: &Array4<f32>,
	multiplier: u32,
	ensemble: bool,
) -> Result<Vec<Array4<f32>>> {
	let count = (multiplier - 1) as usize;
	let mut results = Vec::with_capacity(count);

	// Generate timesteps: 1/N, 2/N, ..., (N-1)/N.
	// Use recursive binary splitting for temporal consistency:
	// e.g. for 4×: first compute t=0.5 (between f0 and f1),
	// then t=0.25 (between f0 and mid), then t=0.75 (between mid and f1).
	recursive_interp(rife, f0, f1, 0.0, 1.0, multiplier, ensemble, &mut results)?;

	Ok(results)
}

/// Recursively subdivide the `[t0, t1]` interval and interpolate.
///
/// Frames are pushed in chronological order (ascending timestep).
#[allow(clippy::too_many_arguments)]
fn recursive_interp(
	rife: &mut RifeSession,
	f0: &Array4<f32>,
	f1: &Array4<f32>,
	t0: f32,
	t1: f32,
	divisions: u32,
	ensemble: bool,
	out: &mut Vec<Array4<f32>>,
) -> Result<()> {
	if divisions <= 1 {
		return Ok(());
	}

	let mid_t = (t0 + t1) / 2.0;

	// Map mid_t back to a relative timestep in [0, 1] between the two frames.
	// Since we always pass the two bounding frames, the timestep for RIFE is 0.5.
	let mid = rife.interpolate(f0, f1, 0.5, ensemble)?;

	let left_divs = divisions / 2;
	let right_divs = divisions - left_divs;

	// Left half: frames between f0 and mid.
	recursive_interp(rife, f0, &mid, t0, mid_t, left_divs, ensemble, out)?;

	// The midpoint frame itself.
	out.push(mid.clone());

	// Right half: frames between mid and f1.
	recursive_interp(rife, &mid, f1, mid_t, t1, right_divs, ensemble, out)?;

	Ok(())
}

// ── Helpers ────────────────────────────────────────────────────────────────

/// Write a complete byte buffer to the writer, handling partial writes.
fn write_bytes(writer: &mut impl Write, data: &[u8]) -> Result<()> {
	writer
		.write_all(data)
		.map_err(|e| anyhow::anyhow!("Failed to write frame to FFmpeg: {e}"))
}

/// Call the progress callback if present.
fn report_progress(
	cb: &Option<Box<dyn Fn(usize, usize) + Send + Sync>>,
	done: usize,
	total: usize,
) {
	if let Some(f) = cb {
		f(done, total);
	}
}
