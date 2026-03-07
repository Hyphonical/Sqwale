//! Frame interpolation pipeline: piped FFmpeg reader → RIFE inference → piped FFmpeg writer.
//!
//! Orchestrates the zero-disk architecture — all frame data flows through
//! in-memory buffers between two FFmpeg child processes, with RIFE inference
//! producing the interpolated frames in between.

pub mod rife;

use anyhow::{Result, bail};
use crossbeam_channel::bounded;
use ndarray::Array4;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;
use std::thread;
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
	let (mut reader_child, reader_stdout) = ffmpeg::spawn_reader(input)?;
	let (mut writer_child, writer_stdin) = ffmpeg::spawn_writer(
		input,
		output,
		info.width,
		info.height,
		&out_fps_str,
		options.crf,
	)?;

	// Bounded channels decouple FFmpeg I/O from inference.
	// Reader → Inference: up to 10 raw frames buffered.
	let (read_tx, read_rx) = bounded::<Vec<u8>>(10);
	// Inference → Writer: up to 20 output frames buffered.
	let (write_tx, write_rx) = bounded::<Vec<u8>>(20);

	// Reader thread: continuously reads raw RGB24 frames from FFmpeg stdout
	// and forwards them into the inference channel.
	let reader_handle = thread::spawn(move || -> Result<()> {
		let mut reader = BufReader::new(reader_stdout);
		loop {
			let mut frame = vec![0u8; frame_size];
			match ffmpeg::read_frame(&mut reader, &mut frame) {
				Ok(true) => {
					if read_tx.send(frame).is_err() {
						// Inference thread dropped the receiver; stop reading.
						break;
					}
				}
				Ok(false) => break, // Clean EOF.
				Err(e) => return Err(e),
			}
		}
		Ok(())
	});

	// Writer thread: receives output frames from the inference channel and
	// writes them sequentially to FFmpeg stdin.
	let writer_handle = thread::spawn(move || -> Result<()> {
		let mut writer = BufWriter::new(writer_stdin);
		for frame in write_rx {
			writer
				.write_all(&frame)
				.map_err(|e| anyhow::anyhow!("Failed to write frame to FFmpeg: {e}"))?;
		}
		writer
			.flush()
			.map_err(|e| anyhow::anyhow!("Failed to flush FFmpeg writer: {e}"))
	});

	// ── Inference (Main Thread) ────────────────────────────────────────────
	// Inference is strictly sequential to preserve frame order; only I/O is
	// decoupled into the reader/writer threads above.

	let mut frames_written: usize = 0;
	let mut frames_read: usize = 0;
	let mut inference_result: Result<()> = Ok(());

	// Receive the first frame; if none arrives the input is empty.
	if let Ok(buf_a) = read_rx.recv() {
		let mut tensor_a = rife::bytes_to_tensor(&buf_a, info.width, info.height);
		frames_read = 1;

		// Log padding info once (it's constant for all frames).
		let pad_h = (32 - (info.height % 32)) % 32;
		let pad_w = (32 - (info.width % 32)) % 32;
		debug!(
			"RIFE padding: {}×{} → {}×{} (pad_h={}, pad_w={})",
			info.width,
			info.height,
			info.width + pad_w,
			info.height + pad_h,
			pad_h,
			pad_w
		);

		// Pass the first frame through to the writer unchanged.
		if write_tx.send(buf_a).is_ok() {
			frames_written += 1;
			report_progress(&options.on_progress, frames_written, total_output_frames);
		} else {
			inference_result = Err(anyhow::anyhow!("Writer channel closed unexpectedly"));
		}

		// Main inference loop: receive frame B, interpolate A→B, send midframes + B.
		'outer: while inference_result.is_ok() {
			if options.cancel.is_cancelled() {
				info!("Interpolation cancelled — finalising output");
				break;
			}

			let buf_b = match read_rx.recv() {
				Ok(frame) => frame,
				Err(_) => break, // EOF: no more input frames.
			};
			frames_read += 1;

			let tensor_b = rife::bytes_to_tensor(&buf_b, info.width, info.height);

			let mids = match generate_midframes(
				rife,
				&tensor_a,
				&tensor_b,
				options.multiplier,
				options.ensemble,
			) {
				Ok(m) => m,
				Err(e) => {
					inference_result = Err(e);
					break;
				}
			};

			for mid in &mids {
				let bytes = rife::tensor_to_bytes(mid, info.width, info.height);
				if write_tx.send(bytes).is_err() {
					inference_result = Err(anyhow::anyhow!("Writer channel closed unexpectedly"));
					break 'outer;
				}
				frames_written += 1;
				report_progress(&options.on_progress, frames_written, total_output_frames);
			}

			// Send frame B and slide the window.
			if write_tx.send(buf_b).is_err() {
				inference_result = Err(anyhow::anyhow!("Writer channel closed unexpectedly"));
				break;
			}
			frames_written += 1;
			report_progress(&options.on_progress, frames_written, total_output_frames);

			tensor_a = tensor_b;
		}
	}

	// Drop channel endpoints: signals the reader to stop sending and the
	// writer to flush once its queue is drained.
	drop(write_tx);
	drop(read_rx);

	// Join threads and collect any errors they encountered.
	let reader_result = reader_handle.join().expect("reader thread panicked");
	let writer_result = writer_handle.join().expect("writer thread panicked");

	// Wait for FFmpeg processes.
	let reader_status = reader_child.wait().ok();
	let writer_wait_result = writer_child
		.wait()
		.map_err(|e| anyhow::anyhow!("FFmpeg writer did not exit cleanly: {e}"));

	if frames_read == 0 {
		bail!("Video contains no frames");
	}

	debug!(
		"Reader exit: {:?}, Writer exit: {:?}",
		reader_status.map(|s| s.code()),
		writer_wait_result.as_ref().ok().and_then(|s| s.code()),
	);

	info!(
		"Interpolation complete: {} input frames → {} output frames",
		frames_read, frames_written
	);

	// Propagate errors in priority order: inference > reader > writer > FFmpeg exit.
	inference_result?;
	reader_result?;
	writer_result?;

	let writer_status = writer_wait_result?;
	if !writer_status.success() {
		bail!(
			"FFmpeg writer exited with status {}",
			writer_status.code().unwrap_or(-1)
		);
	}

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
