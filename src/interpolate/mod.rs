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
use std::time::Instant;
use tracing::{debug, info, trace};

use crate::ffmpeg;
use crate::ffmpeg::ContainerFormat;
use crate::pipeline::CancelToken;

pub use rife::RifeSession;

/// Task sent from the inference thread to the FFmpeg writer thread.
///
/// Keeping this as an enum lets the writer thread perform `tensor_to_bytes`
/// conversion concurrently with the next GPU inference call.
enum WriteTask {
	/// Raw RGB24 bytes — passed through without conversion (input frame).
	Raw(Vec<u8>),
	/// Interpolated f32 tensor — the writer thread converts it to bytes.
	Tensor(Array4<f32>),
}

/// Options controlling the interpolation pipeline.
pub struct InterpolateOptions {
	/// Frame rate multiplier (must be a power of two ≥ 2).
	pub multiplier: u32,

	/// Use ensemble (horizontal-flip averaging) for higher quality.
	pub ensemble: bool,

	/// x264 CRF value for the output encoder.
	pub crf: u32,

	/// Output container format.
	pub container: ContainerFormat,

	/// When `Some(threshold)`, frames whose pixel difference exceeds the
	/// threshold are treated as scene cuts: the last pre-cut frame is
	/// duplicated instead of running RIFE inference.
	///
	/// The threshold is a value in `[0.0, 1.0]` using the same mean-absolute-
	/// difference formula as FFmpeg's `scdet` filter. `0.1` is a sensible
	/// default for hard scene cuts.
	pub scene_detect_threshold: Option<f64>,

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
		output,
		info.width,
		info.height,
		&out_fps_str,
		options.crf,
		options.container,
	)?;

	// Three-stage pipeline decouples I/O, CPU conversion, and GPU inference:
	//
	//  FFmpeg reader ──[read_rx]──► Prep thread (bytes→tensor)
	//                ──[prep_rx]──► Main thread (GPU inference)
	//                ──[write_tx]──► Writer thread (tensor→bytes + FFmpeg stdin)
	//
	// Channel sizing: read buffer is generous so FFmpeg is never stalled;
	// prep buffer is small because each entry holds a full tensor (~25 MB);
	// write buffer matches a few frame-pairs worth of output.
	let (read_tx, read_rx) = bounded::<Vec<u8>>(8);
	let (prep_tx, prep_rx) = bounded::<(Vec<u8>, Array4<f32>)>(4);
	let (write_tx, write_rx) = bounded::<WriteTask>(16);

	// Reader thread: continuously reads raw RGB24 frames from FFmpeg stdout
	// and forwards them into the prep channel.
	let reader_handle = thread::spawn(move || -> Result<()> {
		let mut reader = BufReader::new(reader_stdout);
		loop {
			let mut frame = vec![0u8; frame_size];
			match ffmpeg::read_frame(&mut reader, &mut frame) {
				Ok(true) => {
					if read_tx.send(frame).is_err() {
						// Prep thread dropped the receiver; stop reading.
						break;
					}
				}
				Ok(false) => break, // Clean EOF.
				Err(e) => return Err(e),
			}
		}
		Ok(())
	});

	// Prep thread: converts raw bytes to f32 tensors so that tensor conversion
	// overlaps with GPU inference on the main thread.
	let (prep_w, prep_h) = (info.width, info.height);
	let prep_handle = thread::spawn(move || {
		for buf in read_rx {
			let tensor = rife::bytes_to_tensor(&buf, prep_w, prep_h);
			if prep_tx.send((buf, tensor)).is_err() {
				break;
			}
		}
	});

	// Writer thread: converts interpolated tensors to bytes and writes them to
	// FFmpeg stdin. Running tensor_to_bytes here overlaps post-processing with
	// the next GPU inference call on the main thread.
	let writer_handle = thread::spawn(move || -> Result<()> {
		let mut writer = BufWriter::new(writer_stdin);
		for task in write_rx {
			let bytes = match task {
				WriteTask::Raw(b) => b,
				WriteTask::Tensor(t) => rife::tensor_to_bytes(&t),
			};
			writer
				.write_all(&bytes)
				.map_err(|e| anyhow::anyhow!("Failed to write frame to FFmpeg: {e}"))?;
		}
		writer
			.flush()
			.map_err(|e| anyhow::anyhow!("Failed to flush FFmpeg writer: {e}"))
	});

	// ── Inference (Main Thread) ────────────────────────────────────────────
	// Inference is strictly sequential to preserve frame order.
	// CPU work (bytes↔tensor conversion) is overlapped by the prep and writer
	// threads so the GPU is kept as busy as possible.

	let mut frames_written: usize = 0;
	let mut frames_read: usize = 0;
	let mut inference_result: Result<()> = Ok(());

	// Receive the first pre-converted frame from the prep thread.
	if let Ok((buf_a, tensor_a_init)) = prep_rx.recv() {
		let mut tensor_a = tensor_a_init;
		frames_read = 1;

		// Log padding info once (it’s constant for all frames).
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

		// Track the previous frame’s raw bytes so scene detection can compare
		// them against the incoming frame B. Only allocated when needed.
		let mut prev_buf: Option<Vec<u8>> = options.scene_detect_threshold.map(|_| buf_a.clone());

		// Pass the first frame through to the writer unchanged.
		if write_tx.send(WriteTask::Raw(buf_a)).is_ok() {
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

			let wait_prep_start = Instant::now();
			let (buf_b, tensor_b) = match prep_rx.recv() {
				Ok(frame) => frame,
				Err(_) => break, // EOF: no more input frames.
			};
			let wait_prep_time = wait_prep_start.elapsed();
			frames_read += 1;

			let infer_start = Instant::now();

			// Check for a scene cut before committing to inference.
			let is_scene_cut = prev_buf
				.as_deref()
				.zip(options.scene_detect_threshold)
				.is_some_and(|(prev, threshold)| {
					let score = scene_score(prev, &buf_b);
					trace!(
						"Scene score frame {} (input): {:.4} (threshold={:.3})",
						frames_read, score, threshold
					);
					if score > threshold {
						info!(
							"Scene cut detected at frame {} (score={:.3}, threshold={:.3})",
							frames_read, score, threshold
						);
						true
					} else {
						false
					}
				});

			if is_scene_cut {
				// Duplicate the last pre-cut frame (multiplier − 1) times to
				// maintain the correct output frame count without blending
				// across the cut.
				{
					let dup = prev_buf.as_ref().unwrap();
					for _ in 0..(options.multiplier - 1) {
						if write_tx.send(WriteTask::Raw(dup.clone())).is_err() {
							inference_result =
								Err(anyhow::anyhow!("Writer channel closed unexpectedly"));
							break 'outer;
						}
						frames_written += 1;
						report_progress(&options.on_progress, frames_written, total_output_frames);
					}
				}
				*prev_buf.as_mut().unwrap() = buf_b.clone();
			} else {
				// Normal RIFE inference path.
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

				for mid in mids {
					if write_tx.send(WriteTask::Tensor(mid)).is_err() {
						inference_result =
							Err(anyhow::anyhow!("Writer channel closed unexpectedly"));
						break 'outer;
					}
					frames_written += 1;
					report_progress(&options.on_progress, frames_written, total_output_frames);
				}

				if let Some(ref mut pb) = prev_buf {
					*pb = buf_b.clone();
				}
			}

			// Send frame B and slide the window.
			if write_tx.send(WriteTask::Raw(buf_b)).is_err() {
				inference_result = Err(anyhow::anyhow!("Writer channel closed unexpectedly"));
				break;
			}
			frames_written += 1;
			report_progress(&options.on_progress, frames_written, total_output_frames);

			let infer_time = infer_start.elapsed();
			debug!(
				"✦ Prep Wait: {:.1}ms | Infer: {:.1}ms | (tensor→bytes async in writer)",
				wait_prep_time.as_secs_f64() * 1_000.0,
				infer_time.as_secs_f64() * 1_000.0,
			);

			tensor_a = tensor_b;
		}
	}

	// Drop channel endpoints: closing write_tx drains the writer; closing
	// prep_rx signals the prep thread to stop (it will drain read_rx and exit,
	// which in turn causes the reader thread to stop sending).
	drop(write_tx);
	drop(prep_rx);

	// Join threads and collect any errors they encountered.
	prep_handle.join().expect("prep thread panicked");
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

	// Collect subtrees into temporary buffers so `mid` can be *moved* into
	// `out` rather than cloned. Each Array4 can be tens of MB, so skipping
	// the clone is meaningful for 4× and above.
	let mut left = Vec::new();
	recursive_interp(rife, f0, &mid, t0, mid_t, left_divs, ensemble, &mut left)?;
	let mut right = Vec::new();
	recursive_interp(rife, &mid, f1, mid_t, t1, right_divs, ensemble, &mut right)?;

	// Assemble in chronological order: left ‥ mid ‥ right.
	out.extend(left);
	out.push(mid);
	out.extend(right);

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

/// Compute a scene change score between two raw RGB24 frames using parallel CPU processing.
///
/// Returns a value in `[0.0, 1.0]` where `0.0` means identical frames and
/// `1.0` means maximally different. Uses the mean absolute difference of all
/// channel values, normalised by 255 — the same formula as FFmpeg's `scdet`
/// filter, so thresholds are directly comparable.
///
/// Uses Rayon's parallel iterators to split the computation across all
/// available CPU cores, eliminating the single-threaded bottleneck for
/// large frame buffers (~6.2 MB per 1080p RGB24 frame).
fn scene_score(a: &[u8], b: &[u8]) -> f64 {
	use rayon::prelude::*;

	debug_assert_eq!(a.len(), b.len(), "scene_score: frame size mismatch");
	let sad: u64 = a
		.par_iter()
		.zip(b.par_iter())
		.map(|(&x, &y)| x.abs_diff(y) as u64)
		.sum();
	sad as f64 / (a.len() as f64 * 255.0)
}
