//! Piped FFmpeg wrapper: video probing, raw-frame reader, and raw-frame writer.
//!
//! All frame data flows through OS pipes as raw `RGB24` bytes — no temporary
//! files are created on disk.

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::thread;

use tracing::{debug, info, warn};

// ── Video metadata ─────────────────────────────────────────────────────────

/// Metadata extracted from a video file via `ffprobe`.
pub struct VideoInfo {
	/// Frames per second as a floating point value.
	pub fps: f64,
	/// Frame rate as a rational string (e.g. `"24000/1001"`).
	pub fps_str: String,
	/// Total number of video frames (estimated from duration × fps if needed).
	pub frame_count: usize,
	/// Frame width in pixels.
	pub width: usize,
	/// Frame height in pixels.
	pub height: usize,
	/// Whether the file contains at least one audio stream.
	pub has_audio: bool,
}

// ── ffprobe JSON structures ────────────────────────────────────────────────

#[derive(Deserialize)]
struct ProbeOutput {
	streams: Vec<ProbeStream>,
	#[serde(default)]
	format: Option<ProbeFormat>,
}

#[derive(Deserialize)]
struct ProbeFormat {
	#[serde(default)]
	duration: Option<String>,
}

#[derive(Deserialize)]
struct ProbeStream {
	codec_type: String,
	#[serde(default)]
	width: Option<usize>,
	#[serde(default)]
	height: Option<usize>,
	#[serde(default)]
	r_frame_rate: Option<String>,
	#[serde(default)]
	avg_frame_rate: Option<String>,
	#[serde(default)]
	nb_frames: Option<String>,
	#[serde(default)]
	duration: Option<String>,
}

// ── Public API ─────────────────────────────────────────────────────────────

/// Returns `true` if the host FFmpeg installation exposes the `hevc_nvenc` encoder.
pub(crate) fn supports_nvenc() -> bool {
	match std::process::Command::new("ffmpeg")
		.arg("-encoders")
		.output()
	{
		Ok(output) => String::from_utf8_lossy(&output.stdout).contains("hevc_nvenc"),
		Err(_) => false,
	}
}

/// Probe a video file with `ffprobe` and return its metadata.
pub fn probe(input: &Path) -> Result<VideoInfo> {
	let input_str = input
		.to_str()
		.context("Input path contains invalid UTF-8")?;

	let output = Command::new("ffprobe")
		.args(["-v", "quiet"])
		.args(["-print_format", "json"])
		.args(["-show_streams"])
		.args(["-show_format"])
		.arg(input_str)
		.stdout(Stdio::piped())
		.stderr(Stdio::null())
		.output()
		.context(
			"Failed to run ffprobe. Is FFmpeg installed and on your PATH?\n\
			 Download from: https://ffmpeg.org/download.html",
		)?;

	if !output.status.success() {
		bail!(
			"ffprobe exited with status {}",
			output.status.code().unwrap_or(-1)
		);
	}

	let probe: ProbeOutput =
		serde_json::from_slice(&output.stdout).context("Failed to parse ffprobe JSON output")?;

	let video = probe
		.streams
		.iter()
		.find(|s| s.codec_type == "video")
		.context("No video stream found in file")?;

	let has_audio = probe.streams.iter().any(|s| s.codec_type == "audio");

	let width = video.width.context("Video stream has no width")?;
	let height = video.height.context("Video stream has no height")?;

	let r_frame_rate = video.r_frame_rate.as_deref().unwrap_or("0/1").to_owned();
	let avg_frame_rate = video.avg_frame_rate.as_deref().unwrap_or("0/1").to_owned();

	// Warn on variable frame rate.
	let r_fps = parse_rational_fps(&r_frame_rate);
	let a_fps = parse_rational_fps(&avg_frame_rate);
	if r_fps > 0.0 && a_fps > 0.0 && (r_fps - a_fps).abs() / r_fps > 0.05 {
		warn!(
			"Variable frame rate detected (r_frame_rate={r_frame_rate}, \
			 avg_frame_rate={avg_frame_rate}). The output will be constant frame rate."
		);
	}

	// Use r_frame_rate as the canonical value.
	let fps = r_fps;
	let fps_str = r_frame_rate;

	// Estimate frame count.
	let frame_count = estimate_frame_count(
		video.nb_frames.as_deref(),
		video.duration.as_deref(),
		probe
			.format
			.as_ref()
			.and_then(|format| format.duration.as_deref()),
		fps,
	);

	debug!(
		"Probed video: {}×{}, {} fps ({}), ~{} frames, audio={}",
		width, height, fps, fps_str, frame_count, has_audio
	);

	Ok(VideoInfo {
		fps,
		fps_str,
		frame_count,
		width,
		height,
		has_audio,
	})
}

/// Pipe a child process's stderr to the tracing `debug!` macro.
///
/// Spawns a background thread that reads `stderr` line by line.
/// Call this after taking all other handles from the child.
macro_rules! pipe_stderr_to_debug {
	($child:expr, $target:literal) => {
		if let Some(stderr) = $child.stderr.take() {
			thread::spawn(move || {
				let reader = BufReader::new(stderr);
				for line in reader.lines().map_while(Result::ok) {
					debug!(target: $target, "{}", line);
				}
			});
		}
	};
}

/// Spawn an FFmpeg process that decodes all video frames to raw RGB24 on stdout.
///
/// Uses GPU hardware acceleration (CUDA) for video decoding if available,
/// otherwise falls back to CPU decoding.
pub fn spawn_reader(input: &Path) -> Result<(Child, ChildStdout)> {
	let input_str = input
		.to_str()
		.context("Input path contains invalid UTF-8")?;

	let mut cmd = Command::new("ffmpeg");

	// Enable CUDA hardware acceleration if GPU is available.
	// Note: We intentionally omit -hwaccel_output_format cuda so that FFmpeg
	// downloads decoded frames back to CPU memory for the raw pipe output.
	// Using hwaccel_output_format cuda would keep frames as CUDA surfaces,
	// which cannot be piped as rgb24 without an explicit hwdownload filter.
	if supports_nvenc() {
		cmd.args(["-hwaccel", "cuda"]);
	}

	let mut child = cmd
		.args(["-i", input_str])
		.args(["-f", "rawvideo"])
		.args(["-pix_fmt", "rgb24"])
		.args(["-v", "quiet"])
		.arg("-")
		.stdout(Stdio::piped())
		.stderr(Stdio::piped())
		.spawn()
		.context(
			"Failed to spawn FFmpeg reader. Is FFmpeg installed and on your PATH?\n\
			 Download from: https://ffmpeg.org/download.html",
		)?;

	let stdout = child
		.stdout
		.take()
		.context("Failed to capture FFmpeg reader stdout")?;

	pipe_stderr_to_debug!(&mut child, "ffmpeg::reader");

	debug!("FFmpeg reader spawned for {:?}", input);
	Ok((child, stdout))
}

/// Spawn an FFmpeg process that reads raw RGB24 frames from stdin and encodes to MPEG-TS.
///
/// Audio is intentionally omitted — call [`mux_audio_into`] after the run completes
/// to add the source audio in a single lossless remux pass.
///
/// B-frames are disabled (`-bf 0`) so that any partial `.ts` can be resumed cleanly
/// via [`spawn_writer_append`].
pub fn spawn_writer(
	output: &Path,
	width: usize,
	height: usize,
	output_fps_str: &str,
	crf: u32,
) -> Result<(Child, ChildStdin)> {
	let output_str = output
		.to_str()
		.context("Output path contains invalid UTF-8")?;

	let (mut child, stdin) =
		build_writer_child(output_str, width, height, output_fps_str, crf, None)?;

	// Pipe stderr to tracing in a background thread.
	pipe_stderr_to_debug!(&mut child, "ffmpeg::writer");

	debug!(
		"FFmpeg writer spawned: {}×{} @ {} fps, CRF {}, output {:?}",
		width, height, output_fps_str, crf, output
	);
	Ok((child, stdin))
}

/// Spawn an FFmpeg process that reads raw RGB24 frames from stdin and **appends** encoded
/// MPEG-TS packets to an existing `.ts` file.
///
/// All output timestamps are offset by `ts_offset_secs` so the new packets continue
/// seamlessly from the last frame in the existing file. A background thread forwards
/// the child's stdout directly to the open file in append mode.
pub fn spawn_writer_append(
	append_to: &Path,
	width: usize,
	height: usize,
	output_fps_str: &str,
	crf: u32,
	ts_offset_secs: f64,
) -> Result<(Child, ChildStdin)> {
	// Build the child writing to pipe:1 (stdout).
	let (mut child, stdin) = build_writer_child(
		"pipe:1",
		width,
		height,
		output_fps_str,
		crf,
		Some(ts_offset_secs),
	)?;

	// Capture stdout and forward to the append file in a background thread.
	let stdout = child
		.stdout
		.take()
		.context("Failed to capture FFmpeg writer stdout for append")?;
	let append_path = append_to.to_path_buf();
	thread::spawn(move || {
		let file = std::fs::OpenOptions::new()
			.append(true)
			.open(&append_path)
			.expect("Cannot open output .ts for appending");
		let mut src = BufReader::new(stdout);
		let mut dst = std::io::BufWriter::new(file);
		if let Err(e) = std::io::copy(&mut src, &mut dst) {
			warn!(target: "ffmpeg::writer_append", "Append I/O error: {e}");
		}
	});

	pipe_stderr_to_debug!(&mut child, "ffmpeg::writer");

	debug!(
		"FFmpeg append-writer spawned: {}×{} @ {} fps, CRF {}, ts_offset={:.3}s, append→{:?}",
		width, height, output_fps_str, crf, ts_offset_secs, append_to
	);
	Ok((child, stdin))
}

/// Spawn an FFmpeg process that reads raw RGB24 frames from stdin, starting output
/// timestamps at `start_frame` decoded from the input at the given seek point,
/// and writes to a file from `start_frame`'s position.
/// The raw writer that decodes from start_frame is for `spawn_reader_from`.
///
/// Internal helper shared by `spawn_writer` and `spawn_writer_append`.
fn build_writer_child(
	output_target: &str,
	width: usize,
	height: usize,
	output_fps_str: &str,
	crf: u32,
	ts_offset_secs: Option<f64>,
) -> Result<(Child, ChildStdin)> {
	let size_arg = format!("{width}x{height}");

	let mut cmd = Command::new("ffmpeg");
	cmd.arg("-y")
		.args(["-f", "rawvideo"])
		.args(["-pix_fmt", "rgb24"])
		.args(["-s", &size_arg])
		.args(["-r", output_fps_str])
		.args(["-i", "-"])
		.args(["-map", "0:v"]);

	// Force a standard output pixel format so players can always read the file.
	cmd.args(["-pix_fmt", "yuv420p"]);

	if supports_nvenc() {
		info!("Hardware encoding: using hevc_nvenc (preset p4)");
		cmd.args(["-c:v", "hevc_nvenc"]).args(["-preset", "p4"]);

		if crf == 0 {
			// NVENC requires a specific tune flag for true lossless.
			cmd.args(["-tune", "lossless"]);
		} else {
			// VBR rate control: -cq is respected only when -b:v 0 removes the bitrate cap.
			cmd.args(["-rc", "vbr"])
				.args(["-cq", &crf.to_string()])
				.args(["-b:v", "0"]);
		}
	} else {
		info!("Hardware encoding unavailable: falling back to libx264 (preset fast)");
		cmd.args(["-c:v", "libx264"])
			.args(["-preset", "fast"])
			.args(["-crf", &crf.to_string()]);
	}

	// Disable B-frames so any partial .ts can be resumed without GOP misalignment.
	cmd.args(["-bf", "0"]);

	// Apply PTS offset for append / continue mode.
	if let Some(offset) = ts_offset_secs {
		cmd.args(["-output_ts_offset", &format!("{offset:.6}")]);
	}

	let mut child = cmd
		.args(["-f", "mpegts"])
		.args(["-v", "quiet"])
		.arg(output_target)
		.stdin(Stdio::piped())
		.stdout(if ts_offset_secs.is_some() {
			Stdio::piped() // append mode: capture stdout for the forwarding thread
		} else {
			Stdio::null()
		})
		.stderr(Stdio::piped())
		.spawn()
		.context(
			"Failed to spawn FFmpeg writer. Is FFmpeg installed and on your PATH?\n\
			 Download from: https://ffmpeg.org/download.html",
		)?;

	let stdin = child
		.stdin
		.take()
		.context("Failed to capture FFmpeg writer stdin")?;

	Ok((child, stdin))
}

/// Spawn an FFmpeg reader that begins decoding from `start_frame`, discarding all
/// preceding frames via a `select` filter.
///
/// When `start_frame == 0` this is equivalent to [`spawn_reader`].
///
/// # Performance note
/// All frames before `start_frame` are decoded (but not output). For very long
/// videos with a late resume point this may take some time. A future optimisation
/// would combine `-ss` fast-seek with a small sub-GOP `select` offset.
pub fn spawn_reader_from(input: &Path, start_frame: usize) -> Result<(Child, ChildStdout)> {
	if start_frame == 0 {
		return spawn_reader(input);
	}

	let input_str = input
		.to_str()
		.context("Input path contains invalid UTF-8")?;

	let mut cmd = Command::new("ffmpeg");

	// CUDA hwaccel: frames are on CPU after decode, so the software `select` filter works.
	if supports_nvenc() {
		cmd.args(["-hwaccel", "cuda"]);
	}

	let select_expr = format!("select=gte(n\\,{start_frame})");
	let mut child = cmd
		.args(["-i", input_str])
		.args(["-vf", &select_expr])
		.args(["-f", "rawvideo"])
		.args(["-pix_fmt", "rgb24"])
		.args(["-v", "quiet"])
		.arg("-")
		.stdout(Stdio::piped())
		.stderr(Stdio::piped())
		.spawn()
		.context(
			"Failed to spawn FFmpeg reader. Is FFmpeg installed and on your PATH?\n\
			 Download from: https://ffmpeg.org/download.html",
		)?;

	let stdout = child
		.stdout
		.take()
		.context("Failed to capture FFmpeg reader stdout")?;

	pipe_stderr_to_debug!(&mut child, "ffmpeg::reader");

	debug!(
		"FFmpeg seek-reader spawned for {:?}, start_frame={start_frame}",
		input
	);
	Ok((child, stdout))
}

/// Mux the audio stream from `source` into `video_only_ts`, writing the combined result
/// to `output`. Uses stream copy (no re-encoding).
///
/// If `source` has no audio stream, the video is copied to `output` unchanged.
pub fn mux_audio_into(video_ts: &Path, source: &Path, output: &Path) -> Result<()> {
	let video_str = video_ts
		.to_str()
		.context("Video path contains invalid UTF-8")?;
	let source_str = source
		.to_str()
		.context("Source path contains invalid UTF-8")?;
	let output_str = output
		.to_str()
		.context("Output path contains invalid UTF-8")?;

	let status = Command::new("ffmpeg")
		.arg("-y")
		.args(["-i", video_str])
		.args(["-i", source_str])
		.args(["-map", "0:v"])
		.args(["-map", "1:a?"])
		.args(["-c", "copy"])
		.args(["-f", "mpegts"])
		.args(["-v", "quiet"])
		.arg(output_str)
		.status()
		.context("Failed to run FFmpeg audio mux")?;

	if !status.success() {
		bail!(
			"FFmpeg audio mux exited with status {}",
			status.code().unwrap_or(-1)
		);
	}

	debug!(
		"Audio mux complete: {:?} + {:?} → {:?}",
		video_ts, source, output
	);
	Ok(())
}

/// Trim an MPEG-TS file to at most `frame_count` video frames, writing to `output`.
/// Uses stream copy (no re-encoding). The input file is not modified.
pub fn trim_to_frames(input: &Path, frame_count: usize, output: &Path) -> Result<()> {
	let input_str = input
		.to_str()
		.context("Input path contains invalid UTF-8")?;
	let output_str = output
		.to_str()
		.context("Output path contains invalid UTF-8")?;

	let status = Command::new("ffmpeg")
		.arg("-y")
		.args(["-i", input_str])
		.args(["-map", "0:v"])
		.args(["-frames:v", &frame_count.to_string()])
		.args(["-c", "copy"])
		.args(["-f", "mpegts"])
		.args(["-v", "quiet"])
		.arg(output_str)
		.status()
		.context("Failed to run FFmpeg trim")?;

	if !status.success() {
		bail!(
			"FFmpeg trim exited with status {}",
			status.code().unwrap_or(-1)
		);
	}

	debug!(
		"Trim complete: {:?} → {frame_count} frames → {:?}",
		input, output
	);
	Ok(())
}

/// Multiply a rational FPS string by an integer multiplier.
///
/// `"24000/1001"` × 2 → `"48000/1001"`.
/// Integer rates like `"30"` are treated as `"30/1"`.
pub fn multiply_fps(fps_str: &str, multiplier: u32) -> Result<String> {
	let (num, den) = if let Some((n, d)) = fps_str.split_once('/') {
		let num: u64 = n
			.trim()
			.parse()
			.with_context(|| format!("Invalid FPS numerator: '{n}'"))?;
		let den: u64 = d
			.trim()
			.parse()
			.with_context(|| format!("Invalid FPS denominator: '{d}'"))?;
		(num, den)
	} else {
		let num: u64 = fps_str
			.trim()
			.parse()
			.with_context(|| format!("Invalid FPS value: '{fps_str}'"))?;
		(num, 1)
	};

	Ok(format!("{}/{}", num * multiplier as u64, den))
}

/// Read exactly one raw RGB24 frame from a reader.
///
/// Returns `false` when the stream has ended (no more frames).
pub fn read_frame(reader: &mut impl Read, buf: &mut [u8]) -> Result<bool> {
	let mut filled = 0;
	while filled < buf.len() {
		match reader.read(&mut buf[filled..]) {
			Ok(0) => {
				if filled == 0 {
					return Ok(false); // Clean EOF.
				}
				bail!(
					"Unexpected end of video stream (read {filled}/{} bytes)",
					buf.len()
				);
			}
			Ok(n) => filled += n,
			Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
			Err(e) => return Err(e).context("Failed to read frame from FFmpeg"),
		}
	}
	Ok(true)
}

// ── Helpers ────────────────────────────────────────────────────────────────

/// Parse a rational string like `"24000/1001"` or `"30"` to an `f64`.
fn parse_rational_fps(s: &str) -> f64 {
	if let Some((n, d)) = s.split_once('/') {
		let num: f64 = n.trim().parse().unwrap_or(0.0);
		let den: f64 = d.trim().parse().unwrap_or(1.0);
		if den == 0.0 { 0.0 } else { num / den }
	} else {
		s.trim().parse().unwrap_or(0.0)
	}
}

fn estimate_frame_count(
	nb_frames: Option<&str>,
	stream_duration: Option<&str>,
	format_duration: Option<&str>,
	fps: f64,
) -> usize {
	nb_frames
		.and_then(|value| value.parse::<usize>().ok())
		.or_else(|| {
			parse_duration_seconds(stream_duration)
				.map(|duration| (duration * fps).round() as usize)
		})
		.or_else(|| {
			parse_duration_seconds(format_duration)
				.map(|duration| (duration * fps).round() as usize)
		})
		.unwrap_or(0)
}

fn parse_duration_seconds(value: Option<&str>) -> Option<f64> {
	value.and_then(|duration| duration.parse::<f64>().ok())
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn multiply_fps_rational() {
		assert_eq!(multiply_fps("24000/1001", 2).unwrap(), "48000/1001");
		assert_eq!(multiply_fps("24000/1001", 4).unwrap(), "96000/1001");
	}

	#[test]
	fn multiply_fps_integer() {
		assert_eq!(multiply_fps("30", 2).unwrap(), "60/1");
		assert_eq!(multiply_fps("60", 4).unwrap(), "240/1");
	}

	#[test]
	fn multiply_fps_with_spaces() {
		assert_eq!(multiply_fps(" 24000 / 1001 ", 2).unwrap(), "48000/1001");
	}

	#[test]
	fn parse_rational_fps_values() {
		assert!((parse_rational_fps("24000/1001") - 23.976).abs() < 0.01);
		assert!((parse_rational_fps("30") - 30.0).abs() < 0.01);
		assert!((parse_rational_fps("0/0") - 0.0).abs() < 0.01);
	}

	#[test]
	fn estimate_frame_count_prefers_nb_frames() {
		assert_eq!(
			estimate_frame_count(Some("483"), Some("20.132"), Some("20.141"), 24.0),
			483
		);
	}

	#[test]
	fn estimate_frame_count_uses_stream_duration_when_present() {
		assert_eq!(
			estimate_frame_count(None, Some("20.132"), Some("20.141"), 24.0),
			483
		);
	}

	#[test]
	fn estimate_frame_count_falls_back_to_format_duration() {
		assert_eq!(
			estimate_frame_count(None, None, Some("20.141000"), 24.0),
			483
		);
	}
}
