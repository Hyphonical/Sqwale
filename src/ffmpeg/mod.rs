//! Piped FFmpeg wrapper: video probing, raw-frame reader, and raw-frame writer.
//!
//! All frame data flows through OS pipes as raw `RGB24` bytes — no temporary
//! files are created on disk.

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use tracing::{debug, warn};

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

/// Probe a video file with `ffprobe` and return its metadata.
pub fn probe(input: &Path) -> Result<VideoInfo> {
	let input_str = input
		.to_str()
		.context("Input path contains invalid UTF-8")?;

	let output = Command::new("ffprobe")
		.args(["-v", "quiet"])
		.args(["-print_format", "json"])
		.args(["-show_streams"])
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
	let frame_count = video
		.nb_frames
		.as_deref()
		.and_then(|s| s.parse::<usize>().ok())
		.or_else(|| {
			video
				.duration
				.as_deref()
				.and_then(|s| s.parse::<f64>().ok())
				.map(|dur| (dur * fps).round() as usize)
		})
		.unwrap_or(0);

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

/// Spawn an FFmpeg process that decodes all video frames to raw RGB24 on stdout.
pub fn spawn_reader(input: &Path) -> Result<(Child, ChildStdout)> {
	let input_str = input
		.to_str()
		.context("Input path contains invalid UTF-8")?;

	let mut child = Command::new("ffmpeg")
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

	// Pipe stderr to tracing in a background thread.
	if let Some(stderr) = child.stderr.take() {
		std::thread::spawn(move || {
			let reader = BufReader::new(stderr);
			for line in reader.lines().map_while(Result::ok) {
				debug!(target: "ffmpeg::reader", "{}", line);
			}
		});
	}

	debug!("FFmpeg reader spawned for {:?}", input);
	Ok((child, stdout))
}

/// Spawn an FFmpeg process that reads raw RGB24 frames from stdin and encodes to MKV.
///
/// If the original file has audio, it is copied into the output unchanged.
pub fn spawn_writer(
	input: &Path,
	output: &Path,
	width: usize,
	height: usize,
	output_fps_str: &str,
	crf: u32,
) -> Result<(Child, ChildStdin)> {
	let input_str = input
		.to_str()
		.context("Input path contains invalid UTF-8")?;
	let output_str = output
		.to_str()
		.context("Output path contains invalid UTF-8")?;

	let size_arg = format!("{width}x{height}");

	let mut child = Command::new("ffmpeg")
		.arg("-y")
		.args(["-f", "rawvideo"])
		.args(["-pix_fmt", "rgb24"])
		.args(["-s", &size_arg])
		.args(["-r", output_fps_str])
		.args(["-i", "-"])
		.args(["-i", input_str])
		.args(["-map", "0:v"])
		.args(["-map", "1:a?"])
		.args(["-c:v", "libx264"])
		.args(["-crf", &crf.to_string()])
		.args(["-c:a", "copy"])
		.args(["-v", "quiet"])
		.arg(output_str)
		.stdin(Stdio::piped())
		.stdout(Stdio::null())
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

	// Pipe stderr to tracing in a background thread.
	if let Some(stderr) = child.stderr.take() {
		std::thread::spawn(move || {
			let reader = BufReader::new(stderr);
			for line in reader.lines().map_while(Result::ok) {
				debug!(target: "ffmpeg::writer", "{}", line);
			}
		});
	}

	debug!(
		"FFmpeg writer spawned: {}×{} @ {} fps, CRF {}, output {:?}",
		width, height, output_fps_str, crf, output
	);
	Ok((child, stdin))
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
}
