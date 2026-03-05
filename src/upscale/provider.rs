//! Execution provider management with platform-specific fallback support.

use anyhow::Result;
use colored::Colorize;
use ort::ep::ExecutionProviderDispatch;
use std::str::FromStr;

/// Execution provider for ONNX Runtime inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provider {
	/// Automatic device selection (NPU/GPU/CPU).
	Auto,
	Cpu,
	Cuda,
	TensorRT,
	CoreML,
	XNNPack,
}

impl Provider {
	/// Build execution provider dispatch with automatic CPU fallback.
	/// Returns (dispatch, actual_provider, warning_message).
	pub fn build(self) -> (Vec<ExecutionProviderDispatch>, Self, Option<String>) {
		match self {
			// Auto variant returns empty dispatch to signal auto_device usage
			Self::Auto => (vec![], Self::Auto, None),

			Self::Cpu => (vec![ort::ep::CPU::default().build()], Self::Cpu, None),

			Self::Cuda => {
				#[cfg(any(target_os = "windows", target_os = "linux"))]
				{
					let cuda_available = self.check_availability();
					if cuda_available {
						(
							vec![
								ort::ep::CUDA::default().build(),
								ort::ep::CPU::default().build(),
							],
							Self::Cuda,
							None,
						)
					} else {
						(
							vec![ort::ep::CPU::default().build()],
							Self::Cpu,
							Some(format!(
								"{} CUDA unavailable, using CPU instead\n  {} Ensure NVIDIA GPU drivers and CUDA runtime are installed",
								"⚠".yellow(),
								"Hint:".yellow()
							)),
						)
					}
				}
				#[cfg(not(any(target_os = "windows", target_os = "linux")))]
				{
					(
						vec![ort::ep::CPU::default().build()],
						Self::Cpu,
						Some(format!(
							"{} CUDA not supported on this platform, using CPU",
							"⚠".yellow()
						)),
					)
				}
			}

			Self::TensorRT => {
				#[cfg(any(target_os = "windows", target_os = "linux"))]
				{
					let trt_available = self.check_availability();
					if trt_available {
						(
							vec![
								ort::ep::TensorRT::default().build(),
								ort::ep::CPU::default().build(),
							],
							Self::TensorRT,
							None,
						)
					} else {
						(
							vec![ort::ep::CPU::default().build()],
							Self::Cpu,
							Some(format!(
								"{} TensorRT unavailable, using CPU instead\n  {} Install TensorRT and ensure CUDA is available",
								"⚠".yellow(),
								"Hint:".yellow()
							)),
						)
					}
				}
				#[cfg(not(any(target_os = "windows", target_os = "linux")))]
				{
					(
						vec![ort::ep::CPU::default().build()],
						Self::Cpu,
						Some(format!(
							"{} TensorRT not supported on this platform, using CPU",
							"⚠".yellow()
						)),
					)
				}
			}

			Self::CoreML => {
				#[cfg(target_os = "macos")]
				{
					(
						vec![
							ort::ep::CoreML::default().build(),
							ort::ep::CPU::default().build(),
						],
						Self::CoreML,
						None,
					)
				}
				#[cfg(not(target_os = "macos"))]
				{
					(
						vec![ort::ep::CPU::default().build()],
						Self::Cpu,
						Some(format!(
							"{} CoreML only available on macOS, using CPU",
							"⚠".yellow()
						)),
					)
				}
			}

			Self::XNNPack => {
				#[cfg(target_os = "linux")]
				{
					(
						vec![
							ort::ep::XNNPACK::default().build(),
							ort::ep::CPU::default().build(),
						],
						Self::XNNPack,
						None,
					)
				}
				#[cfg(not(target_os = "linux"))]
				{
					(
						vec![ort::ep::CPU::default().build()],
						Self::Cpu,
						Some(format!(
							"{} XNNPACK only available on Linux, using CPU",
							"⚠".yellow()
						)),
					)
				}
			}
		}
	}

	/// Check if provider is available at runtime (basic heuristic).
	fn check_availability(self) -> bool {
		match self {
			Self::Auto => true,
			Self::Cpu => true,
			Self::Cuda | Self::TensorRT => {
				// Basic check: CUDA requires nvidia-smi to be present
				std::process::Command::new("nvidia-smi").output().is_ok()
			}
			Self::CoreML => cfg!(target_os = "macos"),
			Self::XNNPack => cfg!(target_os = "linux"),
		}
	}

	/// Get display name for provider.
	pub fn name(self) -> &'static str {
		match self {
			Self::Auto => "Auto",
			Self::Cpu => "CPU",
			Self::Cuda => "CUDA",
			Self::TensorRT => "TensorRT",
			Self::CoreML => "CoreML",
			Self::XNNPack => "XNNPACK",
		}
	}
}

impl std::fmt::Display for Provider {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(f, "{}", self.name())
	}
}

impl FromStr for Provider {
	type Err = anyhow::Error;

	fn from_str(s: &str) -> Result<Self> {
		match s.to_lowercase().as_str() {
			"auto" => Ok(Self::Auto),
			"cpu" => Ok(Self::Cpu),
			"cuda" => Ok(Self::Cuda),
			"tensorrt" | "trt" => Ok(Self::TensorRT),
			"coreml" => Ok(Self::CoreML),
			"xnnpack" => Ok(Self::XNNPack),
			_ => anyhow::bail!(
				"{} '{}'\n  {} Valid providers: auto, cpu, cuda, tensorrt, coreml, xnnpack",
				"Unknown provider:".red().bold(),
				s.bright_white(),
				"Hint:".yellow()
			),
		}
	}
}
