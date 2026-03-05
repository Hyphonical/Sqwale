//! Execution provider management with platform-specific fallback support.

use anyhow::Result;
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
	///
	/// Returns `(dispatches, actual_provider, optional_warning)`.
	pub fn build(self) -> (Vec<ExecutionProviderDispatch>, Self, Option<String>) {
		match self {
			Self::Auto => (vec![], Self::Auto, None),

			Self::Cpu => (vec![ort::ep::CPU::default().build()], Self::Cpu, None),

			Self::Cuda => {
				#[cfg(any(target_os = "windows", target_os = "linux"))]
				{
					if self.check_availability() {
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
							Some("⚠ CUDA unavailable — falling back to CPU. Ensure NVIDIA drivers and CUDA runtime are installed.".into()),
						)
					}
				}
				#[cfg(not(any(target_os = "windows", target_os = "linux")))]
				{
					(
						vec![ort::ep::CPU::default().build()],
						Self::Cpu,
						Some("⚠ CUDA is not supported on this platform — using CPU.".into()),
					)
				}
			}

			Self::TensorRT => {
				#[cfg(any(target_os = "windows", target_os = "linux"))]
				{
					if self.check_availability() {
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
							Some("⚠ TensorRT unavailable — falling back to CPU. Install TensorRT and ensure CUDA is available.".into()),
						)
					}
				}
				#[cfg(not(any(target_os = "windows", target_os = "linux")))]
				{
					(
						vec![ort::ep::CPU::default().build()],
						Self::Cpu,
						Some("⚠ TensorRT is not supported on this platform — using CPU.".into()),
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
						Some("⚠ CoreML is only available on macOS — using CPU.".into()),
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
						Some("⚠ XNNPACK is only available on Linux — using CPU.".into()),
					)
				}
			}
		}
	}

	/// Heuristic availability check.
	fn check_availability(self) -> bool {
		match self {
			Self::Auto | Self::Cpu => true,
			Self::Cuda | Self::TensorRT => {
				std::process::Command::new("nvidia-smi").output().is_ok()
			}
			Self::CoreML => cfg!(target_os = "macos"),
			Self::XNNPack => cfg!(target_os = "linux"),
		}
	}

	/// Display name.
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
		f.write_str(self.name())
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
			other => anyhow::bail!(
				"Unknown provider '{other}'. Valid options: auto, cpu, cuda, tensorrt, coreml, xnnpack"
			),
		}
	}
}
