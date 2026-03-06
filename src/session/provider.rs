//! Execution provider selection and platform dispatch.

use anyhow::{Result, bail};
use ort::ep::ExecutionProviderDispatch;
use std::str::FromStr;

/// Execution provider selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ProviderSelection {
	/// Let ORT choose the best available device.
	#[default]
	Auto,
	Cpu,
	Cuda,
	TensorRT,
	DirectML,
	CoreML,
	XNNPack,
}

impl ProviderSelection {
	/// Human-readable name for display.
	pub fn name(self) -> &'static str {
		match self {
			Self::Auto => "auto",
			Self::Cpu => "CPU",
			Self::Cuda => "CUDA",
			Self::TensorRT => "TensorRT",
			Self::DirectML => "DirectML",
			Self::CoreML => "CoreML",
			Self::XNNPack => "XNNPACK",
		}
	}
}

impl std::fmt::Display for ProviderSelection {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.write_str(self.name())
	}
}

impl FromStr for ProviderSelection {
	type Err = anyhow::Error;

	fn from_str(s: &str) -> Result<Self> {
		match s.to_lowercase().as_str() {
			"auto" => Ok(Self::Auto),
			"cpu" => Ok(Self::Cpu),
			"cuda" => Ok(Self::Cuda),
			"tensorrt" | "trt" => Ok(Self::TensorRT),
			"directml" | "dml" => Ok(Self::DirectML),
			"coreml" => Ok(Self::CoreML),
			"xnnpack" => Ok(Self::XNNPack),
			other => bail!(
				"Unknown provider '{other}'. Valid: auto, cpu, cuda, tensorrt, directml, coreml, xnnpack"
			),
		}
	}
}

/// Construct an execution provider dispatch for the given selection.
pub fn make_ep(provider: ProviderSelection) -> Result<ExecutionProviderDispatch> {
	match provider {
		ProviderSelection::Cpu | ProviderSelection::Auto => Ok(ort::ep::CPU::default().build()),
		ProviderSelection::Cuda => {
			#[cfg(any(target_os = "windows", target_os = "linux"))]
			{
				Ok(ort::ep::CUDA::default().build())
			}
			#[cfg(not(any(target_os = "windows", target_os = "linux")))]
			{
				bail!("CUDA is not available on this platform")
			}
		}
		ProviderSelection::TensorRT => {
			#[cfg(any(target_os = "windows", target_os = "linux"))]
			{
				Ok(ort::ep::TensorRT::default().build())
			}
			#[cfg(not(any(target_os = "windows", target_os = "linux")))]
			{
				bail!("TensorRT is not available on this platform")
			}
		}
		ProviderSelection::DirectML => {
			#[cfg(target_os = "windows")]
			{
				Ok(ort::ep::DirectML::default().build())
			}
			#[cfg(not(target_os = "windows"))]
			{
				bail!("DirectML is only available on Windows")
			}
		}
		ProviderSelection::CoreML => {
			#[cfg(target_os = "macos")]
			{
				Ok(ort::ep::CoreML::default().build())
			}
			#[cfg(not(target_os = "macos"))]
			{
				bail!("CoreML is only available on macOS")
			}
		}
		ProviderSelection::XNNPack => {
			#[cfg(target_os = "linux")]
			{
				Ok(ort::ep::XNNPACK::default().build())
			}
			#[cfg(not(target_os = "linux"))]
			{
				bail!("XNNPACK is only available on Linux")
			}
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn parse_valid_providers() {
		assert_eq!(
			ProviderSelection::from_str("auto").unwrap(),
			ProviderSelection::Auto
		);
		assert_eq!(
			ProviderSelection::from_str("cpu").unwrap(),
			ProviderSelection::Cpu
		);
		assert_eq!(
			ProviderSelection::from_str("CUDA").unwrap(),
			ProviderSelection::Cuda
		);
		assert_eq!(
			ProviderSelection::from_str("tensorrt").unwrap(),
			ProviderSelection::TensorRT
		);
		assert_eq!(
			ProviderSelection::from_str("trt").unwrap(),
			ProviderSelection::TensorRT
		);
		assert_eq!(
			ProviderSelection::from_str("directml").unwrap(),
			ProviderSelection::DirectML
		);
		assert_eq!(
			ProviderSelection::from_str("dml").unwrap(),
			ProviderSelection::DirectML
		);
		assert_eq!(
			ProviderSelection::from_str("coreml").unwrap(),
			ProviderSelection::CoreML
		);
		assert_eq!(
			ProviderSelection::from_str("xnnpack").unwrap(),
			ProviderSelection::XNNPack
		);
	}

	#[test]
	fn parse_invalid_provider() {
		assert!(ProviderSelection::from_str("vulkan").is_err());
	}

	#[test]
	fn name_round_trip() {
		assert_eq!(ProviderSelection::Cpu.name(), "CPU");
		assert_eq!(ProviderSelection::Auto.name(), "auto");
	}
}
