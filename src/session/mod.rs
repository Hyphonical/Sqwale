//! ORT session management: model loading and session creation.

mod provider;

pub use provider::{ProviderSelection, make_ep};

use anyhow::Result;
use ort::session::Session;
use ort::session::builder::AutoDevicePolicy;
use std::path::Path;
use tracing::warn;

use crate::inspect::{self, ModelInfo};

/// An ORT session paired with model metadata and the provider that was used.
pub struct SessionContext {
	/// The ORT inference session.
	pub(crate) session: Session,

	/// Metadata extracted from the model (scale, channels, tiling, etc.).
	pub model_info: ModelInfo,

	/// The execution provider that was actually used for this session.
	pub provider_used: ProviderSelection,
}

/// Load an ONNX model: inspect it and create an inference session.
///
/// 1. Calls `inspect_model(path)` to extract metadata.
/// 2. Creates an ORT session with the selected provider.
/// 3. Returns the session context with metadata and actual provider.
pub fn load_model(path: &Path, provider: ProviderSelection) -> Result<SessionContext> {
	let model_info = inspect::inspect_model(path)?;

	let (session, provider_used) = create_session(path, provider)?;

	Ok(SessionContext {
		session,
		model_info,
		provider_used,
	})
}

/// Load an ONNX model from raw bytes: inspect it and create an inference session.
///
/// Equivalent to `load_model` but reads the model from memory instead of a file.
/// Useful for embedded models included via `include_bytes!`.
pub fn load_model_bytes(bytes: &[u8], provider: ProviderSelection) -> Result<SessionContext> {
	let model_info = inspect::inspect_model_bytes(bytes)?;

	let (session, provider_used) = create_session_from_bytes(bytes, provider)?;

	Ok(SessionContext {
		session,
		model_info,
		provider_used,
	})
}

/// Create an ORT session with the given provider, falling back to CPU on failure.
fn create_session(
	path: &Path,
	provider: ProviderSelection,
) -> Result<(Session, ProviderSelection)> {
	match provider {
		ProviderSelection::Auto => {
			let session = Session::builder()
				.map_err(|e| anyhow::anyhow!("Failed to create session builder: {e}"))?
				.with_auto_device(AutoDevicePolicy::MaxPerformance)
				.map_err(|e| anyhow::anyhow!("Failed to configure auto device: {e}"))?
				.commit_from_file(path)
				.map_err(|e| anyhow::anyhow!("Failed to load model: {}: {e}", path.display()))?;
			Ok((session, ProviderSelection::Auto))
		}
		ProviderSelection::Cpu => {
			let ep = make_ep(ProviderSelection::Cpu)?;
			let session = Session::builder()
				.map_err(|e| anyhow::anyhow!("Failed to create session builder: {e}"))?
				.with_execution_providers([ep])
				.map_err(|e| anyhow::anyhow!("Failed to configure CPU provider: {e}"))?
				.commit_from_file(path)
				.map_err(|e| anyhow::anyhow!("Failed to load model: {}: {e}", path.display()))?;
			Ok((session, ProviderSelection::Cpu))
		}
		requested => {
			// Try the requested provider, fall back to CPU on failure.
			let ep = make_ep(requested)?;
			let try_result = Session::builder()
				.map_err(|e| e.to_string())
				.and_then(|b| b.with_execution_providers([ep]).map_err(|e| e.to_string()))
				.and_then(|mut b| b.commit_from_file(path).map_err(|e| e.to_string()));

			match try_result {
				Ok(session) => Ok((session, requested)),
				Err(e) => {
					warn!(
						"{} provider failed ({}), falling back to CPU",
						requested.name(),
						e
					);
					let ep = make_ep(ProviderSelection::Cpu)?;
					let session = Session::builder()
						.map_err(|e| anyhow::anyhow!("Failed to create session builder: {e}"))?
						.with_execution_providers([ep])
						.map_err(|e| anyhow::anyhow!("Failed to configure CPU fallback: {e}"))?
						.commit_from_file(path)
						.map_err(|e| {
							anyhow::anyhow!(
								"Failed to load model with CPU fallback: {}: {e}",
								path.display()
							)
						})?;
					Ok((session, ProviderSelection::Cpu))
				}
			}
		}
	}
}

/// Create an ORT session from raw model bytes with the given provider.
fn create_session_from_bytes(
	bytes: &[u8],
	provider: ProviderSelection,
) -> Result<(Session, ProviderSelection)> {
	match provider {
		ProviderSelection::Auto => {
			let session = Session::builder()
				.map_err(|e| anyhow::anyhow!("Failed to create session builder: {e}"))?
				.with_auto_device(AutoDevicePolicy::MaxPerformance)
				.map_err(|e| anyhow::anyhow!("Failed to configure auto device: {e}"))?
				.commit_from_memory(bytes)
				.map_err(|e| anyhow::anyhow!("Failed to load model from memory: {e}"))?;
			Ok((session, ProviderSelection::Auto))
		}
		ProviderSelection::Cpu => {
			let ep = make_ep(ProviderSelection::Cpu)?;
			let session = Session::builder()
				.map_err(|e| anyhow::anyhow!("Failed to create session builder: {e}"))?
				.with_execution_providers([ep])
				.map_err(|e| anyhow::anyhow!("Failed to configure CPU provider: {e}"))?
				.commit_from_memory(bytes)
				.map_err(|e| anyhow::anyhow!("Failed to load model from memory: {e}"))?;
			Ok((session, ProviderSelection::Cpu))
		}
		requested => {
			// Try the requested provider, fall back to CPU on failure.
			let ep = make_ep(requested)?;
			let try_result = Session::builder()
				.map_err(|e| e.to_string())
				.and_then(|b| b.with_execution_providers([ep]).map_err(|e| e.to_string()))
				.and_then(|mut b| b.commit_from_memory(bytes).map_err(|e| e.to_string()));

			match try_result {
				Ok(session) => Ok((session, requested)),
				Err(e) => {
					warn!(
						"{} provider failed ({}), falling back to CPU",
						requested.name(),
						e
					);
					let ep = make_ep(ProviderSelection::Cpu)?;
					let session = Session::builder()
						.map_err(|e| anyhow::anyhow!("Failed to create session builder: {e}"))?
						.with_execution_providers([ep])
						.map_err(|e| anyhow::anyhow!("Failed to configure CPU fallback: {e}"))?
						.commit_from_memory(bytes)
						.map_err(|e| {
							anyhow::anyhow!(
								"Failed to load model from memory with CPU fallback: {e}"
							)
						})?;
					Ok((session, ProviderSelection::Cpu))
				}
			}
		}
	}
}
