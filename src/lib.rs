//! sqwale — ONNX super-resolution inference library
//!
//! This crate provides both a library API and a CLI tool for running ONNX-based
//! super-resolution models. It focuses on flexibility, performance, and ease of
//! integration with other Rust projects.
//!
//! # Quick Start
//!
//! ## Inspecting Models
//!
//! Use [`inspect_model`] to extract metadata from an ONNX model without creating an inference session:
//!
//! ```no_run
//! use sqwale::inspect_model;
//! use std::path::Path;
//!
//! let info = inspect_model(Path::new("model.onnx"))?;
//! println!("Scale: {}x", info.scale);
//! println!("Input dtype: {}", info.input_dtype);
//! # Ok::<(), anyhow::Error>(())
//! ```
//!
//! ## Upscaling Images
//!
//! Create an [`UpscaleSession`] to upscale images. The session can be reused for multiple images:
//!
//! ```no_run
//! use sqwale::{UpscaleSession, UpscaleOptions, Provider};
//! use image::open;
//! use std::path::Path;
//!
//! let options = UpscaleOptions {
//!     provider: Provider::Cpu,
//! };
//!
//! let session = UpscaleSession::new(Path::new("model.onnx"), &options)?;
//! let input_image = open("input.png")?;
//! let upscaled = session.upscale(input_image)?;
//! upscaled.save("output.png")?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## GPU Acceleration
//!
//! Specify a GPU provider for hardware acceleration. The library automatically falls back to CPU if the provider is unavailable:
//!
//! ```no_run
//! use sqwale::{UpscaleSession, UpscaleOptions, Provider};
//! use std::path::Path;
//!
//! let options = UpscaleOptions {
//!     provider: Provider::Cuda, // Falls back to CPU if CUDA unavailable
//! };
//!
//! let session = UpscaleSession::new(Path::new("model.onnx"), &options)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## FP16 Models
//!
//! The library automatically handles fp16 models by converting between fp32 and fp16 as needed:
//!
//! ```no_run
//! use sqwale::{inspect_model, UpscaleSession, UpscaleOptions, Provider};
//! use std::path::Path;
//!
//! let info = inspect_model(Path::new("model_fp16.onnx"))?;
//! assert_eq!(info.input_dtype, "float16");
//!
//! // Session automatically handles fp32 ↔ fp16 conversion
//! let options = UpscaleOptions { provider: Provider::Cpu };
//! let session = UpscaleSession::new(Path::new("model_fp16.onnx"), &options)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

// ── Public API Exports ─────────────────────────────────────────────────────

// Inspection
pub use inspect::{inspect_model, ColorSpace, ModelInfo, ScaleSource, TileInfo};

// Upscaling
pub use upscale::{Provider, UpscaleOptions, UpscaleSession};

// ── Internal Modules ───────────────────────────────────────────────────────

pub mod inspect;
pub mod upscale;
