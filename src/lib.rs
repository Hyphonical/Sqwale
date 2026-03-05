//! sqwale — ONNX super-resolution inference library
//!
//! This crate provides both a library API and a CLI tool for running ONNX-based
//! super-resolution models. It focuses on flexibility, performance, and ease of
//! integration with other Rust projects.

// ── Public API Exports ─────────────────────────────────────────────────────

// Inspection
pub use inspect::{inspect_model, ColorSpace, ModelInfo, ScaleSource, TileInfo};

// ── Internal Modules ───────────────────────────────────────────────────────

pub mod inspect;
