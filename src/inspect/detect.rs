//! Model detection logic: scale, channels, tiling, and color space inference.

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

use super::proto::{self, IoInfo, NodeInfo};
use super::{ColorSpace, ModelInfo, ScaleSource, TileInfo};

type IoParseResult = (u32, (Option<u64>, Option<u64>), String);

/// Inspect an ONNX model and extract all metadata without creating an inference session.
pub fn inspect_model(path: &Path) -> Result<ModelInfo> {
	let file_bytes =
		fs::read(path).with_context(|| format!("Failed to read model: {}", path.display()))?;
	inspect_model_bytes(&file_bytes)
}

/// Inspect an ONNX model from raw bytes and extract all metadata.
pub fn inspect_model_bytes(file_bytes: &[u8]) -> Result<ModelInfo> {
	let nodes = proto::extract_nodes(file_bytes);
	let opset = proto::extract_opset(file_bytes);
	let metadata = proto::extract_metadata(file_bytes);
	let float_inits = proto::extract_float_initializers(file_bytes);
	let (input_io, output_io) = proto::extract_io_from_proto(file_bytes);

	let (input_channels, input_spatial, input_dtype) = parse_io(input_io)?;
	let (output_channels, output_spatial, output_dtype) =
		parse_io(output_io).unwrap_or_else(|_| (input_channels, (None, None), input_dtype.clone()));

	let (scale, scale_source) = detect_scale(
		&nodes,
		&metadata,
		&float_inits,
		input_spatial,
		output_spatial,
	);
	let tile = detect_tiling(&nodes, input_spatial);
	let color_space = infer_color_space(input_channels);
	let op_fingerprint = compute_op_fingerprint(&nodes);

	Ok(ModelInfo {
		scale,
		scale_source,
		color_space,
		input_channels,
		output_channels,
		tile,
		input_dtype,
		output_dtype,
		opset,
		op_fingerprint,
	})
}

fn parse_io(io: Option<IoInfo>) -> Result<IoParseResult> {
	let (channels, spatial, dtype) = io.ok_or_else(|| anyhow::anyhow!("Missing I/O info"))?;
	Ok((channels, spatial, dtype))
}

fn infer_color_space(channels: u32) -> ColorSpace {
	match channels {
		1 => ColorSpace::Grayscale,
		3 => ColorSpace::Rgb,
		4 => ColorSpace::Rgba,
		n => ColorSpace::Unknown(n),
	}
}

fn detect_scale(
	nodes: &[NodeInfo],
	metadata: &HashMap<String, String>,
	float_inits: &HashMap<String, Vec<f32>>,
	input_spatial: (Option<u64>, Option<u64>),
	output_spatial: (Option<u64>, Option<u64>),
) -> (u32, ScaleSource) {
	if let Some(scale_str) = metadata.get("scale") {
		if let Ok(scale) = scale_str.parse::<u32>() {
			return (scale, ScaleSource::Metadata);
		}
	}

	if let (Some(in_h), Some(out_h)) = (input_spatial.0, output_spatial.0) {
		if in_h > 0 && out_h > 0 && out_h >= in_h {
			let ratio = (out_h / in_h) as u32;
			if ratio > 1 {
				return (ratio, ScaleSource::StaticShapeRatio);
			}
		}
	}

	let mut resize_product: u32 = 1;
	for node in nodes {
		match node.op_type.as_str() {
			"DepthToSpace" => {
				if let Some((_, scale)) =
					node.int_attrs.iter().find(|(name, _)| name == "blocksize")
				{
					return (*scale as u32, ScaleSource::DepthToSpace);
				}
			}
			"ConvTranspose" => {
				if let Some((_, strides)) =
					node.ints_attrs.iter().find(|(name, _)| name == "strides")
				{
					if let Some(&stride) = strides.iter().max() {
						if stride > 1 {
							return (stride as u32, ScaleSource::ConvTransposeStride);
						}
					}
				}
			}
			// Resize op: input[2] = scales tensor (optional), input[3] = sizes tensor.
			// Models sometimes chain multiple Resize nodes (e.g. two 2x nodes for 4x total).
			// Accumulate the product so we report the overall upscale factor.
			"Resize" => {
				if let Some(scale) = detect_resize_scale(node, float_inits) {
					resize_product = resize_product.saturating_mul(scale);
				}
			}
			_ => {}
		}
	}

	if resize_product > 1 {
		return (resize_product, ScaleSource::Resize);
	}

	(1, ScaleSource::Assumed)
}

/// Try to read the spatial scale from a Resize node's scales initializer.
///
/// Returns `Some(scale)` when the node's scales input (index 2) resolves to a
/// float tensor that contains at least one whole-number spatial scale ≥ 2.
/// The maximum value in the tensor is used, so vectors of any length work:
/// `[1, 1, 4, 4]`, `[4, 4]`, or `[1, 4, 4]` all yield `Some(4)`.
fn detect_resize_scale(node: &NodeInfo, float_inits: &HashMap<String, Vec<f32>>) -> Option<u32> {
	// inputs: [data, roi, scales, sizes] — scales is index 2.
	let scales_name = node.inputs.get(2).filter(|s| !s.is_empty())?;
	let scales = float_inits.get(scales_name.as_str())?;
	if scales.is_empty() {
		return None;
	}
	// The spatial scale is the maximum value; batch/channel scales are 1.0.
	let max_scale = scales.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
	if max_scale < 2.0 || max_scale.fract() != 0.0 {
		return None;
	}
	Some(max_scale as u32)
}

fn detect_tiling(nodes: &[NodeInfo], input_spatial: (Option<u64>, Option<u64>)) -> TileInfo {
	let has_dynamic_spatial = input_spatial.0.is_none() || input_spatial.1.is_none();

	let fixed_size = if let (Some(h), Some(w)) = input_spatial {
		Some((h, w))
	} else {
		None
	};

	let alignment = detect_alignment(nodes);

	TileInfo {
		supported: has_dynamic_spatial,
		alignment,
		fixed_size,
	}
}

fn detect_alignment(nodes: &[NodeInfo]) -> Option<u32> {
	for node in nodes {
		if node.op_type == "Reshape" {
			if let Some((_, values)) = node.ints_attrs.iter().find(|(name, _)| name == "shape") {
				for &val in values {
					if val > 1 && val <= 64 && (val & (val - 1)) == 0 {
						return Some(val as u32);
					}
				}
			}
		}
	}
	None
}

fn compute_op_fingerprint(nodes: &[NodeInfo]) -> Vec<(String, usize)> {
	let mut counts: HashMap<String, usize> = HashMap::new();
	for node in nodes {
		*counts.entry(node.op_type.clone()).or_insert(0) += 1;
	}
	let mut pairs: Vec<_> = counts.into_iter().collect();
	pairs.sort_by(|a, b| b.1.cmp(&a.1));
	pairs
}
