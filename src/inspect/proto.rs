//! Minimal hand-rolled protobuf reader for ONNX `ModelProto`.
//!
//! We walk several levels deep to extract:
//!
//! ```text
//! ModelProto   (field 7=graph, field 8=opset_import, field 14=metadata_props)
//! └── GraphProto (field 1=node[], field 5=initializer[], field 11=input[], field 12=output[])
//!     ├── NodeProto   (field 2=output[], field 4=op_type, field 5=attribute[])
//!     │   └── AttributeProto (field 1=name, field 3=i, field 5/6=t TensorProto, field 7=ints[])
//!     └── TensorProto (field 1=dims[], field 2=data_type, field 4=float_data, field 8=name, field 9=raw_data)
//! ```
//!
//! Notable ONNX proto quirks handled here:
//! - `TensorProto.raw_data` is field 9 (not 16 as sometimes documented).
//! - `TensorProto.name` is field 8.
//! - Some exporters (e.g. PyTorch) store the `Constant` op's tensor at
//!   `AttributeProto` field 5 instead of field 6.
//!
//! This avoids `prost` recursion limits and works on any valid ONNX file
//! regardless of large tensor-initialiser blobs.

use std::collections::HashMap;

// ── Public Types ───────────────────────────────────────────────────────────

/// A single graph node with only the attributes we care about.
#[derive(Debug, Clone)]
pub struct NodeInfo {
	pub op_type: String,
	/// Input tensor names (positional, matching the op spec).
	pub inputs: Vec<String>,
	/// `(attr_name, single_int64)` for integer attributes.
	pub int_attrs: Vec<(String, i64)>,
	/// `(attr_name, values)` for packed repeated int64 attributes.
	pub ints_attrs: Vec<(String, Vec<i64>)>,
}

/// I/O tensor info `(channels, (opt_height, opt_width), dtype_string)`.
pub type IoInfo = (u32, (Option<u64>, Option<u64>), String);

// ── Top-level Extraction Helpers ───────────────────────────────────────────

/// Extract float-typed 1-D tensors as `name → Vec<f32>` from raw model bytes.
///
/// Sources:
/// 1. `GraphProto.initializer` (field 5) — named TensorProtos.
/// 2. `Constant` graph nodes — the tensor is embedded in the `value` attribute;
///    the name comes from the node's single output.
///
/// Covers both `float_data` (field 4) and IEEE 754 `raw_data` (field 9).
/// Tensors with more than one dimension (weight matrices etc.) are skipped.
pub fn extract_float_initializers(file_bytes: &[u8]) -> HashMap<String, Vec<f32>> {
	let gb = graph_bytes(file_bytes);
	let graph_fields = iter_fields(&gb);

	// Named initializers in GraphProto.initializer (field 5)
	// TensorProto: name=field 8, raw_data=field 9, float_data=field 4.
	let mut map: HashMap<String, Vec<f32>> = graph_fields
		.iter()
		.filter(|(f, w, _)| *f == 5 && *w == 2)
		.filter_map(|(_, _, b)| {
			let name = iter_fields(b)
				.iter()
				.find(|(f, w, _)| *f == 8 && *w == 2)
				.and_then(|(_, _, nb)| std::str::from_utf8(nb).ok().map(str::to_owned))?;
			let floats = parse_raw_float_tensor(b)?;
			Some((name, floats))
		})
		.collect();

	// Values from Constant op nodes (field 1 = NodeProto)
	for (f, w, b) in &graph_fields {
		if *f == 1 && *w == 2 {
			if let Some((name, floats)) = parse_constant_float_node(b) {
				map.insert(name, floats);
			}
		}
	}

	map
}

/// Extract all graph nodes from raw model bytes.
pub fn extract_nodes(file_bytes: &[u8]) -> Vec<NodeInfo> {
	let graph = graph_bytes(file_bytes);
	iter_fields(&graph)
		.into_iter()
		.filter(|(f, w, _)| *f == 1 && *w == 2)
		.filter_map(|(_, _, b)| parse_node(&b))
		.collect()
}

/// Extract the maximum opset version from raw model bytes.
pub fn extract_opset(file_bytes: &[u8]) -> u64 {
	// ModelProto field 8 = opset_import[] → OperatorSetIdProto field 2 = version
	iter_fields(file_bytes)
		.into_iter()
		.filter(|(f, w, _)| *f == 8 && *w == 2)
		.filter_map(|(_, _, b)| {
			iter_fields(&b)
				.into_iter()
				.find(|(f, w, _)| *f == 2 && *w == 0)
				.map(|(_, _, vb)| bytes_as_i64(&vb) as u64)
		})
		.max()
		.unwrap_or(0)
}

/// Extract `metadata_props` key→value pairs.
pub fn extract_metadata(file_bytes: &[u8]) -> HashMap<String, String> {
	// ModelProto field 14 = metadata_props[] (StringStringEntryProto)
	iter_fields(file_bytes)
		.into_iter()
		.filter(|(f, w, _)| *f == 14 && *w == 2)
		.filter_map(|(_, _, b)| {
			let kv = iter_fields(&b);
			let key = kv
				.iter()
				.find(|(f, w, _)| *f == 1 && *w == 2)
				.and_then(|(_, _, b)| std::str::from_utf8(b).ok().map(str::to_owned))?;
			let val = kv
				.iter()
				.find(|(f, w, _)| *f == 2 && *w == 2)
				.and_then(|(_, _, b)| std::str::from_utf8(b).ok().map(str::to_owned))
				.unwrap_or_default();
			Some((key, val))
		})
		.collect()
}

/// Extract first input and first output `IoInfo` from raw model bytes.
pub fn extract_io_from_proto(file_bytes: &[u8]) -> (Option<IoInfo>, Option<IoInfo>) {
	let gb = graph_bytes(file_bytes);
	let graph_fields = iter_fields(&gb);

	// GraphProto field 11 = input[], field 12 = output[]
	let input = graph_fields
		.iter()
		.find(|(f, w, _)| *f == 11 && *w == 2)
		.and_then(|(_, _, b)| parse_value_info(b));

	let output = graph_fields
		.iter()
		.find(|(f, w, _)| *f == 12 && *w == 2)
		.and_then(|(_, _, b)| parse_value_info(b));

	(input, output)
}

// ── Internal Proto Walking ─────────────────────────────────────────────────

fn graph_bytes(file_bytes: &[u8]) -> Vec<u8> {
	// ModelProto field 7 = graph (GraphProto)
	iter_fields(file_bytes)
		.into_iter()
		.find(|(f, w, _)| *f == 7 && *w == 2)
		.map(|(_, _, b)| b)
		.unwrap_or_default()
}

fn iter_fields(buf: &[u8]) -> Vec<(u32, u8, Vec<u8>)> {
	let mut out = Vec::new();
	let mut pos = 0;

	while pos < buf.len() {
		let Some((tag, consumed)) = read_varint_at(buf, pos) else {
			break;
		};
		pos += consumed;

		let field_num = (tag >> 3) as u32;
		let wire_type = (tag & 0x7) as u8;

		match wire_type {
			0 => {
				let Some((val, consumed)) = read_varint_at(buf, pos) else {
					break;
				};
				pos += consumed;
				out.push((field_num, wire_type, val.to_le_bytes().to_vec()));
			}
			1 => {
				if pos + 8 > buf.len() {
					break;
				}
				out.push((field_num, wire_type, buf[pos..pos + 8].to_vec()));
				pos += 8;
			}
			2 => {
				let Some((len, consumed)) = read_varint_at(buf, pos) else {
					break;
				};
				pos += consumed;
				let len = len as usize;
				if pos + len > buf.len() {
					break;
				}
				out.push((field_num, wire_type, buf[pos..pos + len].to_vec()));
				pos += len;
			}
			5 => {
				if pos + 4 > buf.len() {
					break;
				}
				out.push((field_num, wire_type, buf[pos..pos + 4].to_vec()));
				pos += 4;
			}
			_ => break,
		}
	}

	out
}

fn read_varint_at(buf: &[u8], mut pos: usize) -> Option<(u64, usize)> {
	let mut result: u64 = 0;
	let mut shift = 0u32;
	let start = pos;
	loop {
		if pos >= buf.len() || shift >= 64 {
			return None;
		}
		let b = buf[pos];
		pos += 1;
		result |= ((b & 0x7F) as u64) << shift;
		if b & 0x80 == 0 {
			break;
		}
		shift += 7;
	}
	Some((result, pos - start))
}

fn bytes_as_i64(bytes: &[u8]) -> i64 {
	let mut arr = [0u8; 8];
	let n = bytes.len().min(8);
	arr[..n].copy_from_slice(&bytes[..n]);
	i64::from_le_bytes(arr)
}

fn bytes_as_packed_i64s(bytes: &[u8]) -> Vec<i64> {
	let mut out = Vec::new();
	let mut pos = 0;
	while pos < bytes.len() {
		if let Some((val, consumed)) = read_varint_at(bytes, pos) {
			out.push(val as i64);
			pos += consumed;
		} else {
			break;
		}
	}
	out
}

fn parse_node(bytes: &[u8]) -> Option<NodeInfo> {
	let fields = iter_fields(bytes);

	let op_type = fields
		.iter()
		.find(|(f, w, _)| *f == 4 && *w == 2)
		.and_then(|(_, _, b)| std::str::from_utf8(b).ok().map(str::to_owned))?;

	// NodeProto field 1 = input (repeated string)
	let inputs: Vec<String> = fields
		.iter()
		.filter(|(f, w, _)| *f == 1 && *w == 2)
		.filter_map(|(_, _, b)| std::str::from_utf8(b).ok().map(str::to_owned))
		.collect();

	let mut int_attrs: Vec<(String, i64)> = Vec::new();
	let mut ints_attrs: Vec<(String, Vec<i64>)> = Vec::new();

	for (field_num, wire_type, bytes) in &fields {
		if *field_num != 5 || *wire_type != 2 {
			continue;
		}
		let attr_fields = iter_fields(bytes);
		let name = attr_fields
			.iter()
			.find(|(f, w, _)| *f == 1 && *w == 2)
			.and_then(|(_, _, b)| std::str::from_utf8(b).ok().map(str::to_owned))
			.unwrap_or_default();

		for (af, aw, ab) in &attr_fields {
			match (af, aw) {
				(3, 0) => int_attrs.push((name.clone(), bytes_as_i64(ab))),
				(7, 2) => ints_attrs.push((name.clone(), bytes_as_packed_i64s(ab))),
				_ => {}
			}
		}
	}

	Some(NodeInfo {
		op_type,
		inputs,
		int_attrs,
		ints_attrs,
	})
}

/// Try to decode a `TensorProto` as a 1-D (or scalar) `Vec<f32>`.
///
/// Returns `None` if the tensor is not float32 or has more than one dimension.
/// Handles both `raw_data` (field 9) and packed `float_data` (field 4).
fn parse_raw_float_tensor(bytes: &[u8]) -> Option<Vec<f32>> {
	let fields = iter_fields(bytes);

	// TensorProto field 2 = data_type (int32): 1 = float32
	let data_type = fields
		.iter()
		.find(|(f, w, _)| *f == 2 && *w == 0)
		.map(|(_, _, b)| bytes_as_i64(b) as i32)
		.unwrap_or(0);
	if data_type != 1 {
		return None;
	}

	// Count rank (field 1 = dims, repeated varint — one entry per dimension).
	// Skip weight matrices and other multi-D tensors.
	let n_dims = fields.iter().filter(|(f, w, _)| *f == 1 && *w == 0).count();
	if n_dims > 1 {
		return None;
	}

	// raw_data (field 9 per ONNX TensorProto spec): IEEE 754 LE bytes, 4 per float.
	if let Some((_, _, raw)) = fields.iter().find(|(f, w, _)| *f == 9 && *w == 2) {
		if raw.len() % 4 == 0 {
			return Some(
				raw.chunks_exact(4)
					.map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
					.collect(),
			);
		}
	}

	// float_data (field 4): packed repeated float32.
	if let Some((_, _, packed)) = fields.iter().find(|(f, w, _)| *f == 4 && *w == 2) {
		if packed.len() % 4 == 0 {
			return Some(
				packed
					.chunks_exact(4)
					.map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
					.collect(),
			);
		}
	}

	None
}

/// Extract the float tensor from a `Constant` `NodeProto`.
///
/// Returns `(output_name, floats)` when the node has `op_type = "Constant"`,
/// a single string output, and a `value` attribute containing a float32 1-D
/// tensor (`AttributeProto.t`, field 6).
fn parse_constant_float_node(bytes: &[u8]) -> Option<(String, Vec<f32>)> {
	let fields = iter_fields(bytes);

	let op_type = fields
		.iter()
		.find(|(f, w, _)| *f == 4 && *w == 2)
		.and_then(|(_, _, b)| std::str::from_utf8(b).ok())?;
	if op_type != "Constant" {
		return None;
	}

	// NodeProto field 2 = output (repeated string)
	let output_name: String = fields
		.iter()
		.find(|(f, w, _)| *f == 2 && *w == 2)
		.and_then(|(_, _, b)| std::str::from_utf8(b).ok().map(str::to_owned))?;

	// NodeProto field 5 = attribute (repeated AttributeProto)
	for (af, aw, ab) in &fields {
		if *af != 5 || *aw != 2 {
			continue;
		}
		let attr_fields = iter_fields(ab);

		// AttributeProto field 1 = name, must be "value"
		let attr_name = attr_fields
			.iter()
			.find(|(f, w, _)| *f == 1 && *w == 2)
			.and_then(|(_, _, b)| std::str::from_utf8(b).ok());
		if attr_name != Some("value") {
			continue;
		}

		// AttributeProto field 6 = t (TensorProto) per the ONNX spec.
		// Some ONNX exporters (e.g. PyTorch) store the tensor at field 5 instead.
		// Try both; parse_raw_float_tensor validates the content so false positives
		// are caught by the data_type check inside it.
		for &tensor_field in &[6u32, 5u32] {
			if let Some((_, _, tensor_bytes)) = attr_fields
				.iter()
				.find(|(f, w, _)| *f == tensor_field && *w == 2)
			{
				if let Some(floats) = parse_raw_float_tensor(tensor_bytes) {
					return Some((output_name, floats));
				}
			}
		}
	}

	None
}

fn parse_value_info(bytes: &[u8]) -> Option<IoInfo> {
	// ValueInfoProto field 2 = type (TypeProto)
	let type_bytes = iter_fields(bytes)
		.into_iter()
		.find(|(f, w, _)| *f == 2 && *w == 2)
		.map(|(_, _, b)| b)?;

	// TypeProto field 1 = tensor_type
	let tensor_bytes = iter_fields(&type_bytes)
		.into_iter()
		.find(|(f, w, _)| *f == 1 && *w == 2)
		.map(|(_, _, b)| b)?;

	let tensor_fields = iter_fields(&tensor_bytes);

	let elem_type = tensor_fields
		.iter()
		.find(|(f, w, _)| *f == 1 && *w == 0)
		.map(|(_, _, b)| bytes_as_i64(b) as i32)
		.unwrap_or(0);

	let dtype = elem_type_name(elem_type).to_owned();

	let shape_bytes = tensor_fields
		.iter()
		.find(|(f, w, _)| *f == 2 && *w == 2)
		.map(|(_, _, b)| b.clone())?;

	let dims: Vec<Option<u64>> = iter_fields(&shape_bytes)
		.into_iter()
		.filter(|(f, w, _)| *f == 1 && *w == 2)
		.map(|(_, _, b)| {
			iter_fields(&b)
				.into_iter()
				.find(|(f, w, _)| *f == 1 && *w == 0)
				.map(|(_, _, vb)| bytes_as_i64(&vb) as u64)
		})
		.collect();

	if dims.len() < 4 {
		return None;
	}

	let channels: u32 = dims[1].unwrap_or(0).try_into().ok()?;
	if channels == 0 {
		return None;
	}

	let h = dims[2].filter(|&v| v > 0);
	let w = dims[3].filter(|&v| v > 0);

	Some((channels, (h, w), dtype))
}

fn elem_type_name(t: i32) -> &'static str {
	match t {
		1 => "float32",
		2 => "uint8",
		3 => "int8",
		5 => "int32",
		6 => "int64",
		10 => "float16",
		11 => "float64",
		16 => "bfloat16",
		_ => "unknown",
	}
}
