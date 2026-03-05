//! Minimal hand-rolled protobuf reader for ONNX `ModelProto`.
//!
//! We walk only 4 levels deep to extract:
//!
//! ```text
//! ModelProto (field 7 = graph, field 8 = opset_import, field 14 = metadata_props)
//! └── GraphProto (field 1 = node[], field 11 = input[], field 12 = output[])
//!     └── NodeProto (field 4 = op_type, field 5 = attribute[])
//!         └── AttributeProto (field 1 = name, field 3 = i, field 7 = ints[])
//! ```
//!
//! This avoids `prost` recursion limits and works on any valid ONNX file
//! regardless of large tensor-initialiser blobs.

use std::collections::HashMap;

// ── Public Types ───────────────────────────────────────────────────────────

/// A single graph node with only the attributes we care about.
#[derive(Debug, Clone)]
pub struct NodeInfo {
	pub op_type: String,
	/// `(attr_name, single_int64)` for integer attributes.
	pub int_attrs: Vec<(String, i64)>,
	/// `(attr_name, values)` for packed repeated int64 attributes.
	pub ints_attrs: Vec<(String, Vec<i64>)>,
}

/// I/O tensor info `(channels, (opt_height, opt_width), dtype_string)`.
pub type IoInfo = (u32, (Option<u64>, Option<u64>), String);

// ── Top-level Extraction Helpers ───────────────────────────────────────────

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
		int_attrs,
		ints_attrs,
	})
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
