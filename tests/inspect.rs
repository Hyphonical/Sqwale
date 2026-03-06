use std::path::Path;

use sqwale::inspect::inspect_model;

#[test]
fn inspect_compact_model() {
	let path = Path::new("models/2x_OpenProteus_Compact_i2_70K_fp32.onnx");
	if !path.exists() {
		// Skip if models aren't available (e.g. CI without model artifacts).
		return;
	}
	let info = inspect_model(path).expect("inspection should succeed");

	assert_eq!(info.scale, 2);
	assert_eq!(info.input_channels, 3);
	assert_eq!(info.output_channels, 3);
	assert_eq!(info.input_dtype, "float32");
	assert!(info.opset > 0);
	assert!(!info.op_fingerprint.is_empty());
}
