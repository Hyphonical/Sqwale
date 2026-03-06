//! Image loading, saving, and path utilities.

use anyhow::{Context, Result, bail};
use image::DynamicImage;
use std::path::{Path, PathBuf};

/// Load an image from disk.
pub fn load_image(path: &Path) -> Result<DynamicImage> {
	image::open(path).with_context(|| format!("Failed to load image: {}", path.display()))
}

/// Save an image to disk, inferring format from the file extension.
pub fn save_image(img: &DynamicImage, path: &Path) -> Result<()> {
	if let Some(parent) = path.parent() {
		if !parent.exists() {
			std::fs::create_dir_all(parent).with_context(|| {
				format!("Failed to create output directory: {}", parent.display())
			})?;
		}
	}
	img.save(path)
		.with_context(|| format!("Failed to save image: {}", path.display()))
}

/// Derive a default output path: `{dir}/{stem}_{scale}x.{ext}`.
pub fn default_output_path(input: &Path, scale: u32) -> PathBuf {
	let stem = input.file_stem().unwrap_or_default().to_string_lossy();
	let ext = input.extension().unwrap_or_default().to_string_lossy();
	let parent = input.parent().unwrap_or_else(|| Path::new("."));
	parent.join(format!("{stem}_{scale}x.{ext}"))
}

/// Validate that the output path's extension matches the input's.
///
/// Returns `Ok` if extensions match (case-insensitive) or if the output
/// has no extension (treated as a directory). Errors on mismatch because
/// format conversion is not supported.
pub fn check_extension_match(input: &Path, output: &Path) -> Result<()> {
	let out_ext = match output.extension() {
		Some(e) => e.to_string_lossy().to_lowercase(),
		None => return Ok(()), // No extension → directory, OK
	};
	let in_ext = input
		.extension()
		.unwrap_or_default()
		.to_string_lossy()
		.to_lowercase();

	if in_ext == out_ext {
		Ok(())
	} else {
		bail!(
			"Output extension '.{out_ext}' does not match input '.{in_ext}'. \
			 Format conversion is not supported — use the same extension."
		)
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn default_output_path_basic() {
		let input = Path::new("images/photo.png");
		let result = default_output_path(input, 2);
		assert_eq!(result, PathBuf::from("images/photo_2x.png"));
	}

	#[test]
	fn default_output_path_4x() {
		let input = Path::new("test.jpg");
		let result = default_output_path(input, 4);
		assert_eq!(result.file_name().unwrap().to_str().unwrap(), "test_4x.jpg");
	}

	#[test]
	fn extension_match_same() {
		assert!(check_extension_match(Path::new("a.png"), Path::new("b.png")).is_ok());
	}

	#[test]
	fn extension_match_case_insensitive() {
		assert!(check_extension_match(Path::new("a.PNG"), Path::new("b.png")).is_ok());
	}

	#[test]
	fn extension_match_no_output_ext() {
		assert!(check_extension_match(Path::new("a.png"), Path::new("output_dir")).is_ok());
	}

	#[test]
	fn extension_mismatch() {
		assert!(check_extension_match(Path::new("a.png"), Path::new("b.jpg")).is_err());
	}
}
