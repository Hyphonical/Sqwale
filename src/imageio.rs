//! Image loading, saving, and path utilities.

use anyhow::{Context, Result};
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
}
