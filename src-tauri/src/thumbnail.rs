use fast_image_resize::{images::Image as FirImage, ResizeAlg, ResizeOptions, Resizer, PixelType};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};

const THUMBNAIL_SIZE: u32 = 256;
const WEBP_QUALITY: f32 = 80.0;

/// Computes the SHA-256 hash of a path string, returns hex-encoded
fn path_to_hash(path: &Path) -> String {
    let path_str = path.to_string_lossy();
    let mut hasher = Sha256::new();
    hasher.update(path_str.as_bytes());
    let result = hasher.finalize();
    hex::encode(result)
}

/// Returns the thumbnail file path for a given source image
pub fn thumbnail_path(thumbnails_dir: &Path, source_path: &Path) -> PathBuf {
    let hash = path_to_hash(source_path);
    thumbnails_dir.join(format!("{}.webp", hash))
}

/// Checks if a thumbnail exists and is up-to-date
pub fn thumbnail_is_current(thumb_path: &Path, source_path: &Path) -> bool {
    let thumb_meta = match fs::metadata(thumb_path) {
        Ok(m) => m,
        Err(_) => return false,
    };

    let source_meta = match fs::metadata(source_path) {
        Ok(m) => m,
        Err(_) => return false,
    };

    let thumb_mtime = thumb_meta.modified().ok();
    let source_mtime = source_meta.modified().ok();

    match (thumb_mtime, source_mtime) {
        (Some(t), Some(s)) => t >= s,
        _ => false,
    }
}

/// Generates a thumbnail for the given source image by decoding it from disk.
/// Optionally accepts a reusable Resizer to avoid per-image allocation in batch processing.
pub fn generate_thumbnail(source_path: &Path, thumb_path: &Path, resizer: Option<&mut Resizer>) -> Result<(), String> {
    let (rgb_data, width, height) = crate::embedding::decode_image_to_rgb(source_path)?;
    generate_thumbnail_from_rgb(&rgb_data, width, height, thumb_path, resizer)
}

/// Generates a thumbnail from pre-decoded RGB pixel data.
/// Used in the merged scan pipeline to avoid decoding the same image twice.
pub fn generate_thumbnail_from_rgb(
    rgb_data: &[u8],
    width: u32,
    height: u32,
    thumb_path: &Path,
    resizer: Option<&mut Resizer>,
) -> Result<(), String> {
    // Calculate target dimensions preserving aspect ratio
    let (target_width, target_height) = calculate_thumbnail_dimensions(width, height, THUMBNAIL_SIZE);

    // Resize using fast_image_resize (SIMD-accelerated)
    let resized_rgb = fast_resize_rgb(rgb_data, width, height, target_width, target_height, resizer)?;

    // Encode as WebP
    let encoder = webp::Encoder::from_rgb(&resized_rgb, target_width, target_height);
    let webp_data = encoder.encode(WEBP_QUALITY);

    // Write to disk
    fs::write(thumb_path, &*webp_data)
        .map_err(|e| format!("Failed to write thumbnail '{}': {}", thumb_path.display(), e))?;

    Ok(())
}

/// Calculate thumbnail dimensions preserving aspect ratio.
fn calculate_thumbnail_dimensions(width: u32, height: u32, max_size: u32) -> (u32, u32) {
    if width <= max_size && height <= max_size {
        return (width, height);
    }

    let aspect = width as f32 / height as f32;
    if width > height {
        (max_size, (max_size as f32 / aspect).round() as u32)
    } else {
        ((max_size as f32 * aspect).round() as u32, max_size)
    }
}

/// Resize RGB image data using fast_image_resize (SIMD-accelerated, bilinear).
/// Optionally accepts a reusable Resizer to avoid per-image allocation in batch processing.
pub fn fast_resize_rgb(
    rgb_data: &[u8],
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    resizer: Option<&mut Resizer>,
) -> Result<Vec<u8>, String> {
    if src_width == 0 || src_height == 0 || dst_width == 0 || dst_height == 0 {
        return Err("Invalid image dimensions".to_string());
    }

    // If no resize needed, return original
    if src_width == dst_width && src_height == dst_height {
        return Ok(rgb_data.to_vec());
    }

    let mut rgb_copy = rgb_data.to_vec();
    let src_image = FirImage::from_slice_u8(
        src_width,
        src_height,
        &mut rgb_copy,
        PixelType::U8x3,
    )
    .map_err(|e| format!("Failed to create source image: {}", e))?;

    let mut dst_image = FirImage::new(dst_width, dst_height, PixelType::U8x3);

    let options = ResizeOptions::new().resize_alg(ResizeAlg::Convolution(
        fast_image_resize::FilterType::Bilinear,
    ));

    let mut fallback_resizer;
    let resizer = match resizer {
        Some(r) => r,
        None => {
            fallback_resizer = Resizer::new();
            &mut fallback_resizer
        }
    };

    resizer
        .resize(&src_image, &mut dst_image, Some(&options))
        .map_err(|e| format!("Failed to resize image: {}", e))?;

    Ok(dst_image.into_vec())
}

/// Gets the thumbnail file path for a source image, generating the thumbnail if needed.
/// Returns the absolute path to the thumbnail file for use with Tauri's asset protocol.
pub fn get_thumbnail_path_for_asset(thumbnails_dir: &Path, source_path: &Path) -> Result<String, String> {
    let thumb_path = thumbnail_path(thumbnails_dir, source_path);

    // Generate if missing or stale
    if !thumbnail_is_current(&thumb_path, source_path) {
        generate_thumbnail(source_path, &thumb_path, None)?;
    }

    // Return absolute path as string for asset protocol
    Ok(thumb_path.to_string_lossy().to_string())
}

/// Ensures a thumbnail exists for the given source image, generating if needed.
/// Optionally accepts a reusable Resizer to avoid per-image allocation in batch processing.
pub fn ensure_thumbnail(thumbnails_dir: &Path, source_path: &Path, resizer: Option<&mut Resizer>) -> Result<(), String> {
    let thumb_path = thumbnail_path(thumbnails_dir, source_path);

    if !thumbnail_is_current(&thumb_path, source_path) {
        generate_thumbnail(source_path, &thumb_path, resizer)?;
    }

    Ok(())
}

/// Deletes the thumbnail for a given source image
pub fn delete_thumbnail(thumbnails_dir: &Path, source_path: &Path) -> Result<(), String> {
    let thumb_path = thumbnail_path(thumbnails_dir, source_path);

    if thumb_path.exists() {
        fs::remove_file(&thumb_path)
            .map_err(|e| format!("Failed to delete thumbnail '{}': {}", thumb_path.display(), e))?;
    }

    Ok(())
}

/// Deletes thumbnails for multiple source images
pub fn delete_thumbnails(thumbnails_dir: &Path, source_paths: &[String]) -> Result<(), String> {
    for path_str in source_paths {
        let source_path = Path::new(path_str);
        // Ignore errors for individual deletions - the file might already be gone
        let _ = delete_thumbnail(thumbnails_dir, source_path);
    }
    Ok(())
}
