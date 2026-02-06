use base64::{engine::general_purpose::STANDARD, Engine};
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

/// Generates a thumbnail for the given source image
pub fn generate_thumbnail(source_path: &Path, thumb_path: &Path) -> Result<(), String> {
    let ext = source_path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase())
        .unwrap_or_default();

    // Load and decode - use turbojpeg for JPEGs with scaled decoding
    let (rgba_data, width, height) = if ext == "jpg" || ext == "jpeg" || ext == "jfif" {
        decode_jpeg_for_thumbnail(source_path)?
    } else {
        decode_other_for_thumbnail(source_path)?
    };

    // Calculate target dimensions preserving aspect ratio
    let (target_width, target_height) = calculate_thumbnail_dimensions(width, height, THUMBNAIL_SIZE);

    // Resize using fast_image_resize (SIMD-accelerated)
    let resized_rgba = fast_resize_rgba(&rgba_data, width, height, target_width, target_height)?;

    // Encode as WebP
    let encoder = webp::Encoder::from_rgba(&resized_rgba, target_width, target_height);
    let webp_data = encoder.encode(WEBP_QUALITY);

    // Write to disk
    fs::write(thumb_path, &*webp_data)
        .map_err(|e| format!("Failed to write thumbnail '{}': {}", thumb_path.display(), e))?;

    Ok(())
}

/// Decode a JPEG using turbojpeg with scaled decoding for thumbnails.
fn decode_jpeg_for_thumbnail(path: &Path) -> Result<(Vec<u8>, u32, u32), String> {
    let jpeg_data = fs::read(path)
        .map_err(|e| format!("Failed to read JPEG {}: {}", path.display(), e))?;

    let mut decompressor = turbojpeg::Decompressor::new()
        .map_err(|e| format!("Failed to create JPEG decompressor: {}", e))?;

    let header = decompressor
        .read_header(&jpeg_data)
        .map_err(|e| format!("Failed to read JPEG header {}: {}", path.display(), e))?;

    // Choose scaling factor - we want at least THUMBNAIL_SIZE on the smaller dimension
    let scaling = choose_jpeg_scale(header.width, header.height, THUMBNAIL_SIZE as usize);
    decompressor.set_scaling_factor(scaling);

    let scaled_header = header.scaled(scaling);
    let scaled_width = scaled_header.width;
    let scaled_height = scaled_header.height;

    // Decode to RGBA for WebP encoding
    let mut rgba_data = vec![0u8; scaled_width * scaled_height * 4];
    let image = turbojpeg::Image {
        pixels: &mut rgba_data[..],
        width: scaled_width,
        pitch: scaled_width * 4,
        height: scaled_height,
        format: turbojpeg::PixelFormat::RGBA,
    };

    decompressor
        .decompress(&jpeg_data, image)
        .map_err(|e| format!("Failed to decompress JPEG {}: {}", path.display(), e))?;

    Ok((rgba_data, scaled_width as u32, scaled_height as u32))
}

/// Choose the best JPEG scaling factor for thumbnail generation.
fn choose_jpeg_scale(width: usize, height: usize, target_size: usize) -> turbojpeg::ScalingFactor {
    let min_dim = width.min(height);

    if min_dim / 8 >= target_size {
        turbojpeg::ScalingFactor::ONE_EIGHTH
    } else if min_dim / 4 >= target_size {
        turbojpeg::ScalingFactor::ONE_QUARTER
    } else if min_dim / 2 >= target_size {
        turbojpeg::ScalingFactor::ONE_HALF
    } else {
        turbojpeg::ScalingFactor::ONE
    }
}

/// Decode non-JPEG formats using the image crate.
fn decode_other_for_thumbnail(path: &Path) -> Result<(Vec<u8>, u32, u32), String> {
    let img = image::open(path)
        .map_err(|e| format!("Failed to open image {}: {}", path.display(), e))?;

    let rgba = img.to_rgba8();
    let width = rgba.width();
    let height = rgba.height();
    let rgba_data = rgba.into_raw();

    Ok((rgba_data, width, height))
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

/// Resize RGBA image data using fast_image_resize.
fn fast_resize_rgba(
    rgba_data: &[u8],
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
) -> Result<Vec<u8>, String> {
    if src_width == 0 || src_height == 0 || dst_width == 0 || dst_height == 0 {
        return Err("Invalid image dimensions".to_string());
    }

    // If no resize needed, return original
    if src_width == dst_width && src_height == dst_height {
        return Ok(rgba_data.to_vec());
    }

    let mut rgba_copy = rgba_data.to_vec();
    let src_image = FirImage::from_slice_u8(
        src_width,
        src_height,
        &mut rgba_copy,
        PixelType::U8x4,
    )
    .map_err(|e| format!("Failed to create source image: {}", e))?;

    let mut dst_image = FirImage::new(dst_width, dst_height, PixelType::U8x4);

    let mut resizer = Resizer::new();
    // Use Lanczos3 to match previous quality
    let options = ResizeOptions::new().resize_alg(ResizeAlg::Convolution(
        fast_image_resize::FilterType::Bilinear,
    ));

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
        generate_thumbnail(source_path, &thumb_path)?;
    }

    // Return absolute path as string for asset protocol
    Ok(thumb_path.to_string_lossy().to_string())
}

/// Gets a thumbnail as a base64 data URL, generating it if needed.
/// DEPRECATED: Use get_thumbnail_path_for_asset with convertFileSrc instead for better performance.
#[allow(dead_code)]
pub fn get_thumbnail_base64(thumbnails_dir: &Path, source_path: &Path) -> Result<String, String> {
    let thumb_path = thumbnail_path(thumbnails_dir, source_path);

    // Generate if missing or stale
    if !thumbnail_is_current(&thumb_path, source_path) {
        generate_thumbnail(source_path, &thumb_path)?;
    }

    // Read the thumbnail file
    let data = fs::read(&thumb_path)
        .map_err(|e| format!("Failed to read thumbnail '{}': {}", thumb_path.display(), e))?;

    // Return as base64 data URL
    let base64_data = STANDARD.encode(&data);
    Ok(format!("data:image/webp;base64,{}", base64_data))
}

/// Ensures a thumbnail exists for the given source image, generating if needed
pub fn ensure_thumbnail(thumbnails_dir: &Path, source_path: &Path) -> Result<(), String> {
    let thumb_path = thumbnail_path(thumbnails_dir, source_path);

    if !thumbnail_is_current(&thumb_path, source_path) {
        generate_thumbnail(source_path, &thumb_path)?;
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
