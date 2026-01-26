use base64::{engine::general_purpose::STANDARD, Engine};
use image::imageops::FilterType;
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
    // Load the source image
    let img = image::open(source_path)
        .map_err(|e| format!("Failed to open image '{}': {}", source_path.display(), e))?;

    // Resize to fit within THUMBNAIL_SIZE x THUMBNAIL_SIZE, preserving aspect ratio
    let thumbnail = img.resize(THUMBNAIL_SIZE, THUMBNAIL_SIZE, FilterType::Lanczos3);

    // Encode as WebP
    let rgba = thumbnail.to_rgba8();
    let encoder = webp::Encoder::from_rgba(&rgba, rgba.width(), rgba.height());
    let webp_data = encoder.encode(WEBP_QUALITY);

    // Ensure parent directory exists
    if let Some(parent) = thumb_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create thumbnails directory: {}", e))?;
    }

    // Write to disk
    fs::write(thumb_path, &*webp_data)
        .map_err(|e| format!("Failed to write thumbnail '{}': {}", thumb_path.display(), e))?;

    Ok(())
}

/// Gets a thumbnail as a base64 data URL, generating it if needed
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
