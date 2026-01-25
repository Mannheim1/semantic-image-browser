use base64::{engine::general_purpose::STANDARD, Engine};
use image::imageops::FilterType;
use image::ImageReader;
use std::io::Cursor;
use std::path::Path;

const THUMBNAIL_SIZE: u32 = 256;

pub fn get_thumbnail_base64(path: &Path, _requested_size: u32) -> Result<String, String> {
    // Load and decode the image
    let img = ImageReader::open(path)
        .map_err(|e| format!("Failed to open image '{}': {}", path.display(), e))?
        .with_guessed_format()
        .map_err(|e| format!("Failed to guess format: {}", e))?
        .decode()
        .map_err(|e| format!("Failed to decode image '{}': {}", path.display(), e))?;

    // Resize to thumbnail (maintain aspect ratio, fit within bounds)
    let thumbnail = img.thumbnail(THUMBNAIL_SIZE, THUMBNAIL_SIZE);

    // Encode to PNG in memory
    let mut png_bytes: Vec<u8> = Vec::new();
    thumbnail
        .write_to(&mut Cursor::new(&mut png_bytes), image::ImageFormat::Png)
        .map_err(|e| format!("Failed to encode PNG: {}", e))?;

    // Return as base64 data URL
    let base64_data = STANDARD.encode(&png_bytes);
    Ok(format!("data:image/png;base64,{}", base64_data))
}
