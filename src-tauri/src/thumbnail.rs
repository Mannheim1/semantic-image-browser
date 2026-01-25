use image::imageops::FilterType;
use image::ImageReader;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;
use tauri::{AppHandle, Manager};

const THUMBNAIL_SIZE: u32 = 256;

pub fn thumbnails_dir(app: &AppHandle) -> Result<PathBuf, String> {
    let app_data = app.path().app_local_data_dir().map_err(|e| e.to_string())?;
    Ok(app_data.join("thumbnails"))
}

fn cache_key(path: &Path) -> String {
    let mut hasher = Sha256::new();
    hasher.update(path.to_string_lossy().as_bytes());
    let result = hasher.finalize();
    format!("{:x}", result)
}

pub fn thumbnail_path(app: &AppHandle, source_path: &Path) -> Result<PathBuf, String> {
    let dir = thumbnails_dir(app)?;
    let key = cache_key(source_path);
    Ok(dir.join(format!("{}.jpg", key)))
}

pub fn thumbnail_exists(
    app: &AppHandle,
    source_path: &Path,
    source_modified: SystemTime,
) -> Result<bool, String> {
    let thumb_path = thumbnail_path(app, source_path)?;
    if !thumb_path.exists() {
        return Ok(false);
    }

    let thumb_meta = fs::metadata(&thumb_path).map_err(|e| e.to_string())?;
    let thumb_modified = thumb_meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);

    Ok(thumb_modified >= source_modified)
}

pub fn generate_thumbnail(source_path: &Path, dest_path: &Path) -> Result<(), String> {
    if let Some(parent) = dest_path.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }

    let img = ImageReader::open(source_path)
        .map_err(|e| e.to_string())?
        .with_guessed_format()
        .map_err(|e| e.to_string())?
        .decode()
        .map_err(|e| e.to_string())?;

    let thumbnail = img.resize(THUMBNAIL_SIZE, THUMBNAIL_SIZE, FilterType::Lanczos3);

    thumbnail.save(dest_path).map_err(|e| e.to_string())
}

pub fn cleanup_orphans(
    app: &AppHandle,
    valid_hashes: &std::collections::HashSet<String>,
) -> Result<u32, String> {
    let dir = thumbnails_dir(app)?;
    if !dir.exists() {
        return Ok(0);
    }

    let mut removed = 0;
    for entry in fs::read_dir(&dir).map_err(|e| e.to_string())? {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };

        let path = entry.path();
        if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
            if !valid_hashes.contains(stem) {
                if fs::remove_file(&path).is_ok() {
                    removed += 1;
                }
            }
        }
    }

    Ok(removed)
}

pub fn hash_for_path(path: &Path) -> String {
    cache_key(path)
}
