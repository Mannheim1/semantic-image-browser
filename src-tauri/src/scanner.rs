use std::path::Path;
use std::time::SystemTime;
use walkdir::WalkDir;

const IMAGE_EXTENSIONS: &[&str] = &["jpg", "jpeg", "jfif", "png", "gif", "webp", "bmp", "tiff", "tif", "avif"];

pub struct ScannedFile {
    pub path: String,
    pub file_type: String,
    pub file_size: u64,
    pub created_at: SystemTime,
    pub modified_at: SystemTime,
}

fn is_image_file(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| IMAGE_EXTENSIONS.contains(&ext.to_lowercase().as_str()))
        .unwrap_or(false)
}

fn get_extension(path: &Path) -> String {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_lowercase())
        .unwrap_or_default()
}

pub fn scan_directory(dir: &Path) -> Result<Vec<ScannedFile>, String> {
    let mut files = Vec::new();

    for entry in WalkDir::new(dir)
        .follow_links(false)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        if !path.is_file() || !is_image_file(path) {
            continue;
        }

        let metadata = match entry.metadata() {
            Ok(m) => m,
            Err(_) => continue,
        };

        let created_at = metadata.created().unwrap_or(SystemTime::UNIX_EPOCH);
        let modified_at = metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH);

        files.push(ScannedFile {
            path: path.to_string_lossy().to_string(),
            file_type: get_extension(path),
            file_size: metadata.len(),
            created_at,
            modified_at,
        });
    }

    Ok(files)
}

pub fn system_time_to_millis(time: SystemTime) -> i64 {
    time.duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}
