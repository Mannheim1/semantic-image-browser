//! ONNX Runtime download and management.
//!
//! This module handles downloading the appropriate ONNX Runtime for the current platform.
//! The runtime is downloaded from Microsoft's official GitHub releases and stored in
//! the app's local data directory.

use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use tauri::{AppHandle, Manager};

/// ONNX Runtime version to download.
const ORT_VERSION: &str = "1.23.2";

/// Base URL for ONNX Runtime releases.
const ORT_RELEASE_BASE: &str = "https://github.com/microsoft/onnxruntime/releases/download";

/// Runtime type selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeType {
    Cpu,
    Gpu,
}

impl RuntimeType {
    pub fn as_str(&self) -> &'static str {
        match self {
            RuntimeType::Cpu => "cpu",
            RuntimeType::Gpu => "gpu",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "cpu" => Some(RuntimeType::Cpu),
            "gpu" => Some(RuntimeType::Gpu),
            _ => None,
        }
    }
}

/// Platform and architecture detection.
#[derive(Debug, Clone, Copy)]
pub enum Platform {
    WindowsX64,
    WindowsArm64,
    MacOsX64,
    MacOsArm64,
    MacOsUniversal,
    LinuxX64,
    LinuxArm64,
}

impl Platform {
    /// Detect the current platform.
    pub fn detect() -> Option<Self> {
        #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
        return Some(Platform::WindowsX64);

        #[cfg(all(target_os = "windows", target_arch = "aarch64"))]
        return Some(Platform::WindowsArm64);

        #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
        return Some(Platform::MacOsX64);

        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        return Some(Platform::MacOsArm64);

        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        return Some(Platform::LinuxX64);

        #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
        return Some(Platform::LinuxArm64);

        #[cfg(not(any(
            all(target_os = "windows", target_arch = "x86_64"),
            all(target_os = "windows", target_arch = "aarch64"),
            all(target_os = "macos", target_arch = "x86_64"),
            all(target_os = "macos", target_arch = "aarch64"),
            all(target_os = "linux", target_arch = "x86_64"),
            all(target_os = "linux", target_arch = "aarch64"),
        )))]
        return None;
    }

    /// Check if GPU runtime is available for this platform.
    pub fn gpu_available(&self) -> bool {
        matches!(self, Platform::WindowsX64 | Platform::LinuxX64)
    }

    /// Get the archive filename for this platform and runtime type.
    pub fn archive_filename(&self, runtime_type: RuntimeType) -> Option<String> {
        let name = match (self, runtime_type) {
            // Windows
            (Platform::WindowsX64, RuntimeType::Cpu) => {
                format!("onnxruntime-win-x64-{}.zip", ORT_VERSION)
            }
            (Platform::WindowsX64, RuntimeType::Gpu) => {
                format!("onnxruntime-win-x64-gpu-{}.zip", ORT_VERSION)
            }
            (Platform::WindowsArm64, RuntimeType::Cpu) => {
                format!("onnxruntime-win-arm64-{}.zip", ORT_VERSION)
            }
            (Platform::WindowsArm64, RuntimeType::Gpu) => return None, // No GPU for Windows ARM64

            // macOS (no GPU support)
            (Platform::MacOsX64, RuntimeType::Cpu) => {
                format!("onnxruntime-osx-x86_64-{}.tgz", ORT_VERSION)
            }
            (Platform::MacOsArm64, RuntimeType::Cpu) => {
                format!("onnxruntime-osx-arm64-{}.tgz", ORT_VERSION)
            }
            (Platform::MacOsUniversal, RuntimeType::Cpu) => {
                format!("onnxruntime-osx-universal2-{}.tgz", ORT_VERSION)
            }
            (Platform::MacOsX64 | Platform::MacOsArm64 | Platform::MacOsUniversal, RuntimeType::Gpu) => {
                return None // No GPU for macOS
            }

            // Linux
            (Platform::LinuxX64, RuntimeType::Cpu) => {
                format!("onnxruntime-linux-x64-{}.tgz", ORT_VERSION)
            }
            (Platform::LinuxX64, RuntimeType::Gpu) => {
                format!("onnxruntime-linux-x64-gpu-{}.tgz", ORT_VERSION)
            }
            (Platform::LinuxArm64, RuntimeType::Cpu) => {
                format!("onnxruntime-linux-aarch64-{}.tgz", ORT_VERSION)
            }
            (Platform::LinuxArm64, RuntimeType::Gpu) => return None, // No GPU for Linux ARM64
        };
        Some(name)
    }

    /// Get the download URL for this platform and runtime type.
    pub fn download_url(&self, runtime_type: RuntimeType) -> Option<String> {
        let filename = self.archive_filename(runtime_type)?;
        Some(format!("{}/v{}/{}", ORT_RELEASE_BASE, ORT_VERSION, filename))
    }

    /// Get the expected directory name inside the archive.
    pub fn archive_inner_dir(&self, runtime_type: RuntimeType) -> Option<String> {
        let name = match (self, runtime_type) {
            (Platform::WindowsX64, RuntimeType::Cpu) => format!("onnxruntime-win-x64-{}", ORT_VERSION),
            (Platform::WindowsX64, RuntimeType::Gpu) => format!("onnxruntime-win-x64-gpu-{}", ORT_VERSION),
            (Platform::WindowsArm64, RuntimeType::Cpu) => format!("onnxruntime-win-arm64-{}", ORT_VERSION),
            (Platform::WindowsArm64, RuntimeType::Gpu) => return None,
            (Platform::MacOsX64, RuntimeType::Cpu) => format!("onnxruntime-osx-x86_64-{}", ORT_VERSION),
            (Platform::MacOsArm64, RuntimeType::Cpu) => format!("onnxruntime-osx-arm64-{}", ORT_VERSION),
            (Platform::MacOsUniversal, RuntimeType::Cpu) => format!("onnxruntime-osx-universal2-{}", ORT_VERSION),
            (Platform::MacOsX64 | Platform::MacOsArm64 | Platform::MacOsUniversal, RuntimeType::Gpu) => return None,
            (Platform::LinuxX64, RuntimeType::Cpu) => format!("onnxruntime-linux-x64-{}", ORT_VERSION),
            (Platform::LinuxX64, RuntimeType::Gpu) => format!("onnxruntime-linux-x64-gpu-{}", ORT_VERSION),
            (Platform::LinuxArm64, RuntimeType::Cpu) => format!("onnxruntime-linux-aarch64-{}", ORT_VERSION),
            (Platform::LinuxArm64, RuntimeType::Gpu) => return None,
        };
        Some(name)
    }

    /// Get the library filename for this platform.
    pub fn library_filename(&self) -> &'static str {
        match self {
            Platform::WindowsX64 | Platform::WindowsArm64 => "onnxruntime.dll",
            Platform::MacOsX64 | Platform::MacOsArm64 | Platform::MacOsUniversal => "libonnxruntime.dylib",
            Platform::LinuxX64 | Platform::LinuxArm64 => "libonnxruntime.so",
        }
    }

    /// Check if archives are zip (Windows) or tgz (Unix).
    pub fn uses_zip(&self) -> bool {
        matches!(self, Platform::WindowsX64 | Platform::WindowsArm64)
    }
}

/// Get the directory where ONNX Runtime should be stored.
pub fn ort_dir(app: &AppHandle) -> Result<PathBuf, String> {
    let app_data = app.path().app_local_data_dir().map_err(|e| e.to_string())?;
    Ok(app_data.join("onnxruntime"))
}

/// Get the path to the ONNX Runtime library if it exists.
pub fn get_ort_library_path(app: &AppHandle) -> Result<Option<PathBuf>, String> {
    let platform = Platform::detect().ok_or("Unsupported platform")?;
    let dir = ort_dir(app)?;
    let lib_path = dir.join("lib").join(platform.library_filename());

    if lib_path.exists() {
        Ok(Some(lib_path))
    } else {
        Ok(None)
    }
}

/// Check if ONNX Runtime is installed.
pub fn is_ort_installed(app: &AppHandle) -> Result<bool, String> {
    Ok(get_ort_library_path(app)?.is_some())
}

/// Get the expected download size in bytes for display purposes.
pub fn get_download_size(runtime_type: RuntimeType) -> Option<u64> {
    let platform = Platform::detect()?;

    // Approximate sizes from the release data
    match (platform, runtime_type) {
        (Platform::WindowsX64, RuntimeType::Cpu) => Some(78_000_000),
        (Platform::WindowsX64, RuntimeType::Gpu) => Some(326_000_000),
        (Platform::WindowsArm64, RuntimeType::Cpu) => Some(79_000_000),
        (Platform::MacOsArm64, RuntimeType::Cpu) => Some(10_000_000),
        (Platform::MacOsX64, RuntimeType::Cpu) => Some(12_000_000),
        (Platform::MacOsUniversal, RuntimeType::Cpu) => Some(43_000_000),
        (Platform::LinuxX64, RuntimeType::Cpu) => Some(8_000_000),
        (Platform::LinuxX64, RuntimeType::Gpu) => Some(241_000_000),
        (Platform::LinuxArm64, RuntimeType::Cpu) => Some(7_000_000),
        _ => None,
    }
}

/// Download and extract ONNX Runtime.
///
/// This function downloads the appropriate ONNX Runtime archive for the current platform,
/// extracts it, and moves the library files to the expected location.
pub async fn download_ort(
    app: &AppHandle,
    runtime_type: RuntimeType,
    progress_callback: impl Fn(u64, u64) + Send + 'static,
) -> Result<PathBuf, String> {
    let platform = Platform::detect().ok_or("Unsupported platform")?;

    // Check if GPU is available for this platform
    if runtime_type == RuntimeType::Gpu && !platform.gpu_available() {
        return Err("GPU runtime is not available for this platform".to_string());
    }

    let url = platform
        .download_url(runtime_type)
        .ok_or("No download URL for this platform/runtime combination")?;

    let ort_directory = ort_dir(app)?;

    // Clean up any existing installation
    if ort_directory.exists() {
        fs::remove_dir_all(&ort_directory).map_err(|e| format!("Failed to remove existing ORT directory: {}", e))?;
    }
    fs::create_dir_all(&ort_directory).map_err(|e| format!("Failed to create ORT directory: {}", e))?;

    // Download the archive
    let archive_filename = platform.archive_filename(runtime_type).unwrap();
    let archive_path = ort_directory.join(&archive_filename);

    download_file(&url, &archive_path, progress_callback).await?;

    // Extract the archive
    let inner_dir = platform.archive_inner_dir(runtime_type).unwrap();

    if platform.uses_zip() {
        extract_zip(&archive_path, &ort_directory)?;
    } else {
        extract_tgz(&archive_path, &ort_directory)?;
    }

    // Move contents from inner directory to ort_directory
    let extracted_dir = ort_directory.join(&inner_dir);
    if extracted_dir.exists() {
        move_dir_contents(&extracted_dir, &ort_directory)?;
        fs::remove_dir_all(&extracted_dir)
            .map_err(|e| format!("Failed to remove extracted directory: {}", e))?;
    }

    // Clean up archive
    fs::remove_file(&archive_path)
        .map_err(|e| format!("Failed to remove archive: {}", e))?;

    // Return the path to the library
    let lib_path = ort_directory.join("lib").join(platform.library_filename());
    if !lib_path.exists() {
        return Err(format!("Library not found after extraction: {}", lib_path.display()));
    }

    Ok(lib_path)
}

/// Download a file with progress reporting.
async fn download_file(
    url: &str,
    dest: &Path,
    progress_callback: impl Fn(u64, u64) + Send + 'static,
) -> Result<(), String> {
    let client = reqwest::Client::new();

    let response = client
        .get(url)
        .send()
        .await
        .map_err(|e| format!("Failed to start download: {}", e))?;

    if !response.status().is_success() {
        return Err(format!("Download failed with status: {}", response.status()));
    }

    let total_size = response.content_length().unwrap_or(0);

    let mut file = fs::File::create(dest)
        .map_err(|e| format!("Failed to create file: {}", e))?;

    let mut downloaded: u64 = 0;
    let mut stream = response.bytes_stream();

    use futures_util::StreamExt;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| format!("Error downloading chunk: {}", e))?;
        file.write_all(&chunk)
            .map_err(|e| format!("Error writing to file: {}", e))?;
        downloaded += chunk.len() as u64;
        progress_callback(downloaded, total_size);
    }

    Ok(())
}

/// Extract a zip archive.
fn extract_zip(archive_path: &Path, dest_dir: &Path) -> Result<(), String> {
    let file = fs::File::open(archive_path)
        .map_err(|e| format!("Failed to open archive: {}", e))?;
    let mut archive = zip::ZipArchive::new(file)
        .map_err(|e| format!("Failed to read zip archive: {}", e))?;

    for i in 0..archive.len() {
        let mut file = archive.by_index(i)
            .map_err(|e| format!("Failed to read archive entry: {}", e))?;

        let outpath = match file.enclosed_name() {
            Some(path) => dest_dir.join(path),
            None => continue,
        };

        if file.name().ends_with('/') {
            fs::create_dir_all(&outpath)
                .map_err(|e| format!("Failed to create directory: {}", e))?;
        } else {
            if let Some(parent) = outpath.parent() {
                fs::create_dir_all(parent)
                    .map_err(|e| format!("Failed to create parent directory: {}", e))?;
            }
            let mut outfile = fs::File::create(&outpath)
                .map_err(|e| format!("Failed to create file: {}", e))?;
            io::copy(&mut file, &mut outfile)
                .map_err(|e| format!("Failed to extract file: {}", e))?;
        }
    }

    Ok(())
}

/// Extract a tgz archive.
fn extract_tgz(archive_path: &Path, dest_dir: &Path) -> Result<(), String> {
    let file = fs::File::open(archive_path)
        .map_err(|e| format!("Failed to open archive: {}", e))?;
    let decompressed = flate2::read::GzDecoder::new(file);
    let mut archive = tar::Archive::new(decompressed);

    archive.unpack(dest_dir)
        .map_err(|e| format!("Failed to extract tgz archive: {}", e))?;

    Ok(())
}

/// Move contents of one directory into another.
fn move_dir_contents(src: &Path, dest: &Path) -> Result<(), String> {
    for entry in fs::read_dir(src).map_err(|e| format!("Failed to read directory: {}", e))? {
        let entry = entry.map_err(|e| format!("Failed to read entry: {}", e))?;
        let src_path = entry.path();
        let dest_path = dest.join(entry.file_name());

        if dest_path.exists() {
            if dest_path.is_dir() {
                fs::remove_dir_all(&dest_path)
                    .map_err(|e| format!("Failed to remove existing directory: {}", e))?;
            } else {
                fs::remove_file(&dest_path)
                    .map_err(|e| format!("Failed to remove existing file: {}", e))?;
            }
        }

        fs::rename(&src_path, &dest_path)
            .map_err(|e| format!("Failed to move {}: {}", src_path.display(), e))?;
    }
    Ok(())
}

/// Information about ONNX Runtime status for the frontend.
#[derive(Debug, Clone, serde::Serialize)]
pub struct OrtStatus {
    pub installed: bool,
    pub library_path: Option<String>,
    pub runtime_type: Option<String>,
    pub gpu_available: bool,
    pub platform: String,
}

/// Get the current ONNX Runtime status.
pub fn get_ort_status(app: &AppHandle) -> Result<OrtStatus, String> {
    let platform = Platform::detect().ok_or("Unsupported platform")?;
    let lib_path = get_ort_library_path(app)?;

    // Get runtime type from config
    let cfg = crate::config::load_config(app)?;
    let runtime_type = cfg.runtime_type;

    Ok(OrtStatus {
        installed: lib_path.is_some(),
        library_path: lib_path.map(|p| p.to_string_lossy().to_string()),
        runtime_type,
        gpu_available: platform.gpu_available(),
        platform: format!("{:?}", platform),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_platform_detection() {
        let platform = Platform::detect();
        assert!(platform.is_some());
    }

    #[test]
    fn test_download_urls() {
        let platform = Platform::WindowsX64;
        let url = platform.download_url(RuntimeType::Cpu);
        assert!(url.is_some());
        assert!(url.unwrap().contains("onnxruntime-win-x64-1.23.2.zip"));
    }

    #[test]
    fn test_runtime_type_conversion() {
        assert_eq!(RuntimeType::from_str("cpu"), Some(RuntimeType::Cpu));
        assert_eq!(RuntimeType::from_str("GPU"), Some(RuntimeType::Gpu));
        assert_eq!(RuntimeType::from_str("invalid"), None);
    }
}
