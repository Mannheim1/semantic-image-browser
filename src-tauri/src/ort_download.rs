//! ONNX Runtime download and management.
//!
//! This module handles downloading the appropriate ONNX Runtime for the current platform.
//! The runtime is downloaded from Microsoft's official GitHub releases and stored in
//! the app's local data directory.
//!
//! Runtimes are stored in separate directories to allow multiple runtimes to coexist:
//! - `{app_data}/runtimes/cpu/lib/`
//! - `{app_data}/runtimes/directml/lib/` (TODO: not yet implemented)
//! - `{app_data}/runtimes/cuda/lib/`

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
    DirectMl, // TODO: Download not yet implemented
    Cuda,
}

impl RuntimeType {
    pub fn as_str(&self) -> &'static str {
        match self {
            RuntimeType::Cpu => "cpu",
            RuntimeType::DirectMl => "directml",
            RuntimeType::Cuda => "cuda",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "cpu" => Some(RuntimeType::Cpu),
            "directml" => Some(RuntimeType::DirectMl),
            "cuda" => Some(RuntimeType::Cuda),
            // Legacy support: "gpu" maps to Cuda
            "gpu" => Some(RuntimeType::Cuda),
            _ => None,
        }
    }

    /// Returns all runtime types.
    pub fn all() -> &'static [RuntimeType] {
        &[RuntimeType::Cpu, RuntimeType::DirectMl, RuntimeType::Cuda]
    }

    /// Returns a human-readable display name.
    pub fn display_name(&self) -> &'static str {
        match self {
            RuntimeType::Cpu => "CPU",
            RuntimeType::DirectMl => "GPU (DirectML)",
            RuntimeType::Cuda => "GPU (CUDA)",
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

    /// Check if CUDA runtime is available for this platform.
    pub fn cuda_available(&self) -> bool {
        matches!(self, Platform::WindowsX64 | Platform::LinuxX64)
    }

    /// Check if DirectML runtime is available for this platform.
    /// TODO: DirectML download not yet implemented.
    pub fn directml_available(&self) -> bool {
        matches!(self, Platform::WindowsX64)
    }

    /// Get the archive filename for this platform and runtime type.
    pub fn archive_filename(&self, runtime_type: RuntimeType) -> Option<String> {
        let name = match (self, runtime_type) {
            // Windows
            (Platform::WindowsX64, RuntimeType::Cpu) => {
                format!("onnxruntime-win-x64-{}.zip", ORT_VERSION)
            }
            (Platform::WindowsX64, RuntimeType::Cuda) => {
                format!("onnxruntime-win-x64-gpu-{}.zip", ORT_VERSION)
            }
            (Platform::WindowsX64, RuntimeType::DirectMl) => {
                // TODO: DirectML download not yet implemented
                return None;
            }
            (Platform::WindowsArm64, RuntimeType::Cpu) => {
                format!("onnxruntime-win-arm64-{}.zip", ORT_VERSION)
            }
            (Platform::WindowsArm64, RuntimeType::Cuda | RuntimeType::DirectMl) => return None,

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
            (Platform::MacOsX64 | Platform::MacOsArm64 | Platform::MacOsUniversal, RuntimeType::Cuda | RuntimeType::DirectMl) => {
                return None
            }

            // Linux
            (Platform::LinuxX64, RuntimeType::Cpu) => {
                format!("onnxruntime-linux-x64-{}.tgz", ORT_VERSION)
            }
            (Platform::LinuxX64, RuntimeType::Cuda) => {
                format!("onnxruntime-linux-x64-gpu-{}.tgz", ORT_VERSION)
            }
            (Platform::LinuxX64, RuntimeType::DirectMl) => return None, // DirectML is Windows-only
            (Platform::LinuxArm64, RuntimeType::Cpu) => {
                format!("onnxruntime-linux-aarch64-{}.tgz", ORT_VERSION)
            }
            (Platform::LinuxArm64, RuntimeType::Cuda | RuntimeType::DirectMl) => return None,
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
            (Platform::WindowsX64, RuntimeType::Cuda) => format!("onnxruntime-win-x64-gpu-{}", ORT_VERSION),
            (Platform::WindowsX64, RuntimeType::DirectMl) => return None, // TODO
            (Platform::WindowsArm64, RuntimeType::Cpu) => format!("onnxruntime-win-arm64-{}", ORT_VERSION),
            (Platform::WindowsArm64, RuntimeType::Cuda | RuntimeType::DirectMl) => return None,
            (Platform::MacOsX64, RuntimeType::Cpu) => format!("onnxruntime-osx-x86_64-{}", ORT_VERSION),
            (Platform::MacOsArm64, RuntimeType::Cpu) => format!("onnxruntime-osx-arm64-{}", ORT_VERSION),
            (Platform::MacOsUniversal, RuntimeType::Cpu) => format!("onnxruntime-osx-universal2-{}", ORT_VERSION),
            (Platform::MacOsX64 | Platform::MacOsArm64 | Platform::MacOsUniversal, RuntimeType::Cuda | RuntimeType::DirectMl) => return None,
            (Platform::LinuxX64, RuntimeType::Cpu) => format!("onnxruntime-linux-x64-{}", ORT_VERSION),
            (Platform::LinuxX64, RuntimeType::Cuda) => format!("onnxruntime-linux-x64-gpu-{}", ORT_VERSION),
            (Platform::LinuxX64, RuntimeType::DirectMl) => return None,
            (Platform::LinuxArm64, RuntimeType::Cpu) => format!("onnxruntime-linux-aarch64-{}", ORT_VERSION),
            (Platform::LinuxArm64, RuntimeType::Cuda | RuntimeType::DirectMl) => return None,
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

/// Get the base directory where all runtimes are stored.
pub fn runtimes_base_dir(app: &AppHandle) -> Result<PathBuf, String> {
    let app_data = app.path().app_local_data_dir().map_err(|e| e.to_string())?;
    Ok(app_data.join("runtimes"))
}

/// Get the directory for a specific runtime type.
pub fn runtime_dir(app: &AppHandle, runtime_type: RuntimeType) -> Result<PathBuf, String> {
    let base = runtimes_base_dir(app)?;
    Ok(base.join(runtime_type.as_str()))
}

/// Get the path to the ONNX Runtime library for a specific runtime type, if it exists.
pub fn get_ort_library_path(app: &AppHandle, runtime_type: RuntimeType) -> Result<Option<PathBuf>, String> {
    let platform = Platform::detect().ok_or("Unsupported platform")?;
    let dir = runtime_dir(app, runtime_type)?;
    let lib_path = dir.join("lib").join(platform.library_filename());

    if lib_path.exists() {
        Ok(Some(lib_path))
    } else {
        Ok(None)
    }
}

/// Check if a specific runtime is installed.
pub fn is_runtime_installed(app: &AppHandle, runtime_type: RuntimeType) -> Result<bool, String> {
    Ok(get_ort_library_path(app, runtime_type)?.is_some())
}


/// Get the expected download size in bytes for display purposes.
pub fn get_download_size(runtime_type: RuntimeType) -> Option<u64> {
    let platform = Platform::detect()?;

    // Approximate sizes from the release data
    match (platform, runtime_type) {
        (Platform::WindowsX64, RuntimeType::Cpu) => Some(78_000_000),
        (Platform::WindowsX64, RuntimeType::Cuda) => Some(326_000_000),
        (Platform::WindowsX64, RuntimeType::DirectMl) => None, // TODO: not yet implemented
        (Platform::WindowsArm64, RuntimeType::Cpu) => Some(79_000_000),
        (Platform::MacOsArm64, RuntimeType::Cpu) => Some(10_000_000),
        (Platform::MacOsX64, RuntimeType::Cpu) => Some(12_000_000),
        (Platform::MacOsUniversal, RuntimeType::Cpu) => Some(43_000_000),
        (Platform::LinuxX64, RuntimeType::Cpu) => Some(8_000_000),
        (Platform::LinuxX64, RuntimeType::Cuda) => Some(241_000_000),
        (Platform::LinuxArm64, RuntimeType::Cpu) => Some(7_000_000),
        _ => None,
    }
}

/// Download and extract ONNX Runtime.
///
/// This function downloads the appropriate ONNX Runtime archive for the current platform,
/// extracts it to a runtime-specific directory, and returns the path to the library.
/// Each runtime type is stored in its own directory, allowing multiple runtimes to coexist.
pub async fn download_ort(
    app: &AppHandle,
    runtime_type: RuntimeType,
    progress_callback: impl Fn(u64, u64) + Send + 'static,
) -> Result<PathBuf, String> {
    let platform = Platform::detect().ok_or("Unsupported platform")?;

    // Check if this runtime type is available for this platform
    match runtime_type {
        RuntimeType::Cuda if !platform.cuda_available() => {
            return Err("CUDA runtime is not available for this platform".to_string());
        }
        RuntimeType::DirectMl => {
            // TODO: DirectML download not yet implemented
            return Err("DirectML runtime download is not yet implemented".to_string());
        }
        _ => {}
    }

    let url = platform
        .download_url(runtime_type)
        .ok_or("No download URL for this platform/runtime combination")?;

    let rt_directory = runtime_dir(app, runtime_type)?;

    // Clean up any existing installation for THIS runtime type only
    if rt_directory.exists() {
        fs::remove_dir_all(&rt_directory).map_err(|e| format!("Failed to remove existing runtime directory: {}", e))?;
    }
    fs::create_dir_all(&rt_directory).map_err(|e| format!("Failed to create runtime directory: {}", e))?;

    // Download the archive
    let archive_filename = platform.archive_filename(runtime_type).unwrap();
    let archive_path = rt_directory.join(&archive_filename);

    download_file(&url, &archive_path, progress_callback).await?;

    // Extract the archive
    let inner_dir = platform.archive_inner_dir(runtime_type).unwrap();

    if platform.uses_zip() {
        extract_zip(&archive_path, &rt_directory)?;
    } else {
        extract_tgz(&archive_path, &rt_directory)?;
    }

    // Move contents from inner directory to rt_directory
    let extracted_dir = rt_directory.join(&inner_dir);
    if extracted_dir.exists() {
        move_dir_contents(&extracted_dir, &rt_directory)?;
        fs::remove_dir_all(&extracted_dir)
            .map_err(|e| format!("Failed to remove extracted directory: {}", e))?;
    }

    // Clean up archive
    fs::remove_file(&archive_path)
        .map_err(|e| format!("Failed to remove archive: {}", e))?;

    // Return the path to the library
    let lib_path = rt_directory.join("lib").join(platform.library_filename());
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

/// Calculate the total size of a directory recursively.
fn get_dir_size(path: &Path) -> u64 {
    if !path.exists() {
        return 0;
    }

    let mut size = 0u64;
    if let Ok(entries) = fs::read_dir(path) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                size += entry.metadata().map(|m| m.len()).unwrap_or(0);
            } else if path.is_dir() {
                size += get_dir_size(&path);
            }
        }
    }
    size
}

/// Get the installed size of a runtime.
pub fn get_runtime_installed_size(app: &AppHandle, runtime_type: RuntimeType) -> Result<Option<u64>, String> {
    let dir = runtime_dir(app, runtime_type)?;
    if dir.exists() {
        Ok(Some(get_dir_size(&dir)))
    } else {
        Ok(None)
    }
}

/// Uninstall a runtime by deleting its directory.
pub fn uninstall_runtime(app: &AppHandle, runtime_type: RuntimeType) -> Result<(), String> {
    let dir = runtime_dir(app, runtime_type)?;
    if dir.exists() {
        fs::remove_dir_all(&dir)
            .map_err(|e| format!("Failed to remove runtime directory: {}", e))?;
    }
    Ok(())
}

/// Check if CUDA system dependencies are available.
/// This checks for the required DLLs in the system PATH.
#[cfg(target_os = "windows")]
pub fn check_cuda_dependencies() -> CudaDependencyStatus {
    use std::env;

    // DLLs required by ONNX Runtime CUDA 12
    let cuda_dlls = [
        ("CUDA Runtime", "cudart64_12.dll"),
    ];
    let cublas_dlls = [
        ("cuBLAS", "cublas64_12.dll"),
        ("cuBLAS Lt", "cublasLt64_12.dll"),
    ];
    let cudnn_dlls = [
        ("cuDNN", "cudnn64_9.dll"),
        ("cuDNN Ops", "cudnn_ops64_9.dll"),
        ("cuDNN CNN", "cudnn_cnn64_9.dll"),
    ];

    let path_var = env::var("PATH").unwrap_or_default();
    let paths: Vec<&str> = path_var.split(';').collect();

    let find_dll = |dll_name: &str| -> bool {
        for dir in &paths {
            let dll_path = Path::new(dir).join(dll_name);
            if dll_path.exists() {
                return true;
            }
        }
        false
    };

    let mut dependencies = Vec::new();

    // Check CUDA Runtime
    let cuda_found = cuda_dlls.iter().all(|(_, dll)| find_dll(dll));
    dependencies.push(CudaDependency {
        name: "CUDA Toolkit 12.x".to_string(),
        found: cuda_found,
        details: if cuda_found { None } else { Some("cudart64_12.dll not found in PATH".to_string()) },
    });

    // Check cuBLAS
    let cublas_found = cublas_dlls.iter().all(|(_, dll)| find_dll(dll));
    dependencies.push(CudaDependency {
        name: "cuBLAS".to_string(),
        found: cublas_found,
        details: if cublas_found { None } else { Some("cublas64_12.dll not found in PATH".to_string()) },
    });

    // Check cuDNN
    let cudnn_found = cudnn_dlls.iter().all(|(_, dll)| find_dll(dll));
    dependencies.push(CudaDependency {
        name: "cuDNN 9.x".to_string(),
        found: cudnn_found,
        details: if cudnn_found { None } else { Some("cudnn64_9.dll not found in PATH".to_string()) },
    });

    let all_found = cuda_found && cublas_found && cudnn_found;

    CudaDependencyStatus {
        all_found,
        dependencies,
    }
}

#[cfg(not(target_os = "windows"))]
pub fn check_cuda_dependencies() -> CudaDependencyStatus {
    // On non-Windows, just return empty for now
    // Could be extended to check for .so files on Linux
    CudaDependencyStatus {
        all_found: false,
        dependencies: vec![
            CudaDependency {
                name: "CUDA dependencies".to_string(),
                found: false,
                details: Some("CUDA dependency checking not implemented for this platform".to_string()),
            },
        ],
    }
}

/// Information about a single runtime type for the frontend.
#[derive(Debug, Clone, serde::Serialize)]
pub struct RuntimeInfo {
    pub runtime_type: String,
    pub display_name: String,
    pub installed: bool,
    pub available: bool, // Whether this runtime can be downloaded for this platform
    pub download_size: Option<u64>,
    pub installed_size: Option<u64>, // Actual size on disk if installed
}

/// Information about a CUDA system dependency.
#[derive(Debug, Clone, serde::Serialize)]
pub struct CudaDependency {
    pub name: String,
    pub found: bool,
    pub details: Option<String>,
}

/// Result of checking CUDA system dependencies.
#[derive(Debug, Clone, serde::Serialize)]
pub struct CudaDependencyStatus {
    pub all_found: bool,
    pub dependencies: Vec<CudaDependency>,
}

/// Information about ONNX Runtime status for the frontend.
#[derive(Debug, Clone, serde::Serialize)]
pub struct OrtStatus {
    /// The currently selected runtime type from config
    pub selected_runtime: Option<String>,
    /// Whether the selected runtime is installed
    pub selected_installed: bool,
    /// Path to the selected runtime's library (if installed)
    pub library_path: Option<String>,
    /// Information about all runtime types
    pub runtimes: Vec<RuntimeInfo>,
    /// Platform identifier
    pub platform: String,
}

/// Get the current ONNX Runtime status.
pub fn get_ort_status(app: &AppHandle, selected_runtime: Option<String>) -> Result<OrtStatus, String> {
    let platform = Platform::detect().ok_or("Unsupported platform")?;

    let selected_type = selected_runtime
        .as_deref()
        .and_then(RuntimeType::from_str)
        .unwrap_or(RuntimeType::Cpu);

    // Get library path for selected runtime
    let lib_path = get_ort_library_path(app, selected_type)?;

    // Build info for all runtime types
    let mut runtimes = Vec::new();
    for &rt in RuntimeType::all() {
        let installed = is_runtime_installed(app, rt)?;
        let installed_size = if installed {
            get_runtime_installed_size(app, rt)?
        } else {
            None
        };
        let available = match rt {
            RuntimeType::Cpu => true,
            RuntimeType::Cuda => platform.cuda_available(),
            RuntimeType::DirectMl => false, // TODO: not yet implemented
        };
        runtimes.push(RuntimeInfo {
            runtime_type: rt.as_str().to_string(),
            display_name: rt.display_name().to_string(),
            installed,
            available,
            download_size: get_download_size(rt),
            installed_size,
        });
    }

    Ok(OrtStatus {
        selected_runtime,
        selected_installed: lib_path.is_some(),
        library_path: lib_path.map(|p| p.to_string_lossy().to_string()),
        runtimes,
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

        let cuda_url = platform.download_url(RuntimeType::Cuda);
        assert!(cuda_url.is_some());
        assert!(cuda_url.unwrap().contains("onnxruntime-win-x64-gpu-1.23.2.zip"));

        // DirectML not yet implemented
        let directml_url = platform.download_url(RuntimeType::DirectMl);
        assert!(directml_url.is_none());
    }

    #[test]
    fn test_runtime_type_conversion() {
        assert_eq!(RuntimeType::from_str("cpu"), Some(RuntimeType::Cpu));
        assert_eq!(RuntimeType::from_str("cuda"), Some(RuntimeType::Cuda));
        assert_eq!(RuntimeType::from_str("directml"), Some(RuntimeType::DirectMl));
        // Legacy "gpu" maps to Cuda
        assert_eq!(RuntimeType::from_str("gpu"), Some(RuntimeType::Cuda));
        assert_eq!(RuntimeType::from_str("GPU"), Some(RuntimeType::Cuda));
        assert_eq!(RuntimeType::from_str("invalid"), None);
    }

    #[test]
    fn test_runtime_type_all() {
        let all = RuntimeType::all();
        assert_eq!(all.len(), 3);
        assert!(all.contains(&RuntimeType::Cpu));
        assert!(all.contains(&RuntimeType::DirectMl));
        assert!(all.contains(&RuntimeType::Cuda));
    }
}
