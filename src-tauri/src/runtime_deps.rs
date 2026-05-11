//! First-launch download of CUDA runtime libraries for the GPU backend.
//!
//! The bundled CUDA installer is too large to fit GitHub release size limits,
//! so CUDA libraries (cudart, cuBLAS, cuDNN) are downloaded on first launch
//! and cached under the app's local data dir. Subsequent launches reuse the
//! cache. CPU and CoreML builds do not call into this module.

use futures_util::StreamExt;
use serde::Serialize;
use std::io::Write;
use std::path::{Path, PathBuf};
use tauri::{AppHandle, Emitter, Manager};

#[derive(Clone, Serialize)]
pub struct DownloadProgress {
    pub phase: String,
    pub current_bytes: u64,
    pub total_bytes: u64,
}

struct Archive {
    label: &'static str,
    url: &'static str,
    files: &'static [&'static str],
}

#[cfg(all(target_os = "windows", feature = "backend-cuda"))]
const CUDA_ARCHIVES: &[Archive] = &[
    Archive {
        label: "CUDA Runtime",
        url: "https://developer.download.nvidia.com/compute/cuda/redist/cuda_cudart/windows-x86_64/cuda_cudart-windows-x86_64-12.8.90-archive.zip",
        files: &["cudart64_12.dll"],
    },
    Archive {
        label: "cuBLAS",
        url: "https://developer.download.nvidia.com/compute/cuda/redist/libcublas/windows-x86_64/libcublas-windows-x86_64-12.8.4.1-archive.zip",
        files: &["cublas64_12.dll", "cublasLt64_12.dll"],
    },
    Archive {
        label: "cuFFT",
        url: "https://developer.download.nvidia.com/compute/cuda/redist/libcufft/windows-x86_64/libcufft-windows-x86_64-11.3.3.41-archive.zip",
        files: &["cufft64_11.dll"],
    },
    Archive {
        label: "cuDNN",
        url: "https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/cudnn-windows-x86_64-9.19.0.56_cuda12-archive.zip",
        files: &[
            "cudnn64_9.dll",
            "cudnn_ops64_9.dll",
            "cudnn_cnn64_9.dll",
            "cudnn_graph64_9.dll",
            "cudnn_heuristic64_9.dll",
            "cudnn_engines_precompiled64_9.dll",
            "cudnn_engines_runtime_compiled64_9.dll",
        ],
    },
];

/// Path where downloaded runtime dependencies are cached.
pub fn cuda_runtime_dir(app: &AppHandle) -> Result<PathBuf, String> {
    let base = app.path().app_local_data_dir().map_err(|e| e.to_string())?;
    let dir = base.join("runtime");
    std::fs::create_dir_all(&dir).map_err(|e| format!("create runtime dir: {}", e))?;
    Ok(dir)
}

#[cfg(all(target_os = "windows", feature = "backend-cuda"))]
fn missing_files(dir: &Path) -> Vec<&'static str> {
    CUDA_ARCHIVES
        .iter()
        .flat_map(|a| a.files.iter().copied())
        .filter(|f| !dir.join(f).exists())
        .collect()
}

/// Ensure all required CUDA runtime DLLs are present in the cache dir,
/// downloading and extracting any that are missing, then prepend the cache
/// dir to the process `PATH` so the Windows DLL loader finds the libs via
/// its standard search. Returns the cache dir.
///
/// Why prepend to `PATH` instead of using the modern `AddDllDirectory` API:
/// the legacy `SetDllDirectoryW` call in [`crate::embedding`] keeps `PATH`
/// in the loader's search list, which we rely on so that systems with a
/// pre-installed NVIDIA CUDA toolkit (e.g. the dev environment) still find
/// the libs as a fallback. Adding our cache to `PATH` slots into that same
/// mechanism without disabling the fallback.
///
/// Emits `runtime_deps_progress` events the frontend uses to display status.
#[cfg(all(target_os = "windows", feature = "backend-cuda"))]
pub async fn ensure_cuda_runtime(app: &AppHandle) -> Result<PathBuf, String> {
    let runtime_dir = cuda_runtime_dir(app)?;

    if !missing_files(&runtime_dir).is_empty() {
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::limited(5))
            .build()
            .map_err(|e| format!("build http client: {}", e))?;

        let tmp_dir = runtime_dir.join(".tmp");
        std::fs::create_dir_all(&tmp_dir).map_err(|e| format!("create tmp dir: {}", e))?;

        for archive in CUDA_ARCHIVES {
            if archive.files.iter().all(|f| runtime_dir.join(f).exists()) {
                continue;
            }

            let zip_path = tmp_dir.join(format!("{}.zip", archive.label.replace(' ', "_")));
            download(app, &client, archive.label, archive.url, &zip_path).await?;
            extract(app, archive.label, &zip_path, &runtime_dir, archive.files)?;
            let _ = std::fs::remove_file(&zip_path);
        }

        let still_missing = missing_files(&runtime_dir);
        if !still_missing.is_empty() {
            return Err(format!(
                "after download, still missing: {}",
                still_missing.join(", ")
            ));
        }

        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    prepend_to_path(&runtime_dir);

    let _ = app.emit(
        "runtime_deps_progress",
        DownloadProgress {
            phase: "done".into(),
            current_bytes: 0,
            total_bytes: 0,
        },
    );
    Ok(runtime_dir)
}

#[cfg(all(target_os = "windows", feature = "backend-cuda"))]
fn prepend_to_path(dir: &Path) {
    let dir_str = dir.to_string_lossy().to_string();
    let current = std::env::var("PATH").unwrap_or_default();
    if current
        .split(';')
        .any(|p| p.eq_ignore_ascii_case(&dir_str))
    {
        return;
    }
    let new_path = if current.is_empty() {
        dir_str
    } else {
        format!("{};{}", dir_str, current)
    };
    std::env::set_var("PATH", new_path);
    println!("Prepended to PATH: {}", dir.display());
}

#[cfg(all(target_os = "windows", feature = "backend-cuda"))]
async fn download(
    app: &AppHandle,
    client: &reqwest::Client,
    label: &str,
    url: &str,
    out: &Path,
) -> Result<(), String> {
    let resp = client
        .get(url)
        .send()
        .await
        .map_err(|e| format!("download {}: {}", label, e))?
        .error_for_status()
        .map_err(|e| format!("download {}: {}", label, e))?;

    let total = resp.content_length().unwrap_or(0);
    let phase = format!("Downloading {}", label);

    let _ = app.emit(
        "runtime_deps_progress",
        DownloadProgress {
            phase: phase.clone(),
            current_bytes: 0,
            total_bytes: total,
        },
    );

    let mut file = std::fs::File::create(out).map_err(|e| format!("create {}: {}", out.display(), e))?;
    let mut stream = resp.bytes_stream();
    let mut downloaded: u64 = 0;
    let mut last_emit = std::time::Instant::now();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| format!("download {}: {}", label, e))?;
        file.write_all(&chunk)
            .map_err(|e| format!("write {}: {}", out.display(), e))?;
        downloaded += chunk.len() as u64;

        if last_emit.elapsed() > std::time::Duration::from_millis(150) {
            let _ = app.emit(
                "runtime_deps_progress",
                DownloadProgress {
                    phase: phase.clone(),
                    current_bytes: downloaded,
                    total_bytes: total,
                },
            );
            last_emit = std::time::Instant::now();
        }
    }

    let _ = app.emit(
        "runtime_deps_progress",
        DownloadProgress {
            phase: phase.clone(),
            current_bytes: downloaded,
            total_bytes: total,
        },
    );

    if total > 0 && downloaded != total {
        return Err(format!(
            "download {}: size mismatch (got {} expected {})",
            label, downloaded, total
        ));
    }

    Ok(())
}

#[cfg(all(target_os = "windows", feature = "backend-cuda"))]
fn extract(
    app: &AppHandle,
    label: &str,
    zip_path: &Path,
    out_dir: &Path,
    wanted: &[&str],
) -> Result<(), String> {
    let _ = app.emit(
        "runtime_deps_progress",
        DownloadProgress {
            phase: format!("Extracting {}", label),
            current_bytes: 0,
            total_bytes: 0,
        },
    );

    let file = std::fs::File::open(zip_path).map_err(|e| format!("open {}: {}", zip_path.display(), e))?;
    let mut archive = zip::ZipArchive::new(file).map_err(|e| format!("open zip {}: {}", label, e))?;

    for i in 0..archive.len() {
        let mut entry = archive
            .by_index(i)
            .map_err(|e| format!("read zip entry {}: {}", i, e))?;
        if entry.is_dir() {
            continue;
        }
        let name = entry.name().to_string();
        let basename = name.rsplit(|c| c == '/' || c == '\\').next().unwrap_or("");
        if !wanted.contains(&basename) {
            continue;
        }
        let out_path = out_dir.join(basename);
        let mut out = std::fs::File::create(&out_path)
            .map_err(|e| format!("create {}: {}", out_path.display(), e))?;
        std::io::copy(&mut entry, &mut out)
            .map_err(|e| format!("extract {}: {}", basename, e))?;
    }

    Ok(())
}
