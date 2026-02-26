use lancedb::{Connection, Table};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::RwLock;
use tauri::{AppHandle, Emitter, Manager};
use tauri_plugin_opener::OpenerExt;

mod benchmark;
mod config;
mod database;
mod embedding;
mod image_ops;
mod menu;
mod scan;
mod state;
mod thumbnail;

use config::AppConfig;
use database::{FilterOptions, ImageInfo, SortOptions};
use scan::{ScanProgressState, ScanResult, scan_directory_internal};
use state::{AppState, EmbeddingBackend};

/// Resolve the `bundled/` resource directory.
///
/// In production builds, resources are placed alongside the executable by the
/// Tauri bundler. In dev mode (`cargo tauri dev`), they live in `src-tauri/bundled/`.
fn bundled_dir(app: &AppHandle) -> PathBuf {
    if cfg!(dev) {
        // During `tauri dev`, resolve relative to the Cargo manifest directory
        // which is src-tauri/ — where bundled/ lives.
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("bundled")
    } else {
        app.path().resource_dir()
            .expect("Failed to resolve resource directory")
            .join("bundled")
    }
}

/// Platform-specific ONNX Runtime library filename.
fn ort_lib_filename() -> &'static str {
    if cfg!(target_os = "windows") {
        "onnxruntime.dll"
    } else if cfg!(target_os = "macos") {
        "libonnxruntime.dylib"
    } else {
        "libonnxruntime.so"
    }
}

/// Validates that a path exists and is a directory.
fn validate_directory(path: &str) -> Result<(), String> {
    let p = Path::new(path);

    if !p.exists() {
        return Err(format!("Directory does not exist: {}", p.display()));
    }

    if !p.is_dir() {
        return Err(format!("Path is not a directory: {}", p.display()));
    }

    Ok(())
}

#[tauri::command]
async fn get_config(state: tauri::State<'_, AppState>) -> Result<AppConfig, String> {
    Ok(config::get_config(&state))
}

#[tauri::command]
fn get_embedding_model_status(state: tauri::State<'_, AppState>) -> bool {
    state.embedding_backend.read().map(|p| p.is_some()).unwrap_or(false)
}

#[tauri::command]
async fn add_watched_directory(app: AppHandle, state: tauri::State<'_, AppState>, path: String) -> Result<ScanResult, String> {
    validate_directory(&path)?;

    if !config::get_config(&state).watched_directories.contains(&path) {
        config::update_config(&app, &state, |cfg| {
            cfg.watched_directories.push(path.clone());
        })?;
    }

    let thumb_dir = &state.thumbnails_dir;
    let table = state.table.lock().await;

    let progress = ScanProgressState::new(&app);
    // Clone Arc and String out of the locks before the await point
    let embedding_backend = state.embedding_backend.read().map_err(|e| e.to_string())?.clone();
    let model_id = state.model_id.read().map_err(|e| e.to_string())?.clone();
    let (result, _) = scan_directory_internal(
        &table,
        thumb_dir,
        &path,
        embedding_backend.as_deref(),
        model_id.as_deref(),
        Some(progress),
    ).await?;
    Ok(result)
}

#[tauri::command]
async fn remove_watched_directory(app: AppHandle, state: tauri::State<'_, AppState>, path: String) -> Result<(), String> {
    config::update_config(&app, &state, |cfg| {
        cfg.watched_directories.retain(|p| p != &path);
    })?;

    let thumb_dir = &state.thumbnails_dir;

    let removed_path = Path::new(&path);
    let remaining_dirs: Vec<String> = config::get_config(&state).watched_directories;
    let remaining_dirs: Vec<&Path> = remaining_dirs.iter().map(|d| Path::new(d.as_str())).collect();
    let table = state.table.lock().await;
    let all_paths = database::get_all_paths(&table).await?;
    let to_remove: Vec<String> = all_paths
        .into_iter()
        .filter(|p| {
            let p = Path::new(p);
            p.starts_with(removed_path) && !remaining_dirs.iter().any(|dir| p.starts_with(dir))
        })
        .collect();

    if !to_remove.is_empty() {
        let progress = ScanProgressState::new(&app);
        progress.set_phase("removing");
        progress.set_total(to_remove.len());
        for chunk in to_remove.chunks(500) {
            thumbnail::delete_thumbnails(thumb_dir, chunk)?;
            database::remove_images(&table, chunk).await?;
            progress.increment_by(chunk.len());
        }
    }

    Ok(())
}

#[tauri::command]
async fn rescan_all(app: AppHandle, state: tauri::State<'_, AppState>) -> Result<ScanResult, String> {
    let cfg = config::get_config(&state);
    let thumb_dir = &state.thumbnails_dir;
    let progress = ScanProgressState::new(&app);

    let mut total_result = ScanResult {
        images_found: 0,
        images_added: 0,
        images_updated: 0,
        images_removed: 0,
        errors: Vec::new(),
    };

    let table = state.table.lock().await;
    let mut all_seen_paths: HashSet<String> = HashSet::new();

    // Clone Arc and String out of the locks before the await points
    let embedding_backend = state.embedding_backend.read().map_err(|e| e.to_string())?.clone();
    let model_id = state.model_id.read().map_err(|e| e.to_string())?.clone();
    for dir in &cfg.watched_directories {
        match scan_directory_internal(
            &table,
            thumb_dir,
            dir,
            embedding_backend.as_deref(),
            model_id.as_deref(),
            Some(progress.clone()),
        ).await {
            Ok((result, seen_paths)) => {
                total_result.images_found += result.images_found;
                total_result.images_added += result.images_added;
                total_result.images_updated += result.images_updated;
                total_result.errors.extend(result.errors);
                all_seen_paths.extend(seen_paths);
            }
            Err(e) => {
                total_result.errors.push(format!("Error scanning {}: {}", dir, e));
            }
        }
    }

    let db_paths = database::get_all_paths(&table).await?;
    let to_remove: Vec<String> = db_paths
        .into_iter()
        .filter(|p| !all_seen_paths.contains(p))
        .collect();

    if !to_remove.is_empty() {
        total_result.images_removed = to_remove.len() as u32;
        thumbnail::delete_thumbnails(thumb_dir, &to_remove)?;
        database::remove_images(&table, &to_remove).await?;
    }

    Ok(total_result)
}

#[tauri::command]
async fn get_thumbnail_path(state: tauri::State<'_, AppState>, image_path: String) -> Result<String, String> {
    let thumb_dir = state.thumbnails_dir.clone();

    tokio::task::spawn_blocking(move || {
        let source = Path::new(&image_path);
        thumbnail::get_thumbnail_path_for_asset(&thumb_dir, source)
    })
    .await
    .map_err(|e| format!("Task join error: {}", e))?
}

#[tauri::command]
async fn get_watched_directories(state: tauri::State<'_, AppState>) -> Result<Vec<String>, String> {
    Ok(config::get_config(&state).watched_directories)
}

#[tauri::command]
async fn get_indexed_count(state: tauri::State<'_, AppState>) -> Result<u32, String> {
    let table = state.table.lock().await;
    let paths = database::get_all_paths(&table).await?;
    Ok(paths.len() as u32)
}

#[tauri::command]
async fn get_all_images(state: tauri::State<'_, AppState>) -> Result<Vec<ImageInfo>, String> {
    let table = state.table.lock().await;
    database::get_all_images(&table).await
}

/// Search for images using semantic similarity if the embedding model is available,
/// otherwise fall back to filename search.
///
/// Note: Currently searches using embedding slot 1 only. The schema supports
/// slots 2-5 to allow switching models without recalculating embeddings, but
/// only slot 1 is used for search at this time.
#[tauri::command]
async fn search_images(state: tauri::State<'_, AppState>, query: String) -> Result<Vec<database::SearchResult>, String> {
    let table = state.table.lock().await;
    if query.trim().is_empty() {
        let images = database::get_all_images(&table).await?;
        let results = images
            .into_iter()
            .map(|img| database::SearchResult {
                path: img.path,
                file_type: img.file_type,
                file_size: img.file_size,
                created_at: img.created_at,
                modified_at: img.modified_at,
                sort_score: None,
            })
            .collect();
        return Ok(results);
    }

    // Try to generate text embedding using the backend
    let query_embedding = if let Ok(backend_guard) = state.embedding_backend.read() {
        if let Some(backend) = backend_guard.as_ref() {
            match backend.embed_text(&query) {
                Ok(emb) => Some(emb),
                Err(e) => {
                    eprintln!("Text embedding failed, falling back to filename search: {}", e);
                    None
                }
            }
        } else {
            None
        }
    } else {
        None
    };

    // Use vector search if we have an embedding, otherwise fall back to filename search
    if let Some(embedding) = query_embedding {
        // Return top 100 results, user can scroll through them
        database::search_by_embedding(&table, &embedding, 100).await
    } else {
        database::search_by_filename(&table, &query).await
    }
}

/// Search for images similar to a given image by path.
/// Uses the stored embedding from the database (does not re-compute).
#[tauri::command]
async fn search_similar_images(
    state: tauri::State<'_, AppState>,
    image_path: String,
) -> Result<Vec<database::SearchResult>, String> {
    let table = state.table.lock().await;

    // Get the embedding for the source image
    let embedding = database::get_image_embedding(&table, &image_path)
        .await?
        .ok_or_else(|| format!("Image has no embedding: {}", image_path))?;

    // Search for similar images (return top 100)
    database::search_by_embedding(&table, &embedding, 100).await
}

/// Search for images with filter and sort options.
#[tauri::command]
async fn search_images_filtered(
    state: tauri::State<'_, AppState>,
    query: String,
    filter: FilterOptions,
    sort: SortOptions,
) -> Result<Vec<database::SearchResult>, String> {
    let table = state.table.lock().await;

    // Generate text embedding for the query if available
    let query_embedding = if !query.trim().is_empty() {
        if let Ok(backend_guard) = state.embedding_backend.read() {
            if let Some(backend) = backend_guard.as_ref() {
                match backend.embed_text(&query) {
                    Ok(emb) => Some(emb),
                    Err(e) => {
                        eprintln!("Text embedding failed: {}", e);
                        None
                    }
                }
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    };

    database::search_filtered(
        &table,
        query_embedding.as_deref(),
        &filter,
        &sort,
        100,
    )
    .await
}

/// Get all distinct file types in the database.
#[tauri::command]
async fn get_file_types(state: tauri::State<'_, AppState>) -> Result<Vec<String>, String> {
    let table = state.table.lock().await;
    database::get_file_types(&table).await
}

#[tauri::command]
async fn open_image(app: AppHandle, path: String) -> Result<(), String> {
    app.opener()
        .open_path(&path, None::<&str>)
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn show_in_folder(app: AppHandle, path: String) -> Result<(), String> {
    app.opener()
        .reveal_item_in_dir(&path)
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn delete_all_thumbnails(state: tauri::State<'_, AppState>) -> Result<(), String> {
    let thumb_dir = &state.thumbnails_dir;

    if thumb_dir.exists() {
        std::fs::remove_dir_all(thumb_dir).map_err(|e| e.to_string())?;
    }

    Ok(())
}

#[tauri::command]
async fn clear_database(app: AppHandle, state: tauri::State<'_, AppState>) -> Result<(), String> {
    let db_path = database::db_path(&app)?;

    let mut table = state.table.lock().await;

    if db_path.exists() {
        std::fs::remove_dir_all(&db_path).map_err(|e| e.to_string())?;
    }

    // Recreate the table so the shared state remains valid
    *table = database::get_or_create_table(&state.db).await?;

    Ok(())
}

#[tauri::command]
async fn open_app_data_folder(app: AppHandle) -> Result<(), String> {
    let app_data = app.path().app_local_data_dir().map_err(|e| e.to_string())?;

    // Create the directory if it doesn't exist
    if !app_data.exists() {
        std::fs::create_dir_all(&app_data).map_err(|e| e.to_string())?;
    }

    app.opener()
        .open_path(app_data.to_string_lossy().to_string(), None::<&str>)
        .map_err(|e| e.to_string())
}

/// Return a human-readable label for this build variant, e.g. "Windows x64 (CUDA)".
#[tauri::command]
fn get_build_variant() -> String {
    let os = if cfg!(target_os = "windows") {
        "Windows"
    } else if cfg!(target_os = "macos") {
        "macOS"
    } else {
        "Linux"
    };

    let arch = if cfg!(target_arch = "x86_64") {
        "x64"
    } else if cfg!(target_arch = "aarch64") {
        "ARM64"
    } else {
        std::env::consts::ARCH
    };

    let accel = if cfg!(feature = "backend-cuda") {
        " (CUDA)"
    } else {
        " (CPU)"
    };

    format!("{} {}{}", os, arch, accel)
}

/// Show resolved paths for bundled dependencies.
///
/// Reports the backend feature, ONNX Runtime library, model files,
/// and any backend-specific provider libraries.
#[tauri::command]
fn get_dependency_paths(app: AppHandle) -> Vec<(String, String)> {
    let bundled = bundled_dir(&app);
    let lib_dir = bundled.join("lib");
    let model_path = bundled.join("model");

    let status = |p: &Path| -> String {
        if p.exists() {
            p.display().to_string()
        } else {
            format!("{} (NOT FOUND)", p.display())
        }
    };

    /// Check for a library in the bundled lib dir first, then search PATH.
    /// Only used by backends that require extra provider DLLs (e.g. CUDA).
    #[allow(dead_code)]
    fn find_lib(lib_dir: &Path, filename: &str) -> String {
        let bundled_path = lib_dir.join(filename);
        if bundled_path.exists() {
            return bundled_path.display().to_string();
        }
        let path_var = std::env::var("PATH").unwrap_or_default();
        let sep = if cfg!(target_os = "windows") { ';' } else { ':' };
        for dir in path_var.split(sep) {
            let p = Path::new(dir).join(filename);
            if p.exists() {
                return format!("{} (system)", p.display());
            }
        }
        format!("{} (NOT FOUND)", bundled_path.display())
    }

    let backend_name = if cfg!(feature = "backend-cuda") {
        "CUDA"
    } else {
        "CPU"
    };

    let mut deps = vec![
        ("Backend".into(), backend_name.into()),
        // ONNX Runtime
        ("ONNX Runtime".into(), status(&lib_dir.join(ort_lib_filename()))),
    ];

    // Backend-specific provider libraries
    #[cfg(feature = "backend-cuda")]
    {
        deps.push(("ORT CUDA Provider".into(), find_lib(&lib_dir, if cfg!(target_os = "windows") { "onnxruntime_providers_cuda.dll" } else { "libonnxruntime_providers_cuda.so" })));
        deps.push(("ORT Shared Provider".into(), find_lib(&lib_dir, if cfg!(target_os = "windows") { "onnxruntime_providers_shared.dll" } else { "libonnxruntime_providers_shared.so" })));

        if cfg!(target_os = "windows") {
            for (label, dll) in [
                ("CUDA Runtime", "cudart64_12.dll"),
                ("cuBLAS", "cublas64_12.dll"),
                ("cuBLAS Lt", "cublasLt64_12.dll"),
                ("cuDNN", "cudnn64_9.dll"),
                ("cuDNN Ops", "cudnn_ops64_9.dll"),
                ("cuDNN CNN", "cudnn_cnn64_9.dll"),
            ] {
                deps.push((label.into(), find_lib(&lib_dir, dll)));
            }
        } else if cfg!(target_os = "linux") {
            for (label, so) in [
                ("CUDA Runtime", "libcudart.so.12"),
                ("cuBLAS", "libcublas.so.12"),
                ("cuBLAS Lt", "libcublasLt.so.12"),
                ("cuDNN", "libcudnn.so.9"),
                ("cuDNN Ops", "libcudnn_ops.so.9"),
                ("cuDNN CNN", "libcudnn_cnn.so.9"),
            ] {
                deps.push((label.into(), find_lib(&lib_dir, so)));
            }
        }
    }

    // Model files
    deps.push(("Model directory".into(), status(&model_path)));
    deps.push(("  vision_model.onnx".into(), status(&model_path.join("onnx").join("vision_model.onnx"))));
    deps.push(("  text_model.onnx".into(), status(&model_path.join("onnx").join("text_model.onnx"))));
    deps.push(("  tokenizer.json".into(), status(&model_path.join("tokenizer.json"))));

    deps
}

/// All URLs the CI build pipeline would download from when creating bundled releases.
const BUNDLE_URLS: &[(&str, &str)] = &[
    // ONNX Runtime — per-platform builds
    ("ORT Windows x64 CPU",   "https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-win-x64-1.23.2.zip"),
    ("ORT Windows x64 CUDA",  "https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-win-x64-gpu-1.23.2.zip"),
    ("ORT macOS ARM64",        "https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-osx-arm64-1.23.2.tgz"),
    ("ORT macOS x64",          "https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-osx-x86_64-1.23.2.tgz"),
    ("ORT Linux x64 CPU",     "https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-linux-x64-1.23.2.tgz"),
    ("ORT Linux x64 CUDA",    "https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-linux-x64-gpu-1.23.2.tgz"),

    // SigLIP2 model files (platform-independent)
    ("Model: vision_model.onnx", "https://huggingface.co/onnx-community/siglip2-base-patch16-256-ONNX/resolve/main/onnx/vision_model.onnx"),
    ("Model: text_model.onnx",   "https://huggingface.co/onnx-community/siglip2-base-patch16-256-ONNX/resolve/main/onnx/text_model.onnx"),
    ("Model: tokenizer.json",    "https://huggingface.co/onnx-community/siglip2-base-patch16-256-ONNX/resolve/main/tokenizer.json"),

    // NVIDIA CUDA redistributables (Windows CUDA builds only)
    ("CUDA: cudart (Windows)",  "https://developer.download.nvidia.com/compute/cuda/redist/cuda_cudart/windows-x86_64/cuda_cudart-windows-x86_64-12.8.90-archive.zip"),
    ("CUDA: cublas (Windows)",  "https://developer.download.nvidia.com/compute/cuda/redist/libcublas/windows-x86_64/libcublas-windows-x86_64-12.8.4.1-archive.zip"),
    ("CUDA: cuDNN (Windows)",   "https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/cudnn-windows-x86_64-9.19.0.56_cuda12-archive.zip"),

    // NVIDIA CUDA redistributables (Linux CUDA builds only)
    ("CUDA: cudart (Linux)",  "https://developer.download.nvidia.com/compute/cuda/redist/cuda_cudart/linux-x86_64/cuda_cudart-linux-x86_64-12.8.90-archive.tar.xz"),
    ("CUDA: cublas (Linux)",  "https://developer.download.nvidia.com/compute/cuda/redist/libcublas/linux-x86_64/libcublas-linux-x86_64-12.8.4.1-archive.tar.xz"),
    ("CUDA: cuDNN (Linux)",   "https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-9.19.0.56_cuda12-archive.tar.xz"),
];

/// Test all bundle download URLs by sending HEAD requests.
/// Returns a list of (label, url, status) tuples.
#[tauri::command]
async fn test_bundle_urls() -> Vec<(String, String, String)> {
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::limited(5))
        .build()
        .unwrap_or_default();

    let mut results = Vec::new();
    for (label, url) in BUNDLE_URLS {
        let status = match client.head(*url).send().await {
            Ok(resp) => {
                let code = resp.status();
                if code.is_success() {
                    let size = resp.headers()
                        .get(reqwest::header::CONTENT_LENGTH)
                        .and_then(|v| v.to_str().ok())
                        .and_then(|v| v.parse::<u64>().ok());
                    match size {
                        Some(bytes) => format!("OK ({:.1} MB)", bytes as f64 / 1_048_576.0),
                        None => "OK".into(),
                    }
                } else {
                    format!("FAIL ({})", code)
                }
            }
            Err(e) => format!("ERROR ({})", e),
        };
        results.push((label.to_string(), url.to_string(), status));
    }
    results
}

/// Toggle benchmark CSV logging on or off for this session.
#[tauri::command]
fn toggle_benchmarking() -> bool {
    let new_value = !benchmark::is_enabled();
    benchmark::set_enabled(new_value);
    new_value
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            let cfg = config::load_config(app.handle())?;

            let menu = menu::build_menu(app, &cfg)?;
            app.set_menu(menu)?;

            // Handle menu events
            app.on_menu_event(|app, event| {
                let _ = app.emit("menu-event", event.id().0.as_str());
            });

            let handle = app.handle().clone();

            // Initialize benchmark CSV logger
            let app_data = handle.path().app_local_data_dir().expect("Failed to get app data dir");
            std::fs::create_dir_all(&app_data).ok();
            benchmark::init(&app_data);

            // Compute thumbnails directory once (never changes)
            let thumbnails_dir = app_data.join("thumbnails");

            // Phase 1: Quick initialization (database only) - blocks briefly
            // This is fast and required for the app to function at all
            let (db, table) = tauri::async_runtime::block_on(async {
                let db = database::open_connection(&handle).await?;
                let table = database::get_or_create_table(&db).await?;
                Ok::<(Connection, Table), String>((db, table))
            })
            .expect("Failed to initialize database");

            // Register AppState with empty embedding backend (will be populated async)
            handle.manage(AppState {
                db,
                table: tokio::sync::Mutex::new(table),
                embedding_backend: RwLock::new(None), // Will be Some(Arc<EmbeddingBackend>) after async init
                model_id: RwLock::new(None),
                config: RwLock::new(cfg.clone()),
                thumbnails_dir,
            });

            // Phase 2: Heavy initialization (embedding models) - runs in background
            // This allows the UI to appear immediately while models load
            let handle_for_task = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                // Resolve bundled resource paths
                let bundled = bundled_dir(&handle_for_task);
                let ort_path = bundled.join("lib").join(ort_lib_filename());
                let model_path = bundled.join("model");

                println!("ORT library: {}", ort_path.display());
                println!("Model directory: {}", model_path.display());

                // Initialize ONNX Runtime
                if let Err(e) = embedding::init_ort(&ort_path) {
                    eprintln!("Failed to initialize ONNX Runtime: {}", e);
                    let _ = handle_for_task.emit("model_ready", ());
                    return;
                }

                let model_id = Some("siglip2-base-patch16-256".to_string());

                let (embedding_backend, model_id) = match EmbeddingBackend::load(&model_path) {
                    Ok(backend) => (Some(Arc::new(backend)), model_id),
                    Err(e) => {
                        eprintln!("Failed to load embedding backend: {}", e);
                        (None, None)
                    }
                };

                // Update the AppState with loaded backend
                if let Some(state) = handle_for_task.try_state::<AppState>() {
                    if let Ok(mut backend_guard) = state.embedding_backend.write() {
                        *backend_guard = embedding_backend;
                    }
                    if let Ok(mut id_guard) = state.model_id.write() {
                        *id_guard = model_id;
                    }
                }

                // Emit event to notify frontend that model loading is complete
                let _ = handle_for_task.emit("model_ready", ());
            });

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_config,
            get_embedding_model_status,
            add_watched_directory,
            remove_watched_directory,
            rescan_all,
            get_thumbnail_path,
            get_watched_directories,
            get_indexed_count,
            get_all_images,
            search_images,
            search_similar_images,
            search_images_filtered,
            get_file_types,
            open_image,
            show_in_folder,
            delete_all_thumbnails,
            clear_database,
            open_app_data_folder,
            toggle_benchmarking,
            get_build_variant,
            get_dependency_paths,
            test_bundle_urls
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_validate_directory_exists() {
        let temp = std::env::temp_dir();
        let result = validate_directory(temp.to_str().unwrap());
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_directory_nonexistent() {
        let result = validate_directory("/this/path/definitely/does/not/exist/12345");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("does not exist"));
    }

    #[test]
    fn test_validate_directory_file_not_dir() {
        let temp_dir = std::env::temp_dir();
        let temp_file = temp_dir.join("test_validate_file.txt");
        fs::write(&temp_file, "test").unwrap();

        let result = validate_directory(temp_file.to_str().unwrap());
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not a directory"));

        let _ = fs::remove_file(&temp_file);
    }
}
