use lancedb::{Connection, Table};
use serde::Serialize;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::RwLock;
use tauri::{AppHandle, Emitter, Manager};
use tauri_plugin_opener::OpenerExt;

mod benchmark;
mod config;
mod database;
mod embedding;
mod image_ops;
mod menu;
mod ort_download;
mod scan;
mod state;
mod thumbnail;

use config::AppConfig;
use database::{FilterOptions, ImageInfo, SortOptions};
use embedding::{GpuEmbeddingModel, GPU_BATCH_SIZE};
use scan::{ScanProgressState, ScanResult, scan_directory_internal};
use state::{AppState, EmbeddingBackend, EmbeddingPool, MAX_EMBEDDING_WORKERS};

fn thumbnails_dir(app: &AppHandle, _config: &AppConfig) -> Result<PathBuf, String> {
    let app_data = app.path().app_local_data_dir().map_err(|e| e.to_string())?;
    Ok(app_data.join("thumbnails"))
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

#[derive(Debug, Clone, Serialize)]
pub struct SiglipConfigInfo {
    pub has_text: bool,
    pub has_vision: bool,
    pub text_hidden_size: Option<i64>,
    pub vision_hidden_size: Option<i64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingTestResult {
    pub model_loaded: bool,
    pub image_embedding_dim: Option<usize>,
    pub text_embedding_dim: Option<usize>,
    pub similarity: Option<f32>,
    pub error: Option<String>,
}

#[tauri::command]
fn inspect_siglip_config(path: String) -> Result<SiglipConfigInfo, String> {
    let content = std::fs::read_to_string(&path).map_err(|e| e.to_string())?;
    let json: serde_json::Value = serde_json::from_str(&content).map_err(|e| e.to_string())?;

    let text_config = json.get("text_config");
    let vision_config = json.get("vision_config");

    let text_hidden_size = text_config
        .and_then(|v| v.get("hidden_size"))
        .and_then(|v| v.as_i64());

    let vision_hidden_size = vision_config
        .and_then(|v| v.get("hidden_size"))
        .and_then(|v| v.as_i64());

    Ok(SiglipConfigInfo {
        has_text: text_config.is_some(),
        has_vision: vision_config.is_some(),
        text_hidden_size,
        vision_hidden_size,
    })
}

#[tauri::command]
fn test_embedding(app: AppHandle, model_dir: String, image_path: String, query: String) -> Result<EmbeddingTestResult, String> {
    use embedding::EmbeddingModel;

    let cfg = config::load_config(&app)?;
    let runtime_type = cfg.runtime_type
        .as_deref()
        .and_then(ort_download::RuntimeType::from_str)
        .unwrap_or(ort_download::RuntimeType::Cpu);
    let ort_path = ort_download::get_ort_library_path(&app, runtime_type)?
        .ok_or_else(|| format!("{} runtime is not installed.", runtime_type.display_name()))?;

    let model_path = Path::new(&model_dir);
    let image_file = Path::new(&image_path);

    // Initialize ONNX Runtime using the selected downloaded runtime library.
    if let Err(e) = embedding::init_ort(&ort_path) {
        return Ok(EmbeddingTestResult {
            model_loaded: false,
            image_embedding_dim: None,
            text_embedding_dim: None,
            similarity: None,
            error: Some(format!("Failed to initialize ONNX Runtime: {}", e)),
        });
    }

    // Try to load the model (test always uses CPU for simplicity)
    let mut model = match EmbeddingModel::load(model_path, false) {
        Ok(m) => m,
        Err(e) => {
            return Ok(EmbeddingTestResult {
                model_loaded: false,
                image_embedding_dim: None,
                text_embedding_dim: None,
                similarity: None,
                error: Some(format!("Failed to load model: {}", e)),
            });
        }
    };

    // Try to embed the image
    let image_emb = match model.embed_image(image_file) {
        Ok(emb) => emb,
        Err(e) => {
            return Ok(EmbeddingTestResult {
                model_loaded: true,
                image_embedding_dim: None,
                text_embedding_dim: None,
                similarity: None,
                error: Some(format!("Failed to embed image: {}", e)),
            });
        }
    };

    // Try to embed the text
    let text_emb = match model.embed_text(&query) {
        Ok(emb) => emb,
        Err(e) => {
            return Ok(EmbeddingTestResult {
                model_loaded: true,
                image_embedding_dim: Some(image_emb.len()),
                text_embedding_dim: None,
                similarity: None,
                error: Some(format!("Failed to embed text: {}", e)),
            });
        }
    };

    // Compute cosine similarity (vectors are already L2 normalized, so dot product = cosine sim)
    let similarity: f32 = image_emb.iter().zip(text_emb.iter()).map(|(a, b)| a * b).sum();

    Ok(EmbeddingTestResult {
        model_loaded: true,
        image_embedding_dim: Some(image_emb.len()),
        text_embedding_dim: Some(text_emb.len()),
        similarity: Some(similarity),
        error: None,
    })
}

#[tauri::command]
async fn get_config(app: AppHandle) -> Result<AppConfig, String> {
    config::load_config(&app)
}

#[tauri::command]
async fn set_model_config(app: AppHandle, model_dir: String) -> Result<(), String> {
    let mut cfg = config::load_config(&app)?;
    cfg.model_dir = Some(model_dir);
    config::save_config(&app, &cfg)?;
    Ok(())
}

#[tauri::command]
fn get_embedding_model_status(state: tauri::State<'_, AppState>) -> bool {
    state.embedding_backend.read().map(|p| p.is_some()).unwrap_or(false)
}

#[tauri::command]
async fn add_watched_directory(app: AppHandle, state: tauri::State<'_, AppState>, path: String) -> Result<ScanResult, String> {
    validate_directory(&path)?;
    let mut cfg = config::load_config(&app)?;

    if !cfg.watched_directories.contains(&path) {
        cfg.watched_directories.push(path.clone());
        config::save_config(&app, &cfg)?;
    }

    let thumb_dir = thumbnails_dir(&app, &cfg)?;
    let table = state.table.lock().await;

    let progress = ScanProgressState::new(&app);
    // Clone Arc and String out of the locks before the await point
    let embedding_backend = state.embedding_backend.read().map_err(|e| e.to_string())?.clone();
    let model_id = state.model_id.read().map_err(|e| e.to_string())?.clone();
    let (result, _) = scan_directory_internal(
        &table,
        &thumb_dir,
        &path,
        embedding_backend.as_deref(),
        model_id.as_deref(),
        Some(progress),
    ).await?;
    Ok(result)
}

#[tauri::command]
async fn remove_watched_directory(app: AppHandle, state: tauri::State<'_, AppState>, path: String) -> Result<(), String> {
    let mut cfg = config::load_config(&app)?;
    cfg.watched_directories.retain(|p| p != &path);
    config::save_config(&app, &cfg)?;

    let thumb_dir = thumbnails_dir(&app, &cfg)?;

    let removed_path = Path::new(&path);
    let table = state.table.lock().await;
    let all_paths = database::get_all_paths(&table).await?;
    let to_remove: Vec<String> = all_paths
        .into_iter()
        .filter(|p| Path::new(p).starts_with(removed_path))
        .collect();

    if !to_remove.is_empty() {
        let progress = ScanProgressState::new(&app);
        progress.set_phase("removing");
        progress.set_total(to_remove.len());
        for chunk in to_remove.chunks(500) {
            thumbnail::delete_thumbnails(&thumb_dir, chunk)?;
            database::remove_images(&table, chunk).await?;
            progress.increment_by(chunk.len());
        }
    }

    Ok(())
}

#[tauri::command]
async fn rescan_all(app: AppHandle, state: tauri::State<'_, AppState>) -> Result<ScanResult, String> {
    let cfg = config::load_config(&app)?;
    let thumb_dir = thumbnails_dir(&app, &cfg)?;
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
            &thumb_dir,
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
        thumbnail::delete_thumbnails(&thumb_dir, &to_remove)?;
        database::remove_images(&table, &to_remove).await?;
    }

    Ok(total_result)
}

#[tauri::command]
async fn get_thumbnail_path(app: AppHandle, image_path: String) -> Result<String, String> {
    let cfg = config::load_config(&app)?;
    let thumb_dir = thumbnails_dir(&app, &cfg)?;

    tokio::task::spawn_blocking(move || {
        let source = Path::new(&image_path);
        thumbnail::get_thumbnail_path_for_asset(&thumb_dir, source)
    })
    .await
    .map_err(|e| format!("Task join error: {}", e))?
}

#[tauri::command]
async fn get_watched_directories(app: AppHandle) -> Result<Vec<String>, String> {
    let cfg = config::load_config(&app)?;
    Ok(cfg.watched_directories)
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
async fn delete_all_thumbnails(app: AppHandle) -> Result<(), String> {
    let cfg = config::load_config(&app)?;
    let thumb_dir = thumbnails_dir(&app, &cfg)?;

    if thumb_dir.exists() {
        std::fs::remove_dir_all(&thumb_dir).map_err(|e| e.to_string())?;
    }

    Ok(())
}

#[tauri::command]
async fn clear_database(app: AppHandle, state: tauri::State<'_, AppState>) -> Result<(), String> {
    let cfg = config::load_config(&app)?;
    let db_path = database::db_path(&app, &cfg)?;

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

// ============================================================================
// ONNX Runtime download commands
// ============================================================================

#[derive(Clone, serde::Serialize)]
pub struct OrtDownloadProgress {
    pub downloaded: u64,
    pub total: u64,
}

/// Get the current ONNX Runtime installation status.
#[tauri::command]
fn get_ort_status(app: AppHandle) -> Result<ort_download::OrtStatus, String> {
    ort_download::get_ort_status(&app)
}

/// Get the download size for a runtime type.
#[tauri::command]
fn get_ort_download_size(runtime_type: String) -> Result<Option<u64>, String> {
    let rt = ort_download::RuntimeType::from_str(&runtime_type)
        .ok_or_else(|| format!("Invalid runtime type: {}", runtime_type))?;
    Ok(ort_download::get_download_size(rt))
}

/// Download and install ONNX Runtime.
#[tauri::command]
async fn download_ort(app: AppHandle, runtime_type: String) -> Result<String, String> {
    let rt = ort_download::RuntimeType::from_str(&runtime_type)
        .ok_or_else(|| format!("Invalid runtime type: {}", runtime_type))?;

    // Create a channel for progress updates
    let app_clone = app.clone();
    let progress_callback = move |downloaded: u64, total: u64| {
        let _ = app_clone.emit("ort_download_progress", OrtDownloadProgress { downloaded, total });
    };

    let lib_path = ort_download::download_ort(&app, rt, progress_callback).await?;

    // Save the runtime type to config after successful download
    let mut cfg = config::load_config(&app)?;
    cfg.runtime_type = Some(runtime_type.clone());
    config::save_config(&app, &cfg)?;

    Ok(lib_path.to_string_lossy().to_string())
}

/// Set the runtime type in config (for selecting an already-installed runtime).
#[tauri::command]
async fn set_runtime_type(app: AppHandle, runtime_type: String) -> Result<(), String> {
    let rt = ort_download::RuntimeType::from_str(&runtime_type)
        .ok_or_else(|| format!("Invalid runtime type: {}", runtime_type))?;

    // Verify the runtime is installed
    if !ort_download::is_runtime_installed(&app, rt)? {
        return Err(format!("Runtime '{}' is not installed", runtime_type));
    }

    let mut cfg = config::load_config(&app)?;
    cfg.runtime_type = Some(runtime_type);
    config::save_config(&app, &cfg)?;

    Ok(())
}

/// Check if CUDA runtime is available for this platform.
#[tauri::command]
fn is_cuda_available() -> bool {
    ort_download::Platform::detect()
        .map(|p| p.cuda_available())
        .unwrap_or(false)
}

/// Uninstall a runtime.
#[tauri::command]
async fn uninstall_runtime(app: AppHandle, runtime_type: String) -> Result<(), String> {
    let rt = ort_download::RuntimeType::from_str(&runtime_type)
        .ok_or_else(|| format!("Invalid runtime type: {}", runtime_type))?;

    ort_download::uninstall_runtime(&app, rt)?;

    // If this was the selected runtime, clear the selection
    let mut cfg = config::load_config(&app)?;
    if cfg.runtime_type.as_deref() == Some(runtime_type.as_str()) {
        cfg.runtime_type = None;
        config::save_config(&app, &cfg)?;
    }

    Ok(())
}

/// Check CUDA system dependencies.
#[tauri::command]
fn check_cuda_dependencies() -> ort_download::CudaDependencyStatus {
    ort_download::check_cuda_dependencies()
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

            // Phase 1: Quick initialization (database only) - blocks briefly
            // This is fast and required for the app to function at all
            let (db, table) = tauri::async_runtime::block_on(async {
                let db = database::open_connection(&handle, &cfg).await?;
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
            });

            // TODO: Automatically trigger a full rescan of watched directories on app launch.
            // Phase 2: Heavy initialization (embedding models) - runs in background
            // This allows the UI to appear immediately while models load
            let handle_for_task = app.handle().clone();
            let cfg_for_task = cfg.clone();
            tauri::async_runtime::spawn(async move {
                // Determine the runtime type from config (default to CPU)
                let runtime_type = cfg_for_task.runtime_type
                    .as_deref()
                    .and_then(ort_download::RuntimeType::from_str)
                    .unwrap_or(ort_download::RuntimeType::Cpu);

                // Determine the ONNX Runtime library path from downloaded runtime.
                let ort_path: Option<PathBuf> = match ort_download::get_ort_library_path(&handle_for_task, runtime_type) {
                    Ok(Some(p)) => {
                        println!("Using {} runtime: {}", runtime_type.display_name(), p.display());
                        Some(p)
                    }
                    Ok(None) => {
                        println!("{} runtime not installed. Use Model > Select Runtime to download it.", runtime_type.display_name());
                        None
                    }
                    Err(e) => {
                        eprintln!("Warning: Failed to check for ONNX Runtime: {}", e);
                        None
                    }
                };

                // Check if CUDA runtime is configured (DirectML would need different handling)
                let use_gpu = runtime_type == ort_download::RuntimeType::Cuda;

                // Try to load the embedding backend if ORT and model are available
                let (embedding_backend, model_id) = match (ort_path, &cfg_for_task.model_dir) {
                    (Some(ort_path), Some(model_path)) => {
                        let model_path = Path::new(model_path);

                        // Extract model ID from directory name
                        let model_id = model_path
                            .file_name()
                            .and_then(|n| n.to_str())
                            .map(|s| s.to_string());

                        // Initialize ONNX Runtime
                        if let Err(e) = embedding::init_ort(&ort_path) {
                            eprintln!("Warning: Failed to initialize ONNX Runtime: {}", e);
                            (None, None)
                        } else if use_gpu {
                            // GPU mode: single model with batched inference
                            match GpuEmbeddingModel::load(model_path) {
                                Ok(model) => {
                                    println!("Using GPU backend with batched inference (batch size: {})", GPU_BATCH_SIZE);
                                    (Some(Arc::new(EmbeddingBackend::Gpu(Mutex::new(model)))), model_id)
                                }
                                Err(e) => {
                                    eprintln!("Warning: Failed to load GPU embedding model: {}", e);
                                    (None, None)
                                }
                            }
                        } else {
                            // CPU mode: pool of models for thread-parallel inference
                            let num_workers = std::thread::available_parallelism()
                                .map(|n| n.get().min(MAX_EMBEDDING_WORKERS))
                                .unwrap_or(2);

                            match EmbeddingPool::new(model_path, num_workers, false) {
                                Ok(pool) => {
                                    println!("Using CPU backend with {} parallel workers", pool.len());
                                    (Some(Arc::new(EmbeddingBackend::Cpu(pool))), model_id)
                                }
                                Err(e) => {
                                    eprintln!("Warning: Failed to load CPU embedding model pool: {}", e);
                                    (None, None)
                                }
                            }
                        }
                    }
                    (None, _) => {
                        println!("ONNX Runtime not available. Semantic search will be disabled.");
                        (None, None)
                    }
                    (_, None) => {
                        println!("Model directory not configured. Semantic search will be disabled.");
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
            inspect_siglip_config,
            test_embedding,
            get_config,
            set_model_config,
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
            get_ort_status,
            get_ort_download_size,
            download_ort,
            set_runtime_type,
            uninstall_runtime,
            check_cuda_dependencies,
            is_cuda_available,
            clear_database,
            open_app_data_folder
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
