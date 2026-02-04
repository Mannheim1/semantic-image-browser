use lancedb::{Connection, Table};
use serde::Serialize;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::RwLock;
use tauri::menu::{CheckMenuItemBuilder, MenuBuilder, MenuItemBuilder, PredefinedMenuItem, SubmenuBuilder};
use tauri::{AppHandle, Emitter, Manager};
use tauri_plugin_opener::OpenerExt;

mod config;
mod database;
mod embedding;
mod ort_download;
mod scanner;
mod thumbnail;

use config::AppConfig;
use database::{FilterOptions, ImageInfo, ImageRecord, SortOptions};
use embedding::{EmbeddingModel, GpuEmbeddingModel, GPU_BATCH_SIZE};

/// Maximum number of embedding model instances to keep in the pool.
/// This limits RAM usage (~500MB per model) while enabling parallel processing.
/// 4 workers = ~2GB for models, which is reasonable for most systems.
const MAX_EMBEDDING_WORKERS: usize = 4;

/// Holds either a CPU embedding pool or a GPU embedding model.
/// CPU mode uses multiple model instances for thread-parallel inference.
/// GPU mode uses a single model with batched inference for maximum GPU utilization.
pub enum EmbeddingBackend {
    Cpu(EmbeddingPool),
    Gpu(Mutex<GpuEmbeddingModel>),
}

impl EmbeddingBackend {
    /// Embed text using whichever backend is available.
    pub fn embed_text(&self, text: &str) -> Result<Vec<f32>, String> {
        match self {
            EmbeddingBackend::Cpu(pool) => pool.embed_text(text),
            EmbeddingBackend::Gpu(model) => {
                let mut guard = model.lock().map_err(|e| format!("Failed to lock GPU model: {}", e))?;
                guard.embed_text(text)
            }
        }
    }

    /// Returns true if this is the GPU backend.
    pub fn is_gpu(&self) -> bool {
        matches!(self, EmbeddingBackend::Gpu(_))
    }
}

pub struct AppState {
    pub db: Connection,
    pub table: tokio::sync::Mutex<Table>,
    /// Embedding backend - either CPU pool or GPU model.
    /// Wrapped in RwLock<Option<Arc<...>>> to allow async initialization after app starts.
    /// Arc allows cloning a reference to use across await points.
    pub embedding_backend: RwLock<Option<Arc<EmbeddingBackend>>>,
    /// Model identifier (e.g., "siglip2-base-patch16-256") for database storage.
    /// Wrapped in RwLock to allow async initialization after app starts.
    pub model_id: RwLock<Option<String>>,
}

/// A pool of embedding models for parallel inference.
pub struct EmbeddingPool {
    models: Vec<Mutex<EmbeddingModel>>,
}

impl EmbeddingPool {
    /// Create a new pool with up to `count` model instances.
    /// If `use_gpu` is true, attempts to use CUDA execution provider.
    pub fn new(model_dir: &Path, count: usize, use_gpu: bool) -> Result<Self, String> {
        let mut models = Vec::with_capacity(count);
        for i in 0..count {
            match EmbeddingModel::load(model_dir, use_gpu) {
                Ok(model) => models.push(Mutex::new(model)),
                Err(e) => {
                    // If we failed to load any models, that's an error.
                    // If we loaded at least one, we can continue with fewer workers.
                    if models.is_empty() {
                        return Err(format!("Failed to load embedding model: {}", e));
                    } else {
                        eprintln!("Warning: Could only load {} of {} embedding models: {}", i, count, e);
                        break;
                    }
                }
            }
        }
        let runtime_label = if use_gpu { "GPU" } else { "CPU" };
        println!("Loaded {} embedding model instance(s) for parallel processing ({})", models.len(), runtime_label);
        Ok(Self { models })
    }

    /// Get the number of models in the pool.
    pub fn len(&self) -> usize {
        self.models.len()
    }

    /// Get a reference to a model by index.
    pub fn get(&self, index: usize) -> Option<&Mutex<EmbeddingModel>> {
        self.models.get(index)
    }

    /// Embed text using the first available model (for search queries).
    pub fn embed_text(&self, text: &str) -> Result<Vec<f32>, String> {
        let model = self.models.first().ok_or("No models in pool")?;
        let mut guard = model.lock().map_err(|e| format!("Failed to lock model: {}", e))?;
        guard.embed_text(text)
    }
}

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
pub struct ScanResult {
    pub images_found: u32,
    pub images_added: u32,
    pub images_updated: u32,
    pub images_removed: u32,
    pub errors: Vec<String>,
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

#[derive(Clone, Serialize)]
pub struct ScanProgressPayload {
    pub phase: String,
    pub current: u64,
    pub total: u64,
}

#[derive(Clone)]
pub struct ScanProgressState {
    app: AppHandle,
    phase: Arc<std::sync::RwLock<String>>,
    total: Arc<AtomicUsize>,
    current: Arc<AtomicUsize>,
}

impl ScanProgressState {
    pub fn new(app: &AppHandle) -> Self {
        Self {
            app: app.clone(),
            phase: Arc::new(std::sync::RwLock::new("thumbnails".to_string())),
            total: Arc::new(AtomicUsize::new(0)),
            current: Arc::new(AtomicUsize::new(0)),
        }
    }

    pub fn set_phase(&self, phase: &str) {
        if let Ok(mut p) = self.phase.write() {
            *p = phase.to_string();
        }
        self.current.store(0, Ordering::SeqCst);
        self.total.store(0, Ordering::SeqCst);
    }

    pub fn set_total(&self, count: usize) {
        self.total.store(count, Ordering::SeqCst);
        self.current.store(0, Ordering::SeqCst);
        self.emit();
    }

    pub fn increment(&self) {
        self.current.fetch_add(1, Ordering::SeqCst);
        self.emit();
    }

    fn emit(&self) {
        let phase = self.phase.read().map(|p| p.clone()).unwrap_or_default();
        let current = self.current.load(Ordering::SeqCst) as u64;
        let total = self.total.load(Ordering::SeqCst) as u64;
        let _ = self.app.emit(
            "scan_progress",
            ScanProgressPayload { phase, current, total },
        );
    }
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
fn test_embedding(ort_dylib_path: String, model_dir: String, image_path: String, query: String) -> Result<EmbeddingTestResult, String> {
    use embedding::EmbeddingModel;

    let dylib_path = Path::new(&ort_dylib_path);
    let model_path = Path::new(&model_dir);
    let image_file = Path::new(&image_path);

    // Initialize ONNX Runtime with the provided dylib
    if let Err(e) = embedding::init_ort(dylib_path) {
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
async fn set_model_config(app: AppHandle, ort_dylib_path: String, model_dir: String) -> Result<(), String> {
    let mut cfg = config::load_config(&app)?;
    cfg.ort_dylib_path = Some(ort_dylib_path);
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
        thumbnail::delete_thumbnails(&thumb_dir, &to_remove)?;
        database::remove_images(&table, &to_remove).await?;
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

    // Save the runtime type to config
    let mut cfg = config::load_config(&app)?;
    cfg.runtime_type = Some(runtime_type.clone());
    config::save_config(&app, &cfg)?;

    // Create a channel for progress updates
    let app_clone = app.clone();
    let progress_callback = move |downloaded: u64, total: u64| {
        let _ = app_clone.emit("ort_download_progress", OrtDownloadProgress { downloaded, total });
    };

    let lib_path = ort_download::download_ort(&app, rt, progress_callback).await?;

    Ok(lib_path.to_string_lossy().to_string())
}

/// Check if GPU runtime is available for this platform.
#[tauri::command]
fn is_gpu_available() -> bool {
    ort_download::Platform::detect()
        .map(|p| p.gpu_available())
        .unwrap_or(false)
}

async fn scan_directory_internal(
    table: &lancedb::Table,
    thumb_dir: &Path,
    dir: &str,
    embedding_backend: Option<&EmbeddingBackend>,
    model_id: Option<&str>,
    progress: Option<ScanProgressState>,
) -> Result<(ScanResult, HashSet<String>), String> {
    let dir_path = Path::new(dir);
    let files = tokio::task::spawn_blocking({
        let dir_path = dir_path.to_path_buf();
        move || scanner::scan_directory(&dir_path)
    })
    .await
    .map_err(|e| e.to_string())??;

    let mut result = ScanResult {
        images_found: files.len() as u32,
        images_added: 0,
        images_updated: 0,
        images_removed: 0,
        errors: Vec::new(),
    };

    let db_modified_times = database::get_all_modified_times(table).await?;

    let mut seen_paths: HashSet<String> = HashSet::new();
    let mut to_upsert: Vec<ImageRecord> = Vec::new();
    let mut paths_needing_processing: Vec<(String, scanner::ScannedFile)> = Vec::new();

    for file in files {
        seen_paths.insert(file.path.clone());

        let file_modified_ms = scanner::system_time_to_millis(file.modified_at);

        let needs_update = match db_modified_times.get(&file.path) {
            Some(&db_modified) => {
                if db_modified < file_modified_ms {
                    result.images_updated += 1;
                    true
                } else {
                    false
                }
            }
            None => {
                result.images_added += 1;
                true
            }
        };

        if needs_update {
            paths_needing_processing.push((file.path.clone(), file));
        }
    }

    // Generate thumbnails and embeddings for new/updated images
    if !paths_needing_processing.is_empty() {
        let thumb_dir_clone = thumb_dir.to_path_buf();
        let paths_for_thumbnails: Vec<String> = paths_needing_processing
            .iter()
            .map(|(p, _)| p.clone())
            .collect();
        let progress_for_thumbnails = progress.clone();

        // Set up thumbnail phase progress
        if let Some(progress) = &progress {
            progress.set_phase("thumbnails");
            progress.set_total(paths_for_thumbnails.len());
        }

        // Generate thumbnails in parallel
        let thumbnail_errors = tokio::task::spawn_blocking(move || {
            let errors = std::sync::Mutex::new(Vec::new());

            std::thread::scope(|s| {
                let num_threads = std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(4);
                let chunk_size = (paths_for_thumbnails.len() + num_threads - 1) / num_threads;

                for chunk in paths_for_thumbnails.chunks(chunk_size) {
                    let errors = &errors;
                    let thumb_dir = &thumb_dir_clone;
                    let progress = progress_for_thumbnails.clone();
                    s.spawn(move || {
                        for path_str in chunk {
                            let source_path = Path::new(path_str);
                            if let Err(e) = thumbnail::ensure_thumbnail(thumb_dir, source_path) {
                                errors.lock().unwrap().push(format!("Thumbnail error for {}: {}", path_str, e));
                            }
                            if let Some(progress) = &progress {
                                progress.increment();
                            }
                        }
                    });
                }
            });

            errors.into_inner().unwrap()
        })
        .await
        .map_err(|e| e.to_string())?;

        result.errors.extend(thumbnail_errors);

        let paths_for_embeddings: Vec<String> = paths_needing_processing
            .iter()
            .map(|(p, _)| p.clone())
            .collect();
        let progress_for_embeddings = progress.clone();

        // Generate embeddings using either CPU or GPU backend.
        // CPU: Multiple model instances with thread-parallel inference (one image at a time per thread).
        // GPU: Single model with batched inference (multiple images per forward pass).
        let (embeddings, embedding_errors): (Vec<Option<Vec<f32>>>, Vec<String>) =
            if let Some(backend) = embedding_backend {
                // Set up scanning phase progress
                if let Some(progress) = &progress {
                    progress.set_phase("scanning");
                    progress.set_total(paths_for_embeddings.len());
                }

                match backend {
                    EmbeddingBackend::Cpu(pool) => {
                        // CPU mode: parallel workers, each processing images one at a time
                        let num_workers = pool.len();
                        let chunk_size = (paths_for_embeddings.len() + num_workers - 1) / num_workers;

                        let results = std::sync::Mutex::new(vec![None; paths_for_embeddings.len()]);
                        let errors = std::sync::Mutex::new(Vec::new());

                        std::thread::scope(|s| {
                            for (worker_idx, chunk) in paths_for_embeddings.chunks(chunk_size).enumerate() {
                                let errors = &errors;
                                let results = &results;
                                let start = worker_idx * chunk_size;
                                let chunk = chunk.to_vec();
                                let progress = progress_for_embeddings.clone();

                                let model_mutex = match pool.get(worker_idx) {
                                    Some(m) => m,
                                    None => continue,
                                };

                                s.spawn(move || {
                                    let mut model = match model_mutex.lock() {
                                        Ok(m) => m,
                                        Err(e) => {
                                            errors.lock().unwrap().push(format!(
                                                "Failed to lock embedding model {}: {}",
                                                worker_idx, e
                                            ));
                                            return;
                                        }
                                    };

                                    for (offset, path) in chunk.iter().enumerate() {
                                        let image_path = Path::new(path);
                                        match model.embed_image(image_path) {
                                            Ok(emb) => {
                                                results.lock().unwrap()[start + offset] = Some(emb);
                                            }
                                            Err(e) => {
                                                errors
                                                    .lock()
                                                    .unwrap()
                                                    .push(format!("Embedding error for {}: {}", path, e));
                                            }
                                        }
                                        if let Some(progress) = &progress {
                                            progress.increment();
                                        }
                                    }
                                });
                            }
                        });

                        (results.into_inner().unwrap(), errors.into_inner().unwrap())
                    }
                    EmbeddingBackend::Gpu(model_mutex) => {
                        // GPU mode: batched inference for maximum GPU utilization
                        let mut results: Vec<Option<Vec<f32>>> = vec![None; paths_for_embeddings.len()];
                        let mut errors: Vec<String> = Vec::new();

                        // Lock the GPU model for the entire embedding phase
                        let mut model = match model_mutex.lock() {
                            Ok(m) => m,
                            Err(e) => {
                                let err = format!("Failed to lock GPU embedding model: {}", e);
                                return Err(err);
                            }
                        };

                        // Process images in batches
                        for batch_start in (0..paths_for_embeddings.len()).step_by(GPU_BATCH_SIZE) {
                            let batch_end = (batch_start + GPU_BATCH_SIZE).min(paths_for_embeddings.len());
                            let batch_paths: Vec<&Path> = paths_for_embeddings[batch_start..batch_end]
                                .iter()
                                .map(|p| Path::new(p.as_str()))
                                .collect();

                            let batch_results = model.embed_images_batch(&batch_paths);

                            for (offset, result) in batch_results.into_iter().enumerate() {
                                let idx = batch_start + offset;
                                match result {
                                    Ok(emb) => {
                                        results[idx] = Some(emb);
                                    }
                                    Err(e) => {
                                        errors.push(format!(
                                            "Embedding error for {}: {}",
                                            paths_for_embeddings[idx], e
                                        ));
                                    }
                                }
                                if let Some(progress) = &progress_for_embeddings {
                                    progress.increment();
                                }
                            }
                        }

                        (results, errors)
                    }
                }
            } else {
                (vec![None; paths_for_embeddings.len()], Vec::new())
            };

        result.errors.extend(embedding_errors);

        let mut embedding_iter = embeddings.into_iter();

        for (path, file) in paths_needing_processing {
            let file_modified_ms = scanner::system_time_to_millis(file.modified_at);

            let embedding = embedding_iter.next().unwrap_or(None);
            let emb_model_id = embedding.as_ref().and_then(|_| model_id.map(|s| s.to_string()));

            to_upsert.push(ImageRecord {
                path,
                file_type: file.file_type,
                file_size: file.file_size,
                created_at: scanner::system_time_to_millis(file.created_at),
                modified_at: file_modified_ms,
                visual_embedding: embedding,
                model_id: emb_model_id,
            });
        }
    }

    if !to_upsert.is_empty() {
        database::upsert_images(table, to_upsert).await?;
    }

    Ok((result, seen_paths))
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            let cfg = config::load_config(app.handle())?;

            // Build the menu bar
            // Use & before a letter to create a mnemonic (Alt+letter shortcut on Windows)
            let file_menu = SubmenuBuilder::new(app, "&File")
                .item(&MenuItemBuilder::new("&Add Folder...").id("add_folder").accelerator("CmdOrCtrl+O").build(app)?)
                .item(&MenuItemBuilder::new("&Rescan All").id("rescan").accelerator("CmdOrCtrl+R").build(app)?)
                .item(&MenuItemBuilder::new("&Manage Folders...").id("manage_folders").build(app)?)
                .separator()
                .item(&MenuItemBuilder::new("&View Files").id("view_files").build(app)?)
                .separator()
                .item(&MenuItemBuilder::new("Clear &Thumbnails").id("clear_thumbnails").build(app)?)
                .item(&MenuItemBuilder::new("Clear &Database").id("clear_database").build(app)?)
                .separator()
                .item(&PredefinedMenuItem::quit(app, None)?)
                .build()?;

            let edit_menu = SubmenuBuilder::new(app, "&Edit")
                .item(&PredefinedMenuItem::undo(app, None)?)
                .item(&PredefinedMenuItem::redo(app, None)?)
                .separator()
                .item(&PredefinedMenuItem::cut(app, None)?)
                .item(&PredefinedMenuItem::copy(app, None)?)
                .item(&PredefinedMenuItem::paste(app, None)?)
                .item(&PredefinedMenuItem::select_all(app, None)?)
                .build()?;

            let search_menu = SubmenuBuilder::new(app, "&Search")
                .item(&CheckMenuItemBuilder::new("&Lexical OCR").id("ocr_lexical").build(app)?)
                .item(&CheckMenuItemBuilder::new("&Semantic OCR").id("ocr_semantic").build(app)?)
                .build()?;

            let model_menu = SubmenuBuilder::new(app, "&Model")
                .item(&MenuItemBuilder::new("&Runtime settings...").id("model_settings").build(app)?)
                .build()?;

            let view_menu = SubmenuBuilder::new(app, "&View")
                .item(&MenuItemBuilder::new("&Relevance").id("sort_relevance").build(app)?)
                .separator()
                .item(&MenuItemBuilder::new("Date &Created \u{2191}").id("sort_created_asc").build(app)?)
                .item(&MenuItemBuilder::new("Date C&reated \u{2193}").id("sort_created_desc").build(app)?)
                .separator()
                .item(&MenuItemBuilder::new("Date &Modified \u{2191}").id("sort_modified_asc").build(app)?)
                .item(&MenuItemBuilder::new("Date Mo&dified \u{2193}").id("sort_modified_desc").build(app)?)
                .separator()
                .item(&MenuItemBuilder::new("File &Size \u{2191}").id("sort_size_asc").build(app)?)
                .item(&MenuItemBuilder::new("File Si&ze \u{2193}").id("sort_size_desc").build(app)?)
                .build()?;

            let help_menu = SubmenuBuilder::new(app, "&Help")
                .item(&MenuItemBuilder::new("&About").id("about").build(app)?)
                .item(&MenuItemBuilder::new("View &Controls").id("view_controls").build(app)?)
                .build()?;

            let menu = if cfg.debug_mode {
                let debug_menu = SubmenuBuilder::new(app, "&Debug")
                    .item(&MenuItemBuilder::new("Debug mode enabled").id("debug_mode_enabled").build(app)?)
                    .build()?;
                MenuBuilder::new(app)
                    .items(&[&file_menu, &edit_menu, &search_menu, &model_menu, &view_menu, &help_menu, &debug_menu])
                    .build()?
            } else {
                MenuBuilder::new(app)
                    .items(&[&file_menu, &edit_menu, &search_menu, &model_menu, &view_menu, &help_menu])
                    .build()?
            };

            app.set_menu(menu)?;

            // Handle menu events
            app.on_menu_event(|app, event| {
                let _ = app.emit("menu-event", event.id().0.as_str());
            });

            let handle = app.handle().clone();

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

            // Phase 2: Heavy initialization (embedding models) - runs in background
            // This allows the UI to appear immediately while models load
            let handle_for_task = app.handle().clone();
            let cfg_for_task = cfg.clone();
            tauri::async_runtime::spawn(async move {
                // Determine the ONNX Runtime library path:
                // 1. If ort_dylib_path is set in config (dev override), use that
                // 2. Otherwise, check if runtime is downloaded to app data
                let ort_path: Option<PathBuf> = if let Some(override_path) = &cfg_for_task.ort_dylib_path {
                    let p = PathBuf::from(override_path);
                    if p.exists() {
                        println!("Using ONNX Runtime from config override: {}", p.display());
                        Some(p)
                    } else {
                        eprintln!("Warning: Configured ort_dylib_path does not exist: {}", override_path);
                        None
                    }
                } else {
                    // Check for downloaded runtime
                    match ort_download::get_ort_library_path(&handle_for_task) {
                        Ok(Some(p)) => {
                            println!("Using downloaded ONNX Runtime: {}", p.display());
                            Some(p)
                        }
                        Ok(None) => {
                            println!("ONNX Runtime not installed. Use the settings to download it.");
                            None
                        }
                        Err(e) => {
                            eprintln!("Warning: Failed to check for ONNX Runtime: {}", e);
                            None
                        }
                    }
                };

                // Check if GPU runtime is configured
                let use_gpu = cfg_for_task.runtime_type.as_deref() == Some("gpu");

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
            is_gpu_available,
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
