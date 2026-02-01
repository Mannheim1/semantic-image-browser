use lancedb::{Connection, Table};
use serde::Serialize;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use tauri::{AppHandle, Manager};
use tauri_plugin_opener::OpenerExt;

mod config;
mod database;
mod embedding;
mod scanner;
mod thumbnail;

use config::AppConfig;
use database::{ImageInfo, ImageRecord};
use embedding::EmbeddingModel;

pub struct AppState {
    pub db: Connection,
    pub table: tokio::sync::Mutex<Table>,
    /// The embedding model, if configured and loaded successfully.
    /// Wrapped in Mutex because inference requires &mut self.
    pub embedding_model: Option<Mutex<EmbeddingModel>>,
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

    // Try to load the model
    let mut model = match EmbeddingModel::load(model_path) {
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
    state.embedding_model.is_some()
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

    // Get model ID from config (the directory name is a reasonable identifier)
    let model_id = cfg.model_dir.as_ref().and_then(|p| {
        Path::new(p).file_name().and_then(|n| n.to_str()).map(|s| s.to_string())
    });

    let (result, _) = scan_directory_internal(
        &table,
        &thumb_dir,
        &path,
        state.embedding_model.as_ref(),
        model_id.as_deref(),
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

    // Get model ID from config (the directory name is a reasonable identifier)
    let model_id = cfg.model_dir.as_ref().and_then(|p| {
        Path::new(p).file_name().and_then(|n| n.to_str()).map(|s| s.to_string())
    });

    let mut total_result = ScanResult {
        images_found: 0,
        images_added: 0,
        images_updated: 0,
        images_removed: 0,
        errors: Vec::new(),
    };

    let table = state.table.lock().await;
    let mut all_seen_paths: HashSet<String> = HashSet::new();

    for dir in &cfg.watched_directories {
        match scan_directory_internal(
            &table,
            &thumb_dir,
            dir,
            state.embedding_model.as_ref(),
            model_id.as_deref(),
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
async fn search_images(state: tauri::State<'_, AppState>, query: String) -> Result<Vec<ImageInfo>, String> {
    let table = state.table.lock().await;
    if query.trim().is_empty() {
        return database::get_all_images(&table).await;
    }

    // Try to generate text embedding (must release mutex before await)
    let query_embedding = if let Some(model) = &state.embedding_model {
        match model.lock() {
            Ok(mut m) => match m.embed_text(&query) {
                Ok(emb) => Some(emb),
                Err(e) => {
                    eprintln!("Text embedding failed, falling back to filename search: {}", e);
                    None
                }
            },
            Err(e) => {
                eprintln!("Failed to lock embedding model: {}", e);
                None
            }
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
        .reveal_item_in_dir(&app_data)
        .map_err(|e| e.to_string())
}

async fn scan_directory_internal(
    table: &lancedb::Table,
    thumb_dir: &Path,
    dir: &str,
    embedding_model: Option<&Mutex<EmbeddingModel>>,
    model_id: Option<&str>,
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

        // Generate thumbnails in parallel (existing logic)
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
                    s.spawn(move || {
                        for path_str in chunk {
                            let source_path = Path::new(path_str);
                            if let Err(e) = thumbnail::ensure_thumbnail(thumb_dir, source_path) {
                                errors.lock().unwrap().push(format!("Thumbnail error for {}: {}", path_str, e));
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

        // Generate embeddings (sequential for now - model requires &mut self)
        // Note: Embedding generation is slower than thumbnails, but we keep it simple.
        // Future optimization: batch inference or use a separate thread with message passing.
        for (path, file) in paths_needing_processing {
            let file_modified_ms = scanner::system_time_to_millis(file.modified_at);

            let (embedding, emb_model_id) = if let Some(model) = embedding_model {
                let image_path = Path::new(&path);
                match model.lock() {
                    Ok(mut m) => match m.embed_image(image_path) {
                        Ok(emb) => (Some(emb), model_id.map(|s| s.to_string())),
                        Err(e) => {
                            result.errors.push(format!("Embedding error for {}: {}", path, e));
                            (None, None)
                        }
                    },
                    Err(e) => {
                        result.errors.push(format!("Failed to lock embedding model: {}", e));
                        (None, None)
                    }
                }
            } else {
                (None, None)
            };

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
            let handle = app.handle().clone();
            tauri::async_runtime::block_on(async {
                let cfg = config::load_config(&handle)?;
                let db = database::open_connection(&handle, &cfg).await?;
                let table = database::get_or_create_table(&db).await?;

                // Try to load the embedding model if configured
                let embedding_model = match (&cfg.ort_dylib_path, &cfg.model_dir) {
                    (Some(ort_path), Some(model_path)) => {
                        let ort_path = Path::new(ort_path);
                        let model_path = Path::new(model_path);

                        // Initialize ONNX Runtime
                        if let Err(e) = embedding::init_ort(ort_path) {
                            eprintln!("Warning: Failed to initialize ONNX Runtime: {}", e);
                            None
                        } else {
                            // Load the model
                            match EmbeddingModel::load(model_path) {
                                Ok(model) => {
                                    println!("Embedding model loaded successfully");
                                    Some(Mutex::new(model))
                                }
                                Err(e) => {
                                    eprintln!("Warning: Failed to load embedding model: {}", e);
                                    None
                                }
                            }
                        }
                    }
                    _ => {
                        println!("Embedding model not configured (set ort_dylib_path and model_dir in config)");
                        None
                    }
                };

                handle.manage(AppState {
                    db,
                    table: tokio::sync::Mutex::new(table),
                    embedding_model,
                });
                Ok::<(), String>(())
            })
            .expect("Failed to initialize database");
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
            open_image,
            show_in_folder,
            delete_all_thumbnails,
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
