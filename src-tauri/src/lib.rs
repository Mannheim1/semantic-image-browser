use lancedb::{Connection, Table};
use ort::session::Session;
use serde::Serialize;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use tauri::{AppHandle, Manager};
use tauri_plugin_opener::OpenerExt;

mod config;
mod database;
mod scanner;
mod thumbnail;

use config::AppConfig;
use database::{ImageInfo, ImageRecord};

pub struct AppState {
    pub db: Connection,
    pub table: tokio::sync::Mutex<Table>,
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

#[tauri::command]
fn test_onnx() -> Result<String, String> {
    let _builder = Session::builder().map_err(|e| e.to_string())?;
    Ok("ONNX Runtime initialized successfully!".to_string())
}

#[tauri::command]
async fn get_config(app: AppHandle) -> Result<AppConfig, String> {
    config::load_config(&app)
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
    let (result, _) = scan_directory_internal(&table, &thumb_dir, &path).await?;
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
        match scan_directory_internal(&table, &thumb_dir, dir).await {
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

#[tauri::command]
async fn search_images(state: tauri::State<'_, AppState>, query: String) -> Result<Vec<ImageInfo>, String> {
    let table = state.table.lock().await;
    if query.trim().is_empty() {
        database::get_all_images(&table).await
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

    let mut seen_paths: HashSet<String> = HashSet::new();
    let mut to_upsert: Vec<ImageRecord> = Vec::new();
    let mut paths_needing_thumbnails: Vec<String> = Vec::new();

    for file in files {
        seen_paths.insert(file.path.clone());

        let file_modified_ms = scanner::system_time_to_millis(file.modified_at);

        let needs_update = match database::get_image_by_path(table, &file.path).await {
            Ok(Some(db_modified)) => {
                // File exists in DB, check if it was modified
                if db_modified < file_modified_ms {
                    result.images_updated += 1;
                    true
                } else {
                    false
                }
            }
            Ok(None) => {
                // File not in DB - add it
                result.images_added += 1;
                true
            }
            Err(e) if e.contains("null byte") => {
                // Path validation error - skip this specific file
                result.errors.push(format!("Invalid path '{}': {}", file.path, e));
                continue;
            }
            Err(e) => {
                // Database infrastructure error - fail entire scan to prevent data loss
                return Err(format!(
                    "Database error while checking '{}': {}. Scan aborted to prevent data loss. \
                    This usually indicates a database connection problem or corruption.",
                    file.path, e
                ));
            }
        };

        if needs_update {
            paths_needing_thumbnails.push(file.path.clone());
            to_upsert.push(ImageRecord {
                path: file.path.clone(),
                file_type: file.file_type,
                file_size: file.file_size,
                created_at: scanner::system_time_to_millis(file.created_at),
                modified_at: file_modified_ms,
            });
        }
    }

    // Generate thumbnails for new/updated images in parallel
    if !paths_needing_thumbnails.is_empty() {
        let thumb_dir_clone = thumb_dir.to_path_buf();
        let thumbnail_errors = tokio::task::spawn_blocking(move || {
            use std::sync::Mutex;
            let errors = Mutex::new(Vec::new());

            std::thread::scope(|s| {
                let num_threads = std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(4);
                let chunk_size = (paths_needing_thumbnails.len() + num_threads - 1) / num_threads;

                for chunk in paths_needing_thumbnails.chunks(chunk_size) {
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
                handle.manage(AppState { db, table: tokio::sync::Mutex::new(table) });
                Ok::<(), String>(())
            })
            .expect("Failed to initialize database");
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            test_onnx,
            get_config,
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
