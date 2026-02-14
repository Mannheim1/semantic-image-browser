//! Image scanning and processing pipeline.
//!
//! This module owns the full scan lifecycle:
//! - Directory traversal to discover image files
//! - Diffing against the database to find new/modified images
//! - Decode-once pipeline: shared RGB data feeds both thumbnail generation and embedding
//! - CPU mode (thread-parallel), GPU mode (batched inference), and thumbnail-only mode
//! - Progress reporting to the frontend via Tauri events

use serde::Serialize;
use std::collections::HashSet;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::SystemTime;
use tauri::{AppHandle, Emitter};
use walkdir::WalkDir;

use crate::benchmark::{self, PreprocessTiming};
use crate::database::{self, ImageRecord};
use crate::embedding::{GpuEmbeddingModel, GPU_BATCH_SIZE, PreprocessedBatch, preprocess_image_from_rgb, IMAGE_SIZE};
use crate::image_ops::decode_image_to_rgb;
use crate::state::EmbeddingBackend;
use crate::thumbnail;

// ── Directory scanning ──────────────────────────────────────────────

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
        .max_depth(1)
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

// ── Progress reporting ──────────────────────────────────────────────

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
            phase: Arc::new(std::sync::RwLock::new("scanning".to_string())),
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

    pub fn increment_by(&self, count: usize) {
        self.current.fetch_add(count, Ordering::SeqCst);
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

// ── Scan results ────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct ScanResult {
    pub images_found: u32,
    pub images_added: u32,
    pub images_updated: u32,
    pub images_removed: u32,
    pub errors: Vec<String>,
}

// ── Scan pipeline ───────────────────────────────────────────────────

pub async fn scan_directory_internal(
    table: &lancedb::Table,
    thumb_dir: &Path,
    dir: &str,
    embedding_backend: Option<&EmbeddingBackend>,
    model_id: Option<&str>,
    progress: Option<ScanProgressState>,
) -> Result<(ScanResult, HashSet<String>), String> {
    benchmark::begin_scan_session();

    let dir_path = Path::new(dir);
    let files = tokio::task::spawn_blocking({
        let dir_path = dir_path.to_path_buf();
        move || scan_directory(&dir_path)
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
    let mut paths_needing_processing: Vec<(String, ScannedFile)> = Vec::new();

    for file in files {
        seen_paths.insert(file.path.clone());

        let file_modified_ms = system_time_to_millis(file.modified_at);

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

    // Process images: decode once, then generate thumbnail + embedding from shared RGB data.
    // This avoids decoding each image twice (once for thumbnail, once for embedding).
    if !paths_needing_processing.is_empty() {
        // Ensure thumbnails directory exists once before parallel processing
        std::fs::create_dir_all(thumb_dir)
            .map_err(|e| format!("Failed to create thumbnails directory: {}", e))?;

        let num_images = paths_needing_processing.len();

        // Set up progress tracking
        if let Some(progress) = &progress {
            progress.set_phase("scanning");
            progress.set_total(num_images);
        }

        // Collect inputs for processing
        let processing_inputs: Vec<(String, String, u64)> = paths_needing_processing
            .iter()
            .map(|(p, f)| (p.clone(), f.file_type.clone(), f.file_size))
            .collect();

        let embeddings: Vec<Option<Vec<f32>>>;
        let processing_errors: Vec<String>;

        match embedding_backend {
            Some(EmbeddingBackend::Cpu(pool)) => {
                // CPU mode: each worker decodes once, generates thumbnail, then runs embedding inference.
                let num_workers = pool.len();
                let thumb_dir_ref = thumb_dir;

                let emb_results = std::sync::Mutex::new(vec![None; num_images]);
                let errors = std::sync::Mutex::new(Vec::new());

                // Round-robin distribution with original indices
                let mut thread_buckets: Vec<Vec<(usize, &(String, String, u64))>> =
                    (0..num_workers).map(|_| Vec::new()).collect();
                for (i, item) in processing_inputs.iter().enumerate() {
                    thread_buckets[i % num_workers].push((i, item));
                }

                let progress_ref = &progress;

                std::thread::scope(|s| {
                    for (worker_idx, bucket) in thread_buckets.into_iter().enumerate() {
                        let errors = &errors;
                        let emb_results = &emb_results;

                        let model_mutex = match pool.get(worker_idx) {
                            Some(m) => m,
                            None => continue,
                        };

                        s.spawn(move || {
                            let mut model = match model_mutex.lock() {
                                Ok(m) => m,
                                Err(e) => {
                                    errors.lock().unwrap().push(format!(
                                        "Failed to lock embedding model {}: {}", worker_idx, e
                                    ));
                                    return;
                                }
                            };

                            let mut resizer = fast_image_resize::Resizer::new();

                            for (original_idx, (path_str, file_type, file_size)) in &bucket {
                                let source_path = Path::new(path_str.as_str());
                                let start = std::time::Instant::now();

                                // Decode once
                                let decode_result = decode_image_to_rgb(source_path);
                                let decode_time = start.elapsed();

                                let (rgb_data, width, height) = match decode_result {
                                    Ok(data) => data,
                                    Err(e) => {
                                        errors.lock().unwrap().push(format!(
                                            "Decode error for {}: {}", path_str, e
                                        ));
                                        if let Some(progress) = progress_ref {
                                            progress.increment();
                                        }
                                        continue;
                                    }
                                };

                                // Generate thumbnail from decoded RGB data
                                let thumb_start = std::time::Instant::now();
                                let thumb_path = thumbnail::thumbnail_path(thumb_dir_ref, source_path);
                                if !thumbnail::thumbnail_is_current(&thumb_path, source_path) {
                                    if let Err(e) = thumbnail::generate_thumbnail_from_rgb(
                                        &rgb_data, width, height, &thumb_path, Some(&mut resizer),
                                    ) {
                                        errors.lock().unwrap().push(format!(
                                            "Thumbnail error for {}: {}", path_str, e
                                        ));
                                    }
                                }
                                let thumb_time = thumb_start.elapsed();

                                // Preprocess for embedding from same decoded RGB data
                                let filename = source_path.file_name()
                                    .unwrap_or_default().to_string_lossy();

                                match preprocess_image_from_rgb(
                                    &rgb_data, width, height, &filename, file_type, *file_size, Some(&mut resizer),
                                ) {
                                    Ok((pixel_values, mut timing)) => {
                                        timing.decode = decode_time;
                                        timing.thumbnail = thumb_time;

                                        let inference_start = std::time::Instant::now();
                                        match model.embed_preprocessed(&pixel_values) {
                                            Ok(emb) => {
                                                let inference_time = inference_start.elapsed();
                                                benchmark::log_image(&timing, inference_time, "cpu");
                                                emb_results.lock().unwrap()[*original_idx] = Some(emb);
                                            }
                                            Err(e) => {
                                                errors.lock().unwrap().push(format!(
                                                    "Embedding error for {}: {}", path_str, e
                                                ));
                                            }
                                        }
                                    }
                                    Err(e) => {
                                        errors.lock().unwrap().push(format!(
                                            "Preprocess error for {}: {}", path_str, e
                                        ));
                                    }
                                }

                                if let Some(progress) = progress_ref {
                                    progress.increment();
                                }
                            }
                        });
                    }
                });

                embeddings = emb_results.into_inner().unwrap();
                processing_errors = errors.into_inner().unwrap();
            }

            Some(EmbeddingBackend::Gpu(model_mutex)) => {
                // GPU mode: producer threads decode + thumbnail + preprocess in parallel via Rayon,
                // send preprocessed tensors through a bounded channel.
                // Consumer collects batches and runs GPU inference.
                use std::sync::mpsc;

                let mut emb_results: Vec<Option<Vec<f32>>> = vec![None; num_images];
                let mut errors: Vec<String> = Vec::new();
                let thumb_dir_owned = thumb_dir.to_path_buf();

                let mut model = match model_mutex.lock() {
                    Ok(m) => m,
                    Err(e) => {
                        return Err(format!("Failed to lock GPU embedding model: {}", e));
                    }
                };

                let (tx, rx) = mpsc::sync_channel::<(usize, Result<(Vec<f32>, PreprocessTiming), String>)>(GPU_BATCH_SIZE * 2);

                let progress_for_consumer = progress.clone();

                // Producer: decode, thumbnail, and preprocess in parallel
                let inputs_owned: Vec<(usize, String, String, u64)> = processing_inputs
                    .iter()
                    .enumerate()
                    .map(|(i, (p, ft, fs))| (i, p.clone(), ft.clone(), *fs))
                    .collect();

                std::thread::spawn(move || {
                    use rayon::prelude::*;
                    inputs_owned.par_iter().for_each(|(idx, path_str, file_type, file_size)| {
                        let source_path = Path::new(path_str.as_str());
                        let start = std::time::Instant::now();

                        // Decode once
                        let decode_result = decode_image_to_rgb(source_path);
                        let decode_time = start.elapsed();

                        let result = match decode_result {
                            Ok((rgb_data, width, height)) => {
                                // Generate thumbnail from decoded RGB data
                                let thumb_start = std::time::Instant::now();
                                let thumb_path = thumbnail::thumbnail_path(&thumb_dir_owned, source_path);
                                if !thumbnail::thumbnail_is_current(&thumb_path, source_path) {
                                    // Thumbnail errors are non-fatal for the embedding pipeline
                                    let _ = thumbnail::generate_thumbnail_from_rgb(
                                        &rgb_data, width, height, &thumb_path, None,
                                    );
                                }
                                let thumb_time = thumb_start.elapsed();

                                // Preprocess for embedding
                                let filename = source_path.file_name()
                                    .unwrap_or_default().to_string_lossy();
                                match preprocess_image_from_rgb(
                                    &rgb_data, width, height, &filename, file_type, *file_size, None,
                                ) {
                                    Ok((pixel_values, mut timing)) => {
                                        timing.decode = decode_time;
                                        timing.thumbnail = thumb_time;
                                        Ok((pixel_values, timing))
                                    }
                                    Err(e) => Err(e),
                                }
                            }
                            Err(e) => Err(e),
                        };

                        let _ = tx.send((*idx, result));
                    });
                    // tx drops here, closing the channel
                });

                // Consumer: collect preprocessed images into batches, run GPU inference
                let paths_for_errors: Vec<String> = processing_inputs.iter().map(|(p, _, _)| p.clone()).collect();
                let pixels_per_image = 3 * (IMAGE_SIZE as usize * IMAGE_SIZE as usize);
                let mut batch_pixel_data: Vec<f32> = Vec::with_capacity(GPU_BATCH_SIZE * pixels_per_image);
                let mut batch_indices: Vec<usize> = Vec::with_capacity(GPU_BATCH_SIZE);
                let mut batch_timings: Vec<PreprocessTiming> = Vec::with_capacity(GPU_BATCH_SIZE);

                let flush_batch = |model: &mut std::sync::MutexGuard<'_, GpuEmbeddingModel>,
                                   pixel_data: &mut Vec<f32>,
                                   indices: &mut Vec<usize>,
                                   timings: &mut Vec<PreprocessTiming>,
                                   results: &mut Vec<Option<Vec<f32>>>,
                                   errors: &mut Vec<String>,
                                   paths: &[String],
                                   progress: &Option<ScanProgressState>| {
                    if indices.is_empty() {
                        return;
                    }

                    let batch = PreprocessedBatch {
                        pixel_data: std::mem::take(pixel_data),
                        valid_indices: (0..indices.len()).collect(),
                        timings: std::mem::take(timings).into_iter().map(Some).collect(),
                        errors: vec![None; indices.len()],
                        count: indices.len(),
                    };

                    let batch_results = model.infer_batch(&batch);

                    for (batch_pos, result) in batch_results.into_iter().enumerate() {
                        let original_idx = indices[batch_pos];
                        match result {
                            Ok(emb) => {
                                results[original_idx] = Some(emb);
                            }
                            Err(e) => {
                                errors.push(format!(
                                    "Embedding error for {}: {}",
                                    paths[original_idx], e
                                ));
                            }
                        }
                        if let Some(progress) = progress {
                            progress.increment();
                        }
                    }

                    indices.clear();
                };

                for (original_idx, result) in rx.iter() {
                    match result {
                        Ok((pixels, timing)) => {
                            batch_pixel_data.extend(pixels);
                            batch_indices.push(original_idx);
                            batch_timings.push(timing);

                            if batch_indices.len() >= GPU_BATCH_SIZE {
                                flush_batch(
                                    &mut model, &mut batch_pixel_data, &mut batch_indices,
                                    &mut batch_timings, &mut emb_results, &mut errors,
                                    &paths_for_errors, &progress_for_consumer,
                                );
                            }
                        }
                        Err(e) => {
                            errors.push(format!(
                                "Processing error for {}: {}",
                                paths_for_errors[original_idx], e
                            ));
                            if let Some(progress) = &progress_for_consumer {
                                progress.increment();
                            }
                        }
                    }
                }

                // Flush remaining images (final partial batch)
                flush_batch(
                    &mut model, &mut batch_pixel_data, &mut batch_indices,
                    &mut batch_timings, &mut emb_results, &mut errors,
                    &paths_for_errors, &progress_for_consumer,
                );

                embeddings = emb_results;
                processing_errors = errors;
            }

            None => {
                // No embedding backend — just generate thumbnails
                let thumb_dir_clone = thumb_dir.to_path_buf();
                let progress_for_thumbs = progress.clone();

                let thumb_errors = tokio::task::spawn_blocking(move || {
                    let errors = std::sync::Mutex::new(Vec::new());

                    std::thread::scope(|s| {
                        let num_threads = std::thread::available_parallelism()
                            .map(|n| n.get())
                            .unwrap_or(4);

                        let mut thread_buckets: Vec<Vec<&(String, String, u64)>> =
                            (0..num_threads).map(|_| Vec::new()).collect();
                        for (i, item) in processing_inputs.iter().enumerate() {
                            thread_buckets[i % num_threads].push(item);
                        }

                        for bucket in thread_buckets {
                            let errors = &errors;
                            let thumb_dir = &thumb_dir_clone;
                            let progress = progress_for_thumbs.clone();
                            s.spawn(move || {
                                let mut resizer = fast_image_resize::Resizer::new();
                                for (path_str, _file_type, _file_size) in bucket {
                                    let source_path = Path::new(path_str.as_str());
                                    if let Err(e) = thumbnail::ensure_thumbnail(
                                        thumb_dir, source_path, Some(&mut resizer),
                                    ) {
                                        errors.lock().unwrap().push(format!(
                                            "Thumbnail error for {}: {}", path_str, e
                                        ));
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

                embeddings = vec![None; num_images];
                processing_errors = thumb_errors;
            }
        }

        result.errors.extend(processing_errors);

        let mut embedding_iter = embeddings.into_iter();

        for (path, file) in paths_needing_processing {
            let file_modified_ms = system_time_to_millis(file.modified_at);

            let embedding = embedding_iter.next().unwrap_or(None);
            let emb_model_id = embedding.as_ref().and_then(|_| model_id.map(|s| s.to_string()));

            to_upsert.push(ImageRecord {
                path,
                file_type: file.file_type,
                file_size: file.file_size,
                created_at: system_time_to_millis(file.created_at),
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
