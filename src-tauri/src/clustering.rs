//! HDBSCAN clustering and 2D UMAP-style embedding.
//!
//! Pulls all embeddings out of the LanceDB table, runs the chosen algorithm,
//! and persists the result as JSON in the app data dir. The frontend reads the
//! cached result; recomputing is a user-triggered action via the menu.

use std::path::PathBuf;

use arrow_array::{Array, FixedSizeListArray, Float32Array, RecordBatch, StringArray};
use futures::TryStreamExt;
use hnsw_rs::prelude::{DistL2, Hnsw};
use lancedb::query::{ExecutableQuery, QueryBase, Select};
use lancedb::Table;
use ndarray::Array2;
use rand::Rng;
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Manager};
use umap_rs::{Umap, UmapConfig};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterAssignment {
    pub path: String,
    /// HDBSCAN label. -1 means noise (unassigned).
    pub cluster_id: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterResult {
    pub assignments: Vec<ClusterAssignment>,
    /// Number of distinct (non-noise) clusters.
    pub num_clusters: u32,
    /// Total points considered (matches len of assignments).
    pub num_points: u32,
    /// Timestamp (ms since epoch) when this result was produced.
    pub computed_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UmapPoint {
    pub path: String,
    pub x: f32,
    pub y: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UmapResult {
    pub points: Vec<UmapPoint>,
    pub computed_at: i64,
}

/// Where the cluster JSON file lives in the app data dir.
fn clusters_path(app: &AppHandle) -> Result<PathBuf, String> {
    Ok(app
        .path()
        .app_local_data_dir()
        .map_err(|e| e.to_string())?
        .join("clusters.json"))
}

fn umap_path(app: &AppHandle) -> Result<PathBuf, String> {
    Ok(app
        .path()
        .app_local_data_dir()
        .map_err(|e| e.to_string())?
        .join("umap.json"))
}

fn now_ms() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// Pull every (path, embedding_1) pair out of the table.
/// Rows without an embedding are silently skipped.
async fn load_all_embeddings(table: &Table) -> Result<(Vec<String>, Vec<Vec<f32>>), String> {
    let batches: Vec<RecordBatch> = table
        .query()
        .select(Select::Columns(vec![
            "path".to_string(),
            "embedding_1".to_string(),
        ]))
        .execute()
        .await
        .map_err(|e| e.to_string())?
        .try_collect()
        .await
        .map_err(|e| e.to_string())?;

    let mut paths = Vec::new();
    let mut embeddings: Vec<Vec<f32>> = Vec::new();

    for batch in batches {
        let path_col = batch
            .column_by_name("path")
            .ok_or("path column missing")?
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or("path not a string array")?;
        let emb_col = batch
            .column_by_name("embedding_1")
            .ok_or("embedding_1 column missing")?
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .ok_or("embedding_1 not a fixed-size list")?;

        for i in 0..batch.num_rows() {
            if path_col.is_null(i) || emb_col.is_null(i) {
                continue;
            }
            let values = emb_col.value(i);
            let f32_arr = values
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or("embedding values are not f32")?;
            let vec: Vec<f32> = (0..f32_arr.len()).map(|j| f32_arr.value(j)).collect();
            paths.push(path_col.value(i).to_string());
            embeddings.push(vec);
        }
    }

    Ok((paths, embeddings))
}

/// Run HDBSCAN clustering on all stored embeddings.
pub async fn compute_clusters(app: &AppHandle, table: &Table) -> Result<ClusterResult, String> {
    use hdbscan::Hdbscan;

    let (paths, embeddings) = load_all_embeddings(table).await?;
    if embeddings.is_empty() {
        return Err("No embeddings to cluster. Add a folder first.".into());
    }
    if embeddings.len() < 5 {
        return Err(format!(
            "Need at least 5 images to cluster, found {}.",
            embeddings.len()
        ));
    }

    let labels = tokio::task::spawn_blocking(move || {
        let clusterer = Hdbscan::default_hyper_params(&embeddings);
        clusterer.cluster().map_err(|e| format!("HDBSCAN failed: {:?}", e))
    })
    .await
    .map_err(|e| format!("clustering task join error: {}", e))??;

    if labels.len() != paths.len() {
        return Err(format!(
            "HDBSCAN returned {} labels for {} inputs",
            labels.len(),
            paths.len()
        ));
    }

    let mut max_label: i32 = -1;
    let assignments: Vec<ClusterAssignment> = paths
        .into_iter()
        .zip(labels.iter())
        .map(|(path, &label)| {
            if label >= 0 && label > max_label {
                max_label = label;
            }
            ClusterAssignment {
                path,
                cluster_id: label,
            }
        })
        .collect();

    let result = ClusterResult {
        num_points: assignments.len() as u32,
        num_clusters: (max_label + 1).max(0) as u32,
        assignments,
        computed_at: now_ms(),
    };

    let json = serde_json::to_vec_pretty(&result).map_err(|e| e.to_string())?;
    let path = clusters_path(app)?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    std::fs::write(&path, &json).map_err(|e| e.to_string())?;

    Ok(result)
}

/// Run UMAP-style 2D embedding via annembed.
pub async fn compute_umap(app: &AppHandle, table: &Table) -> Result<UmapResult, String> {
    let (paths, embeddings) = load_all_embeddings(table).await?;
    if embeddings.is_empty() {
        return Err("No embeddings to project. Add a folder first.".into());
    }
    if embeddings.len() < 20 {
        return Err(format!(
            "Need at least 20 images for a 2D map, found {}.",
            embeddings.len()
        ));
    }

    let coords = tokio::task::spawn_blocking(move || run_umap(&embeddings))
        .await
        .map_err(|e| format!("UMAP task join error: {}", e))??;

    if coords.len() != paths.len() {
        return Err(format!(
            "UMAP returned {} points for {} inputs",
            coords.len(),
            paths.len()
        ));
    }

    let points: Vec<UmapPoint> = paths
        .into_iter()
        .zip(coords.into_iter())
        .map(|(path, (x, y))| UmapPoint { path, x, y })
        .collect();

    let result = UmapResult {
        points,
        computed_at: now_ms(),
    };

    let json = serde_json::to_vec_pretty(&result).map_err(|e| e.to_string())?;
    let path = umap_path(app)?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    std::fs::write(&path, &json).map_err(|e| e.to_string())?;

    Ok(result)
}

/// CPU-bound UMAP body: build an HNSW kNN graph, then run umap-rs.
fn run_umap(embeddings: &[Vec<f32>]) -> Result<Vec<(f32, f32)>, String> {
    let n = embeddings.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let dim = embeddings[0].len();
    if dim == 0 {
        return Err("Embeddings have zero dimensions".into());
    }
    for v in embeddings {
        if v.len() != dim {
            return Err("Embeddings have inconsistent dimensions".into());
        }
    }

    let config = UmapConfig::default();
    let n_neighbors = config.graph.n_neighbors;
    if n <= n_neighbors {
        return Err(format!(
            "Need more than {} images for UMAP (have {}).",
            n_neighbors, n
        ));
    }

    // Build HNSW index for fast kNN.
    let ef_c = 50usize;
    let max_nb_connection = 70usize;
    let nb_layer = 16usize.min(((n as f32).ln().trunc() as usize).max(1));
    let hnsw = Hnsw::<f32, DistL2>::new(max_nb_connection, n, nb_layer, ef_c, DistL2 {});
    let data_with_id: Vec<(&Vec<f32>, usize)> = embeddings.iter().zip(0..n).collect();
    hnsw.parallel_insert(&data_with_id);

    // Query each point for its k nearest neighbours. Search for k+1 then drop
    // the self-match — UMAP expects neighbours, not the point itself.
    let ef_search = (n_neighbors * 4).max(50);
    let results = hnsw.parallel_search(embeddings, n_neighbors + 1, ef_search);

    let mut knn_indices = Array2::<u32>::zeros((n, n_neighbors));
    let mut knn_dists = Array2::<f32>::zeros((n, n_neighbors));
    for (i, neighbours) in results.iter().enumerate() {
        let mut col = 0usize;
        for nb in neighbours.iter() {
            if col == n_neighbors {
                break;
            }
            if nb.d_id == i {
                continue; // skip self-match
            }
            knn_indices[[i, col]] = nb.d_id as u32;
            knn_dists[[i, col]] = nb.distance;
            col += 1;
        }
        if col < n_neighbors {
            return Err(format!(
                "HNSW returned only {} neighbours for point {} (need {})",
                col, i, n_neighbors
            ));
        }
    }

    // Pack input data into the contiguous Array2 umap-rs wants.
    let mut data = Array2::<f32>::zeros((n, dim));
    for (i, v) in embeddings.iter().enumerate() {
        for (j, &x) in v.iter().enumerate() {
            data[[i, j]] = x;
        }
    }

    // Random initialisation in [-10, 10] (standard for UMAP).
    let mut rng = rand::rng();
    let init = Array2::<f32>::from_shape_simple_fn((n, config.n_components), || {
        rng.random_range(-10.0..10.0)
    });

    // Fit can panic on bad input — convert that into a clean Result.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let umap = Umap::new(config);
        umap.fit(
            data.view(),
            knn_indices.view(),
            knn_dists.view(),
            init.view(),
        )
    }))
    .map_err(|_| "UMAP fitting panicked".to_string())?;

    let emb = result.embedding();
    if emb.ncols() < 2 {
        return Err("UMAP produced fewer than 2 output dimensions".into());
    }
    let coords: Vec<(f32, f32)> = (0..emb.nrows())
        .map(|i| (emb[[i, 0]], emb[[i, 1]]))
        .collect();
    Ok(coords)
}

pub fn load_cluster_result(app: &AppHandle) -> Result<Option<ClusterResult>, String> {
    let path = clusters_path(app)?;
    if !path.exists() {
        return Ok(None);
    }
    let bytes = std::fs::read(&path).map_err(|e| e.to_string())?;
    let result: ClusterResult = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    Ok(Some(result))
}

pub fn load_umap_result(app: &AppHandle) -> Result<Option<UmapResult>, String> {
    let path = umap_path(app)?;
    if !path.exists() {
        return Ok(None);
    }
    let bytes = std::fs::read(&path).map_err(|e| e.to_string())?;
    let result: UmapResult = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    Ok(Some(result))
}

/// Best-effort: delete any cached cluster/UMAP results.
/// Called when the DB is cleared so stale assignments don't show up.
pub fn clear_cached_results(app: &AppHandle) {
    if let Ok(p) = clusters_path(app) {
        let _ = std::fs::remove_file(p);
    }
    if let Ok(p) = umap_path(app) {
        let _ = std::fs::remove_file(p);
    }
}

