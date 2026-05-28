//! HDBSCAN clustering and 2D UMAP-style embedding.
//!
//! Pulls all embeddings out of the LanceDB table, runs the chosen algorithm,
//! and persists the result as JSON in the app data dir. The frontend reads the
//! cached result; recomputing is a user-triggered action via the menu.
//!
//! Pipeline:
//!   1. Build a single HNSW kNN graph from the 768-dim SigLIP embeddings.
//!   2. UMAP-project that to 2D for the map view (min_dist=0.3, PCA init).
//!   3. UMAP-project that to 5D for clustering (min_dist=0.0, PCA init) —
//!      density-based clustering on 2D loses too much neighbourhood
//!      information; 5D preserves it. This matches the standard "UMAP→HDBSCAN
//!      pipeline" recipe used by BERTopic/Top2Vec.
//!   4. HDBSCAN on the 5D coords; assignments saved per-path.

use std::path::PathBuf;

use arrow_array::{Array, FixedSizeListArray, Float32Array, RecordBatch, StringArray};
use futures::TryStreamExt;
use hnsw_rs::prelude::{DistL2, Hnsw};
use lancedb::query::{ExecutableQuery, QueryBase, Select};
use lancedb::Table;
use ndarray::Array2;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Manager};
use umap_rs::{GraphParams, ManifoldParams, Umap, UmapConfig};

/// kNN neighbourhood size used to build the graph. Same for both UMAP runs so
/// they can share a single kNN graph.
const N_NEIGHBORS: usize = 10;

/// Output dimensionality of the UMAP fed to HDBSCAN. 2D loses too much
/// structure; ~5D is the sweet spot for clustering (per BERTopic literature).
const CLUSTER_UMAP_DIM: usize = 5;

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

/// HDBSCAN on a 5D UMAP projection. Builds the kNN graph once and reuses it
/// for both the cluster-feeding UMAP and (if needed) the viz UMAP cache.
pub async fn compute_clusters(app: &AppHandle, table: &Table) -> Result<ClusterResult, String> {
    use hdbscan::Hdbscan;

    let (paths, embeddings) = load_all_embeddings(table).await?;
    if embeddings.is_empty() {
        return Err("No embeddings to cluster. Add a folder first.".into());
    }
    if embeddings.len() <= N_NEIGHBORS {
        return Err(format!(
            "Need more than {} images to cluster, found {}.",
            N_NEIGHBORS,
            embeddings.len()
        ));
    }

    let cached_viz = load_umap_result(app).ok().flatten();
    let viz_cache_valid = cached_viz
        .as_ref()
        .map(|u| umap_matches_paths(u, &paths))
        .unwrap_or(false);

    let app_clone = app.clone();
    let paths_for_task = paths.clone();
    let cached_viz_for_task = cached_viz.clone();

    let (cluster_labels, viz_result) = tokio::task::spawn_blocking(
        move || -> Result<(Vec<i32>, UmapResult), String> {
            // Validate dimensions before doing real work.
            let dim = embeddings[0].len();
            if dim == 0 {
                return Err("Embeddings have zero dimensions".into());
            }
            for v in &embeddings {
                if v.len() != dim {
                    return Err("Embeddings have inconsistent dimensions".into());
                }
            }

            // Build the kNN graph once — reused by both UMAP calls.
            let (knn_indices, knn_dists) = build_knn(&embeddings)?;

            // Refresh the viz UMAP if the cache is stale, so the map view and
            // the clusters stay in sync.
            let viz_result = if viz_cache_valid {
                cached_viz_for_task.expect("viz_cache_valid implies Some")
            } else {
                let arr = run_umap(&embeddings, &knn_indices, &knn_dists, 2, 0.3)?;
                let points: Vec<UmapPoint> = paths_for_task
                    .iter()
                    .enumerate()
                    .map(|(i, p)| UmapPoint {
                        path: p.clone(),
                        x: arr[[i, 0]],
                        y: arr[[i, 1]],
                    })
                    .collect();
                let result = UmapResult {
                    points,
                    computed_at: now_ms(),
                };
                save_umap_result(&app_clone, &result)?;
                result
            };

            // Higher-dim UMAP feeds HDBSCAN. Tight (min_dist=0.0) so neighbours
            // pack together, which is what density-based clustering needs.
            let cluster_arr =
                run_umap(&embeddings, &knn_indices, &knn_dists, CLUSTER_UMAP_DIM, 0.0)?;

            let cluster_points: Vec<Vec<f32>> = (0..cluster_arr.nrows())
                .map(|i| (0..cluster_arr.ncols()).map(|j| cluster_arr[[i, j]]).collect())
                .collect();

            let clusterer = Hdbscan::default_hyper_params(&cluster_points);
            let labels = clusterer
                .cluster()
                .map_err(|e| format!("HDBSCAN failed: {:?}", e))?;

            Ok((labels, viz_result))
        },
    )
    .await
    .map_err(|e| format!("clustering task join error: {}", e))??;

    if cluster_labels.len() != paths.len() {
        return Err(format!(
            "HDBSCAN returned {} labels for {} inputs",
            cluster_labels.len(),
            paths.len()
        ));
    }

    let mut max_label: i32 = -1;
    let assignments: Vec<ClusterAssignment> = paths
        .iter()
        .zip(cluster_labels.iter())
        .map(|(path, &label)| {
            if label >= 0 && label > max_label {
                max_label = label;
            }
            ClusterAssignment {
                path: path.clone(),
                cluster_id: label,
            }
        })
        .collect();

    // Use viz_result so we keep it warm in the build (the JSON cache is the
    // real reason we ran it).
    let _ = viz_result;

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

/// Compute and cache the 2D UMAP used by the map view.
pub async fn compute_umap(app: &AppHandle, table: &Table) -> Result<UmapResult, String> {
    let (paths, embeddings) = load_all_embeddings(table).await?;
    if embeddings.is_empty() {
        return Err("No embeddings to project. Add a folder first.".into());
    }
    if embeddings.len() <= N_NEIGHBORS {
        return Err(format!(
            "Need more than {} images for a 2D map, found {}.",
            N_NEIGHBORS,
            embeddings.len()
        ));
    }

    let arr = tokio::task::spawn_blocking(move || -> Result<Array2<f32>, String> {
        let (knn_indices, knn_dists) = build_knn(&embeddings)?;
        run_umap(&embeddings, &knn_indices, &knn_dists, 2, 0.3)
    })
    .await
    .map_err(|e| format!("UMAP task join error: {}", e))??;

    if arr.nrows() != paths.len() {
        return Err(format!(
            "UMAP returned {} points for {} inputs",
            arr.nrows(),
            paths.len()
        ));
    }

    let points: Vec<UmapPoint> = paths
        .into_iter()
        .enumerate()
        .map(|(i, path)| UmapPoint {
            path,
            x: arr[[i, 0]],
            y: arr[[i, 1]],
        })
        .collect();

    let result = UmapResult {
        points,
        computed_at: now_ms(),
    };

    save_umap_result(app, &result)?;
    Ok(result)
}

fn save_umap_result(app: &AppHandle, result: &UmapResult) -> Result<(), String> {
    let json = serde_json::to_vec_pretty(result).map_err(|e| e.to_string())?;
    let path = umap_path(app)?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    std::fs::write(&path, &json).map_err(|e| e.to_string())
}

/// True iff the cached UMAP has exactly the same paths in the same order.
fn umap_matches_paths(cached: &UmapResult, paths: &[String]) -> bool {
    if cached.points.len() != paths.len() {
        return false;
    }
    cached
        .points
        .iter()
        .zip(paths.iter())
        .all(|(p, q)| &p.path == q)
}

/// Build a kNN graph via HNSW. Returns (indices, distances) arrays of shape
/// (n, N_NEIGHBORS), with each row's self-match removed.
fn build_knn(embeddings: &[Vec<f32>]) -> Result<(Array2<u32>, Array2<f32>), String> {
    let n = embeddings.len();

    let ef_c = 50usize;
    let max_nb_connection = 70usize;
    let nb_layer = 16usize.min(((n as f32).ln().trunc() as usize).max(1));
    let hnsw = Hnsw::<f32, DistL2>::new(max_nb_connection, n, nb_layer, ef_c, DistL2 {});
    let data_with_id: Vec<(&Vec<f32>, usize)> = embeddings.iter().zip(0..n).collect();
    hnsw.parallel_insert(&data_with_id);

    let ef_search = (N_NEIGHBORS * 4).max(50);
    let results = hnsw.parallel_search(embeddings, N_NEIGHBORS + 1, ef_search);

    let mut knn_indices = Array2::<u32>::zeros((n, N_NEIGHBORS));
    let mut knn_dists = Array2::<f32>::zeros((n, N_NEIGHBORS));
    for (i, neighbours) in results.iter().enumerate() {
        let mut col = 0usize;
        for nb in neighbours.iter() {
            if col == N_NEIGHBORS {
                break;
            }
            if nb.d_id == i {
                continue;
            }
            knn_indices[[i, col]] = nb.d_id as u32;
            knn_dists[[i, col]] = nb.distance;
            col += 1;
        }
        if col < N_NEIGHBORS {
            return Err(format!(
                "HNSW returned only {} neighbours for point {} (need {})",
                col, i, N_NEIGHBORS
            ));
        }
    }
    Ok((knn_indices, knn_dists))
}

/// Run UMAP with PCA initialisation. Caller provides the pre-built kNN graph.
/// Returns the embedded coords as (n × n_components).
fn run_umap(
    embeddings: &[Vec<f32>],
    knn_indices: &Array2<u32>,
    knn_dists: &Array2<f32>,
    n_components: usize,
    min_dist: f32,
) -> Result<Array2<f32>, String> {
    let n = embeddings.len();
    let dim = embeddings[0].len();

    let config = UmapConfig {
        n_components,
        graph: GraphParams {
            n_neighbors: N_NEIGHBORS,
            ..Default::default()
        },
        manifold: ManifoldParams {
            min_dist,
            ..Default::default()
        },
        ..Default::default()
    };

    // Pack input data into the contiguous Array2 umap-rs wants.
    let mut data = Array2::<f32>::zeros((n, dim));
    for (i, v) in embeddings.iter().enumerate() {
        for (j, &x) in v.iter().enumerate() {
            data[[i, j]] = x;
        }
    }

    let init = pca_init(embeddings, n_components)?;

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
    if emb.ncols() < n_components {
        return Err(format!(
            "UMAP produced {} dims, expected {}",
            emb.ncols(),
            n_components
        ));
    }
    Ok(emb.to_owned())
}

/// PCA initialisation: project centered embeddings onto their top-N principal
/// components. Uses power iteration on the implicit covariance matrix — we
/// never materialise the dim×dim covariance, we just multiply by X^T X / n via
/// two passes over the data per iteration.
fn pca_init(embeddings: &[Vec<f32>], n_components: usize) -> Result<Array2<f32>, String> {
    let n = embeddings.len();
    let dim = embeddings[0].len();
    const ITERS: usize = 30;

    // Column means.
    let mut mean = vec![0.0f32; dim];
    for v in embeddings {
        for (i, x) in v.iter().enumerate() {
            mean[i] += x;
        }
    }
    for x in mean.iter_mut() {
        *x /= n as f32;
    }

    // Find n_components principal components via power iteration with deflation.
    let mut components: Vec<Vec<f32>> = Vec::with_capacity(n_components);
    for _ in 0..n_components {
        let pc = power_iteration(embeddings, &mean, &components, ITERS);
        components.push(pc);
    }

    // Project each (centered) embedding onto each component.
    let rows: Vec<Vec<f32>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let v = &embeddings[i];
            components
                .iter()
                .map(|pc| {
                    let mut s = 0.0f32;
                    for j in 0..dim {
                        s += (v[j] - mean[j]) * pc[j];
                    }
                    s
                })
                .collect()
        })
        .collect();

    let mut init = Array2::<f32>::zeros((n, n_components));
    for (i, row) in rows.into_iter().enumerate() {
        for (j, x) in row.into_iter().enumerate() {
            init[[i, j]] = x;
        }
    }

    standardise_columns(&mut init, 5.0);
    Ok(init)
}

/// Power iteration for top eigenvector of (X^T X / n), where X is the row-wise
/// centered embedding matrix. `deflate` holds previously-found eigenvectors;
/// the iteration projects them out at every step (Gram-Schmidt) so we get the
/// next-largest eigenvector each call.
fn power_iteration(
    embeddings: &[Vec<f32>],
    mean: &[f32],
    deflate: &[Vec<f32>],
    iters: usize,
) -> Vec<f32> {
    let n = embeddings.len();
    let dim = embeddings[0].len();

    let mut v: Vec<f32> = (0..dim)
        .map(|i| (i as f32 * 0.1).sin() + 0.5)
        .collect();
    normalise(&mut v);
    project_out(&mut v, deflate);
    normalise(&mut v);

    for _ in 0..iters {
        // u[i] = sum_d (X[i][d] - mean[d]) * v[d]
        let u: Vec<f32> = (0..n)
            .into_par_iter()
            .map(|i| {
                let row = &embeddings[i];
                let mut s = 0.0f32;
                for d in 0..dim {
                    s += (row[d] - mean[d]) * v[d];
                }
                s
            })
            .collect();

        // w[d] = sum_i u[i] * (X[i][d] - mean[d]) / n
        let mut w: Vec<f32> = (0..dim)
            .into_par_iter()
            .map(|d| {
                let mut s = 0.0f32;
                for i in 0..n {
                    s += u[i] * (embeddings[i][d] - mean[d]);
                }
                s / n as f32
            })
            .collect();

        project_out(&mut w, deflate);
        let norm = norm_of(&w);
        if norm < 1e-10 {
            return v;
        }
        for x in w.iter_mut() {
            *x /= norm;
        }
        v = w;
    }

    v
}

fn normalise(v: &mut [f32]) {
    let n = norm_of(v);
    if n > 1e-12 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

fn norm_of(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn project_out(v: &mut [f32], basis: &[Vec<f32>]) {
    for b in basis {
        let dot: f32 = v.iter().zip(b.iter()).map(|(a, c)| a * c).sum();
        for (x, c) in v.iter_mut().zip(b.iter()) {
            *x -= dot * c;
        }
    }
}

/// Centre each column on 0 and scale to roughly the given standard deviation.
fn standardise_columns(arr: &mut Array2<f32>, target_std: f32) {
    let n = arr.nrows();
    for col in 0..arr.ncols() {
        let mut sum = 0.0f32;
        let mut sumsq = 0.0f32;
        for i in 0..n {
            let v = arr[[i, col]];
            sum += v;
            sumsq += v * v;
        }
        let mean = sum / n as f32;
        let var = (sumsq / n as f32) - mean * mean;
        let std = var.max(0.0).sqrt().max(1e-8);
        let scale = target_std / std;
        for i in 0..n {
            arr[[i, col]] = (arr[[i, col]] - mean) * scale;
        }
    }
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
