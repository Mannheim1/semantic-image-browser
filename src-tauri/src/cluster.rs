//! Clustering and 2D projection pipeline.
//!
//! Projects the stored SigLIP image embeddings down to 2D with Barnes-Hut
//! t-SNE, then clusters that same 2D projection with HDBSCAN. Clustering on the
//! projected coordinates (rather than the raw 768-dim vectors) keeps the 2D map
//! and the cluster browser consistent: a visual blob on the map is exactly one
//! browser cluster.

use std::path::Path;

use serde::{Deserialize, Serialize};

use hdbscan::{DistanceMetric, Hdbscan, HdbscanHyperParams};

/// Below this many indexed images the projection/clustering is not meaningful
/// (and t-SNE's neighbour sampling needs a reasonable population to draw from).
const MIN_IMAGES: usize = 20;

/// The automatic minimum cluster size for a library of `n` images: ~1% of the
/// collection, bounded so tiny libraries still cluster and huge ones don't
/// fragment. Shared by `compute` and the Compute Clusters dialog's defaults.
pub fn default_min_cluster_size(n: usize) -> usize {
    (n / 100).clamp(5, 50)
}

/// The automatic `min_samples` for a given minimum cluster size: half of it, so
/// membership is loose enough to reclaim border images without changing how many
/// clusters form. Shared by `compute` and the Compute Clusters dialog's defaults.
pub fn default_min_samples(min_cluster_size: usize) -> usize {
    (min_cluster_size / 2).max(1)
}

/// The automatic `epsilon`: no distance threshold, leaving HDBSCAN's own cluster
/// selection untouched.
pub const DEFAULT_EPSILON: f64 = 0.0;

/// The hyper parameter values the app would pick automatically for a library of
/// `n` images, so the Compute Clusters dialog can pre-fill each field with the
/// value it actually uses. Maximum cluster size has no default (no cap).
#[derive(Clone, Serialize)]
pub struct DefaultParams {
    pub min_cluster_size: usize,
    pub min_samples: usize,
    pub epsilon: f64,
}

/// Gather every auto-default in one place for the Compute Clusters dialog.
pub fn default_params(n: usize) -> DefaultParams {
    let min_cluster_size = default_min_cluster_size(n);
    DefaultParams {
        min_cluster_size,
        min_samples: default_min_samples(min_cluster_size),
        epsilon: DEFAULT_EPSILON,
    }
}

/// One image placed in the 2D projection and assigned to a cluster.
#[derive(Clone, Serialize, Deserialize)]
pub struct ClusterPoint {
    pub path: String,
    pub x: f32,
    pub y: f32,
    /// Cluster id (0-based). `-1` means HDBSCAN left the point unclustered (noise).
    pub cluster: i32,
}

/// Full result of a clustering run, cached in app state for both views.
#[derive(Clone, Serialize, Deserialize)]
pub struct ClusterResult {
    pub points: Vec<ClusterPoint>,
    pub num_clusters: usize,
    pub num_noise: usize,
}

/// Lightweight summary returned to the caller that triggered the computation.
#[derive(Clone, Serialize)]
pub struct ClusterSummary {
    pub num_clusters: usize,
    pub num_noise: usize,
    pub num_images: usize,
}

/// Run Barnes-Hut t-SNE (to 2D) followed by HDBSCAN on the given
/// `(path, embedding)` pairs.
///
/// `min_cluster_size`, `max_cluster_size`, `min_samples`, and `epsilon` are
/// optional overrides for the HDBSCAN hyper parameters of the same name; `None`
/// keeps the auto default (see [`default_params`]) or no cap (max).
///
/// This is CPU-bound and blocking; call it from a blocking context.
pub fn compute(
    data: Vec<(String, Vec<f32>)>,
    min_cluster_size: Option<usize>,
    max_cluster_size: Option<usize>,
    min_samples: Option<usize>,
    epsilon: Option<f64>,
) -> Result<ClusterResult, String> {
    let n = data.len();
    if n < MIN_IMAGES {
        return Err(format!(
            "Need at least {} indexed images to compute clusters (have {}).",
            MIN_IMAGES, n
        ));
    }

    let dim = data[0].1.len();
    for (_, emb) in &data {
        if emb.len() != dim {
            return Err("Embeddings have inconsistent dimensions".to_string());
        }
    }

    // Borrow each stored embedding as a slice for t-SNE's sample list.
    let samples: Vec<&[f32]> = data.iter().map(|(_, emb)| emb.as_slice()).collect();

    // t-SNE samples roughly 3 * perplexity neighbours, so perplexity must stay
    // below n / 3. Scale it with the library and cap at the usual default of 30.
    let perplexity = (((n - 1) as f32) / 3.0).clamp(2.0, 30.0);

    let mut tsne = bhtsne::tSNE::new(&samples);
    tsne.embedding_dim(2)
        .perplexity(perplexity)
        .epochs(1000)
        .barnes_hut(0.5, |a, b| {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f32>()
                .sqrt()
        });
    // Row-major, length n * 2.
    let embedding = tsne.embedding();

    // Cluster the 2D projection so map regions and browser groups match.
    let coords: Vec<Vec<f32>> = (0..n)
        .map(|i| vec![embedding[i * 2], embedding[i * 2 + 1]])
        .collect();

    // Scale the minimum cluster size with the library so large collections do
    // not produce a swarm of tiny clusters, while small ones still cluster. A
    // caller-supplied value (from the Compute Clusters dialog) overrides this.
    let min_cluster_size = min_cluster_size
        .filter(|&v| v >= 2)
        .unwrap_or_else(|| default_min_cluster_size(n));
    // `min_samples` is the density bar a point must clear to be pulled into a
    // cluster: lower it to reclaim border images (less noise), raise it for
    // tighter cores (more noise). Defaults to half of `min_cluster_size`.
    let min_samples = min_samples
        .filter(|&v| v >= 1)
        .unwrap_or_else(|| default_min_samples(min_cluster_size));
    // `epsilon` is a distance threshold that merges clusters splitting below it,
    // pulling nearby fragments and stray points together. 0.0 leaves HDBSCAN's
    // own selection untouched.
    let epsilon = epsilon.filter(|&v| v >= 0.0).unwrap_or(DEFAULT_EPSILON);
    let mut builder = HdbscanHyperParams::builder()
        .min_cluster_size(min_cluster_size)
        .min_samples(min_samples)
        .epsilon(epsilon)
        .dist_metric(DistanceMetric::Euclidean);
    // Capping cluster size makes HDBSCAN reject an oversized blob and select its
    // smaller sub-clusters instead, breaking up one dominant group.
    if let Some(max) = max_cluster_size.filter(|&v| v >= 2) {
        builder = builder.max_cluster_size(max);
    }
    let hyper = builder.build();
    let labels = Hdbscan::new(&coords, hyper)
        .cluster()
        .map_err(|e| format!("HDBSCAN clustering failed: {}", e))?;

    let mut max_label = -1i32;
    let mut num_noise = 0usize;
    let points: Vec<ClusterPoint> = data
        .into_iter()
        .enumerate()
        .map(|(i, (path, _))| {
            let cluster = labels[i];
            if cluster < 0 {
                num_noise += 1;
            } else if cluster > max_label {
                max_label = cluster;
            }
            ClusterPoint {
                path,
                x: embedding[i * 2],
                y: embedding[i * 2 + 1],
                cluster,
            }
        })
        .collect();

    let num_clusters = (max_label + 1) as usize;

    Ok(ClusterResult {
        points,
        num_clusters,
        num_noise,
    })
}

/// Write a clustering result to disk as JSON so it survives between sessions.
pub fn save(path: &Path, result: &ClusterResult) -> Result<(), String> {
    let json = serde_json::to_vec(result).map_err(|e| e.to_string())?;
    std::fs::write(path, json).map_err(|e| e.to_string())
}

/// Load a previously saved clustering result, if one exists. A missing or
/// unreadable file simply yields `None` — the user can recompute.
pub fn load(path: &Path) -> Option<ClusterResult> {
    let bytes = std::fs::read(path).ok()?;
    serde_json::from_slice(&bytes).ok()
}
