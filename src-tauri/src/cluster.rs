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
/// This is CPU-bound and blocking; call it from a blocking context.
pub fn compute(data: Vec<(String, Vec<f32>)>) -> Result<ClusterResult, String> {
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
    // not produce a swarm of tiny clusters, while small ones still cluster.
    let min_cluster_size = (n / 100).clamp(5, 50);
    // `min_samples` is the density bar a point must clear to be pulled into a
    // cluster. Left unset it defaults to `min_cluster_size`, which is strict
    // enough to shave off border images that sit visually and semantically
    // inside a cluster (the grey dots mixed into a coloured blob on the map).
    // Halving it loosens *membership* — reclaiming those border images — without
    // changing how many clusters form (`min_cluster_size` still gates that).
    let min_samples = (min_cluster_size / 2).max(1);
    let hyper = HdbscanHyperParams::builder()
        .min_cluster_size(min_cluster_size)
        .min_samples(min_samples)
        .dist_metric(DistanceMetric::Euclidean)
        .build();
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
