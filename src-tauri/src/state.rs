//! Application state types shared across the codebase.
//!
//! Contains the embedding backend (model instances + inference config),
//! and the main AppState managed by Tauri.

use lancedb::{Connection, Table};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::RwLock;

use crate::config::AppConfig;
use crate::embedding::EmbeddingModel;

/// Controls how the scan pipeline orchestrates inference.
pub struct InferenceConfig {
    /// Number of model instances. CPU: multiple for thread-parallel. GPU: 1.
    pub model_instances: usize,
    /// Images per inference call. 1 = single-image. >1 = batched.
    pub batch_size: usize,
    /// Use pipeline mode: rayon preprocessing + dedicated inference thread(s).
    /// True for accelerator backends (CUDA, CoreML) where CPU preprocessing and
    /// accelerator inference should run concurrently.
    pub pipeline: bool,
}

/// Return the inference config for the active backend feature.
pub fn inference_config() -> InferenceConfig {
    #[cfg(feature = "backend-cuda")]
    {
        InferenceConfig {
            model_instances: 1,
            batch_size: 32,
            pipeline: true,
        }
    }
    #[cfg(feature = "backend-coreml")]
    {
        InferenceConfig {
            model_instances: 2,
            batch_size: 1,
            pipeline: true,
        }
    }
    #[cfg(all(not(feature = "backend-cuda"), not(feature = "backend-coreml")))]
    {
        // Cap at 4 workers to limit RAM (~500MB per model instance)
        const MAX_WORKERS: usize = 4;
        InferenceConfig {
            model_instances: std::thread::available_parallelism()
                .map(|n| n.get().min(MAX_WORKERS))
                .unwrap_or(2),
            batch_size: 1,
            pipeline: false,
        }
    }
}

/// Holds loaded embedding model instances and the inference config.
/// CPU mode uses multiple model instances for thread-parallel inference.
/// GPU mode uses a single model instance — the GPU handles parallelism internally.
pub struct EmbeddingBackend {
    pub models: Vec<Mutex<EmbeddingModel>>,
    pub config: InferenceConfig,
}

impl EmbeddingBackend {
    /// Load a new backend with the given config.
    /// Creates `config.model_instances` copies of the model.
    pub fn load(
        model_dir: &std::path::Path,
        #[cfg(feature = "backend-coreml")] cache_dir: &std::path::Path,
    ) -> Result<Self, String> {
        let config = inference_config();
        let mut models = Vec::with_capacity(config.model_instances);

        for i in 0..config.model_instances {
            match EmbeddingModel::load(
                model_dir,
                #[cfg(feature = "backend-coreml")]
                cache_dir,
            ) {
                Ok(model) => models.push(Mutex::new(model)),
                Err(e) => {
                    if models.is_empty() {
                        return Err(format!("Failed to load embedding model: {}", e));
                    } else {
                        eprintln!(
                            "Warning: Could only load {} of {} embedding models: {}",
                            i, config.model_instances, e
                        );
                        break;
                    }
                }
            }
        }

        println!(
            "Loaded {} embedding model instance(s) (batch_size={})",
            models.len(),
            config.batch_size
        );

        Ok(Self { models, config })
    }

    /// Embed text using any available model instance (for search queries).
    /// Tries each instance with try_lock first, falling back to blocking on the first.
    pub fn embed_text(&self, text: &str) -> Result<Vec<f32>, String> {
        // Try to find an unlocked instance (avoids blocking during scans)
        for model in &self.models {
            if let Ok(mut guard) = model.try_lock() {
                return guard.embed_text(text);
            }
        }
        // All busy — block on the first one
        let model = self.models.first().ok_or("No models loaded")?;
        let mut guard = model
            .lock()
            .map_err(|e| format!("Failed to lock model: {}", e))?;
        guard.embed_text(text)
    }
}

pub struct AppState {
    pub db: Connection,
    pub table: tokio::sync::Mutex<Table>,
    /// Embedding backend — model instances + inference config.
    /// Wrapped in RwLock<Option<Arc<...>>> to allow async initialization after app starts.
    /// Arc allows cloning a reference to use across await points.
    pub embedding_backend: RwLock<Option<Arc<EmbeddingBackend>>>,
    /// Model identifier (e.g., "siglip2-base-patch16-256") for database storage.
    /// Wrapped in RwLock to allow async initialization after app starts.
    pub model_id: RwLock<Option<String>>,
    /// Cached config — avoids reading config.json from disk on every command.
    pub config: RwLock<AppConfig>,
    /// Thumbnails directory path, computed once at startup.
    pub thumbnails_dir: PathBuf,
    /// Most recent clustering result, populated by the button-triggered
    /// `compute_clusters` command and read by the cluster browser / 2D map views.
    pub clusters: RwLock<Option<crate::cluster::ClusterResult>>,
}
