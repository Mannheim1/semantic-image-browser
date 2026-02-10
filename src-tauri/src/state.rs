//! Application state types shared across the codebase.
//!
//! Contains the embedding backend abstraction (CPU pool vs GPU model),
//! and the main AppState managed by Tauri.

use lancedb::{Connection, Table};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::RwLock;

use crate::config::AppConfig;
use crate::embedding::{EmbeddingModel, GpuEmbeddingModel};

/// Maximum number of embedding model instances to keep in the pool.
/// This limits RAM usage (~500MB per model) while enabling parallel processing.
/// 4 workers = ~2GB for models, which is reasonable for most systems.
pub const MAX_EMBEDDING_WORKERS: usize = 4;

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
    /// Cached config — avoids reading config.json from disk on every command.
    pub config: RwLock<AppConfig>,
    /// Thumbnails directory path, computed once at startup.
    pub thumbnails_dir: PathBuf,
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
