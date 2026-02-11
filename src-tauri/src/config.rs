use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use tauri::{AppHandle, Manager};

use crate::state::AppState;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AppConfig {
    pub watched_directories: Vec<String>,
    /// Runtime type: "cpu" or "gpu". Determines which ONNX Runtime to download/use.
    pub runtime_type: Option<String>,
    /// Path to the SigLIP2 ONNX model directory
    pub model_dir: Option<String>,
    /// Enables debug-only UI (menu item at startup).
    #[serde(default)]
    pub debug_mode: bool,
    /// Enables benchmark CSV logging during scans.
    #[serde(default)]
    pub benchmarking: bool,
}

pub fn config_path(app: &AppHandle) -> Result<PathBuf, String> {
    let app_data = app.path().app_local_data_dir().map_err(|e| e.to_string())?;
    Ok(app_data.join("config.json"))
}

pub fn load_config(app: &AppHandle) -> Result<AppConfig, String> {
    let path = config_path(app)?;
    if !path.exists() {
        return Ok(AppConfig::default());
    }
    let content = fs::read_to_string(&path).map_err(|e| e.to_string())?;
    serde_json::from_str(&content).map_err(|e| e.to_string())
}

fn save_config(app: &AppHandle, config: &AppConfig) -> Result<(), String> {
    let path = config_path(app)?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let content = serde_json::to_string_pretty(config).map_err(|e| e.to_string())?;
    fs::write(&path, content).map_err(|e| e.to_string())
}

/// Read config from the in-memory cache.
pub fn get_config(state: &AppState) -> AppConfig {
    state.config.read().unwrap().clone()
}

/// Apply a mutation to the cached config and write through to disk.
pub fn update_config(app: &AppHandle, state: &AppState, f: impl FnOnce(&mut AppConfig)) -> Result<(), String> {
    let mut cfg = state.config.write().map_err(|e| e.to_string())?;
    f(&mut cfg);
    save_config(app, &cfg)
}
