//! Embedding generation using SigLIP2 ONNX models.
//!
//! This module handles:
//! - Loading vision and text ONNX models
//! - Image preprocessing (resize, normalize)
//! - Text tokenization
//! - Generating embeddings for images and text queries

use ort::execution_providers::CUDAExecutionProvider;
use ort::session::Session;
use ort::value::Value;
use std::path::Path;
use std::sync::OnceLock;
use tokenizers::Tokenizer;

use crate::benchmark::{self, PreprocessTiming};
use crate::database::VISUAL_EMBEDDING_DIM;
use crate::image_ops::{decode_image_to_rgb, fast_resize_rgb};

/// Stores the result of ORT initialization (success or error message).
/// Using OnceLock ensures the result is computed once and remembered,
/// so subsequent calls return the same result (including errors).
static ORT_INIT_RESULT: OnceLock<Result<(), String>> = OnceLock::new();

/// Initialize the ONNX Runtime with the library at the given path.
/// This must be called before creating any sessions when using load-dynamic.
/// Safe to call multiple times - only the first call has effect.
/// Subsequent calls return the same result as the first call.
///
/// On Windows, this also adds the library's parent directory to the DLL search path
/// so that CUDA dependencies (cuBLAS, cuDNN) are found when loading the GPU provider.
pub fn init_ort(dylib_path: &Path) -> Result<(), String> {
    ORT_INIT_RESULT
        .get_or_init(|| {
            // On Windows, add the library directory to DLL search path so CUDA deps are found
            #[cfg(target_os = "windows")]
            if let Some(lib_dir) = dylib_path.parent() {
                add_dll_directory(lib_dir)?;
            }

            match ort::init_from(dylib_path) {
                Ok(builder) => {
                    builder.commit();
                    Ok(())
                }
                Err(e) => Err(format!(
                    "Failed to initialize ONNX Runtime from {}: {}",
                    dylib_path.display(),
                    e
                )),
            }
        })
        .clone()
}

/// Add a directory to the Windows DLL search path.
/// This ensures CUDA libraries bundled with the app are found before system-installed ones.
#[cfg(target_os = "windows")]
fn add_dll_directory(dir: &Path) -> Result<(), String> {
    use std::os::windows::ffi::OsStrExt;

    // Convert path to wide string for Windows API
    let wide_path: Vec<u16> = dir.as_os_str().encode_wide().chain(std::iter::once(0)).collect();

    // SetDllDirectoryW adds a directory to the search path for LoadLibrary calls
    // Returns non-zero on success
    let result = unsafe { windows_sys::Win32::System::LibraryLoader::SetDllDirectoryW(wide_path.as_ptr()) };

    if result == 0 {
        Err(format!(
            "Failed to set DLL directory to {}: Windows error {}",
            dir.display(),
            std::io::Error::last_os_error()
        ))
    } else {
        println!("Added DLL search directory: {}", dir.display());
        Ok(())
    }
}

/// Image size expected by the SigLIP2 model (256x256 for siglip2-base-patch16-256)
pub const IMAGE_SIZE: u32 = 256;

/// Maximum sequence length for text tokenization
const MAX_SEQ_LENGTH: usize = 64;

/// Normalization mean (per channel)
const IMAGE_MEAN: [f32; 3] = [0.5, 0.5, 0.5];

/// Normalization std (per channel)
const IMAGE_STD: [f32; 3] = [0.5, 0.5, 0.5];

/// Holds the loaded ONNX sessions for vision and text encoding.
pub struct EmbeddingModel {
    vision_session: Session,
    text_session: Session,
    tokenizer: Tokenizer,
}

impl EmbeddingModel {
    /// Load the embedding model from the given directory.
    ///
    /// Expects the directory to contain:
    /// - `onnx/vision_model.onnx`
    /// - `onnx/text_model.onnx`
    /// - `tokenizer.json`
    ///
    /// If `use_gpu` is true, attempts to use CUDA execution provider.
    pub fn load(model_dir: &Path, use_gpu: bool) -> Result<Self, String> {
        let vision_path = model_dir.join("onnx").join("vision_model.onnx");
        let text_path = model_dir.join("onnx").join("text_model.onnx");
        let tokenizer_path = model_dir.join("tokenizer.json");

        // Verify files exist
        if !vision_path.exists() {
            return Err(format!("Vision model not found: {}", vision_path.display()));
        }
        if !text_path.exists() {
            return Err(format!("Text model not found: {}", text_path.display()));
        }
        if !tokenizer_path.exists() {
            return Err(format!("Tokenizer not found: {}", tokenizer_path.display()));
        }

        // Load vision model
        let vision_session = if use_gpu {
            Session::builder()
                .map_err(|e| format!("Failed to create vision session builder: {}", e))?
                .with_execution_providers([CUDAExecutionProvider::default().build().error_on_failure()])
                .map_err(|e| format!("Failed to register CUDA execution provider for vision model: {}. Make sure CUDA 11.8+ and cuDNN are installed.", e))?
                .commit_from_file(&vision_path)
                .map_err(|e| format!("Failed to load vision model: {}", e))?
        } else {
            Session::builder()
                .map_err(|e| format!("Failed to create vision session builder: {}", e))?
                .commit_from_file(&vision_path)
                .map_err(|e| format!("Failed to load vision model: {}", e))?
        };

        // Load text model
        let text_session = if use_gpu {
            Session::builder()
                .map_err(|e| format!("Failed to create text session builder: {}", e))?
                .with_execution_providers([CUDAExecutionProvider::default().build().error_on_failure()])
                .map_err(|e| format!("Failed to register CUDA execution provider for text model: {}. Make sure CUDA 11.8+ and cuDNN are installed.", e))?
                .commit_from_file(&text_path)
                .map_err(|e| format!("Failed to load text model: {}", e))?
        } else {
            Session::builder()
                .map_err(|e| format!("Failed to create text session builder: {}", e))?
                .commit_from_file(&text_path)
                .map_err(|e| format!("Failed to load text model: {}", e))?
        };

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| format!("Failed to load tokenizer: {}", e))?;

        Ok(Self {
            vision_session,
            text_session,
            tokenizer,
        })
    }

    /// Generate an embedding for an image file.
    ///
    /// Returns a 768-dimensional L2-normalized vector.
    pub fn embed_image(&mut self, image_path: &Path) -> Result<Vec<f32>, String> {
        use std::time::Instant;

        // Preprocess the image
        let (pixel_values, timing) = preprocess_image(image_path)?;

        // Run inference
        let inference_start = Instant::now();
        let embedding = self.run_vision_inference(&pixel_values)?;
        let inference_time = inference_start.elapsed();

        benchmark::log_image(&timing, inference_time, "cpu");

        // L2 normalize
        Ok(l2_normalize(&embedding))
    }

    /// Generate an embedding from pre-computed pixel values (NCHW float tensor).
    /// Used in the merged scan pipeline where decoding/preprocessing is done separately.
    ///
    /// Returns a 768-dimensional L2-normalized vector.
    pub fn embed_preprocessed(&mut self, pixel_values: &[f32]) -> Result<Vec<f32>, String> {
        let embedding = self.run_vision_inference(pixel_values)?;
        Ok(l2_normalize(&embedding))
    }

    /// Generate an embedding for a text query.
    ///
    /// Returns a 768-dimensional L2-normalized vector.
    pub fn embed_text(&mut self, query: &str) -> Result<Vec<f32>, String> {
        // Tokenize the text
        let input_ids = self.tokenize(query)?;

        // Run inference
        let embedding = self.run_text_inference(&input_ids)?;

        // L2 normalize
        Ok(l2_normalize(&embedding))
    }

    /// Tokenize text for the model.
    ///
    /// Applies lowercasing and pads/truncates to MAX_SEQ_LENGTH.
    fn tokenize(&self, text: &str) -> Result<Vec<i64>, String> {
        // SigLIP2 was trained with lowercased text
        let text_lower = text.to_lowercase();

        let encoding = self
            .tokenizer
            .encode(text_lower, true)
            .map_err(|e| format!("Tokenization failed: {}", e))?;

        let mut ids: Vec<i64> = encoding
            .get_ids()
            .iter()
            .map(|&id| id as i64)
            .collect();

        // Truncate if necessary
        if ids.len() > MAX_SEQ_LENGTH {
            ids.truncate(MAX_SEQ_LENGTH);
        }

        // Pad if necessary (pad token id is 0 for this model)
        while ids.len() < MAX_SEQ_LENGTH {
            ids.push(0);
        }

        Ok(ids)
    }

    /// Run the vision model to get the pooler output.
    fn run_vision_inference(&mut self, pixel_values: &[f32]) -> Result<Vec<f32>, String> {
        // Create input tensor with shape [1, 3, 256, 256]
        let shape = [1_i64, 3, IMAGE_SIZE as i64, IMAGE_SIZE as i64];
        let input_tensor = Value::from_array((shape, pixel_values.to_vec()))
            .map_err(|e| format!("Failed to create vision input tensor: {}", e))?;

        // Run inference
        let outputs = self
            .vision_session
            .run(ort::inputs!["pixel_values" => input_tensor])
            .map_err(|e| format!("Vision inference failed: {}", e))?;

        // Extract pooler_output (second output)
        let pooler_output = outputs
            .get("pooler_output")
            .ok_or("pooler_output not found in vision model outputs")?;

        let (_shape, data) = pooler_output
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("Failed to extract pooler_output tensor: {}", e))?;

        // Get the embedding (shape should be [1, 768])
        let embedding: Vec<f32> = data.iter().copied().collect();

        if embedding.len() != VISUAL_EMBEDDING_DIM as usize {
            return Err(format!(
                "Unexpected embedding dimension: expected {}, got {}",
                VISUAL_EMBEDDING_DIM,
                embedding.len()
            ));
        }

        Ok(embedding)
    }

    /// Run the text model to get the pooler output.
    fn run_text_inference(&mut self, input_ids: &[i64]) -> Result<Vec<f32>, String> {
        // Create input tensor with shape [1, sequence_length]
        let shape = [1_i64, input_ids.len() as i64];
        let input_tensor = Value::from_array((shape, input_ids.to_vec()))
            .map_err(|e| format!("Failed to create text input tensor: {}", e))?;

        // Run inference
        let outputs = self
            .text_session
            .run(ort::inputs!["input_ids" => input_tensor])
            .map_err(|e| format!("Text inference failed: {}", e))?;

        // Extract pooler_output
        let pooler_output = outputs
            .get("pooler_output")
            .ok_or("pooler_output not found in text model outputs")?;

        let (_shape, data) = pooler_output
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("Failed to extract pooler_output tensor: {}", e))?;

        // Get the embedding (shape should be [1, 768])
        let embedding: Vec<f32> = data.iter().copied().collect();

        if embedding.len() != VISUAL_EMBEDDING_DIM as usize {
            return Err(format!(
                "Unexpected embedding dimension: expected {}, got {}",
                VISUAL_EMBEDDING_DIM,
                embedding.len()
            ));
        }

        Ok(embedding)
    }
}

/// Preprocess an image for the SigLIP2 vision model.
///
/// Steps:
/// 1. Load and decode the image (using turbojpeg with scaled decoding for JPEGs)
/// 2. Convert to RGB
/// 3. Resize to 256x256 (using fast_image_resize with SIMD)
/// 4. Convert to float and rescale to [0, 1]
/// 5. Normalize with mean=0.5, std=0.5 (resulting in [-1, 1])
/// 6. Return as flat NCHW format [1, 3, 256, 256] = 196608 floats
///
/// Also returns timing data for benchmark logging.
pub fn preprocess_image(path: &Path) -> Result<(Vec<f32>, PreprocessTiming), String> {
    let start = std::time::Instant::now();

    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase())
        .unwrap_or_default();

    let file_size_bytes = std::fs::metadata(path)
        .map(|m| m.len())
        .unwrap_or(0);

    let (rgb_data, width, height) = decode_image_to_rgb(path)?;
    let decode_time = start.elapsed();

    let filename = path.file_name().unwrap_or_default().to_string_lossy();
    let (pixel_values, mut timing) =
        preprocess_image_from_rgb(&rgb_data, width, height, &filename, &ext, file_size_bytes, None)?;
    timing.decode = decode_time;

    Ok((pixel_values, timing))
}

/// Preprocess pre-decoded RGB image data into an embedding tensor.
/// Same as preprocess_image but skips the decode step (uses provided RGB data).
/// The decode_time in the returned timing will be zero.
pub fn preprocess_image_from_rgb(
    rgb_data: &[u8],
    width: u32,
    height: u32,
    file: &str,
    file_type: &str,
    file_size_bytes: u64,
    resizer: Option<&mut fast_image_resize::Resizer>,
) -> Result<(Vec<f32>, PreprocessTiming), String> {
    use std::time::Instant;
    let start = Instant::now();

    // Resize to IMAGE_SIZE x IMAGE_SIZE using fast_image_resize (SIMD-accelerated)
    let resized_rgb = fast_resize_rgb(rgb_data, width, height, IMAGE_SIZE, IMAGE_SIZE, resizer)?;
    let resize_time = start.elapsed();

    // Convert to NCHW float tensor with normalization
    let tensor_start = Instant::now();
    let pixel_values = rgb_to_nchw_normalized(&resized_rgb, IMAGE_SIZE, IMAGE_SIZE);
    let tensor_time = tensor_start.elapsed();

    let timing = PreprocessTiming {
        file: file.to_string(),
        file_type: file_type.to_string(),
        file_size_bytes,
        source_width: width,
        source_height: height,
        decode: std::time::Duration::ZERO,
        thumbnail: std::time::Duration::ZERO,
        resize: resize_time,
        tensor: tensor_time,
    };

    Ok((pixel_values, timing))
}


/// Convert RGB u8 data to NCHW float tensor with normalization.
fn rgb_to_nchw_normalized(rgb_data: &[u8], width: u32, height: u32) -> Vec<f32> {
    let mut pixel_values = vec![0.0f32; (3 * width * height) as usize];

    for c in 0..3 {
        for y in 0..height as usize {
            for x in 0..width as usize {
                let src_idx = (y * width as usize + x) * 3 + c;
                let value = rgb_data[src_idx] as f32 / 255.0;
                let normalized = (value - IMAGE_MEAN[c]) / IMAGE_STD[c];
                let dst_idx = c * (width * height) as usize + y * width as usize + x;
                pixel_values[dst_idx] = normalized;
            }
        }
    }

    pixel_values
}

/// L2 normalize a vector.
fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter().map(|x| x / norm).collect()
    } else {
        v.to_vec()
    }
}

/// Default batch size for GPU inference.
/// 32 is a good balance for a 4070 Ti (12GB VRAM).
/// Each image at 256x256 float32 RGB = ~768KB, so 32 images = ~24MB batch tensor.
pub const GPU_BATCH_SIZE: usize = 32;

/// Result of parallel preprocessing for a batch of images.
/// Contains the concatenated pixel data ready for GPU inference,
/// along with per-image tracking info for error handling and logging.
pub struct PreprocessedBatch {
    /// Concatenated f32 pixel data for all successfully preprocessed images.
    pub pixel_data: Vec<f32>,
    /// Indices (into the original path slice) of images that preprocessed successfully.
    pub valid_indices: Vec<usize>,
    /// Per-image preprocessing timings (None for images that failed).
    pub timings: Vec<Option<PreprocessTiming>>,
    /// Per-image error messages for images that failed preprocessing.
    pub errors: Vec<Option<String>>,
    /// Total number of images in this batch.
    pub count: usize,
}

/// GPU-optimized embedding model that processes images in batches.
/// Uses a single ONNX session to maximize GPU utilization.
pub struct GpuEmbeddingModel {
    vision_session: Session,
    text_session: Session,
    tokenizer: Tokenizer,
}

impl GpuEmbeddingModel {
    /// Load the embedding model for GPU batched inference.
    ///
    /// Unlike EmbeddingModel which creates multiple instances for CPU parallelism,
    /// GpuEmbeddingModel uses a single session since the GPU handles parallelism internally.
    pub fn load(model_dir: &Path) -> Result<Self, String> {
        let vision_path = model_dir.join("onnx").join("vision_model.onnx");
        let text_path = model_dir.join("onnx").join("text_model.onnx");
        let tokenizer_path = model_dir.join("tokenizer.json");

        // Verify files exist
        if !vision_path.exists() {
            return Err(format!("Vision model not found: {}", vision_path.display()));
        }
        if !text_path.exists() {
            return Err(format!("Text model not found: {}", text_path.display()));
        }
        if !tokenizer_path.exists() {
            return Err(format!("Tokenizer not found: {}", tokenizer_path.display()));
        }

        // Load vision model with CUDA
        let vision_session = Session::builder()
            .map_err(|e| format!("Failed to create vision session builder: {}", e))?
            .with_execution_providers([CUDAExecutionProvider::default().build().error_on_failure()])
            .map_err(|e| format!("Failed to register CUDA execution provider for vision model: {}. Make sure CUDA 11.8+ and cuDNN are installed.", e))?
            .commit_from_file(&vision_path)
            .map_err(|e| format!("Failed to load vision model: {}", e))?;

        // Load text model with CUDA
        let text_session = Session::builder()
            .map_err(|e| format!("Failed to create text session builder: {}", e))?
            .with_execution_providers([CUDAExecutionProvider::default().build().error_on_failure()])
            .map_err(|e| format!("Failed to register CUDA execution provider for text model: {}. Make sure CUDA 11.8+ and cuDNN are installed.", e))?
            .commit_from_file(&text_path)
            .map_err(|e| format!("Failed to load text model: {}", e))?;

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| format!("Failed to load tokenizer: {}", e))?;

        println!("Loaded GPU embedding model (single session for batched inference)");
        Ok(Self {
            vision_session,
            text_session,
            tokenizer,
        })
    }

    /// Run GPU inference on a preprocessed batch, returning per-image embeddings.
    ///
    /// Takes a `PreprocessedBatch` from `preprocess_batch()` and runs the vision model.
    /// Returns a Vec of Results, one per image in the original batch (preserving order).
    pub fn infer_batch(&mut self, batch: &PreprocessedBatch) -> Vec<Result<Vec<f32>, String>> {
        use std::time::Instant;

        let mut results: Vec<Result<Vec<f32>, String>> = Vec::with_capacity(batch.count);
        for i in 0..batch.count {
            if let Some(err) = &batch.errors[i] {
                results.push(Err(err.clone()));
            } else {
                results.push(Err("Not processed".to_string()));
            }
        }

        if batch.valid_indices.is_empty() {
            return results;
        }

        // Run batched inference
        let num_valid = batch.valid_indices.len() as i64;
        let shape = [num_valid, 3, IMAGE_SIZE as i64, IMAGE_SIZE as i64];

        let input_tensor = match Value::from_array((shape, batch.pixel_data.clone())) {
            Ok(t) => t,
            Err(e) => {
                let err = format!("Failed to create batched vision input tensor: {}", e);
                for &idx in &batch.valid_indices {
                    results[idx] = Err(err.clone());
                }
                return results;
            }
        };

        let inference_start = Instant::now();
        let outputs = match self.vision_session.run(ort::inputs!["pixel_values" => input_tensor]) {
            Ok(o) => o,
            Err(e) => {
                let err = format!("Batched vision inference failed: {}", e);
                for &idx in &batch.valid_indices {
                    results[idx] = Err(err.clone());
                }
                return results;
            }
        };
        let inference_time = inference_start.elapsed();

        let pooler_output = match outputs.get("pooler_output") {
            Some(o) => o,
            None => {
                let err = "pooler_output not found in vision model outputs".to_string();
                for &idx in &batch.valid_indices {
                    results[idx] = Err(err.clone());
                }
                return results;
            }
        };

        let (_shape, data) = match pooler_output.try_extract_tensor::<f32>() {
            Ok(d) => d,
            Err(e) => {
                let err = format!("Failed to extract batched pooler_output tensor: {}", e);
                for &idx in &batch.valid_indices {
                    results[idx] = Err(err.clone());
                }
                return results;
            }
        };

        // Split the batched output into individual embeddings
        let embedding_dim = VISUAL_EMBEDDING_DIM as usize;
        let flat_data: Vec<f32> = data.iter().copied().collect();

        // Compute per-image share of inference time for logging
        let per_image_inference = inference_time / batch.valid_indices.len() as u32;

        for (batch_idx, &original_idx) in batch.valid_indices.iter().enumerate() {
            let start = batch_idx * embedding_dim;
            let end = start + embedding_dim;
            if end <= flat_data.len() {
                let embedding: Vec<f32> = flat_data[start..end].to_vec();
                results[original_idx] = Ok(l2_normalize(&embedding));
            } else {
                results[original_idx] = Err(format!(
                    "Embedding index out of bounds: expected {}..{}, got len {}",
                    start, end, flat_data.len()
                ));
            }

            // Log per-image timing
            if let Some(timing) = &batch.timings[original_idx] {
                benchmark::log_image(timing, per_image_inference, "gpu_batch");
            }
        }

        results
    }

    /// Generate an embedding for a text query.
    ///
    /// Text queries are typically single items, so no batching needed here.
    pub fn embed_text(&mut self, query: &str) -> Result<Vec<f32>, String> {
        let input_ids = self.tokenize(query)?;

        // Create input tensor with shape [1, sequence_length]
        let shape = [1_i64, input_ids.len() as i64];
        let input_tensor = Value::from_array((shape, input_ids))
            .map_err(|e| format!("Failed to create text input tensor: {}", e))?;

        let outputs = self
            .text_session
            .run(ort::inputs!["input_ids" => input_tensor])
            .map_err(|e| format!("Text inference failed: {}", e))?;

        let pooler_output = outputs
            .get("pooler_output")
            .ok_or("pooler_output not found in text model outputs")?;

        let (_shape, data) = pooler_output
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("Failed to extract pooler_output tensor: {}", e))?;

        let embedding: Vec<f32> = data.iter().copied().collect();

        if embedding.len() != VISUAL_EMBEDDING_DIM as usize {
            return Err(format!(
                "Unexpected embedding dimension: expected {}, got {}",
                VISUAL_EMBEDDING_DIM,
                embedding.len()
            ));
        }

        Ok(l2_normalize(&embedding))
    }

    /// Tokenize text for the model.
    fn tokenize(&self, text: &str) -> Result<Vec<i64>, String> {
        let text_lower = text.to_lowercase();

        let encoding = self
            .tokenizer
            .encode(text_lower, true)
            .map_err(|e| format!("Tokenization failed: {}", e))?;

        let mut ids: Vec<i64> = encoding
            .get_ids()
            .iter()
            .map(|&id| id as i64)
            .collect();

        if ids.len() > MAX_SEQ_LENGTH {
            ids.truncate(MAX_SEQ_LENGTH);
        }

        while ids.len() < MAX_SEQ_LENGTH {
            ids.push(0);
        }

        Ok(ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l2_normalize() {
        let v = vec![3.0, 4.0];
        let normalized = l2_normalize(&v);
        assert!((normalized[0] - 0.6).abs() < 1e-6);
        assert!((normalized[1] - 0.8).abs() < 1e-6);

        // Check that it's unit length
        let length: f32 = normalized.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((length - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_l2_normalize_zero() {
        let v = vec![0.0, 0.0, 0.0];
        let normalized = l2_normalize(&v);
        assert_eq!(normalized, vec![0.0, 0.0, 0.0]);
    }
}
