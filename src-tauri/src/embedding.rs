//! Embedding generation using SigLIP2 ONNX models.
//!
//! This module handles:
//! - Loading vision and text ONNX models
//! - Image preprocessing (resize, normalize)
//! - Text tokenization
//! - Generating embeddings for images and text queries

#[cfg(feature = "backend-cuda")]
use ort::ep::CUDAExecutionProvider;
#[cfg(feature = "backend-coreml")]
use ort::ep::CoreMLExecutionProvider;
use ort::session::Session;
#[cfg(feature = "backend-coreml")]
use ort::session::builder::SessionBuilder;
use ort::value::TensorRef;
use std::path::Path;
use std::sync::OnceLock;
use tokenizers::Tokenizer;

use crate::benchmark::{self, PreprocessTiming};
use crate::database::VISUAL_EMBEDDING_DIM;
use crate::image_ops::fast_resize_rgb;

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

/// Return the execution providers for the active backend feature.
///
/// - `backend-cuda`: CUDA provider (errors on failure to load)
/// - `backend-coreml`: CoreML provider (errors on failure to load)
/// - `backend-cpu` (default): empty list → ORT uses its built-in CPU provider
fn execution_providers(
    #[cfg(feature = "backend-coreml")] cache_dir: &Path,
) -> Vec<ort::execution_providers::ExecutionProviderDispatch> {
    #[cfg(feature = "backend-cuda")]
    {
        vec![CUDAExecutionProvider::default().build().error_on_failure()]
    }
    #[cfg(feature = "backend-coreml")]
    {
        use ort::ep::coreml::{ComputeUnits, ModelFormat, SpecializationStrategy};
        vec![
            CoreMLExecutionProvider::default()
                .with_model_format(ModelFormat::NeuralNetwork)
                .with_static_input_shapes(true)
                .with_compute_units(ComputeUnits::CPUAndNeuralEngine)
                .with_specialization_strategy(SpecializationStrategy::FastPrediction)
                .with_profile_compute_plan(true)
                .with_model_cache_dir(cache_dir.to_string_lossy())
                .build()
                .error_on_failure(),
        ]
    }
    #[cfg(all(not(feature = "backend-cuda"), not(feature = "backend-coreml")))]
    {
        vec![]
    }
}

/// Build an ONNX session for the given model file, using the active backend's execution providers.
enum ModelKind {
    Vision,
    Text,
}

#[cfg(feature = "backend-coreml")]
fn apply_coreml_dimension_overrides(
    mut builder: SessionBuilder,
    label: &str,
    kind: ModelKind,
) -> Result<SessionBuilder, String> {
    let overrides: &[(&str, i64)] = match kind {
        // Vision preprocessing always emits [1, 3, 256, 256].
        ModelKind::Vision => &[
            ("batch_size", 1),
            ("num_channels", 3),
            ("height", IMAGE_SIZE as i64),
            ("width", IMAGE_SIZE as i64),
        ],
        // Text path always pads/truncates to MAX_SEQ_LENGTH.
        ModelKind::Text => &[("batch_size", 1), ("sequence_length", MAX_SEQ_LENGTH as i64)],
    };

    for (name, size) in overrides {
        builder = builder
            .with_dimension_override(*name, *size)
            .map_err(|e| format!("Failed to set {}={} override for {}: {}", name, size, label, e))?;
    }

    Ok(builder)
}

/// Build an ONNX session for the given model file, using the active backend's execution providers.
fn build_session(
    model_path: &Path,
    label: &str,
    kind: ModelKind,
    #[cfg(feature = "backend-coreml")] cache_dir: &Path,
) -> Result<Session, String> {
    let providers = execution_providers(
        #[cfg(feature = "backend-coreml")]
        cache_dir,
    );
    let builder = Session::builder()
        .map_err(|e| format!("Failed to create {} session builder: {}", label, e))?;

    let builder = if providers.is_empty() {
        builder
    } else {
        builder
            .with_execution_providers(providers)
            .map_err(|e| format!("Failed to register execution providers for {}: {}", label, e))?
    };

    #[cfg(feature = "backend-coreml")]
    let builder = apply_coreml_dimension_overrides(builder, label, kind)?;

    builder
        .commit_from_file(model_path)
        .map_err(|e| format!("Failed to load {}: {}", label, e))
}

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
    /// The execution provider is determined at compile time by the active backend feature.
    pub fn load(
        model_dir: &Path,
        #[cfg(feature = "backend-coreml")] cache_dir: &Path,
    ) -> Result<Self, String> {
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

        let vision_session = build_session(
            &vision_path,
            "vision model",
            ModelKind::Vision,
            #[cfg(feature = "backend-coreml")]
            cache_dir,
        )?;
        let text_session = build_session(
            &text_path,
            "text model",
            ModelKind::Text,
            #[cfg(feature = "backend-coreml")]
            cache_dir,
        )?;

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| format!("Failed to load tokenizer: {}", e))?;

        Ok(Self {
            vision_session,
            text_session,
            tokenizer,
        })
    }

    /// Generate an embedding from pre-computed pixel values (NCHW float tensor).
    /// Used in the scan pipeline where decoding/preprocessing is done separately.
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
        let input_ids = self.tokenize(query)?;
        let embedding = self.run_text_inference(&input_ids)?;
        Ok(l2_normalize(&embedding))
    }

    /// Run batched vision inference on a preprocessed batch, returning per-image embeddings.
    ///
    /// Takes a `PreprocessedBatch` and runs the vision model once for the whole batch.
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

        let input_tensor = match TensorRef::from_array_view((shape, &*batch.pixel_data)) {
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

        // Compute per-image share of inference time for logging
        let per_image_inference = inference_time / batch.valid_indices.len() as u32;

        for (batch_idx, &original_idx) in batch.valid_indices.iter().enumerate() {
            let start = batch_idx * embedding_dim;
            let end = start + embedding_dim;
            if end <= data.len() {
                results[original_idx] = Ok(l2_normalize(&data[start..end]));
            } else {
                results[original_idx] = Err(format!(
                    "Embedding index out of bounds: expected {}..{}, got len {}",
                    start, end, data.len()
                ));
            }

            // Log per-image timing
            if let Some(timing) = &batch.timings[original_idx] {
                benchmark::log_image(timing, per_image_inference, "batch");
            }
        }

        results
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
        let input_tensor = TensorRef::from_array_view((shape, pixel_values))
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
        let input_tensor = TensorRef::from_array_view((shape, input_ids))
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

/// Preprocess pre-decoded RGB image data into an embedding tensor.
/// The decode_time in the returned timing will be zero (caller should set it).
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
pub(crate) fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter().map(|x| x / norm).collect()
    } else {
        v.to_vec()
    }
}

/// Result of parallel preprocessing for a batch of images.
/// Contains the concatenated pixel data ready for batched inference,
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
