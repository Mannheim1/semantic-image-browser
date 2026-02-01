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

use crate::database::VISUAL_EMBEDDING_DIM;

/// Stores the result of ORT initialization (success or error message).
/// Using OnceLock ensures the result is computed once and remembered,
/// so subsequent calls return the same result (including errors).
static ORT_INIT_RESULT: OnceLock<Result<(), String>> = OnceLock::new();

/// Initialize the ONNX Runtime with the library at the given path.
/// This must be called before creating any sessions when using load-dynamic.
/// Safe to call multiple times - only the first call has effect.
/// Subsequent calls return the same result as the first call.
pub fn init_ort(dylib_path: &Path) -> Result<(), String> {
    ORT_INIT_RESULT
        .get_or_init(|| {
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

/// Image size expected by the SigLIP2 model (256x256 for siglip2-base-patch16-256)
const IMAGE_SIZE: u32 = 256;

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
        // Preprocess the image
        let pixel_values = preprocess_image(image_path)?;

        // Run inference
        let embedding = self.run_vision_inference(&pixel_values)?;

        // L2 normalize
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
/// 1. Load and decode the image
/// 2. Convert to RGB
/// 3. Resize to 256x256
/// 4. Convert to float and rescale to [0, 1]
/// 5. Normalize with mean=0.5, std=0.5 (resulting in [-1, 1])
/// 6. Return as flat NCHW format [1, 3, 256, 256] = 196608 floats
fn preprocess_image(path: &Path) -> Result<Vec<f32>, String> {
    // Load image
    let img = image::open(path)
        .map_err(|e| format!("Failed to open image {}: {}", path.display(), e))?;

    // Convert to RGB
    let img = img.to_rgb8();

    // Resize to IMAGE_SIZE x IMAGE_SIZE using bilinear interpolation
    // (resample mode 2 in preprocessor_config.json = BILINEAR)
    let img = image::imageops::resize(
        &img,
        IMAGE_SIZE,
        IMAGE_SIZE,
        image::imageops::FilterType::Triangle, // Bilinear
    );

    // Convert to NCHW float tensor with normalization
    // Layout: [batch=1][channel][height][width]
    let mut pixel_values = vec![0.0f32; (1 * 3 * IMAGE_SIZE * IMAGE_SIZE) as usize];

    for c in 0..3 {
        for y in 0..IMAGE_SIZE as usize {
            for x in 0..IMAGE_SIZE as usize {
                let pixel = img.get_pixel(x as u32, y as u32);
                // Rescale [0, 255] -> [0, 1] then normalize: (x - mean) / std
                let value = pixel[c] as f32 / 255.0;
                let normalized = (value - IMAGE_MEAN[c]) / IMAGE_STD[c];
                // Index: batch * (C*H*W) + channel * (H*W) + y * W + x
                let idx = c * (IMAGE_SIZE * IMAGE_SIZE) as usize + y * IMAGE_SIZE as usize + x;
                pixel_values[idx] = normalized;
            }
        }
    }

    Ok(pixel_values)
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
