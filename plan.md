# Implementation Plan: Native CoreML Inference

## Overview

Replace ORT-based CoreML inference with direct CoreML calls via `objc2-core-ml`. Six steps, in implementation order. Windows/CUDA paths are completely untouched.

---

## Step 1: Conversion & Validation Script

**File:** `scripts/convert_to_coreml.py` (new)

This single script converts, validates, and compiles both models.

**What it does:**

1. Load `vision_model.onnx` and `text_model.onnx` from `src-tauri/bundled/model/onnx/`
2. Convert each to MLProgram via `coremltools.convert()` with:
   - `convert_to="mlprogram"`
   - `minimum_deployment_target=ct.target.macOS13`
   - `compute_precision=ct.precision.FLOAT16`
   - Vision input shape: `(1, 3, 256, 256)` as `ct.TensorType`
   - Text input shape: `(1, 64)` as `ct.TensorType` with `dtype=ct.converters.mil.mil.types.int32`
3. Inspect output feature names. If not `pooler_output`, rename with `ct.utils.rename_feature()`.
4. **Validate:** Load each converted CoreML model and the original ONNX model (via `onnxruntime`). Feed identical random input to both. Assert cosine similarity of outputs > 0.99. Exit non-zero on failure.
5. Save `.mlpackage` to temp location.
6. **Compile:** Run `xcrun coremlcompiler compile <model>.mlpackage <output_dir>/` to produce `.mlmodelc` directories. This eliminates the 30-60s first-launch compilation delay for users.
7. Zip each `.mlmodelc` directory into `src-tauri/bundled/model/vision_model.mlmodelc.zip` and `text_model.mlmodelc.zip`.

**Dependencies:** `pip install coremltools onnxruntime numpy`

---

## Step 2: Cargo.toml Changes

**File:** `src-tauri/Cargo.toml`

1. Make `ort` optional — CoreML builds don't need it:
   ```toml
   ort = { version = "2.0.0-rc.11", default-features = false, features = ["load-dynamic"], optional = true }
   ```

2. Update feature flags to gate `ort`:
   ```toml
   backend-cpu = ["dep:ort"]
   backend-cuda = ["dep:ort"]
   backend-coreml = []
   ```

3. Add macOS-specific CoreML dependencies:
   ```toml
   [target.'cfg(target_os = "macos")'.dependencies]
   objc2 = "0.6"
   objc2-foundation = { version = "0.3", features = ["NSArray", "NSDictionary", "NSString", "NSURL", "NSError", "NSNumber"] }
   objc2-core-ml = { version = "0.3", features = ["MLModel", "MLMultiArray", "MLMultiArrayDataType", "MLFeatureValue", "MLDictionaryFeatureProvider", "MLModelConfiguration", "MLComputeUnits", "MLFeatureProvider"] }
   ```

   Note: The exact feature flags for `objc2-foundation` and `objc2-core-ml` will need to be verified against the crate docs during implementation. The types listed above are what we need; the feature names that enable them may differ slightly.

---

## Step 3: embedding.rs Rewrite

**File:** `src-tauri/src/embedding.rs`

This is the largest change. The strategy: use `#[cfg]` to swap between ORT and CoreML implementations while sharing all preprocessing, tokenization, and normalization code.

### Imports

Gate all ORT imports behind `#[cfg(not(feature = "backend-coreml"))]`:
```rust
#[cfg(feature = "backend-cuda")]
use ort::ep::CUDAExecutionProvider;
#[cfg(not(feature = "backend-coreml"))]
use ort::session::Session;
#[cfg(not(feature = "backend-coreml"))]
use ort::value::TensorRef;
```

Add CoreML imports behind `#[cfg(feature = "backend-coreml")]`:
```rust
#[cfg(feature = "backend-coreml")]
use objc2::rc::Retained;
#[cfg(feature = "backend-coreml")]
use objc2_core_ml::{MLModel, MLMultiArray, ...};
```

### `init_ort()` and `ORT_INIT_RESULT`

Gate entirely behind `#[cfg(not(feature = "backend-coreml"))]`. CoreML is a system framework — no initialization needed.

### `execution_providers()`, `apply_coreml_dimension_overrides()`, `build_session()`

Delete the `#[cfg(feature = "backend-coreml")]` branches from these functions. They're ORT concepts that don't apply. The remaining branches (`backend-cuda`, default CPU) stay as-is.

### `EmbeddingModel` struct

```rust
pub struct EmbeddingModel {
    #[cfg(not(feature = "backend-coreml"))]
    vision_session: Session,
    #[cfg(not(feature = "backend-coreml"))]
    text_session: Session,
    #[cfg(feature = "backend-coreml")]
    vision_model: Retained<MLModel>,
    #[cfg(feature = "backend-coreml")]
    text_model: Retained<MLModel>,
    tokenizer: Tokenizer,
}
```

### `EmbeddingModel::load()` — CoreML path

1. Locate `vision_model.mlmodelc.zip` and `text_model.mlmodelc.zip` in the model directory.
2. Extract each zip to the cache directory (passed as `cache_dir` parameter, same as today). Use a marker file (e.g., `.extracted`) to skip if already done. Extract to a temp subdirectory first, then rename atomically for crash safety.
3. Create `MLModelConfiguration`, set `computeUnits` to `MLComputeUnits::CPUAndNeuralEngine`.
4. Load each model with `MLModel::modelWithContentsOfURL:configuration:error:`.
5. Load tokenizer (same as current code).

### `embed_preprocessed()` and `embed_text()` — unchanged signatures

These just call `run_vision_inference` / `run_text_inference` + `l2_normalize`. No `#[cfg]` needed in these methods — the dispatch happens in the private methods they call.

### `run_vision_inference()` — CoreML path

1. Create `MLMultiArray` with shape `[1, 3, 256, 256]` and datatype `Float32`.
2. Copy `pixel_values: &[f32]` into the MLMultiArray via `getMutableBytesWithHandler:`.
3. Wrap in `MLFeatureValue::featureValueWithMultiArray:`.
4. Create `MLDictionaryFeatureProvider` with key `"pixel_values"` → the feature value.
5. Call `vision_model.predictionFromFeatures:error:`.
6. Extract `"pooler_output"` from the result via `featureValueForName:` → `multiArrayValue`.
7. Copy the output float data into a `Vec<f32>` and return.

### `run_text_inference()` — CoreML path

Same pattern as vision, but:
- Shape: `[1, 64]`
- Datatype: `Int32` (not Float32)
- Input key: `"input_ids"`
- Input data: convert `&[i64]` token IDs to `i32` before copying into MLMultiArray

### `infer_batch()` — CoreML path

CoreML config uses `batch_size: 1`, so this method is never called at runtime. But it must compile. The CoreML version loops over valid images and calls `embed_preprocessed` individually:

```rust
#[cfg(feature = "backend-coreml")]
pub fn infer_batch(&mut self, batch: &PreprocessedBatch) -> Vec<Result<Vec<f32>, String>> {
    // Initialize results with errors for failed images
    // For each valid index, slice pixel_data and call embed_preprocessed
    // Log per-image timing
}
```

### Everything else in embedding.rs — UNCHANGED

- `tokenize()` — pure Rust
- `preprocess_image_from_rgb()` — pure Rust
- `rgb_to_nchw_normalized()` — pure Rust
- `l2_normalize()` — pure Rust
- `PreprocessedBatch` struct — pure Rust
- Tests — pure Rust

---

## Step 4: lib.rs Changes

**File:** `src-tauri/src/lib.rs`

### `ort_lib_filename()`

Gate behind `#[cfg(not(feature = "backend-coreml"))]`.

### Phase 2 init (async task, lines 669-722)

Gate the ORT-specific parts:
```rust
// Only resolve and init ORT for non-CoreML backends
#[cfg(not(feature = "backend-coreml"))]
{
    let ort_path = bundled.join("lib").join(ort_lib_filename());
    println!("ORT library: {}", ort_path.display());
    if let Err(e) = embedding::init_ort(&ort_path) {
        eprintln!("Failed to initialize ONNX Runtime: {}", e);
        let _ = handle_for_task.emit("model_ready", ());
        return;
    }
}
```

The `model_path`, `coreml_cache_dir`, and `EmbeddingBackend::load()` calls stay as-is (they're already correctly parameterized).

### `get_dependency_paths()`

Add a `#[cfg(feature = "backend-coreml")]` branch that reports `.mlmodelc.zip` paths instead of ORT library and ONNX model paths.

---

## Step 5: CI Workflow Changes

**File:** `.github/workflows/release.yml`

### macOS matrix entry

Remove `ort-url`, `ort-lib`, `ort-inner-dir` keys (not needed — CoreML is a system framework).

### ORT download step

Add condition: `if: runner.os != 'macOS'` (or check for `matrix.ort-url`).

### New step: Install coremltools (after system deps, macOS only)

```yaml
- name: Install coremltools
  if: runner.os == 'macOS'
  run: pip install coremltools onnxruntime numpy
```

### New step: Convert models (after model download, macOS only)

```yaml
- name: Convert ONNX to CoreML
  if: runner.os == 'macOS'
  run: python scripts/convert_to_coreml.py
```

This runs the conversion, validation, compilation, and zipping. If validation fails (cosine similarity < 0.99), the script exits non-zero and the CI job fails.

### TAURI_CONFIG for macOS

Change the macOS entry's bundled resources from:
```json
{"bundle":{"resources":["bundled/lib/*","bundled/model/onnx/*","bundled/model/tokenizer.json"]}}
```
to:
```json
{"bundle":{"resources":["bundled/model/*.mlmodelc.zip","bundled/model/tokenizer.json"]}}
```

Windows entries are completely unchanged.

### Making TAURI_CONFIG per-platform

Currently `TAURI_CONFIG` is a single value for all builds. It needs to differ for macOS. Add a `tauri-config` key to each matrix entry, then reference `${{ matrix.tauri-config }}` in the build step. Windows entries keep the current value; macOS gets the new one.

---

## Step 6: Local Dev Setup

For development, the conversion must be run once locally:

```bash
pip install coremltools onnxruntime numpy
python scripts/convert_to_coreml.py
```

This produces `src-tauri/bundled/model/vision_model.mlmodelc.zip` and `text_model.mlmodelc.zip`. These are gitignored (like the ONNX models and ORT dylib).

Then build with:
```bash
npm run tauri dev -- --features backend-coreml
```

---

## Files Changed Summary

| File | Change |
|------|--------|
| `scripts/convert_to_coreml.py` | **New** — conversion, validation, compilation |
| `src-tauri/Cargo.toml` | Make `ort` optional, add objc2 deps, update features |
| `src-tauri/src/embedding.rs` | Rewrite model loading + inference for CoreML path |
| `src-tauri/src/lib.rs` | Gate ORT init, update dependency paths |
| `.github/workflows/release.yml` | Per-platform TAURI_CONFIG, add conversion step, skip ORT download |

## Files NOT Changed

| File | Why |
|------|-----|
| `state.rs` | Wraps `EmbeddingModel` via public API only |
| `scan.rs` | Calls `embed_preprocessed()` — same signature |
| `database.rs` | No relation to inference |
| `config.rs` | No relation to inference |
| All frontend code | No relation to inference |
