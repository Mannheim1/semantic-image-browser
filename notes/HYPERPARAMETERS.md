# Scan/Inference Hyperparameters and Tunables

This file lists scan/inference tunables and where they are defined.

## Backend/variant selection

- `backend-cpu`, `backend-cuda`, `backend-coreml` feature flags (compile-time variant switch)  
  Location: `src-tauri/Cargo.toml:19`
- default variant is `backend-cpu`  
  Location: `src-tauri/Cargo.toml:19`

## Core inference orchestration

- `model_instances` per backend:
  - CUDA: `1`
  - CoreML: `2`
  - CPU: `available_parallelism().min(4)` fallback `2`  
  Location: `src-tauri/src/state.rs:28`
- `batch_size` per backend:
  - CUDA: `32`
  - CoreML: `1`
  - CPU: `1`  
  Location: `src-tauri/src/state.rs:33`
- `pipeline` mode per backend:
  - CUDA/CoreML: `true`
  - CPU: `false`  
  Location: `src-tauri/src/state.rs:34`
- CPU max worker cap `MAX_WORKERS = 4`  
  Location: `src-tauri/src/state.rs:48`

## CoreML execution provider tunables

- `ModelFormat` (currently `NeuralNetwork`)  
  Location: `src-tauri/src/embedding.rs:113`
- `RequireStaticInputShapes` (currently `true`)  
  Location: `src-tauri/src/embedding.rs:114`
- `MLComputeUnits` (currently `CPUAndNeuralEngine`)  
  Location: `src-tauri/src/embedding.rs:115`
- `SpecializationStrategy` (currently `FastPrediction`)  
  Location: `src-tauri/src/embedding.rs:116`
- `ProfileComputePlan` (currently `true`)  
  Location: `src-tauri/src/embedding.rs:117`
- CoreML static dimension overrides:
  - Vision: `batch_size=1`, `num_channels=3`, `height=256`, `width=256`
  - Text: `batch_size=1`, `sequence_length=64`  
  Location: `src-tauri/src/embedding.rs:135`

## Shared model/preprocessing hyperparameters

- embedding image size `IMAGE_SIZE = 256`  
  Location: `src-tauri/src/embedding.rs:87`
- text max tokens `MAX_SEQ_LENGTH = 64`  
  Location: `src-tauri/src/embedding.rs:90`
- image normalization mean `[0.5, 0.5, 0.5]`  
  Location: `src-tauri/src/embedding.rs:93`
- image normalization std `[0.5, 0.5, 0.5]`  
  Location: `src-tauri/src/embedding.rs:96`

## Scan-discovery tunables

- allowed image extensions list: `jpg`, `jpeg`, `jfif`, `png`, `gif`, `webp`, `bmp`, `tiff`, `tif`, `avif`  
  Location: `src-tauri/src/scan.rs:28`
- directory traversal depth `max_depth(1)`  
  Location: `src-tauri/src/scan.rs:56`
- symlink behavior `follow_links(false)`  
  Location: `src-tauri/src/scan.rs:57`

## Pipeline queue/concurrency tunables

- CUDA batched pipeline channel capacity `batch_size * 2`  
  Location: `src-tauri/src/scan.rs:389`
- multi-consumer pipeline channel capacity per consumer `4`  
  Location: `src-tauri/src/scan.rs:520`
- thumbnail-only mode thread count `available_parallelism()` fallback `4`  
  Location: `src-tauri/src/scan.rs:627`

## Thumbnail-generation tunables

- thumbnail max dimension `THUMBNAIL_SIZE = 256`  
  Location: `src-tauri/src/thumbnail.rs:8`
- thumbnail WebP quality `WEBP_QUALITY = 80.0`  
  Location: `src-tauri/src/thumbnail.rs:9`

## Decode/resize strategy tunables

- JPEG scaled decode target size `256` (for choosing 1/1, 1/2, 1/4, 1/8 decode)  
  Location: `src-tauri/src/image_ops.rs:50`
- resize filter switch threshold `ratio > 4`:
  - `Box` for large downscale
  - `Bilinear` otherwise  
  Location: `src-tauri/src/image_ops.rs:213`

## Operational toggles affecting scan behavior

- benchmark logging runtime toggle (`toggle_benchmarking`)  
  Location: `src-tauri/src/lib.rs:581`
- benchmark backend label by variant (`cuda`/`coreml`/`cpu`)  
  Location: `src-tauri/src/benchmark.rs:51`
