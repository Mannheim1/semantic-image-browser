# Plan: Native CoreML Inference for macOS

## Problem

The current macOS backend routes SigLIP2 inference through ONNX Runtime's CoreML Execution
Provider (EP). Profiling reveals a fundamental fragmentation issue: ORT's CoreML EP rejects every
attention-layer matrix multiplication (`Q @ K^T` and `scores @ V`) with:

```
gemm_op_builder.cc: MatMul B input must be a constant initializer
```

Because both inputs to these MatMuls are dynamic runtime tensors (not static weights), ORT's
CoreML EP cannot map them to CoreML's `BatchedMatMul` layer and falls back to ORT's CPU EP for
those nodes. With 12+ transformer layers each containing 2 such MatMuls, this produces **26–28
CoreML subgraph partitions per model** separated by CPU-executed nodes. Each partition boundary
requires a CPU↔ANE data transfer, negating much of the benefit of the Neural Engine.

Switching ORT's `ModelFormat` to `MLProgram` was tested and fails immediately:

```
Unable to parse ML Program: in operation /vision_model/embeddings/patch_embedding/Conv:
Required param 'pad' is missing
```

This is a bug in ORT's MLProgram op builder for Conv — not fixable without patching ORT itself.

## Rationale for Native CoreML

`coremltools` converts ONNX models to CoreML's MLProgram format correctly. Its converter handles
dynamic-input MatMul by mapping to `mb.matmul`, which ANE supports natively. The result is a
single (or very small number of) CoreML subgraph(s) covering the full model, eliminating the
CPU↔ANE context switching entirely.

The tradeoff is that ORT is no longer used on the macOS path — inference must call CoreML
framework directly. The `objc2-core-ml` crate provides Rust bindings generated from Apple's
CoreML headers, making this feasible without a Swift bridge. CoreML itself is a macOS system
framework, so no dylib needs to be shipped with the app.

Windows (CPU) and Windows (CUDA) builds are completely unaffected — they continue using ORT as
before.

## Plan

### Step 1: Conversion script

Write `scripts/convert_to_coreml.py`. This script:

- Takes `vision_model.onnx` and `text_model.onnx` as input
- Converts each to an MLProgram `.mlpackage` using `coremltools.convert()` with:
  - `convert_to="mlprogram"`
  - `minimum_deployment_target=ct.target.macOS13`
  - `compute_precision=ct.precision.FLOAT16`
  - Fixed input shapes matching what the app uses: `[1, 3, 256, 256]` for vision, `[1, 64]` for
    text
- Zips each `.mlpackage` directory into `vision_model.mlpackage.zip` and
  `text_model.mlpackage.zip`

Float16 is used because it is what ANE operates on natively. SigLIP2 was trained in float32 but
the embedding similarity comparisons are cosine-based and robust to float16 quantization.

### Step 2: Local testing setup

Run the conversion script once locally against the models in `src-tauri/bundled/model/onnx/`:

```bash
pip install coremltools
python scripts/convert_to_coreml.py
```

Output zips go to `src-tauri/bundled/model/`. The existing `onnx/` subdirectory stays in place
for non-CoreML builds. `npm run tauri dev -- --features backend-coreml` then picks up the zips.

### Step 3: GitHub workflow changes (macOS job only)

In `.github/workflows/release.yml`, for the `macOS ARM64 (CoreML)` matrix entry:

1. **Remove** the ORT download step (or gate it to non-macOS). CoreML is a system framework; no
   dylib is shipped.
2. **Add** a `pip install coremltools` step after the macOS system dependencies step.
3. **Keep** the ONNX model download — the conversion script consumes these files.
4. **Add** a conversion step that runs `scripts/convert_to_coreml.py`.
5. **Update** the `TAURI_CONFIG` for the macOS matrix entry to bundle
   `bundled/model/*.mlpackage.zip` and `bundled/model/tokenizer.json` instead of
   `bundled/model/onnx/*` and `bundled/lib/*`.

The `TAURI_CONFIG` env var is already set per-build in the workflow's `Build Tauri app` step. The
macOS matrix entry gets its own value; Windows entries are unchanged.

### Step 4: Rust backend-coreml rewrite

Replace the ORT-based code behind `#[cfg(feature = "backend-coreml")]` in `embedding.rs`:

1. **Remove** ORT imports and session setup for the CoreML path. `init_ort` becomes a no-op
   (or is skipped) on macOS.
2. **Add** `objc2-core-ml` (and `objc2`, `objc2-foundation`) to `Cargo.toml` under
   `[target.'cfg(target_os = "macos")'.dependencies]`.
3. **On first use**, extract `vision_model.mlpackage.zip` and `text_model.mlpackage.zip` from
   bundled resources into the app's data directory (same directory currently used for the CoreML
   cache). Skip extraction if already present.
4. **Load models** via `MLModel::modelWithContentsOfURL:configuration:error:`.
5. **Run inference** by wrapping the existing preprocessed `f32` pixel tensor as an `MLMultiArray`
   and calling `MLModel::predictionFromFeatures:error:`. Extract `pooler_output` from the result.
6. **Preserve the existing public API** (`embed_preprocessed`, `embed_text`, `infer_batch`) so
   `scan.rs` and `state.rs` require no changes. The pipeline mode (2 model instances,
   producer-consumer) stays the same.

### Step 5: Verify

Run the verbose ORT log test again after conversion to confirm partition count. Since ORT is no
longer used on the macOS path, the verification instead checks:

- No extraction errors on first launch
- Embeddings are numerically similar to the ORT/CPU baseline (cosine similarity > 0.99)
- Search results are qualitatively correct
