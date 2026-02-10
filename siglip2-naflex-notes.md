# SigLIP2 NaFlex Integration Notes

Notes on switching from `siglip2-base-patch16-256` to `siglip2-base-patch16-naflex`. NaFlex is designed to work with various aspect ratios instead of forcing all images to a fixed 256x256 square.

---

## Model Files on Disk

### Current model (working)
```
C:\Dev\test\siglip2-base-patch16-256-ONNX\onnx\
  vision_model.onnx        354.8 MB
  text_model.onnx          1077.1 MB
  (+ quantized variants)
```

### NaFlex ONNX export (incomplete)
```
C:\Dev\test\siglip2-base-patch16-naflex-ONNX\onnx\
  vision_model.onnx        142 KB (stub — weights in external data file)
  vision_model.onnx_data   354.5 MB
  (+ fp16, q4, q4f16 variants with their own _data files)
```

**`text_model.onnx` is missing.** The existing export was done by the transformers.js conversion script, which does not yet support siglip2/naflex (see https://github.com/huggingface/transformers.js/issues/1402). Optimum CLI also does not support the `siglip2` model type yet.

### NaFlex original safetensors (complete)
```
C:\Dev\test\siglip2-base-patch16-naflex\
  model.safetensors        1.50 GB (contains both text and vision weights)
  config.json
  tokenizer.json
  preprocessor_config.json
```

---

## Getting the Missing Text Model

The 256 and NaFlex models have **separately trained weights** — the text_model.onnx from the 256 variant cannot be reused. The text encoder must be exported from the NaFlex safetensors.

Run this script from `C:\Dev\test`:

```python
import torch
from transformers import AutoModel

model = AutoModel.from_pretrained("siglip2-base-patch16-naflex")
text_model = model.text_model
text_model.eval()

dummy_input_ids = torch.zeros(1, 64, dtype=torch.long)

torch.onnx.export(
    text_model,
    (dummy_input_ids,),
    "siglip2-base-patch16-naflex-ONNX/onnx/text_model.onnx",
    input_names=["input_ids"],
    output_names=["last_hidden_state", "pooler_output"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
        "pooler_output": {0: "batch_size"},
    },
    opset_version=17,
)
print("Done")
```

After export, verify with:
```python
import onnx
model = onnx.load("siglip2-base-patch16-naflex-ONNX/onnx/text_model.onnx")
for inp in model.graph.input:
    shape = [d.dim_value or d.dim_param for d in inp.type.tensor_type.shape.dim]
    print(f"  {inp.name}: {shape}")
for out in model.graph.output:
    shape = [d.dim_value or d.dim_param for d in out.type.tensor_type.shape.dim]
    print(f"  {out.name}: {shape}")
```

Expected: `input_ids: [batch_size, sequence_length]` in, `pooler_output: [batch_size, 768]` out.

---

## Vision Model Input Differences

### Current model (256)
```
pixel_values: [batch_size, 3, 256, 256]   (FLOAT)  — raw NCHW pixels
```
One input. Images resized to fixed 256x256 square.

### NaFlex model
```
pixel_values:          [batch_size, max_num_patches, 768]   (FLOAT)  — flattened 16x16x3 patches
pixel_attention_mask:  [batch_size, max_num_patches]         (INT64)  — 1=real patch, 0=padding
spatial_shapes:        [batch_size, 2]                       (INT64)  — (height_patches, width_patches)
```
Three inputs. The 768 dimension is `16 * 16 * 3 = 768` raw pixel values per patch (not a learned embedding). The patch embedding linear layer (768→768) is inside the ONNX model.

### Text model (both variants)
```
input_ids: [batch_size, sequence_length]   (INT64)
```
Same interface. No changes needed for text inference.

### Output (both variants)
```
pooler_output: [batch_size, 768]   (FLOAT)
```
Same 768-dimensional embeddings. Same L2 normalization. Compatible with the existing LanceDB schema.

---

## NaFlex Preprocessing Pipeline

Replaces the current resize-to-256x256 → normalize → NCHW pipeline.

### Step 1: Compute target size
Find the largest scale factor where:
- Both dimensions are multiples of `patch_size` (16)
- Total patches = `(height / 16) * (width / 16)` ≤ `max_num_patches` (256)
- Aspect ratio is preserved

This is done via binary search on the scale factor, rounding each dimension up to the nearest multiple of 16.

### Step 2: Resize
Resize the image to the computed target dimensions (bilinear). The result will NOT be square — it preserves the original aspect ratio.

### Step 3: Normalize
Same formula as the 256 model: `(pixel / 255.0 - 0.5) / 0.5` → range [-1, 1].

### Step 4: Patchify
Split the image into 16x16 pixel patches. Each patch becomes a flat vector of 768 floats (16 * 16 * 3, in HWC order within each patch). The total number of real patches = `(H / 16) * (W / 16)`.

### Step 5: Pad and build masks
- Pad the patch array to exactly `max_num_patches` (256) with zero vectors
- Build `pixel_attention_mask`: array of 256 INT64 values, 1 for each real patch, 0 for padding
- Build `spatial_shapes`: `[height / 16, width / 16]` as two INT64 values

### Reference implementation
The Python reference is in `transformers/models/siglip2/image_processing_siglip2.py`:
- `get_image_size_for_max_num_patches()` — step 1
- `convert_image_to_patches()` — step 4
- `pad_along_first_dim()` — step 5

---

## Required Code Changes in `embedding.rs`

### Constants
Replace:
```rust
pub const IMAGE_SIZE: u32 = 256;
```
With:
```rust
pub const PATCH_SIZE: u32 = 16;
pub const MAX_NUM_PATCHES: usize = 256;
```

### Image preprocessing
Replace `preprocess_image` / `preprocess_image_from_rgb` with the 5-step pipeline above. The output changes from a single `Vec<f32>` of length `3 * 256 * 256 = 196608` to three tensors:
- `pixel_values: Vec<f32>` of length `MAX_NUM_PATCHES * 768 = 196608` (same total size, different layout)
- `pixel_attention_mask: Vec<i64>` of length `MAX_NUM_PATCHES`
- `spatial_shapes: Vec<i64>` of length `2`

### Vision inference (`run_vision_inference`)
Change from:
```rust
let shape = [1, 3, IMAGE_SIZE as i64, IMAGE_SIZE as i64];
let input_tensor = Value::from_array((shape, pixel_values));
self.vision_session.run(ort::inputs!["pixel_values" => input_tensor])
```
To:
```rust
let pv_shape = [1, MAX_NUM_PATCHES as i64, 768];
let pv_tensor = Value::from_array((pv_shape, pixel_values));
let mask_shape = [1, MAX_NUM_PATCHES as i64];
let mask_tensor = Value::from_array((mask_shape, attention_mask));
let ss_shape = [1, 2];
let ss_tensor = Value::from_array((ss_shape, spatial_shapes));
self.vision_session.run(ort::inputs![
    "pixel_values" => pv_tensor,
    "pixel_attention_mask" => mask_tensor,
    "spatial_shapes" => ss_tensor
])
```

### GPU batched inference (`infer_batch` in `GpuEmbeddingModel`)
Same three-input change, with batch dimension:
- `pixel_values: [batch, MAX_NUM_PATCHES, 768]`
- `pixel_attention_mask: [batch, MAX_NUM_PATCHES]`
- `spatial_shapes: [batch, 2]`

### Text inference
**No changes needed.** Same `input_ids` input, same `pooler_output` output, same 768 dimensions.

### JPEG scaled decoding
The `choose_jpeg_scale` function currently targets `IMAGE_SIZE` (256) as the minimum dimension. For NaFlex, the target size varies per image (it depends on aspect ratio). The function should target the computed NaFlex dimensions instead, or just decode at full resolution and let the resize handle it.

---

## NaFlex `preprocessor_config.json`

```json
{
  "image_processor_type": "Siglip2ImageProcessorFast",
  "processor_class": "Siglip2Processor",
  "do_resize": true,
  "do_rescale": true,
  "do_normalize": true,
  "rescale_factor": 0.00392156862745098,
  "image_mean": [0.5, 0.5, 0.5],
  "image_std": [0.5, 0.5, 0.5],
  "resample": 2,
  "patch_size": 16,
  "max_num_patches": 256,
  "default_to_square": true
}
```

Key parameters: `patch_size=16`, `max_num_patches=256`. The `size` field is null (no fixed size, unlike the 256 model which specifies `{"height": 256, "width": 256}`).

---

## NaFlex ONNX Graph Internals

The vision model ONNX graph (380 nodes) includes:
- `model.embeddings.patch_embedding.MatMul.weight: [768, 768]` — projects flattened raw patches to embeddings
- `model.embeddings.patch_embedding.Add.bias: [768]`
- `/model/embeddings/pos_embed/base_weight: [16, 16, 768]` — 2D positional embedding base
- Attention mask processing (first ops in graph: Cast → Sub → Mul to convert mask to attention bias)

The patch embedding is a simple linear layer (MatMul + bias) that maps 768 raw pixel values to 768 embedding dimensions. Positional embeddings are interpolated from a 16x16 base grid to match the actual spatial dimensions of each image.
