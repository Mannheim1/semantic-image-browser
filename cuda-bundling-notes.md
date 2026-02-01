# CUDA Library Bundling for GPU Runtime

## Problem

The GPU runtime requires CUDA libraries (cuBLAS, cuDNN) to be installed system-wide. Users must:
1. Install CUDA Toolkit 12.x
2. Install cuDNN 9.x
3. Manually add cuDNN to PATH

This is a poor user experience. Many apps bundle these DLLs directly.

## Solution

Download and bundle the required CUDA DLLs alongside the ONNX Runtime GPU package.

### Required DLLs (Windows x64, CUDA 12, cuDNN 9)

From ONNX Runtime GPU 1.23.2 error messages, the required libraries are:
- `cublasLt64_12.dll` (from CUDA Toolkit / cuBLAS)
- `cublas64_12.dll` (from CUDA Toolkit / cuBLAS)
- `cudnn64_9.dll` (from cuDNN)
- `cudnn_ops64_9.dll` (from cuDNN)
- `cudnn_cnn64_9.dll` (from cuDNN)

### Where to Get Them

**Option 1: NVIDIA Redistributable Packages (Recommended)**

NVIDIA provides redistributable packages specifically for bundling:
- cuBLAS: https://developer.nvidia.com/cublas
- cuDNN: https://developer.nvidia.com/cudnn

These are smaller than the full toolkit and explicitly licensed for redistribution.

**Option 2: Extract from Full Installers**

Extract the specific DLLs from:
- CUDA Toolkit 12.x installer
- cuDNN 9.x package

### Implementation Approach

1. **Host the DLLs**: Upload the required DLLs to a GitHub Release (same as ONNX Runtime)

2. **Modify `ort_download.rs`**: When downloading GPU runtime, also download a CUDA dependencies package

3. **Storage layout**:
   ```
   {app_data}/onnxruntime/
   ├── lib/
   │   ├── onnxruntime.dll
   │   ├── onnxruntime_providers_cuda.dll
   │   ├── onnxruntime_providers_shared.dll
   │   ├── cublasLt64_12.dll      # bundled
   │   ├── cublas64_12.dll        # bundled
   │   ├── cudnn64_9.dll          # bundled
   │   ├── cudnn_ops64_9.dll      # bundled
   │   └── cudnn_cnn64_9.dll      # bundled
   ```

4. **DLL Loading**: ONNX Runtime should find the DLLs if they're in the same directory as `onnxruntime_providers_cuda.dll`. If not, we may need to call `SetDllDirectory` or modify PATH at runtime before loading.

### Licensing Considerations

- cuBLAS and cuDNN are redistributable under NVIDIA's license
- Must include NVIDIA's license/copyright notices
- Check exact terms at: https://docs.nvidia.com/cuda/eula/

### Package Size Impact

Approximate sizes:
- cuBLAS DLLs: ~150 MB
- cuDNN DLLs: ~700 MB

Total GPU runtime with CUDA deps: ~1.2 GB

This is large but acceptable for a one-time download. Consider:
- Showing download size clearly in UI
- Using compression if hosting ourselves
- Progress indicator during download (already implemented)

### Alternative: Lazy Loading

Instead of bundling, we could:
1. Attempt to load GPU runtime
2. If it fails with missing DLL error, show user-friendly message with download link
3. Provide a button to download CUDA dependencies separately

This keeps initial download smaller but requires two steps for GPU users.

### Recommendation

Bundle the CUDA DLLs with the GPU runtime download. Create a single `onnxruntime-win-x64-gpu-cuda12-cudnn9.zip` package containing everything needed. Host on GitHub Releases.

This follows the KISS principle - one download, everything works.
