# Semantic Image Search Desktop App

## Development Phases

### Phase 1: Dependencies & Skeleton
Get Tauri + Svelte + LanceDB + ONNX Runtime compiling and running in a minimal app.

### Phase 2: Data Layer
File scanning, database schema, OS thumbnail integration.

### Phase 3: UI Shell
Search bar, thumbnail grid, context menus (with placeholder filename search).

### Phase 4: Visual Embedding Search
CLIP/SigLIP model integration for image-to-vector and text-to-vector.

### Phase 5: OCR Pipeline
Tesseract integration, text extraction, storage.

### Phase 6: OCR Search
Both lexical (BM25/FTS) and semantic (text embedding) search.

### Phase 7: Integration
Connect all three search modes to UI, result source indicators, settings.

---

## Features

### Core Search
- Single search bar for semantic queries
- Results displayed as thumbnail grid with filenames
- Double-click opens image in OS default viewer
- Right-click "Show in folder" opens file explorer with image selected
- Right-click "Find similar" searches using that image as input

### Search Modes
- Visual embedding search: matches query text against image content
- OCR lexical search: keyword matching against extracted text (BM25)
- OCR semantic search: embedding similarity against extracted text

Setting to control OCR behavior: disabled, lexical only, semantic only, or both.

Result thumbnails have background color indicating which search mode(s) produced the match.

### Filtering & Sorting
- Filter by file type, date range, file size
- Sort by date created, date modified, file size, relevance
- Search scope setting to limit results to a subdirectory

### Storage
All data is local. Images, embeddings, and metadata stored on disk. Thumbnails are retrieved on-demand from the OS (not stored by the app).

Data location: OS standard app data directory (`app_local_data_dir`), with an optional setting to specify a custom location.

### Stretch Goals (not implemented initially)
- Remote image hosting: images on NAS with only thumbnails and database local
- macOS support

---

## Architecture

### Stack
- Desktop framework: Tauri v2
- Frontend: Svelte + TypeScript
- Vector database: LanceDB (Rust client)
- ML inference: ONNX Runtime
- OCR: Tesseract via leptess
- Thumbnails: OS-native APIs (see Thumbnail System below)

### How Embedding Search Works
Vision-language models (like CLIP, SigLIP) encode both images and text into vectors in a shared space. Similar concepts end up near each other regardless of whether they came from an image or text.

- **Indexing**: Each image is encoded into a vector and stored
- **Search**: The query text is encoded into a vector and compared against stored image vectors
- **Constraint**: The same model must be used for both indexing and search. Embeddings from different models are incompatible (each model has its own vector space).

### Data Flow
1. **Indexing**: Images encoded → embeddings stored in LanceDB
2. **Search**: Query text encoded → vector similarity search in LanceDB
3. **Display**: Results return paths → thumbnails fetched on-demand from OS

### Indexing Flow
1. User selects one or more directories via native folder picker dialog
2. App recursively scans selected directories for images
3. New/modified images are processed (embeddings generated, metadata extracted)
4. Deleted images are removed from the database
5. Periodic re-scan keeps the database in sync with the filesystem

Searches query the database only—no filesystem access at search time (except thumbnail retrieval).

### Configuration Storage
Watched directories and app settings stored in a JSON file in the app data directory (separate from LanceDB).

### Database Schema (LanceDB/Arrow)
- path: Utf8 (file path)
- visual_embedding: FixedSizeList[Float32, N]
- ocr_text: Utf8, nullable (FTS indexed)
- ocr_embedding: FixedSizeList[Float32, M], nullable
- file_type: Utf8
- file_size: Int64
- created_at: Timestamp
- modified_at: Timestamp

Indexes: IVF-PQ on visual_embedding, IVF-PQ on ocr_embedding, FTS on ocr_text.

### Thumbnail System

Thumbnails are fetched on-demand from the operating system rather than generated and cached by the app. This approach:
- Eliminates thumbnail storage and cache invalidation complexity
- Leverages OS-optimized thumbnail extraction and caching
- Provides automatic freshness (OS handles invalidation when files change)
- Gracefully falls back to file icons for unsupported formats or corrupted files

#### Windows Implementation
Uses `IShellItemImageFactory::GetImage` from the Windows Shell API:
1. Create `IShellItem` from file path via `SHCreateItemFromParsingName`
2. Query for `IShellItemImageFactory` interface
3. Call `GetImage` with desired size and `SIIGBF_RESIZETOFIT` flag
4. Convert returned `HBITMAP` to PNG bytes
5. Return as base64 data URL to frontend

**Memory management:** The `HBITMAP` must be freed with `DeleteObject` after conversion. Windows has a per-process limit of ~10,000 GDI objects; failing to free handles will eventually crash the app.

**Threading:** `GetImage` can block during extraction. Must run on background thread (Tauri commands handle this automatically).

**Failure modes:**
- File not found → Error returned, frontend shows placeholder
- Corrupted file → Falls back to file type icon (handled by OS)
- Unsupported format → Falls back to file type icon (handled by OS)
- Permission denied → Error returned, frontend shows placeholder

#### macOS Implementation (Stretch Goal)
Uses Quick Look API (`QLThumbnailGenerator`):
1. Create `QLThumbnailGenerator.Request` with file URL and size
2. Call `generateBestRepresentation`
3. Convert `CGImage` to PNG bytes
4. Return as base64 data URL to frontend

Same interface as Windows: `get_thumbnail(path, size) -> Result<base64_png, Error>`

### Build Output
Single Windows executable. ONNX model bundled. No Python, no background services, no console windows.
