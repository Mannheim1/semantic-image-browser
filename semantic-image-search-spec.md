# Semantic Image Search Desktop App

## Features

### Core Search
- Single search bar for semantic queries
- Results displayed as thumbnail grid with filenames
- Double-click opens image in OS default viewer
- Right-click "Show in folder" opens file explorer with image selected
- Right-click "Find similar" searches using that image as input

### Search Modes
- Visual embedding search: matches query text against image content via CLIP
- OCR lexical search: keyword matching against extracted text (BM25)
- OCR semantic search: embedding similarity against extracted text

Setting to control OCR behavior: disabled, lexical only, semantic only, or both.

Result thumbnails have background color indicating which search mode(s) produced the match.

### Filtering & Sorting
- Filter by file type, date range, file size
- Sort by date created, date modified, file size, relevance
- Search scope setting to limit results to a subdirectory

### Storage
All data is local. Images, embeddings, metadata, and thumbnails stored on disk.

### Stretch Goals (not implemented initially)
- Remote image hosting: images on NAS with only thumbnails and database local
- macOS support

---

## Architecture

### Stack
- Desktop framework: Tauri v2
- Frontend: Svelte + TypeScript + Tailwind CSS
- Vector database: LanceDB (Rust client)
- ML inference: ONNX Runtime
- OCR: Tesseract via leptess
- Thumbnails: Windows Shell API

### Data Flow
A search query is encoded into a vector. LanceDB performs vector search on visual embeddings and optionally FTS/vector search on OCR data. Results return paths and match source indicators. Thumbnails load from local cache. Opening an image launches the OS default viewer.

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

### Build Output
Single Windows executable. No Python, no background services, no console windows.
