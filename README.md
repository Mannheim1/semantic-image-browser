# semantic-image-browser

Search your images with natural language — fully local, no setup required.

![Screenshot](screenshot.png)

---

## Features

- **Natural language search** — type "dog on a beach" or "blueprint diagram" and find matching images by meaning, not filename
- **Image similarity search** — right-click any image to find visually similar ones
- **Fully local and private** — no account, no API key, no internet connection required; nothing leaves your machine
- **Zero setup** — SigLIP2 model and ONNX Runtime are bundled in the installer

---

## Installation

Download the installer for your platform from the [releases page](https://github.com/Mannheim1/semantic-image-browser/releases).

| Platform | Variant | Notes |
|---|---|---|
| Windows x64 | CPU | |
| Windows x64 | CUDA | Requires NVIDIA GPU with CUDA 12 support |
| macOS ARM64 | CPU | |
| Linux x64 | CPU | |
| Linux x64 | CUDA | Requires NVIDIA GPU with CUDA 12 support |

Use the CPU build if you don't have an NVIDIA GPU. The CUDA build is otherwise identical but indexes images much faster.

**To uninstall:** use your system's standard uninstall mechanism. App data (index, thumbnails, config) is stored in your local app data folder and can be deleted manually if desired.

---

## Usage

1. Add a folder via **File → Add Folder**
2. Wait for indexing to complete
3. Type a query and press Enter

Double-click an image to open it. Right-click for additional options.

---

## License

[GPL-3.0-only](LICENSE)
