# semantic-image-browser

<img src="src-tauri/icons/icon.png" alt="semantic-image-browser icon" width="64" />

This is a desktop app for Windows and Mac that allows you to search your images with natural language queries. It uses SigLIP 2, a vision-language encoding model by Google DeepMind. No setup or internet connection is required.

## Installation

Download and run the installer for your platform from the [releases page](https://github.com/Mannheim1/semantic-image-browser/releases). The CUDA variant requires an NVIDIA GPU with CUDA 12 support.

This app stores its data in `C:\Users\<your-username>\AppData\Local\com.mannheim.semantic-image-browser` on Windows and `~/Library/Application Support/com.mannheim.semantic-image-browser` on Mac. To completely remove this app from your computer, delete these folders after uninstalling semantic-image-browser.

## Usage

1. Wait for CUDA dependencies to download an install if running CUDA version
2. Add a folder by clicking **File → Add Folder**. This adds all images in the folder (excluding subfolders) to semantic-image-browser's database.
3. Wait for indexing to complete. Large folders may take a while to add.
4. Type any query in the search bar and the app will show you the images in its database that are closest to the query.
5. View which folders are in the database by clicking **File → Manage Folders**.
6. Remove a folder by clicking the X next to the name of the folder you wish to remove.

Additional usage information can be found by clicking **Help → View Controls**.

## License

[GPL-3.0-only](LICENSE)

