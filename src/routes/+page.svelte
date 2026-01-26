<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";
  import { open } from "@tauri-apps/plugin-dialog";

  interface ScanResult {
    images_found: number;
    images_added: number;
    images_updated: number;
    images_removed: number;
    errors: string[];
  }

  interface ImageInfo {
    path: string;
    file_type: string;
    file_size: number;
    created_at: number;
    modified_at: number;
  }

  let watchedDirectories = $state<string[]>([]);
  let indexedCount = $state(0);
  let scanResult = $state<ScanResult | null>(null);
  let isScanning = $state(false);
  let statusMsg = $state("");
  let onnxMsg = $state("");
  let images = $state<ImageInfo[]>([]);
  let thumbnails = $state<Record<string, string | null>>({});

  async function loadData() {
    try {
      watchedDirectories = await invoke("get_watched_directories");
      indexedCount = await invoke("get_indexed_count");
      images = await invoke("get_all_images");

      // Load thumbnails sequentially to avoid resource exhaustion
      const newThumbnails: Record<string, string | null> = { ...thumbnails };
      for (const img of images) {
        if (!(img.path in newThumbnails)) {
          try {
            const dataUrl: string = await invoke("get_thumbnail", { imagePath: img.path });
            newThumbnails[img.path] = dataUrl;
          } catch (e) {
            console.error(`Failed to get thumbnail for ${img.path}:`, e);
            newThumbnails[img.path] = null;
          }
        }
      }
      thumbnails = newThumbnails;
    } catch (e) {
      statusMsg = `Error loading data: ${e}`;
    }
  }

  async function addDirectory() {
    try {
      const selected = await open({
        directory: true,
        multiple: false,
        title: "Select folder to watch"
      });

      if (selected) {
        isScanning = true;
        statusMsg = "Scanning...";
        scanResult = await invoke("add_watched_directory", { path: selected });
        await loadData();
        statusMsg = "Scan complete!";
        isScanning = false;
      }
    } catch (e) {
      statusMsg = `Error: ${e}`;
      isScanning = false;
    }
  }

  async function removeDirectory(path: string) {
    try {
      await invoke("remove_watched_directory", { path });
      await loadData();
      statusMsg = `Removed ${path}`;
    } catch (e) {
      statusMsg = `Error: ${e}`;
    }
  }

  async function rescanAll() {
    try {
      isScanning = true;
      statusMsg = "Rescanning all directories...";
      scanResult = await invoke("rescan_all");
      await loadData();
      statusMsg = "Rescan complete!";
      isScanning = false;
    } catch (e) {
      statusMsg = `Error: ${e}`;
      isScanning = false;
    }
  }

  async function testOnnx() {
    try {
      onnxMsg = await invoke("test_onnx");
    } catch (e) {
      onnxMsg = `Error: ${e}`;
    }
  }

  $effect(() => {
    loadData();
  });
</script>

<main class="container">
  <h1>Semantic Image Browser</h1>
  <p class="subtitle">Phase 2: Data Layer Test</p>

  <section>
    <h2>Watched Directories</h2>
    <div class="row">
      <button onclick={addDirectory} disabled={isScanning}>Add Directory</button>
      <button onclick={rescanAll} disabled={isScanning || watchedDirectories.length === 0}>
        Rescan All
      </button>
    </div>

    {#if watchedDirectories.length === 0}
      <p class="empty">No directories added yet. Click "Add Directory" to start.</p>
    {:else}
      <ul class="dir-list">
        {#each watchedDirectories as dir}
          <li>
            <span class="dir-path">{dir}</span>
            <button class="remove-btn" onclick={() => removeDirectory(dir)} disabled={isScanning}>
              Remove
            </button>
          </li>
        {/each}
      </ul>
    {/if}
  </section>

  <section>
    <h2>Index Status</h2>
    <p><strong>Indexed images:</strong> {indexedCount}</p>
    {#if statusMsg}
      <p class="status">{statusMsg}</p>
    {/if}
  </section>

  {#if scanResult}
    <section>
      <h2>Last Scan Result</h2>
      <ul class="scan-result">
        <li>Images found: {scanResult.images_found}</li>
        <li>Images added: {scanResult.images_added}</li>
        <li>Images updated: {scanResult.images_updated}</li>
        <li>Images removed: {scanResult.images_removed}</li>
      </ul>
      {#if scanResult.errors.length > 0}
        <details>
          <summary>Errors ({scanResult.errors.length})</summary>
          <ul class="errors">
            {#each scanResult.errors as error}
              <li>{error}</li>
            {/each}
          </ul>
        </details>
      {/if}
    </section>
  {/if}

  <section>
    <h2>Dependency Tests</h2>
    <div class="row">
      <button onclick={testOnnx}>Test ONNX</button>
    </div>
    {#if onnxMsg}
      <p class="status">{onnxMsg}</p>
    {/if}
  </section>

  <section>
    <h2>Debug: Indexed Images</h2>
    {#if images.length === 0}
      <p class="empty">No images indexed yet.</p>
    {:else}
      <div class="image-grid">
        {#each images as img}
          <div class="image-card">
            {#if thumbnails[img.path]}
              <img src={thumbnails[img.path]} alt={img.path} class="thumbnail" />
            {:else if thumbnails[img.path] === null}
              <div class="thumbnail-placeholder thumbnail-error">Failed</div>
            {:else}
              <div class="thumbnail-placeholder">Loading...</div>
            {/if}
            <div class="image-info">
              <span class="filename" title={img.path}>{img.path.split(/[\\/]/).pop()}</span>
              <span class="meta">{img.file_type} · {(img.file_size / 1024).toFixed(1)} KB</span>
            </div>
          </div>
        {/each}
      </div>
    {/if}
  </section>
</main>

<style>
:root {
  font-family: Inter, Avenir, Helvetica, Arial, sans-serif;
  font-size: 16px;
  line-height: 24px;
  font-weight: 400;
  color: #0f0f0f;
  background-color: #f6f6f6;
  font-synthesis: none;
  text-rendering: optimizeLegibility;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

.container {
  max-width: 800px;
  margin: 0 auto;
  padding: 2rem;
}

h1 {
  text-align: center;
  margin-bottom: 0.5rem;
}

.subtitle {
  text-align: center;
  color: #666;
  margin-bottom: 2rem;
}

section {
  background: #fff;
  border-radius: 8px;
  padding: 1.5rem;
  margin-bottom: 1.5rem;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

h2 {
  margin-top: 0;
  margin-bottom: 1rem;
  font-size: 1.25rem;
}

.row {
  display: flex;
  gap: 0.5rem;
  margin-bottom: 1rem;
}

button {
  border-radius: 8px;
  border: 1px solid transparent;
  padding: 0.6em 1.2em;
  font-size: 1em;
  font-weight: 500;
  font-family: inherit;
  color: #0f0f0f;
  background-color: #e8e8e8;
  cursor: pointer;
  transition: all 0.2s;
}

button:hover:not(:disabled) {
  background-color: #d0d0d0;
  border-color: #396cd8;
}

button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.empty {
  color: #666;
  font-style: italic;
}

.dir-list {
  list-style: none;
  padding: 0;
  margin: 0;
}

.dir-list li {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.75rem;
  background: #f9f9f9;
  border-radius: 4px;
  margin-bottom: 0.5rem;
}

.dir-path {
  font-family: monospace;
  font-size: 0.9rem;
  word-break: break-all;
}

.remove-btn {
  padding: 0.3em 0.8em;
  font-size: 0.85em;
  background-color: #ffebee;
  color: #c62828;
}

.remove-btn:hover:not(:disabled) {
  background-color: #ffcdd2;
}

.status {
  padding: 0.5rem;
  background: #e3f2fd;
  border-radius: 4px;
  font-size: 0.9rem;
}

.scan-result {
  list-style: none;
  padding: 0;
  margin: 0;
}

.scan-result li {
  padding: 0.25rem 0;
}

details {
  margin-top: 1rem;
}

summary {
  cursor: pointer;
  color: #c62828;
}

.errors {
  font-size: 0.85rem;
  color: #c62828;
  max-height: 200px;
  overflow-y: auto;
}

.image-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  gap: 1rem;
}

.image-card {
  background: #f0f0f0;
  border-radius: 8px;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

.thumbnail {
  width: 100%;
  aspect-ratio: 1;
  object-fit: cover;
}

.thumbnail-placeholder {
  width: 100%;
  aspect-ratio: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #ddd;
  color: #666;
  font-size: 0.8rem;
}

.thumbnail-error {
  background: #ffebee;
  color: #c62828;
}

.image-info {
  padding: 0.5rem;
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.filename {
  font-size: 0.85rem;
  font-weight: 500;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.meta {
  font-size: 0.75rem;
  color: #666;
}

@media (prefers-color-scheme: dark) {
  :root {
    color: #f6f6f6;
    background-color: #1a1a1a;
  }

  section {
    background: #2a2a2a;
  }

  button {
    color: #f6f6f6;
    background-color: #3a3a3a;
  }

  button:hover:not(:disabled) {
    background-color: #4a4a4a;
  }

  .dir-list li {
    background: #333;
  }

  .status {
    background: #1e3a5f;
  }

  .remove-btn {
    background-color: #5c2a2a;
    color: #ffcdd2;
  }

  .remove-btn:hover:not(:disabled) {
    background-color: #7c3a3a;
  }

  .image-card {
    background: #3a3a3a;
  }

  .thumbnail-placeholder {
    background: #444;
    color: #999;
  }

  .thumbnail-error {
    background: #5c2a2a;
    color: #ffcdd2;
  }

  .meta {
    color: #999;
  }
}
</style>
