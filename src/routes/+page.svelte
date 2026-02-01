<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";
  import { convertFileSrc } from "@tauri-apps/api/core";
  import { listen } from "@tauri-apps/api/event";
  import { open } from "@tauri-apps/plugin-dialog";
  import { onMount } from "svelte";

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

  interface SiglipConfigInfo {
    has_text: boolean;
    has_vision: boolean;
    text_hidden_size: number | null;
    vision_hidden_size: number | null;
  }

  interface EmbeddingTestResult {
    model_loaded: boolean;
    image_embedding_dim: number | null;
    text_embedding_dim: number | null;
    similarity: number | null;
    error: string | null;
  }

  interface ScanProgressPayload {
    current: number;
    total: number;
  }

  let searchQuery = $state("");
  let images = $state<ImageInfo[]>([]);
  let thumbnails = $state<Record<string, string | null>>({});
  let isLoading = $state(false);
  let watchedDirectories = $state<string[]>([]);
  let indexedCount = $state(0);
  let siglipInfo = $state<SiglipConfigInfo | null>(null);
  let siglipError = $state("");
  let embeddingResult = $state<EmbeddingTestResult | null>(null);
  let embeddingTesting = $state(false);
  let embeddingModelLoaded = $state(false);
  let lastScanDurationMs = $state<number | null>(null);
  let scanProgress = $state<ScanProgressPayload | null>(null);

  // Embedding test modal state
  let showEmbeddingModal = $state(false);
  let embeddingInputs = $state({
    ortDylibPath: "C:\\Dev\\onnxruntime-win-x64-1.23.2\\lib\\onnxruntime.dll",
    modelDir: "C:\\Dev\\test\\siglip2-base-patch16-256-ONNX",
    imagePath: "",
    query: "a photo of a cat"
  });

  // Settings menu state
  let showSettingsMenu = $state(false);
  let showOcrMenu = $state(false);
  let ocrMode = $state<"disabled" | "lexical" | "semantic" | "both">("disabled");

  // Context menu state
  let contextMenu = $state<{ x: number; y: number; image: ImageInfo } | null>(null);

  // Debounce timer
  let searchTimeout: ReturnType<typeof setTimeout> | null = null;

  async function loadThumbnails(imagesToLoad: ImageInfo[]) {
    const newThumbnails: Record<string, string | null> = { ...thumbnails };
    const missing = imagesToLoad.filter(img => !(img.path in newThumbnails));

    await Promise.all(missing.map(async (img) => {
      try {
        const thumbPath: string = await invoke("get_thumbnail_path", { imagePath: img.path });
        newThumbnails[img.path] = convertFileSrc(thumbPath);
      } catch {
        newThumbnails[img.path] = null;
      }
    }));

    thumbnails = newThumbnails;
  }

  async function search(query: string) {
    isLoading = true;
    try {
      images = await invoke("search_images", { query });
      await loadThumbnails(images);
    } catch (e) {
      console.error("Search failed:", e);
    }
    isLoading = false;
  }

  function handleSearchInput(e: Event) {
    const value = (e.target as HTMLInputElement).value;
    searchQuery = value;

    if (searchTimeout) clearTimeout(searchTimeout);
    searchTimeout = setTimeout(() => search(value), 300);
  }

  async function loadInitialData() {
    watchedDirectories = await invoke("get_watched_directories");
    indexedCount = await invoke("get_indexed_count");
    embeddingModelLoaded = await invoke("get_embedding_model_status");
    await search("");
  }

  async function addDirectory() {
    const selected = await open({
      directory: true,
      multiple: false,
      title: "Select folder to watch"
    });

    if (selected) {
      isLoading = true;
      scanProgress = { current: 0, total: 0 };
      const start = performance.now();
      await invoke("add_watched_directory", { path: selected });
      lastScanDurationMs = performance.now() - start;
      await loadInitialData();
      isLoading = false;
    }
  }

  async function removeDirectory(path: string) {
    await invoke("remove_watched_directory", { path });
    await loadInitialData();
  }

  async function rescanAll() {
    isLoading = true;
    scanProgress = { current: 0, total: 0 };
    const start = performance.now();
    await invoke("rescan_all");
    lastScanDurationMs = performance.now() - start;
    await loadInitialData();
    isLoading = false;
  }

  async function openImage(path: string) {
    await invoke("open_image", { path });
  }

  async function showInFolder(path: string) {
    await invoke("show_in_folder", { path });
  }

  async function deleteAllThumbnails() {
    await invoke("delete_all_thumbnails");
    thumbnails = {};
    await loadInitialData();
  }

  async function clearDatabase() {
    await invoke("clear_database");
    thumbnails = {};
    await loadInitialData();
  }

  async function openAppDataFolder() {
    await invoke("open_app_data_folder");
  }

  async function inspectSiglipConfig() {
    const selected = await open({
      directory: false,
      multiple: false,
      title: "Select SigLIP config.json",
      filters: [{ name: "Config", extensions: ["json"] }]
    });

    if (!selected || Array.isArray(selected)) {
      return;
    }

    try {
      siglipInfo = await invoke("inspect_siglip_config", { path: selected });
      siglipError = "";
    } catch (e) {
      siglipInfo = null;
      siglipError = String(e);
    }
  }

  function openEmbeddingModal() {
    showEmbeddingModal = true;
    showSettingsMenu = false;
  }

  async function runEmbeddingTest() {
    const { ortDylibPath, modelDir, imagePath, query } = embeddingInputs;

    if (!ortDylibPath || !modelDir || !imagePath || !query) {
      embeddingResult = {
        model_loaded: false,
        image_embedding_dim: null,
        text_embedding_dim: null,
        similarity: null,
        error: "All fields are required"
      };
      return;
    }

    embeddingTesting = true;
    embeddingResult = null;

    try {
      embeddingResult = await invoke("test_embedding", { ortDylibPath, modelDir, imagePath, query });
    } catch (e) {
      embeddingResult = {
        model_loaded: false,
        image_embedding_dim: null,
        text_embedding_dim: null,
        similarity: null,
        error: String(e)
      };
    }

    embeddingTesting = false;
  }

  async function saveModelConfig() {
    const { ortDylibPath, modelDir } = embeddingInputs;
    if (!ortDylibPath || !modelDir) {
      alert("Both ONNX Runtime DLL path and Model Directory are required");
      return;
    }
    try {
      await invoke("set_model_config", { ortDylibPath, modelDir });
      alert("Model configuration saved. Restart the app to load the model.");
    } catch (e) {
      alert("Failed to save config: " + String(e));
    }
  }

  function handleImageDblClick(img: ImageInfo) {
    openImage(img.path);
  }

  function handleContextMenu(e: MouseEvent, img: ImageInfo) {
    e.preventDefault();
    contextMenu = { x: e.clientX, y: e.clientY, image: img };
  }

  function closeContextMenu() {
    contextMenu = null;
  }

  function handleWindowClick() {
    closeContextMenu();
    showSettingsMenu = false;
    showOcrMenu = false;
  }

  function getFilename(path: string): string {
    return path.split(/[\\/]/).pop() || path;
  }

  function getScanningLabel(defaultLabel: string): string {
    if (!isLoading) {
      return defaultLabel;
    }
    if (scanProgress && scanProgress.total > 0) {
      return `Scanning ${scanProgress.current}/${scanProgress.total} images...`;
    }
    return "Scanning...";
  }

  function formatDuration(ms: number): string {
    if (ms < 1000) {
      return `${Math.round(ms)} ms`;
    }
    const seconds = ms / 1000;
    if (seconds < 60) {
      return `${seconds.toFixed(1)} s`;
    }
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.round(seconds % 60);
    return `${minutes}m ${remainingSeconds}s`;
  }

  $effect(() => {
    loadInitialData();
  });

  onMount(() => {
    const unlistenPromise = listen<ScanProgressPayload>("scan_progress", (event) => {
      scanProgress = event.payload;
    });

    return () => {
      unlistenPromise.then((unlisten) => unlisten());
    };
  });
</script>

<svelte:window onclick={handleWindowClick} />

<div class="app">
  <header class="toolbar">
    <input
      type="text"
      class="search-input"
      placeholder="Search images..."
      value={searchQuery}
      oninput={handleSearchInput}
    />

    <div class="toolbar-buttons">
      <div class="dropdown">
        <button
          class="toolbar-btn"
          onclick={(e) => { e.stopPropagation(); showOcrMenu = !showOcrMenu; showSettingsMenu = false; }}
        >
          OCR: {ocrMode}
        </button>
        {#if showOcrMenu}
          <div class="dropdown-menu" onclick={(e) => e.stopPropagation()}>
            <button class="dropdown-item" class:active={ocrMode === "disabled"} onclick={() => { ocrMode = "disabled"; showOcrMenu = false; }}>Disabled</button>
            <button class="dropdown-item" class:active={ocrMode === "lexical"} onclick={() => { ocrMode = "lexical"; showOcrMenu = false; }}>Lexical</button>
            <button class="dropdown-item" class:active={ocrMode === "semantic"} onclick={() => { ocrMode = "semantic"; showOcrMenu = false; }}>Semantic</button>
            <button class="dropdown-item" class:active={ocrMode === "both"} onclick={() => { ocrMode = "both"; showOcrMenu = false; }}>Both</button>
          </div>
        {/if}
      </div>

      <div class="dropdown">
        <button
          class="toolbar-btn gear-btn"
          onclick={(e) => { e.stopPropagation(); showSettingsMenu = !showSettingsMenu; showOcrMenu = false; }}
        >
          &#9881;
        </button>
        {#if showSettingsMenu}
          <div class="dropdown-menu settings-menu" onclick={(e) => e.stopPropagation()}>
            <div class="menu-section">
              <div class="menu-header">Watched Directories</div>
              {#if watchedDirectories.length === 0}
                <div class="menu-empty">No directories</div>
              {:else}
                {#each watchedDirectories as dir}
                  <div class="dir-item">
                    <span class="dir-path" title={dir}>{dir}</span>
                    <button class="dir-remove" onclick={() => removeDirectory(dir)}>×</button>
                  </div>
                {/each}
              {/if}
              <button class="menu-btn" onclick={addDirectory} disabled={isLoading}>
                {getScanningLabel("Add Directory")}
              </button>
              <button class="menu-btn" onclick={rescanAll} disabled={isLoading || watchedDirectories.length === 0}>
                {getScanningLabel("Rescan All")}
              </button>
              {#if lastScanDurationMs !== null}
                <div class="menu-info">Last scan: {formatDuration(lastScanDurationMs)}</div>
              {/if}
            </div>
            <div class="menu-section">
              <div class="menu-header">Debug</div>
              <div class="menu-info">Indexed: {indexedCount} images</div>
              <div class="menu-info">Embedding model: {embeddingModelLoaded ? "✓ Loaded" : "✗ Not configured"}</div>
              <button class="menu-btn" onclick={openAppDataFolder}>Open App Data Folder</button>
              <button class="menu-btn" onclick={deleteAllThumbnails}>Delete All Thumbnails</button>
              <button class="menu-btn" onclick={clearDatabase}>Clear Database</button>
              <button class="menu-btn" onclick={inspectSiglipConfig}>Inspect SigLIP Config</button>
              <button class="menu-btn" onclick={openEmbeddingModal}>
                Test Embedding
              </button>
              {#if siglipError}
                <div class="menu-info">SigLIP error: {siglipError}</div>
              {/if}
              {#if siglipInfo}
                <div class="menu-info">SigLIP config:</div>
                <div class="menu-info">Text tower: {siglipInfo.has_text ? "yes" : "no"}</div>
                <div class="menu-info">Vision tower: {siglipInfo.has_vision ? "yes" : "no"}</div>
                <div class="menu-info">Text hidden size: {siglipInfo.text_hidden_size ?? "n/a"}</div>
                <div class="menu-info">Vision hidden size: {siglipInfo.vision_hidden_size ?? "n/a"}</div>
              {/if}
              {#if embeddingResult}
                <div class="menu-info" style="margin-top: 8px;">Embedding test:</div>
                <div class="menu-info">Model loaded: {embeddingResult.model_loaded ? "yes" : "no"}</div>
                {#if embeddingResult.image_embedding_dim}
                  <div class="menu-info">Image embedding: {embeddingResult.image_embedding_dim} dims</div>
                {/if}
                {#if embeddingResult.text_embedding_dim}
                  <div class="menu-info">Text embedding: {embeddingResult.text_embedding_dim} dims</div>
                {/if}
                {#if embeddingResult.similarity !== null}
                  <div class="menu-info">Similarity: {embeddingResult.similarity.toFixed(4)}</div>
                {/if}
                {#if embeddingResult.error}
                  <div class="menu-info" style="color: #ff6b6b;">Error: {embeddingResult.error}</div>
                {/if}
              {/if}
            </div>
          </div>
        {/if}
      </div>
    </div>
  </header>

  <main class="grid-container">
    {#if images.length === 0 && !isLoading}
      <div class="empty-state">
        {#if watchedDirectories.length === 0}
          <p>No directories added. Click the gear icon to add a directory.</p>
        {:else}
          <p>No images found.</p>
        {/if}
      </div>
    {:else}
      <div class="image-grid">
        {#each images as img}
          <div
            class="image-cell"
            ondblclick={() => handleImageDblClick(img)}
            oncontextmenu={(e) => handleContextMenu(e, img)}
          >
            {#if thumbnails[img.path]}
              <img src={thumbnails[img.path]} alt="" class="thumbnail" />
            {:else if thumbnails[img.path] === null}
              <div class="thumbnail-placeholder">!</div>
            {:else}
              <div class="thumbnail-placeholder"></div>
            {/if}
            <span class="filename">{getFilename(img.path)}</span>
          </div>
        {/each}
      </div>
    {/if}
  </main>

  {#if contextMenu}
    <div
      class="context-menu"
      style="left: {contextMenu.x}px; top: {contextMenu.y}px;"
      onclick={(e) => e.stopPropagation()}
    >
      <button class="context-item" onclick={() => { openImage(contextMenu!.image.path); closeContextMenu(); }}>
        Open
      </button>
      <button class="context-item" onclick={() => { showInFolder(contextMenu!.image.path); closeContextMenu(); }}>
        Show in folder
      </button>
      <button class="context-item disabled" disabled>
        Find similar
      </button>
    </div>
  {/if}

  {#if showEmbeddingModal}
    <div class="modal-overlay" onclick={() => showEmbeddingModal = false}>
      <div class="modal" onclick={(e) => e.stopPropagation()}>
        <div class="modal-header">
          <span>Test Embedding</span>
          <button class="modal-close" onclick={() => showEmbeddingModal = false}>×</button>
        </div>
        <div class="modal-body">
          <label class="modal-label">
            ONNX Runtime DLL
            <input
              type="text"
              class="modal-input"
              bind:value={embeddingInputs.ortDylibPath}
              placeholder="C:\path\to\onnxruntime.dll"
            />
          </label>
          <label class="modal-label">
            Model Directory
            <input
              type="text"
              class="modal-input"
              bind:value={embeddingInputs.modelDir}
              placeholder="C:\path\to\siglip2-model"
            />
          </label>
          <label class="modal-label">
            Image Path
            <input
              type="text"
              class="modal-input"
              bind:value={embeddingInputs.imagePath}
              placeholder="C:\path\to\image.jpg"
            />
          </label>
          <label class="modal-label">
            Text Query
            <input
              type="text"
              class="modal-input"
              bind:value={embeddingInputs.query}
              placeholder="a photo of a cat"
            />
          </label>
        </div>
        <div class="modal-footer">
          <div class="modal-result">
            App model status: {embeddingModelLoaded ? "✓ Loaded" : "✗ Not loaded"}
          </div>
          {#if embeddingResult}
            <div class="modal-result">
              {#if embeddingResult.error}
                <span class="result-error">Error: {embeddingResult.error}</span>
              {:else}
                <span>Test: ✓ | Image: {embeddingResult.image_embedding_dim}d | Text: {embeddingResult.text_embedding_dim}d | Similarity: {embeddingResult.similarity?.toFixed(4)}</span>
              {/if}
            </div>
          {/if}
          <div class="modal-buttons">
            <button class="modal-btn" onclick={runEmbeddingTest} disabled={embeddingTesting}>
              {embeddingTesting ? "Testing..." : "Run Test"}
            </button>
            <button class="modal-btn" onclick={saveModelConfig}>
              Save Config
            </button>
          </div>
        </div>
      </div>
    </div>
  {/if}
</div>

<style>
  :root {
    --bg-base: #1a1412;
    --bg-toolbar: #2a2220;
    --bg-hover: #3a3230;
    --text-primary: #e0d6d0;
    --text-secondary: #a09590;
    --border-color: #3a3230;
    --source-visual: #1a1412;
    --source-ocr-lexical: #1a2e1a;
    --source-ocr-semantic: #1a1a2e;
  }

  :global(body) {
    margin: 0;
    padding: 0;
    background: var(--bg-base);
    color: var(--text-primary);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    font-size: 14px;
    overflow: hidden;
  }

  .app {
    display: flex;
    flex-direction: column;
    height: 100vh;
    width: 100vw;
  }

  .toolbar {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 6px;
    background: var(--bg-toolbar);
    border-bottom: 1px solid var(--border-color);
    flex-shrink: 0;
  }

  .search-input {
    flex: 1;
    height: 34px;
    padding: 0 12px;
    background: var(--bg-base);
    border: 1px solid var(--border-color);
    border-radius: 4px;
    color: var(--text-primary);
    font-size: 14px;
    outline: none;
    box-sizing: border-box;
  }

  .search-input:focus {
    border-color: #5a5250;
  }

  .search-input::placeholder {
    color: var(--text-secondary);
  }

  .toolbar-buttons {
    display: flex;
    gap: 4px;
  }

  .toolbar-btn {
    height: 34px;
    padding: 0 12px;
    background: var(--bg-base);
    border: 1px solid var(--border-color);
    border-radius: 4px;
    color: var(--text-primary);
    cursor: pointer;
    font-size: 13px;
    box-sizing: border-box;
  }

  .toolbar-btn:hover {
    background: var(--bg-hover);
  }

  .gear-btn {
    font-size: 16px;
    width: 34px;
    padding: 0;
  }

  .dropdown {
    position: relative;
  }

  .dropdown-menu {
    position: absolute;
    top: 100%;
    right: 0;
    margin-top: 4px;
    background: var(--bg-toolbar);
    border: 1px solid var(--border-color);
    border-radius: 4px;
    min-width: 120px;
    z-index: 100;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
  }

  .dropdown-item {
    display: block;
    width: 100%;
    padding: 8px 12px;
    background: none;
    border: none;
    color: var(--text-primary);
    text-align: left;
    cursor: pointer;
    font-size: 13px;
  }

  .dropdown-item:hover {
    background: var(--bg-hover);
  }

  .dropdown-item.active {
    background: var(--bg-base);
  }

  .settings-menu {
    min-width: 280px;
    padding: 8px 0;
  }

  .menu-section {
    padding: 8px 12px;
    border-bottom: 1px solid var(--border-color);
  }

  .menu-section:last-child {
    border-bottom: none;
  }

  .menu-header {
    font-size: 11px;
    text-transform: uppercase;
    color: var(--text-secondary);
    margin-bottom: 8px;
  }

  .menu-empty {
    color: var(--text-secondary);
    font-style: italic;
    font-size: 13px;
    margin-bottom: 8px;
  }

  .menu-info {
    color: var(--text-secondary);
    font-size: 13px;
  }

  .menu-btn {
    display: block;
    width: 100%;
    padding: 6px 10px;
    margin-top: 6px;
    background: var(--bg-base);
    border: 1px solid var(--border-color);
    border-radius: 4px;
    color: var(--text-primary);
    cursor: pointer;
    font-size: 13px;
  }

  .menu-btn:hover:not(:disabled) {
    background: var(--bg-hover);
  }

  .menu-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .dir-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 4px 0;
  }

  .dir-path {
    flex: 1;
    font-size: 12px;
    font-family: monospace;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .dir-remove {
    background: none;
    border: none;
    color: var(--text-secondary);
    cursor: pointer;
    font-size: 16px;
    padding: 0 4px;
  }

  .dir-remove:hover {
    color: #ff6b6b;
  }

  .grid-container {
    flex: 1;
    overflow-y: auto;
    overflow-x: hidden;
  }

  .empty-state {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100%;
    color: var(--text-secondary);
  }

  .image-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
    gap: 2px;
  }

  .image-cell {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 4px;
    cursor: pointer;
  }

  .image-cell:hover {
    background: var(--bg-hover);
  }

  .thumbnail {
    width: 100%;
    aspect-ratio: 1;
    object-fit: cover;
    border-radius: 2px;
  }

  .thumbnail-placeholder {
    width: 100%;
    aspect-ratio: 1;
    background: var(--bg-toolbar);
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--text-secondary);
    border-radius: 2px;
  }

  .filename {
    display: block;
    width: 100%;
    margin-top: 4px;
    font-size: 11px;
    color: var(--text-secondary);
    text-align: center;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .context-menu {
    position: fixed;
    background: var(--bg-toolbar);
    border: 1px solid var(--border-color);
    border-radius: 4px;
    min-width: 140px;
    z-index: 200;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    padding: 4px 0;
  }

  .context-item {
    display: block;
    width: 100%;
    padding: 8px 12px;
    background: none;
    border: none;
    color: var(--text-primary);
    text-align: left;
    cursor: pointer;
    font-size: 13px;
  }

  .context-item:hover:not(:disabled) {
    background: var(--bg-hover);
  }

  .context-item.disabled {
    color: var(--text-secondary);
    cursor: not-allowed;
  }

  .modal-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.6);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 300;
  }

  .modal {
    background: var(--bg-toolbar);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    width: 500px;
    max-width: 90vw;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
  }

  .modal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 16px;
    border-bottom: 1px solid var(--border-color);
    font-weight: 500;
  }

  .modal-close {
    background: none;
    border: none;
    color: var(--text-secondary);
    font-size: 20px;
    cursor: pointer;
    padding: 0 4px;
  }

  .modal-close:hover {
    color: var(--text-primary);
  }

  .modal-body {
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .modal-label {
    display: flex;
    flex-direction: column;
    gap: 4px;
    font-size: 12px;
    color: var(--text-secondary);
  }

  .modal-input {
    padding: 8px 10px;
    background: var(--bg-base);
    border: 1px solid var(--border-color);
    border-radius: 4px;
    color: var(--text-primary);
    font-size: 13px;
    font-family: monospace;
  }

  .modal-input:focus {
    outline: none;
    border-color: #5a5250;
  }

  .modal-footer {
    padding: 12px 16px;
    border-top: 1px solid var(--border-color);
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .modal-result {
    font-size: 12px;
    padding: 8px;
    background: var(--bg-base);
    border-radius: 4px;
  }

  .result-error {
    color: #ff6b6b;
  }

  .modal-buttons {
    display: flex;
    gap: 8px;
  }

  .modal-btn {
    padding: 8px 16px;
    background: var(--bg-base);
    border: 1px solid var(--border-color);
    border-radius: 4px;
    color: var(--text-primary);
    cursor: pointer;
    font-size: 13px;
  }

  .modal-btn:hover:not(:disabled) {
    background: var(--bg-hover);
  }

  .modal-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
</style>
