<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";
  import { convertFileSrc } from "@tauri-apps/api/core";
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

  let searchQuery = $state("");
  let images = $state<ImageInfo[]>([]);
  let thumbnails = $state<Record<string, string | null>>({});
  let isLoading = $state(false);
  let watchedDirectories = $state<string[]>([]);
  let indexedCount = $state(0);

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
      await invoke("add_watched_directory", { path: selected });
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
    await invoke("rescan_all");
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

  $effect(() => {
    loadInitialData();
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
                {isLoading ? "Scanning..." : "Add Directory"}
              </button>
              <button class="menu-btn" onclick={rescanAll} disabled={isLoading || watchedDirectories.length === 0}>
                {isLoading ? "Scanning..." : "Rescan All"}
              </button>
            </div>
            <div class="menu-section">
              <div class="menu-header">Debug</div>
              <div class="menu-info">Indexed: {indexedCount} images</div>
              <button class="menu-btn" onclick={openAppDataFolder}>Open App Data Folder</button>
              <button class="menu-btn" onclick={deleteAllThumbnails}>Delete All Thumbnails</button>
              <button class="menu-btn" onclick={clearDatabase}>Clear Database</button>
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
</style>
