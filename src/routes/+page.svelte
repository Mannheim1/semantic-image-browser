<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";
  import { convertFileSrc } from "@tauri-apps/api/core";
  import { listen } from "@tauri-apps/api/event";
  import { open, message } from "@tauri-apps/plugin-dialog";
  import { onMount, tick } from "svelte";

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
    sort_score?: number | null;
  }

  interface ScanProgressPayload {
    phase: string;
    current: number;
    total: number;
  }

  let searchQuery = $state("");
  let images = $state<ImageInfo[]>([]);
  let thumbnails = $state<Record<string, string | null>>({});
  let isLoading = $state(false);
  let isScanning = $state(false);
  let watchedDirectories = $state<string[]>([]);
  let indexedCount = $state(0);
  let embeddingModelLoaded = $state(false);
  let modelLoading = $state(true);
  let lastScanDurationMs = $state<number | null>(null);
  let scanProgress = $state<ScanProgressPayload | null>(null);
  let scanOperation = $state<"adding" | "removing">("adding");

  function startScan(operation: "adding" | "removing") {
    scanOperation = operation;
    isScanning = true;
    scanProgress = { phase: operation === "removing" ? "scan" : "thumbnails", current: 0, total: 0 };
  }

  function endScan() {
    scanProgress = null;
    isScanning = false;
  }
  let isPanelOpen = $state(false);
  let selectedIndex = $state<number | null>(null);
  let selectedImage = $state<ImageInfo | null>(null);
  let panelWidthPct = $state<number | null>(null);
  let isResizingPanel = $state(false);
  let resultsRowEl: HTMLDivElement | null = null;
  let gridContainerEl: HTMLElement | null = null;
  let imageCellEls = $state<(HTMLButtonElement | null)[]>([]);

  let showFoldersModal = $state(false);
  let ocrLexical = $state(false);
  let ocrSemantic = $state(false);
  let showAboutModal = $state(false);
  let showViewControlsModal = $state(false);

  // Sort state
  type SortField = "relevance" | "created_at" | "modified_at" | "file_size";
  let sortField = $state<SortField>("relevance");
  let sortAscending = $state(true);

  // Track if any sort is active (for filtered search)
  let hasActiveFilters = $derived(sortField !== "relevance");

  // Context menu state
  let contextMenu = $state<{ x: number; y: number; image: ImageInfo } | null>(null);

  // Similar search state - when set, we're showing results similar to this image
  let similarToImage = $state<ImageInfo | null>(null);

  let searchBarPlaceholder = $derived(
    modelLoading
      ? "Loading model..."
      : isScanning
        ? `${scanOperation === "removing" ? "Removing" : "Adding"} ${scanProgress?.current ?? 0}/${scanProgress?.total ?? 0} images...`
        : similarToImage
          ? `similar to: ${getFilename(similarToImage.path)}`
          : `Search ${indexedCount} images...`
  );

  let searchBarDisabled = $derived(isScanning || modelLoading);

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

  function buildFilterOptions() {
    return {
      file_types: [],
      min_size: null,
      max_size: null,
      min_created: null,
      max_created: null,
      min_modified: null,
      max_modified: null,
    };
  }

  function buildSortOptions() {
    return {
      field: sortField,
      ascending: sortAscending,
    };
  }

  async function search(query: string) {
    closePanel();
    gridContainerEl?.scrollTo({ top: 0, behavior: "auto" });
    isLoading = true;
    try {
      // Use filtered search if any filters are active, otherwise use simple search
      if (hasActiveFilters) {
        images = await invoke("search_images_filtered", {
          query,
          filter: buildFilterOptions(),
          sort: buildSortOptions(),
        });
      } else {
        images = await invoke("search_images", { query });
      }
      await loadThumbnails(images);
    } catch (e) {
      console.error("Search failed:", e);
    }
    isLoading = false;
  }

  function handleSearchInput(e: Event) {
    if (similarToImage) {
      similarToImage = null;
    }
    const value = (e.target as HTMLInputElement).value;
    searchQuery = value;

    if (searchTimeout) clearTimeout(searchTimeout);
    searchTimeout = setTimeout(() => search(value), 300);
  }

  async function loadInitialData() {
    watchedDirectories = await invoke("get_watched_directories");
    indexedCount = await invoke("get_indexed_count");
    // Only update embeddingModelLoaded if we're not in the initial loading phase
    // (model_ready event will set the final value)
    if (!modelLoading) {
      embeddingModelLoaded = await invoke("get_embedding_model_status");
    }
    await search("");
  }

  function formatBytes(bytes: number): string {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  async function addDirectory() {
    const selected = await open({
      directory: true,
      multiple: false,
      title: "Select folder to watch"
    });

    if (selected) {
      startScan("adding");
      const start = performance.now();
      await invoke("add_watched_directory", { path: selected });
      lastScanDurationMs = performance.now() - start;
      console.log(`Scan completed in ${formatDuration(lastScanDurationMs)}`);
      endScan();
      await loadInitialData();
    }
  }

  async function removeDirectory(path: string) {
    startScan("removing");
    await invoke("remove_watched_directory", { path });
    endScan();
    await loadInitialData();
  }

  async function rescanAll() {
    startScan("adding");
    const start = performance.now();
    await invoke("rescan_all");
    lastScanDurationMs = performance.now() - start;
    console.log(`Scan completed in ${formatDuration(lastScanDurationMs)}`);
    endScan();
    await loadInitialData();
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
  }

  async function clearDatabase() {
    await invoke("clear_database");
    thumbnails = {};
    await loadInitialData();
  }

  async function openAppDataFolder() {
    await invoke("open_app_data_folder");
  }

  async function showDependencyPaths() {
    const deps: [string, string][] = await invoke("get_dependency_paths");
    const text = deps.map(([name, path]) => `${name}:\n  ${path}`).join("\n\n");
    await message(text, { title: "Dependency Paths" });
  }

  async function testBundleUrls() {
    await message("Testing all bundle download URLs...\nThis may take a moment.", { title: "Test Bundle URLs" });
    const results: [string, string, string][] = await invoke("test_bundle_urls");
    const text = results.map(([label, _url, status]) => `${status.startsWith("OK") ? "✓" : "✗"} ${label}: ${status}`).join("\n");
    await message(text, { title: "Bundle URL Test Results" });
  }

  function handleImageDblClick(img: ImageInfo) {
    openImage(img.path);
  }

  function handleContextMenu(e: MouseEvent, img: ImageInfo) {
    e.preventDefault();
    contextMenu = { x: e.clientX, y: e.clientY, image: img };
  }

  async function findSimilar(img: ImageInfo) {
    isLoading = true;
    closePanel();
    gridContainerEl?.scrollTo({ top: 0, behavior: "auto" });
    try {
      images = await invoke("search_similar_images", { imagePath: img.path });
      similarToImage = img;
      await loadThumbnails(images);
    } catch (e) {
      console.error("Find similar failed:", e);
      alert("Failed to find similar images: " + String(e));
    }
    isLoading = false;
  }

  function closeContextMenu() {
    contextMenu = null;
  }

  async function openPanelAtIndex(index: number) {
    if (index < 0 || index >= images.length) return;
    if (panelWidthPct === null) {
      panelWidthPct = 50;
    }
    selectedIndex = index;
    selectedImage = images[index];
    isPanelOpen = true;
    await tick();
    imageCellEls[index]?.scrollIntoView({ block: "center", inline: "nearest" });
  }

  function closePanel() {
    isPanelOpen = false;
    selectedIndex = null;
    selectedImage = null;
    isResizingPanel = false;
  }

  function handleImageClick(index: number) {
    openPanelAtIndex(index);
  }

  function handleKeyDown(e: KeyboardEvent) {
    if (!isPanelOpen || selectedIndex === null || images.length === 0) return;
    const target = e.target as HTMLElement | null;
    const tag = target?.tagName?.toLowerCase();
    if (tag === "input" || tag === "textarea" || target?.isContentEditable) return;

    if (e.key === "ArrowRight") {
      e.preventDefault();
      openPanelAtIndex(Math.min(selectedIndex + 1, images.length - 1));
    }
    if (e.key === "ArrowLeft") {
      e.preventDefault();
      openPanelAtIndex(Math.max(selectedIndex - 1, 0));
    }
  }

  function handlePanelResizeStart(e: MouseEvent) {
    if (!isPanelOpen) return;
    e.preventDefault();
    isResizingPanel = true;
  }

  function handleWindowMouseMove(e: MouseEvent) {
    if (!isResizingPanel || !resultsRowEl) return;
    const rect = resultsRowEl.getBoundingClientRect();
    const clampedX = Math.max(rect.left, Math.min(e.clientX, rect.right));
    const panelWidth = rect.right - clampedX;
    const pct = (panelWidth / rect.width) * 100;
    panelWidthPct = Math.max(20, Math.min(80, pct));
  }

  function handleWindowMouseUp() {
    if (!isResizingPanel) return;
    isResizingPanel = false;
  }

  function handleWindowClick() {
    closeContextMenu();
  }

  function getFilename(path: string): string {
    return path.split(/[\\/]/).pop() || path;
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

  function formatDate(timestampMs: number): string {
    return new Date(timestampMs).toLocaleString();
  }

  $effect(() => {
    loadInitialData();
  });

  onMount(() => {
    const unlistenScanPromise = listen<ScanProgressPayload>("scan_progress", (event) => {
      scanProgress = event.payload;
    });

    const unlistenMenuPromise = listen<string>("menu-event", (event) => {
      handleMenuEvent(event.payload);
    });

    const unlistenModelReadyPromise = listen<void>("model_ready", async () => {
      modelLoading = false;
      embeddingModelLoaded = await invoke("get_embedding_model_status");
    });

    return () => {
      unlistenScanPromise.then((unlisten) => unlisten());
      unlistenMenuPromise.then((unlisten) => unlisten());
      unlistenModelReadyPromise.then((unlisten) => unlisten());
    };
  });

  function handleMenuEvent(menuId: string) {
    switch (menuId) {
      case "add_folder":
        addDirectory();
        break;
      case "rescan":
        rescanAll();
        break;
      case "view_files":
        openAppDataFolder();
        break;
      case "manage_folders":
        showFoldersModal = true;
        break;
      case "clear_thumbnails":
        deleteAllThumbnails();
        break;
      case "clear_database":
        clearDatabase();
        break;
      case "ocr_lexical":
        ocrLexical = !ocrLexical;
        break;
      case "ocr_semantic":
        ocrSemantic = !ocrSemantic;
        break;
      case "sort_relevance":
        sortField = "relevance";
        sortAscending = true;
        search(searchQuery);
        break;
      case "sort_created_asc":
        sortField = "created_at";
        sortAscending = true;
        search(searchQuery);
        break;
      case "sort_created_desc":
        sortField = "created_at";
        sortAscending = false;
        search(searchQuery);
        break;
      case "sort_modified_asc":
        sortField = "modified_at";
        sortAscending = true;
        search(searchQuery);
        break;
      case "sort_modified_desc":
        sortField = "modified_at";
        sortAscending = false;
        search(searchQuery);
        break;
      case "sort_size_asc":
        sortField = "file_size";
        sortAscending = true;
        search(searchQuery);
        break;
      case "sort_size_desc":
        sortField = "file_size";
        sortAscending = false;
        search(searchQuery);
        break;
      case "toggle_benchmarking":
        invoke("toggle_benchmarking");
        break;
      case "show_dependency_paths":
        showDependencyPaths();
        break;
      case "test_bundle_urls":
        testBundleUrls();
        break;
      case "about":
        showAboutModal = true;
        break;
      case "view_controls":
        showViewControlsModal = true;
        break;
    }
  }
</script>

<svelte:window
  onclick={handleWindowClick}
  onkeydown={handleKeyDown}
  onmousemove={handleWindowMouseMove}
  onmouseup={handleWindowMouseUp}
/>

<div class="app">
  <header class="toolbar">
    <div class="search-wrapper">
      <input
        type="text"
        class="search-input"
        placeholder={searchBarPlaceholder}
        value={similarToImage ? "" : searchQuery}
        oninput={handleSearchInput}
        disabled={searchBarDisabled}
      />
    </div>

  </header>

  <div class="results-row" bind:this={resultsRowEl}>
    <main
      class="grid-container"
      class:panel-open={isPanelOpen}
      class:panel-closed={!isPanelOpen}
      style={isPanelOpen ? `width: ${100 - (panelWidthPct ?? 50)}%` : undefined}
      bind:this={gridContainerEl}
    >
      {#if images.length === 0 && !isLoading}
        <div class="empty-state">
          {#if watchedDirectories.length === 0}
            <p>No directories added. Use File → Add Folder to get started.</p>
          {:else}
            <p>No images found.</p>
          {/if}
        </div>
      {:else}
        <div class="image-grid">
          {#each images as img, index}
            <button
              class="image-cell"
              class:selected={selectedImage?.path === img.path}
              onclick={() => handleImageClick(index)}
              ondblclick={() => handleImageDblClick(img)}
              oncontextmenu={(e) => handleContextMenu(e, img)}
              bind:this={imageCellEls[index]}
            >
              {#if thumbnails[img.path]}
                <img src={thumbnails[img.path]} alt="" class="thumbnail" />
              {:else if thumbnails[img.path] === null}
                <div class="thumbnail-placeholder">!</div>
              {:else}
                <div class="thumbnail-placeholder"></div>
              {/if}
              <span class="filename">{getFilename(img.path)}</span>
            </button>
          {/each}
        </div>
      {/if}
    </main>

    {#if isPanelOpen && selectedImage}
      <div class="panel-resizer" role="separator" onmousedown={handlePanelResizeStart}></div>
      <aside class="image-panel" style="width: {panelWidthPct ?? 50}%;">
        <div class="panel-header">
          <div class="panel-title" title={selectedImage.path}>{getFilename(selectedImage.path)}</div>
          <button class="panel-close" onclick={closePanel}>X</button>
        </div>
        <div class="panel-body">
          <div class="panel-image-wrapper">
            <img src={convertFileSrc(selectedImage.path)} alt="" class="panel-image" />
          </div>
          <div class="panel-meta">
            <div class="meta-row"><span>Path</span><span>{selectedImage.path}</span></div>
            <div class="meta-row"><span>Type</span><span>{selectedImage.file_type}</span></div>
            <div class="meta-row"><span>Size</span><span>{formatBytes(selectedImage.file_size)}</span></div>
            <div class="meta-row"><span>Created</span><span>{formatDate(selectedImage.created_at)}</span></div>
            <div class="meta-row"><span>Modified</span><span>{formatDate(selectedImage.modified_at)}</span></div>
            {#if selectedImage.sort_score !== null && selectedImage.sort_score !== undefined}
              <div class="meta-row"><span>Distance</span><span>{selectedImage.sort_score.toFixed(4)}</span></div>
            {/if}
          </div>
        </div>
      </aside>
    {/if}
  </div>

  {#if contextMenu}
    <div
      class="context-menu"
      style="left: {contextMenu.x}px; top: {contextMenu.y}px;"
      onclick={(e) => e.stopPropagation()}
    >
      <button class="context-item" onclick={() => { showInFolder(contextMenu!.image.path); closeContextMenu(); }}>
        Show in folder
      </button>
      <button
        class="context-item"
        class:disabled={!embeddingModelLoaded}
        disabled={!embeddingModelLoaded}
        onclick={() => { findSimilar(contextMenu!.image); closeContextMenu(); }}
      >
        Find similar
      </button>
    </div>
  {/if}

  {#if showFoldersModal}
    <!-- svelte-ignore a11y_click_events_have_key_events -->
    <div class="modal-overlay" role="presentation" onclick={() => showFoldersModal = false}>
      <div class="modal" onclick={(e) => e.stopPropagation()}>
        <div class="modal-header">
          <span>Manage Folders</span>
          <button class="modal-close" onclick={() => showFoldersModal = false}>×</button>
        </div>
        <div class="modal-body">
          {#if watchedDirectories.length === 0}
            <div class="folders-empty">No folders added yet.</div>
          {:else}
            <div class="folders-list">
              {#each watchedDirectories as dir}
                <div class="folder-item">
                  <span class="folder-path" title={dir}>{dir}</span>
                  <button class="folder-remove" onclick={() => removeDirectory(dir)}>×</button>
                </div>
              {/each}
            </div>
          {/if}
        </div>
      </div>
    </div>
  {/if}

  {#if showAboutModal}
    <!-- svelte-ignore a11y_click_events_have_key_events -->
    <div class="modal-overlay" role="presentation" onclick={() => showAboutModal = false}>
      <div class="modal" onclick={(e) => e.stopPropagation()}>
        <div class="modal-header">
          <span>About</span>
          <button class="modal-close" onclick={() => showAboutModal = false}>×</button>
        </div>
        <div class="modal-body">
          <p class="text">
            Semantic Image Search helps you find images using natural-language queries by indexing local folders,
            generating thumbnails, and ranking results with visual embeddings. The app is built with Tauri v2 and
            a Svelte + TypeScript frontend, with a Rust backend that uses LanceDB for vector search, ONNX Runtime
            for model inference, and Tesseract for OCR.
          </p>
        </div>
      </div>
    </div>
  {/if}

  {#if showViewControlsModal}
    <!-- svelte-ignore a11y_click_events_have_key_events -->
    <div class="modal-overlay" role="presentation" onclick={() => showViewControlsModal = false}>
      <div class="modal" onclick={(e) => e.stopPropagation()}>
        <div class="modal-header">
          <span>View Controls</span>
          <button class="modal-close" onclick={() => showViewControlsModal = false}>×</button>
        </div>
        <div class="modal-body">
          <div class="text">
            <div>Click a thumbnail to open the detail side panel.</div>
            <div>Double-click a thumbnail to open the image in your default viewer.</div>
            <div>Right-click a thumbnail to open the context menu.</div>
            <div>With the panel open, use Left/Right arrow keys to move to the previous or next image.</div>
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
    --scrollbar-thumb: #6a615c;
    --scrollbar-thumb-hover: #8a7c6a;
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

  :global(*) {
    scrollbar-color: var(--scrollbar-thumb) var(--bg-base);
  }

  :global(*::-webkit-scrollbar-track) {
    background: var(--bg-base);
  }

  :global(*::-webkit-scrollbar-thumb) {
    background: var(--scrollbar-thumb);
    border-radius: 8px;
  }

  :global(*::-webkit-scrollbar-thumb:hover) {
    background: var(--scrollbar-thumb-hover);
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
    width: 100%;
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

  .search-wrapper {
    flex: 1;
  }

  .search-input:focus {
    border-color: #5a5250;
  }

  .search-input::placeholder {
    color: var(--text-secondary);
  }

  .search-input:disabled {
    cursor: not-allowed;
    opacity: 0.7;
  }

  .search-input:disabled::placeholder {
    font-style: italic;
  }


















  .grid-container {
    overflow-y: auto;
    overflow-x: hidden;
  }

  .grid-container.panel-closed {
    flex: 1 1 auto;
    width: 100%;
  }

  .grid-container.panel-open {
    flex: 0 0 auto;
  }

  .results-row {
    flex: 1;
    display: flex;
    min-height: 0;
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
    background: none;
    border: none;
    color: inherit;
    font: inherit;
  }

  .image-cell:hover {
    background: var(--bg-hover);
  }

  .image-cell.selected {
    background: var(--bg-hover);
    outline: 1px solid #5a5250;
    outline-offset: -1px;
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

  .image-panel {
    flex: 0 0 auto;
    display: flex;
    flex-direction: column;
    background: var(--bg-toolbar);
    border-left: 1px solid var(--border-color);
    min-width: 0;
  }

  .panel-resizer {
    width: 12px;
    cursor: ew-resize;
    background: transparent;
    align-self: stretch;
    position: relative;
    z-index: 1;
  }

  .panel-resizer::before {
    content: "";
    position: absolute;
    top: 0;
    bottom: 0;
    left: 50%;
    width: 1px;
    background: var(--border-color);
    transform: translateX(-50%);
  }

  .panel-resizer:hover {
    background: var(--bg-hover);
  }

  .panel-resizer:hover::before {
    background: #5a5250;
    width: 3px;
  }

  .panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 12px;
    border-bottom: 1px solid var(--border-color);
  }

  .panel-title {
    font-size: 13px;
    font-weight: 500;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .panel-close {
    background: none;
    border: none;
    color: var(--text-secondary);
    font-size: 18px;
    cursor: pointer;
    padding: 0 4px;
    user-select: none;
  }

  .panel-close:hover {
    color: var(--text-primary);
  }

  .panel-body {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 10px;
    padding: 10px 12px;
    overflow: auto;
  }

  .panel-image-wrapper {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    background: var(--bg-base);
    border: 1px solid var(--border-color);
    border-radius: 6px;
    padding: 8px;
    min-height: 200px;
  }

  .panel-image {
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
  }

  .panel-meta {
    display: flex;
    flex-direction: column;
    gap: 6px;
    font-size: 12px;
  }

  .meta-row {
    display: flex;
    gap: 8px;
  }

  .meta-row span:first-child {
    color: var(--text-secondary);
    width: 70px;
    flex-shrink: 0;
  }

  .meta-row span:last-child {
    color: var(--text-primary);
    word-break: break-all;
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

  /* Folders Modal Styles */
  .folders-empty {
    color: var(--text-secondary);
    font-style: italic;
    text-align: center;
    padding: 20px;
  }

  .folders-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .folder-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    background: var(--bg-base);
    border: 1px solid var(--border-color);
    border-radius: 4px;
  }

  .folder-path {
    flex: 1;
    font-size: 13px;
    font-family: monospace;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .folder-remove {
    background: none;
    border: none;
    color: var(--text-secondary);
    cursor: pointer;
    font-size: 18px;
    padding: 0 4px;
    line-height: 1;
  }

  .folder-remove:hover {
    color: #ff6b6b;
  }

  .text {
    margin: 0;
    color: var(--text-secondary);
    line-height: 1.5;
    font-size: 13px;
  }

</style>
