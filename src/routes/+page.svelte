<script lang="ts">
  import "$lib/theme.css";
  import { invoke } from "@tauri-apps/api/core";
  import { convertFileSrc } from "@tauri-apps/api/core";
  import { emit, listen } from "@tauri-apps/api/event";
  import { getCurrentWebview } from "@tauri-apps/api/webview";
  import { getCurrentWindow } from "@tauri-apps/api/window";
  import { open, message } from "@tauri-apps/plugin-dialog";
  import { onMount, tick, untrack } from "svelte";

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

  interface DownloadProgressPayload {
    phase: string;
    current_bytes: number;
    total_bytes: number;
  }
  let depsProgress = $state<DownloadProgressPayload | null>(null);

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

  let ocrLexical = $state(false);
  let ocrSemantic = $state(false);
  let skipNextDirsChanged = false;


  let zoomLevel = $state(1);

  // Sort state
  type SortField = "relevance" | "created_at" | "modified_at" | "file_size";
  let sortField = $state<SortField>("relevance");
  let sortAscending = $state(true);

  // Track if any sort is active (for filtered search)
  let hasActiveFilters = $derived(sortField !== "relevance");

  // Context menu state
  let contextMenu = $state<{ x: number; y: number; image: ImageInfo } | null>(null);
  let contextMenuEl = $state<HTMLElement | null>(null);
  let contextMenuTrigger: HTMLElement | null = null;

  $effect(() => {
    if (contextMenu && contextMenuEl) {
      contextMenuEl.querySelector<HTMLElement>('button:not([disabled])')?.focus();
    }
  });

  // Similar search state - when set, we're showing results similar to this image
  let similarToImage = $state<ImageInfo | null>(null);

  function formatMB(bytes: number): string {
    return (bytes / 1_048_576).toFixed(0);
  }

  let searchBarPlaceholder = $derived(
    depsProgress && depsProgress.phase !== "done"
      ? depsProgress.total_bytes > 0
        ? `${depsProgress.phase} (${formatMB(depsProgress.current_bytes)}/${formatMB(depsProgress.total_bytes)} MB)...`
        : `${depsProgress.phase}...`
      : modelLoading
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
  let latestSearchRequestId = 0;

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
    const requestId = ++latestSearchRequestId;
    closePanel();
    selectedIndex = null;
    selectedImage = null;
    scrollToIndex(null);
    isLoading = true;
    try {
      let nextImages: ImageInfo[];
      // Use filtered search if any filters are active, otherwise use simple search
      if (hasActiveFilters) {
        nextImages = await invoke("search_images_filtered", {
          query,
          filter: buildFilterOptions(),
          sort: buildSortOptions(),
        });
      } else {
        nextImages = await invoke("search_images", { query });
      }
      if (requestId !== latestSearchRequestId) return;
      images = nextImages;
      await loadThumbnails(nextImages);
    } catch (e) {
      console.error("Search failed:", e);
    } finally {
      if (requestId === latestSearchRequestId) {
        isLoading = false;
      }
    }
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
      multiple: true,
      title: "Select folder(s) to watch"
    });

    if (selected) {
      const paths = Array.isArray(selected) ? selected : [selected];
      startScan("adding");
      const start = performance.now();
      for (const path of paths) {
        await invoke("add_watched_directory", { path });
      }
      lastScanDurationMs = performance.now() - start;
      console.log(`Scan completed in ${formatDuration(lastScanDurationMs)}`);
      endScan();
      await loadInitialData();
      skipNextDirsChanged = true;
      emit("directories-changed");
    }
  }

  async function removeDirectory(path: string) {
    startScan("removing");
    await invoke("remove_watched_directory", { path });
    endScan();
    await loadInitialData();
    skipNextDirsChanged = true;
    emit("directories-changed");
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

  function showAbout() {
    invoke("open_popup", { route: "/about", title: "About", width: 390, height: 255, resizable: false });
  }

  function showViewControls() {
    invoke("open_popup", { route: "/view-controls", title: "View Controls", width: 460, height: 300, resizable: false });
  }

  async function showDependencyPaths() {
    const deps: [string, string][] = await invoke("get_dependency_paths");
    const text = deps.map(([name, path]) => `${name}:\n  ${path}`).join("\n\n");
    await message(text, { title: "Dependency Paths" });
  }

  async function testBundleUrls() {
    const results: [string, string, string][] = await invoke("test_bundle_urls");
    const text = results.map(([label, _url, status]) => `${status.startsWith("OK") ? "✓" : "✗"} ${label}: ${status}`).join("\n");
    await message(text, { title: "Bundle URL Test Results" });
  }

  function handleContextMenu(e: MouseEvent, img: ImageInfo) {
    e.preventDefault();
    contextMenuTrigger = e.currentTarget as HTMLElement;
    let { clientX: x, clientY: y } = e;
    if (x === 0 && y === 0) {
      const rect = contextMenuTrigger.getBoundingClientRect();
      x = rect.left;
      y = rect.bottom;
    }
    contextMenu = { x, y, image: img };
  }

  async function findSimilar(img: ImageInfo) {
    closePanel();
    selectedIndex = null;
    selectedImage = null;
    isLoading = true;
    try {
      images = await invoke("search_similar_images", { imagePath: img.path });
      similarToImage = img;
      await loadThumbnails(images);
    } catch (e) {
      console.error("Find similar failed:", e);
      alert("Failed to find similar images: " + String(e));
    }
    isLoading = false;
    scrollToIndex(null);
  }

  function closeContextMenu() {
    contextMenu = null;
    contextMenuTrigger?.focus();
    contextMenuTrigger = null;
  }

  function handleContextMenuKeydown(e: KeyboardEvent) {
    if (!contextMenuEl) return;
    const items = Array.from(contextMenuEl.querySelectorAll<HTMLElement>('button:not([disabled])'));
    const index = items.indexOf(document.activeElement as HTMLElement);

    if (e.key === 'Escape') {
      e.stopPropagation();
      closeContextMenu();
    } else if (e.key === 'ArrowDown') {
      e.preventDefault();
      e.stopPropagation();
      items[(index + 1) % items.length]?.focus();
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      e.stopPropagation();
      items[(index - 1 + items.length) % items.length]?.focus();
    } else if (e.key === 'Tab') {
      closeContextMenu();
    } else {
      e.stopPropagation();
    }
  }

  function selectIndex(index: number) {
    if (index < 0 || index >= images.length) return;
    selectedIndex = index;
    selectedImage = images[index];
  }

  function scrollToIndex(index: number | null) {
    if (index === null) {
      gridContainerEl?.scrollTo({ top: 0, behavior: "auto" });
    } else {
      imageCellEls[index]?.scrollIntoView({ block: "nearest", inline: "nearest" });
    }
  }

  function navigateTo(index: number) {
    selectIndex(index);
    scrollToIndex(index);
    imageCellEls[index]?.focus();
  }

  // Re-scroll selected image into view when layout changes
  $effect(() => {
    isPanelOpen; // track panel open/close
    const idx = untrack(() => selectedIndex);
    if (idx !== null) tick().then(() => scrollToIndex(idx));
  });

  function openPanelAtIndex(index: number) {
    if (index < 0 || index >= images.length) return;
    if (panelWidthPct === null) panelWidthPct = 50;
    isPanelOpen = true;
    navigateTo(index);
  }

  function closePanel() {
    isPanelOpen = false;
    isResizingPanel = false;
  }

  function handleKeyDown(e: KeyboardEvent) {
    if (e.key === "Escape" && isPanelOpen) {
      e.preventDefault();
      closePanel();
      return;
    }

    if (selectedIndex === null || images.length === 0) return;
    const target = e.target as HTMLElement | null;
    const tag = target?.tagName?.toLowerCase();
    if (tag === "input" || tag === "textarea" || target?.isContentEditable) return;

    if (e.key === "ArrowRight") {
      e.preventDefault();
      navigateTo(Math.min(selectedIndex + 1, images.length - 1));
    }
    if (e.key === "ArrowLeft") {
      e.preventDefault();
      navigateTo(Math.max(selectedIndex - 1, 0));
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
      if (!isScanning) {
        scanOperation = event.payload.phase === "scan" ? "removing" : "adding";
        isScanning = true;
      }
    });

    const unlistenMenuPromise = listen<string>("menu-event", (event) => {
      handleMenuEvent(event.payload);
    });

    const unlistenModelReadyPromise = listen<void>("model_ready", async () => {
      modelLoading = false;
      depsProgress = null;
      embeddingModelLoaded = await invoke("get_embedding_model_status");
    });

    const unlistenDepsPromise = listen<DownloadProgressPayload>("runtime_deps_progress", (event) => {
      depsProgress = event.payload.phase === "done" ? null : event.payload;
    });

    const unlistenDirsChangedPromise = listen<void>("directories-changed", () => {
      if (skipNextDirsChanged) {
        skipNextDirsChanged = false;
        return;
      }
      endScan();
      loadInitialData();
    });

    return () => {
      unlistenScanPromise.then((unlisten) => unlisten());
      unlistenMenuPromise.then((unlisten) => unlisten());
      unlistenModelReadyPromise.then((unlisten) => unlisten());
      unlistenDepsPromise.then((unlisten) => unlisten());
      unlistenDirsChangedPromise.then((unlisten) => unlisten());
    };
  });

  async function handleMenuEvent(menuId: string) {
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
        invoke("open_popup", { route: "/manage-folders", title: "Manage Folders", width: 550, height: 400, resizable: true });
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
      case "compute_clusters":
        try {
          await invoke("compute_clusters");
          await message("Clusters computed. Open Cluster → Browse Clusters to view.", { title: "Cluster computation" });
        } catch (e) {
          await message(String(e), { title: "Cluster computation failed", kind: "error" });
        }
        break;
      case "compute_umap":
        try {
          await invoke("compute_umap");
          await message("2D Map computed. Open Cluster → Show 2D Map to view.", { title: "2D Map" });
        } catch (e) {
          await message(String(e), { title: "2D Map failed", kind: "error" });
        }
        break;
      case "browse_clusters":
        invoke("open_popup", { route: "/clusters", title: "Browse Clusters", width: 900, height: 700, resizable: true });
        break;
      case "show_map":
        invoke("open_popup", { route: "/map", title: "2D Map", width: 900, height: 700, resizable: true });
        break;
      case "toggle_benchmarking":
        invoke("toggle_benchmarking");
        break;
      case "toggle_slow_scan":
        invoke("toggle_slow_scan");
        break;
      case "show_dependency_paths":
        showDependencyPaths();
        break;
      case "test_bundle_urls":
        testBundleUrls();
        break;
      case "about":
        showAbout();
        break;
      case "view_controls":
        showViewControls();
        break;
      case "zoom_in":
        zoomLevel = Math.min(3, zoomLevel + 0.1);
        await getCurrentWebview().setZoom(zoomLevel);
        break;
      case "zoom_out":
        zoomLevel = Math.max(0.2, zoomLevel - 0.1);
        await getCurrentWebview().setZoom(zoomLevel);
        break;
      case "reset_zoom":
        zoomLevel = 1;
        await getCurrentWebview().setZoom(zoomLevel);
        break;
      case "toggle_fullscreen": {
        const currentWindow = getCurrentWindow();
        const isFullscreen = await currentWindow.isFullscreen();
        await currentWindow.setFullscreen(!isFullscreen);
        break;
      }
    }
  }
</script>

<svelte:window
  onclick={handleWindowClick}
  onkeydown={handleKeyDown}
  onmousemove={handleWindowMouseMove}
  onmouseup={handleWindowMouseUp}
  onresize={() => { if (selectedIndex !== null) scrollToIndex(selectedIndex); }}
/> <!-- onresize: keep selected image visible on reflow -->

<div class="app">
  <header class="toolbar">
    <div class="search-wrapper">
      <input
        type="text"
        class="search-input"
        placeholder={searchBarPlaceholder}
        value={similarToImage ? "" : searchQuery}
        oninput={handleSearchInput}
        spellcheck={false}
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
              onclick={() => openPanelAtIndex(index)}
              onfocus={() => selectIndex(index)}
              ondblclick={() => openImage(img.path)}
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
      <div
        class="panel-resizer"
        role="slider"
        aria-label="Panel width"
        aria-valuenow={panelWidthPct ?? 50}
        aria-valuemin={20}
        aria-valuemax={80}
        tabindex="0"
        onmousedown={handlePanelResizeStart}
        onkeydown={(e) => {
          if (e.key === 'ArrowLeft') panelWidthPct = Math.min(80, (panelWidthPct ?? 50) + 5);
          else if (e.key === 'ArrowRight') panelWidthPct = Math.max(20, (panelWidthPct ?? 50) - 5);
        }}
      ></div>
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
      role="menu"
      tabindex="-1"
      style="left: {contextMenu.x}px; top: {contextMenu.y}px;"
      bind:this={contextMenuEl}
      onclick={(e) => e.stopPropagation()}
      onkeydown={handleContextMenuKeydown}
    >
      <button role="menuitem" class="context-item" onclick={() => { showInFolder(contextMenu!.image.path); closeContextMenu(); }}>
        Show in folder
      </button>
      <button
        role="menuitem"
        class="context-item"
        class:disabled={!embeddingModelLoaded}
        disabled={!embeddingModelLoaded}
        onclick={() => { const img = contextMenu!.image; closeContextMenu(); findSimilar(img); }}
      >
        Find similar
      </button>
    </div>
  {/if}

</div>

<style>
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

</style>
