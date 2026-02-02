<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";
  import { convertFileSrc } from "@tauri-apps/api/core";
  import { listen } from "@tauri-apps/api/event";
  import { open } from "@tauri-apps/plugin-dialog";
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

  interface OrtStatus {
    installed: boolean;
    library_path: string | null;
    runtime_type: string | null;
    gpu_available: boolean;
    platform: string;
  }

  interface OrtDownloadProgress {
    downloaded: number;
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
  let isPanelOpen = $state(false);
  let selectedIndex = $state<number | null>(null);
  let selectedImage = $state<ImageInfo | null>(null);
  let panelWidthPct = $state<number | null>(null);
  let isResizingPanel = $state(false);
  let resultsRowEl: HTMLDivElement | null = null;
  let gridContainerEl: HTMLElement | null = null;
  let imageCellEls = $state<(HTMLDivElement | null)[]>([]);

  // ONNX Runtime state
  let ortStatus = $state<OrtStatus | null>(null);
  let showOrtModal = $state(false);
  let ortDownloading = $state(false);
  let ortDownloadProgress = $state<OrtDownloadProgress | null>(null);
  let ortDownloadError = $state<string | null>(null);
  let selectedRuntimeType = $state<"cpu" | "gpu">("cpu");

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
  let showFilterMenu = $state(false);
  let ocrMode = $state<"disabled" | "lexical" | "semantic" | "both">("disabled");

  // Filter & Sort state
  type SortField = "relevance" | "created_at" | "modified_at" | "file_size";
  let sortField = $state<SortField>("relevance");
  let sortAscending = $state(true);
  let filterDateFrom = $state<string>("");
  let filterDateTo = $state<string>("");

  // Track if any filter is active
  let hasActiveFilters = $derived(
    filterDateFrom !== "" ||
    filterDateTo !== "" ||
    sortField !== "relevance"
  );

  // Context menu state
  let contextMenu = $state<{ x: number; y: number; image: ImageInfo } | null>(null);

  // Similar search state - when set, we're showing results similar to this image
  let similarToImage = $state<ImageInfo | null>(null);

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
      min_modified: filterDateFrom ? new Date(filterDateFrom).getTime() : null,
      max_modified: filterDateTo ? new Date(filterDateTo + "T23:59:59").getTime() : null,
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
    const value = (e.target as HTMLInputElement).value;
    searchQuery = value;

    if (searchTimeout) clearTimeout(searchTimeout);
    searchTimeout = setTimeout(() => search(value), 300);
  }

  async function loadInitialData() {
    watchedDirectories = await invoke("get_watched_directories");
    indexedCount = await invoke("get_indexed_count");
    embeddingModelLoaded = await invoke("get_embedding_model_status");
    ortStatus = await invoke("get_ort_status");
    await search("");
  }

  function formatBytes(bytes: number): string {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  async function openOrtModal() {
    showOrtModal = true;
    showSettingsMenu = false;
    ortDownloadError = null;
    ortDownloadProgress = null;
    // Refresh status
    ortStatus = await invoke("get_ort_status");
  }

  let ortNeedsRestart = $state(false);

  async function downloadOrt() {
    ortDownloading = true;
    ortDownloadError = null;
    ortDownloadProgress = { downloaded: 0, total: 0 };

    try {
      await invoke("download_ort", { runtimeType: selectedRuntimeType });
      // Refresh status after download
      ortStatus = await invoke("get_ort_status");
      ortDownloadProgress = null;
      ortNeedsRestart = true;
    } catch (e) {
      ortDownloadError = String(e);
    }

    ortDownloading = false;
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

  function viewDistanceScore(img: ImageInfo) {
    if (img.sort_score === null || img.sort_score === undefined) {
      return;
    }
    alert(`Distance: ${img.sort_score.toFixed(4)}`);
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

  function clearSimilarSearch() {
    similarToImage = null;
    searchQuery = "";
    search("");
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
    showSettingsMenu = false;
    showOcrMenu = false;
    showFilterMenu = false;
  }

  function clearFilters() {
    filterDateFrom = "";
    filterDateTo = "";
    sortField = "relevance";
    sortAscending = true;
    search(searchQuery);
  }

  function setSortField(field: SortField) {
    sortField = field;
    search(searchQuery);
  }

  function setSortDirection(ascending: boolean) {
    sortAscending = ascending;
    search(searchQuery);
  }

  function setDateFrom(value: string) {
    filterDateFrom = value;
    search(searchQuery);
  }

  function setDateTo(value: string) {
    filterDateTo = value;
    search(searchQuery);
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

    const unlistenOrtPromise = listen<OrtDownloadProgress>("ort_download_progress", (event) => {
      ortDownloadProgress = event.payload;
    });

    return () => {
      unlistenScanPromise.then((unlisten) => unlisten());
      unlistenOrtPromise.then((unlisten) => unlisten());
    };
  });
</script>

<svelte:window
  onclick={handleWindowClick}
  onkeydown={handleKeyDown}
  onmousemove={handleWindowMouseMove}
  onmouseup={handleWindowMouseUp}
/>

<div class="app">
  <header class="toolbar">
    <input
      type="text"
      class="search-input"
      placeholder="Search images..."
      value={similarToImage ? `similar to: ${getFilename(similarToImage.path)}` : searchQuery}
      oninput={handleSearchInput}
      onfocus={() => { if (similarToImage) clearSimilarSearch(); }}
    />

    <div class="toolbar-buttons">
      <div class="dropdown">
        <button
          class="toolbar-btn"
          onclick={(e) => { e.stopPropagation(); showOcrMenu = !showOcrMenu; showSettingsMenu = false; showFilterMenu = false; }}
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
          class="toolbar-btn"
          class:has-filters={hasActiveFilters}
          onclick={(e) => { e.stopPropagation(); showFilterMenu = !showFilterMenu; showOcrMenu = false; showSettingsMenu = false; }}
        >
          Sort/Filter{hasActiveFilters ? " *" : ""}
        </button>
        {#if showFilterMenu}
          <div class="dropdown-menu filter-menu" onclick={(e) => e.stopPropagation()}>
            <div class="menu-section">
              <div class="menu-header">Sort By</div>
              <div class="sort-options">
                <button class="dropdown-item" class:active={sortField === "relevance"} onclick={() => setSortField("relevance")}>Relevance</button>
                <button class="dropdown-item" class:active={sortField === "created_at"} onclick={() => setSortField("created_at")}>Date Created</button>
                <button class="dropdown-item" class:active={sortField === "modified_at"} onclick={() => setSortField("modified_at")}>Date Modified</button>
                <button class="dropdown-item" class:active={sortField === "file_size"} onclick={() => setSortField("file_size")}>File Size</button>
              </div>
              {#if sortField !== "relevance"}
                <div class="sort-direction">
                  <button class="dropdown-item" class:active={sortAscending} onclick={() => setSortDirection(true)}>Ascending</button>
                  <button class="dropdown-item" class:active={!sortAscending} onclick={() => setSortDirection(false)}>Descending</button>
                </div>
              {/if}
            </div>

            <div class="menu-section">
              <div class="menu-header">Date Modified</div>
              <div class="date-filters">
                <label class="filter-label">
                  <span>From</span>
                  <input
                    type="date"
                    class="filter-input"
                    value={filterDateFrom}
                    onchange={(e) => setDateFrom((e.target as HTMLInputElement).value)}
                  />
                </label>
                <label class="filter-label">
                  <span>To</span>
                  <input
                    type="date"
                    class="filter-input"
                    value={filterDateTo}
                    onchange={(e) => setDateTo((e.target as HTMLInputElement).value)}
                  />
                </label>
              </div>
            </div>

            {#if hasActiveFilters}
              <div class="menu-section">
                <button class="menu-btn" onclick={clearFilters}>Clear All</button>
              </div>
            {/if}
          </div>
        {/if}
      </div>

      <div class="dropdown">
        <button
          class="toolbar-btn gear-btn"
          onclick={(e) => { e.stopPropagation(); showSettingsMenu = !showSettingsMenu; showOcrMenu = false; showFilterMenu = false; }}
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
              <div class="menu-header">Runtime</div>
              <div class="menu-info">
                ONNX Runtime: {ortStatus?.installed ? `✓ ${ortStatus.runtime_type?.toUpperCase() ?? "Installed"}` : "✗ Not installed"}
              </div>
              <div class="menu-info">Embedding model: {embeddingModelLoaded ? "✓ Loaded" : "✗ Not configured"}</div>
              <button class="menu-btn" onclick={openOrtModal}>
                {ortStatus?.installed ? "Manage Runtime" : "Setup Runtime"}
              </button>
            </div>
            <div class="menu-section">
              <div class="menu-header">Debug</div>
              <div class="menu-info">Indexed: {indexedCount} images</div>
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
            <p>No directories added. Click the gear icon to add a directory.</p>
          {:else}
            <p>No images found.</p>
          {/if}
        </div>
      {:else}
        <div class="image-grid">
          {#each images as img, index}
            <div
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
            </div>
          {/each}
        </div>
      {/if}
    </main>

    {#if isPanelOpen && selectedImage}
      <div class="panel-resizer" onmousedown={handlePanelResizeStart}></div>
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
        class:disabled={searchQuery.trim().length === 0}
        disabled={searchQuery.trim().length === 0}
        onclick={() => { viewDistanceScore(contextMenu!.image); closeContextMenu(); }}
      >
        View distance
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

  {#if showOrtModal}
    <div class="modal-overlay" onclick={() => { if (!ortDownloading) showOrtModal = false; }}>
      <div class="modal" onclick={(e) => e.stopPropagation()}>
        <div class="modal-header">
          <span>ONNX Runtime Setup</span>
          <button class="modal-close" onclick={() => { if (!ortDownloading) showOrtModal = false; }} disabled={ortDownloading}>×</button>
        </div>
        <div class="modal-body">
          {#if ortStatus?.installed}
            <div class="ort-status ort-status-ok">
              <div class="ort-status-icon">✓</div>
              <div class="ort-status-text">
                <div>ONNX Runtime is installed</div>
                <div class="ort-status-detail">{ortStatus.library_path}</div>
              </div>
            </div>
          {:else}
            <div class="ort-status ort-status-missing">
              <div class="ort-status-icon">!</div>
              <div class="ort-status-text">
                <div>ONNX Runtime is not installed</div>
                <div class="ort-status-detail">Required for semantic image search</div>
              </div>
            </div>
          {/if}

          <div class="ort-info">
            <div class="ort-info-row">
              <span>Platform:</span>
              <span>{ortStatus?.platform ?? "Unknown"}</span>
            </div>
            <div class="ort-info-row">
              <span>GPU Support:</span>
              <span>{ortStatus?.gpu_available ? "Available" : "Not available"}</span>
            </div>
          </div>

          {#if !ortDownloading}
            <div class="ort-runtime-select">
              <div class="ort-runtime-label">Select Runtime Type:</div>
              <label class="ort-runtime-option">
                <input type="radio" bind:group={selectedRuntimeType} value="cpu" />
                <div class="ort-runtime-info">
                  <span class="ort-runtime-name">CPU (Recommended)</span>
                  <span class="ort-runtime-desc">Works on all systems, ~78 MB download</span>
                </div>
              </label>
              {#if ortStatus?.gpu_available}
                <label class="ort-runtime-option">
                  <input type="radio" bind:group={selectedRuntimeType} value="gpu" />
                  <div class="ort-runtime-info">
                    <span class="ort-runtime-name">GPU (NVIDIA CUDA)</span>
                    <span class="ort-runtime-desc">Faster processing, ~326 MB download. Requires NVIDIA GPU with CUDA 11.8+</span>
                  </div>
                </label>
              {/if}
            </div>
          {/if}

          {#if ortDownloading && ortDownloadProgress}
            <div class="ort-progress">
              <div class="ort-progress-text">
                Downloading... {formatBytes(ortDownloadProgress.downloaded)}
                {#if ortDownloadProgress.total > 0}
                  / {formatBytes(ortDownloadProgress.total)}
                {/if}
              </div>
              <div class="ort-progress-bar">
                <div
                  class="ort-progress-fill"
                  style="width: {ortDownloadProgress.total > 0 ? (ortDownloadProgress.downloaded / ortDownloadProgress.total * 100) : 0}%"
                ></div>
              </div>
            </div>
          {/if}

          {#if ortDownloadError}
            <div class="ort-error">
              Error: {ortDownloadError}
            </div>
          {/if}

          {#if ortNeedsRestart}
            <div class="ort-restart-notice">
              Runtime downloaded successfully. Please restart the app to use the new runtime.
            </div>
          {/if}
        </div>
        <div class="modal-footer">
          <div class="modal-buttons">
            <button
              class="modal-btn modal-btn-primary"
              onclick={downloadOrt}
              disabled={ortDownloading}
            >
              {#if ortDownloading}
                Downloading...
              {:else if ortStatus?.installed}
                Reinstall Runtime
              {:else}
                Download Runtime
              {/if}
            </button>
            <button
              class="modal-btn"
              onclick={() => showOrtModal = false}
              disabled={ortDownloading}
            >
              {ortStatus?.installed ? "Close" : "Cancel"}
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

  .modal-btn-primary {
    background: #3a5a3a;
    border-color: #4a6a4a;
  }

  .modal-btn-primary:hover:not(:disabled) {
    background: #4a6a4a;
  }

  /* ORT Modal Styles */
  .ort-status {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px;
    border-radius: 6px;
    margin-bottom: 12px;
  }

  .ort-status-ok {
    background: rgba(58, 90, 58, 0.3);
    border: 1px solid #4a6a4a;
  }

  .ort-status-missing {
    background: rgba(90, 58, 58, 0.3);
    border: 1px solid #6a4a4a;
  }

  .ort-status-icon {
    font-size: 24px;
    width: 32px;
    text-align: center;
  }

  .ort-status-ok .ort-status-icon {
    color: #6a9a6a;
  }

  .ort-status-missing .ort-status-icon {
    color: #9a6a6a;
  }

  .ort-status-text {
    flex: 1;
  }

  .ort-status-detail {
    font-size: 11px;
    color: var(--text-secondary);
    margin-top: 2px;
    word-break: break-all;
  }

  .ort-info {
    background: var(--bg-base);
    border-radius: 6px;
    padding: 12px;
    margin-bottom: 12px;
  }

  .ort-info-row {
    display: flex;
    justify-content: space-between;
    font-size: 13px;
    padding: 4px 0;
  }

  .ort-info-row span:first-child {
    color: var(--text-secondary);
  }

  .ort-runtime-select {
    margin-top: 12px;
  }

  .ort-runtime-label {
    font-size: 12px;
    color: var(--text-secondary);
    margin-bottom: 8px;
  }

  .ort-runtime-option {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 10px;
    background: var(--bg-base);
    border: 1px solid var(--border-color);
    border-radius: 6px;
    margin-bottom: 8px;
    cursor: pointer;
  }

  .ort-runtime-option:hover {
    border-color: #5a5250;
  }

  .ort-runtime-option input[type="radio"] {
    margin-top: 3px;
  }

  .ort-runtime-info {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .ort-runtime-name {
    font-size: 14px;
  }

  .ort-runtime-desc {
    font-size: 12px;
    color: var(--text-secondary);
  }

  .ort-progress {
    margin-top: 12px;
  }

  .ort-progress-text {
    font-size: 13px;
    margin-bottom: 6px;
    color: var(--text-secondary);
  }

  .ort-progress-bar {
    height: 8px;
    background: var(--bg-base);
    border-radius: 4px;
    overflow: hidden;
  }

  .ort-progress-fill {
    height: 100%;
    background: #5a8a5a;
    transition: width 0.2s ease;
  }

  .ort-error {
    margin-top: 12px;
    padding: 10px;
    background: rgba(90, 58, 58, 0.3);
    border: 1px solid #6a4a4a;
    border-radius: 6px;
    color: #ff6b6b;
    font-size: 13px;
  }

  .ort-restart-notice {
    margin-top: 12px;
    padding: 10px;
    background: rgba(58, 90, 90, 0.3);
    border: 1px solid #4a6a6a;
    border-radius: 6px;
    color: #6ac;
    font-size: 13px;
  }

  /* Filter Menu Styles */
  .toolbar-btn.has-filters {
    background: rgba(58, 90, 58, 0.3);
    border-color: #4a6a4a;
  }

  .filter-menu {
    min-width: 200px;
    padding: 8px 0;
  }

  .sort-options {
    display: flex;
    flex-direction: column;
  }

  .sort-direction {
    margin-top: 4px;
    padding-top: 4px;
    border-top: 1px solid var(--border-color);
    display: flex;
    flex-direction: column;
  }

  .date-filters {
    display: flex;
    gap: 8px;
  }

  .filter-label {
    display: flex;
    flex-direction: column;
    gap: 4px;
    flex: 1;
    font-size: 12px;
    color: var(--text-secondary);
  }

  .filter-input {
    padding: 6px 8px;
    background: var(--bg-base);
    border: 1px solid var(--border-color);
    border-radius: 4px;
    color: var(--text-primary);
    font-size: 12px;
    width: 100%;
    box-sizing: border-box;
  }

  .filter-input:focus {
    outline: none;
    border-color: #5a5250;
  }

  .filter-input::-webkit-calendar-picker-indicator {
    filter: invert(0.7);
  }
</style>
