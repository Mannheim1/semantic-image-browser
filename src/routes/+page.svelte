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

  interface ScanProgressPayload {
    phase: string;
    current: number;
    total: number;
  }

  interface RuntimeInfo {
    runtime_type: string;
    display_name: string;
    installed: boolean;
    available: boolean;
    download_size: number | null;
    installed_size: number | null;
  }

  interface OrtStatus {
    selected_runtime: string | null;
    selected_installed: boolean;
    library_path: string | null;
    runtimes: RuntimeInfo[];
    platform: string;
  }

  interface OrtDownloadProgress {
    downloaded: number;
    total: number;
  }

  interface CudaDependency {
    name: string;
    found: boolean;
    details: string | null;
  }

  interface CudaDependencyStatus {
    all_found: boolean;
    dependencies: CudaDependency[];
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
  let ortError = $state<string | null>(null);
  // Track desired install state for each runtime (for the checklist)
  let desiredInstallState = $state<Record<string, boolean>>({});
  let cudaDependencies = $state<CudaDependencyStatus | null>(null);
  let showCudaGuide = $state(false);

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

  // Hide the search value during scanning/loading so placeholder can show progress
  let displaySearchValue = $derived(
    isScanning || modelLoading
      ? ""
      : (similarToImage ? `similar to: ${getFilename(similarToImage.path)}` : searchQuery)
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
    ortError = null;
    ortDownloadProgress = null;
    showCudaGuide = false;
    // Refresh status
    ortStatus = await invoke("get_ort_status");
    // Initialize desired state from current installed state
    desiredInstallState = {};
    for (const rt of ortStatus?.runtimes ?? []) {
      desiredInstallState[rt.runtime_type] = rt.installed;
    }
    // Check CUDA dependencies
    cudaDependencies = await invoke("check_cuda_dependencies");
  }

  let ortNeedsRestart = $state(false);

  // Check if desired state differs from current state
  function hasChanges(): boolean {
    if (!ortStatus) return false;
    for (const rt of ortStatus.runtimes) {
      if (desiredInstallState[rt.runtime_type] !== rt.installed) {
        return true;
      }
    }
    return false;
  }

  // Get summary of changes to apply
  function getChangesSummary(): { toInstall: string[], toUninstall: string[] } {
    const toInstall: string[] = [];
    const toUninstall: string[] = [];
    if (!ortStatus) return { toInstall, toUninstall };

    for (const rt of ortStatus.runtimes) {
      const desired = desiredInstallState[rt.runtime_type];
      if (desired && !rt.installed) {
        toInstall.push(rt.display_name);
      } else if (!desired && rt.installed) {
        toUninstall.push(rt.display_name);
      }
    }
    return { toInstall, toUninstall };
  }

  async function applyRuntimeChanges() {
    if (!ortStatus) return;

    ortDownloading = true;
    ortError = null;

    try {
      // Process uninstalls first
      for (const rt of ortStatus.runtimes) {
        const desired = desiredInstallState[rt.runtime_type];
        if (!desired && rt.installed) {
          await invoke("uninstall_runtime", { runtimeType: rt.runtime_type });
        }
      }

      // Then process installs
      for (const rt of ortStatus.runtimes) {
        const desired = desiredInstallState[rt.runtime_type];
        if (desired && !rt.installed && rt.available) {
          ortDownloadProgress = { downloaded: 0, total: 0 };
          await invoke("download_ort", { runtimeType: rt.runtime_type });
        }
      }

      // Refresh status
      ortStatus = await invoke("get_ort_status");
      ortDownloadProgress = null;

      // Update desired state to match new reality
      for (const rt of ortStatus?.runtimes ?? []) {
        desiredInstallState[rt.runtime_type] = rt.installed;
      }

      ortNeedsRestart = true;
    } catch (e) {
      ortError = String(e);
      // Refresh status to show actual state
      ortStatus = await invoke("get_ort_status");
    }

    ortDownloading = false;
  }

  async function handleRuntimeMenuClick(runtimeType: string) {
    // Refresh status to get current installation state
    ortStatus = await invoke("get_ort_status");
    const runtime = ortStatus?.runtimes?.find(r => r.runtime_type === runtimeType);

    if (!runtime?.available) {
      // DirectML not yet available
      return;
    }

    if (runtime?.installed) {
      // Runtime is installed - switch to it
      try {
        await invoke("set_runtime_type", { runtimeType });
        ortStatus = await invoke("get_ort_status");
        ortNeedsRestart = true;
        // Show the modal to display the restart notice
        showOrtModal = true;
      } catch (e) {
        console.error("Failed to set runtime type:", e);
      }
    } else {
      // Runtime not installed - open modal to install
      openOrtModal();
      desiredInstallState[runtimeType] = true;
    }
  }

  async function addDirectory() {
    const selected = await open({
      directory: true,
      multiple: false,
      title: "Select folder to watch"
    });

    if (selected) {
      isScanning = true;
      scanProgress = { phase: "thumbnails", current: 0, total: 0 };
      const start = performance.now();
      await invoke("add_watched_directory", { path: selected });
      lastScanDurationMs = performance.now() - start;
      console.log(`Scan completed in ${formatDuration(lastScanDurationMs)}`);
      scanProgress = null;
      isScanning = false;
      await loadInitialData();
    }
  }

  async function removeDirectory(path: string) {
    await invoke("remove_watched_directory", { path });
    await loadInitialData();
  }

  async function rescanAll() {
    isScanning = true;
    scanProgress = { phase: "thumbnails", current: 0, total: 0 };
    const start = performance.now();
    await invoke("rescan_all");
    lastScanDurationMs = performance.now() - start;
    console.log(`Scan completed in ${formatDuration(lastScanDurationMs)}`);
    scanProgress = null;
    isScanning = false;
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

    const unlistenOrtPromise = listen<OrtDownloadProgress>("ort_download_progress", (event) => {
      ortDownloadProgress = event.payload;
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
      unlistenOrtPromise.then((unlisten) => unlisten());
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
      case "model_settings":
        openOrtModal();
        break;
      case "runtime_cpu":
        handleRuntimeMenuClick("cpu");
        break;
      case "runtime_directml":
        handleRuntimeMenuClick("directml");
        break;
      case "runtime_cuda":
        handleRuntimeMenuClick("cuda");
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
        placeholder={modelLoading
          ? "Loading model..."
          : isScanning && scanProgress && scanProgress.total > 0
            ? (scanProgress.phase === "thumbnails"
              ? `Creating ${scanProgress.current}/${scanProgress.total} thumbnails...`
              : `Scanning ${scanProgress.current}/${scanProgress.total} images...`)
            : `Search ${indexedCount} images...`}
        value={displaySearchValue}
        oninput={handleSearchInput}
        onfocus={() => { if (similarToImage) clearSimilarSearch(); }}
        disabled={isScanning || modelLoading}
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
        class:disabled={!embeddingModelLoaded}
        disabled={!embeddingModelLoaded}
        onclick={() => { findSimilar(contextMenu!.image); closeContextMenu(); }}
      >
        Find similar
      </button>
    </div>
  {/if}

  {#if showOrtModal}
    <div class="modal-overlay" onclick={() => { if (!ortDownloading) showOrtModal = false; }}>
      <div class="modal modal-wide" onclick={(e) => e.stopPropagation()}>
        <div class="modal-header">
          <span>Runtime Settings</span>
          <button class="modal-close" onclick={() => { if (!ortDownloading) showOrtModal = false; }} disabled={ortDownloading}>×</button>
        </div>
        <div class="modal-body">
          <!-- Installed Runtimes Section -->
          <div class="ort-section">
            <div class="ort-section-title">Installed Runtimes</div>
            <div class="ort-runtime-list">
              {#each ortStatus?.runtimes ?? [] as runtime}
                <label class="ort-runtime-row" class:disabled={!runtime.available}>
                  <input
                    type="checkbox"
                    checked={desiredInstallState[runtime.runtime_type] ?? false}
                    disabled={!runtime.available || ortDownloading}
                    onchange={(e) => {
                      desiredInstallState[runtime.runtime_type] = (e.target as HTMLInputElement).checked;
                      desiredInstallState = { ...desiredInstallState };
                    }}
                  />
                  <span class="ort-runtime-name">{runtime.display_name}</span>
                  <span class="ort-runtime-size">
                    {#if !runtime.available}
                      <span class="ort-coming-soon">Coming soon</span>
                    {:else if runtime.installed && runtime.installed_size}
                      {formatBytes(runtime.installed_size)}
                    {:else if runtime.download_size}
                      ~{formatBytes(runtime.download_size)}
                    {/if}
                  </span>
                </label>
              {/each}
            </div>

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

            {#if ortError}
              <div class="ort-error">
                Error: {ortError}
              </div>
            {/if}

            {#if ortNeedsRestart}
              <div class="ort-restart-notice">
                Changes applied. Please restart the app for them to take effect.
              </div>
            {/if}

            <div class="ort-apply-row">
              <button
                class="modal-btn modal-btn-primary"
                onclick={applyRuntimeChanges}
                disabled={ortDownloading || !hasChanges()}
              >
                {#if ortDownloading}
                  Applying...
                {:else}
                  {@const changes = getChangesSummary()}
                  {#if changes.toInstall.length > 0 && changes.toUninstall.length > 0}
                    Install {changes.toInstall.length} & Uninstall {changes.toUninstall.length}
                  {:else if changes.toInstall.length > 0}
                    Install {changes.toInstall.length} runtime{changes.toInstall.length > 1 ? 's' : ''}
                  {:else if changes.toUninstall.length > 0}
                    Uninstall {changes.toUninstall.length} runtime{changes.toUninstall.length > 1 ? 's' : ''}
                  {:else}
                    Apply Changes
                  {/if}
                {/if}
              </button>
            </div>
          </div>

          <!-- CUDA Dependencies Section -->
          {#if (desiredInstallState['cuda'] || ortStatus?.runtimes?.find(r => r.runtime_type === 'cuda')?.installed)}
            <div class="ort-section">
              <div class="ort-section-title">CUDA System Dependencies</div>
              <div class="ort-dep-list">
                {#each cudaDependencies?.dependencies ?? [] as dep}
                  <div class="ort-dep-row">
                    <span class="ort-dep-icon" class:found={dep.found} class:missing={!dep.found}>
                      {dep.found ? '✓' : '✗'}
                    </span>
                    <span class="ort-dep-name">{dep.name}</span>
                    {#if !dep.found && dep.details}
                      <span class="ort-dep-detail">{dep.details}</span>
                    {/if}
                  </div>
                {/each}
              </div>

              {#if !cudaDependencies?.all_found}
                <button
                  class="ort-guide-toggle"
                  onclick={() => showCudaGuide = !showCudaGuide}
                >
                  {showCudaGuide ? 'Hide' : 'Show'} installation guide
                </button>

                {#if showCudaGuide}
                  <div class="ort-guide">
                    <p><strong>To use the CUDA runtime, install these dependencies:</strong></p>
                    <ol>
                      <li>
                        <strong>CUDA Toolkit 12.x</strong><br>
                        Download from <a href="https://developer.nvidia.com/cuda-downloads" target="_blank">NVIDIA CUDA Downloads</a>
                      </li>
                      <li>
                        <strong>cuDNN 9.x</strong><br>
                        Download from <a href="https://developer.nvidia.com/cudnn" target="_blank">NVIDIA cuDNN</a> (requires NVIDIA account)<br>
                        Extract and add the <code>bin</code> folder to your system PATH
                      </li>
                    </ol>
                    <p>After installing, restart this app and reopen this dialog to verify.</p>
                  </div>
                {/if}
              {/if}
            </div>
          {/if}

          <div class="ort-info-note">
            <span class="ort-info-icon">ℹ</span>
            Use <strong>Model → Select Runtime</strong> in the menu bar to choose which runtime to use.
          </div>
        </div>
        <div class="modal-footer">
          <div class="modal-buttons">
            <button
              class="modal-btn"
              onclick={() => showOrtModal = false}
              disabled={ortDownloading}
            >
              Close
            </button>
          </div>
        </div>
      </div>
    </div>
  {/if}

  {#if showFoldersModal}
    <div class="modal-overlay" onclick={() => showFoldersModal = false}>
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
    <div class="modal-overlay" onclick={() => showAboutModal = false}>
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
    <div class="modal-overlay" onclick={() => showViewControlsModal = false}>
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

  .modal-footer {
    padding: 12px 16px;
    border-top: 1px solid var(--border-color);
    display: flex;
    flex-direction: column;
    gap: 8px;
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
  .modal-wide {
    width: 480px;
  }

  .ort-section {
    background: var(--bg-base);
    border-radius: 6px;
    padding: 16px;
    margin-bottom: 12px;
  }

  .ort-section-title {
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 12px;
    color: var(--text-primary);
  }

  .ort-runtime-list {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .ort-runtime-row {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 10px;
    border-radius: 4px;
    cursor: pointer;
  }

  .ort-runtime-row:hover:not(.disabled) {
    background: rgba(255, 255, 255, 0.05);
  }

  .ort-runtime-row.disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .ort-runtime-row input[type="checkbox"] {
    width: 16px;
    height: 16px;
    cursor: pointer;
  }

  .ort-runtime-row input[type="checkbox"]:disabled {
    cursor: not-allowed;
  }

  .ort-runtime-name {
    flex: 1;
    font-size: 14px;
  }

  .ort-runtime-size {
    font-size: 12px;
    color: var(--text-secondary);
    min-width: 80px;
    text-align: right;
  }

  .ort-coming-soon {
    font-style: italic;
    color: var(--text-secondary);
  }

  .ort-apply-row {
    margin-top: 12px;
    display: flex;
    justify-content: flex-end;
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
    background: rgba(0, 0, 0, 0.3);
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

  /* CUDA Dependencies Styles */
  .ort-dep-list {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .ort-dep-row {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 10px;
    font-size: 13px;
  }

  .ort-dep-icon {
    width: 18px;
    text-align: center;
    font-weight: bold;
  }

  .ort-dep-icon.found {
    color: #4ade80;
  }

  .ort-dep-icon.missing {
    color: #f87171;
  }

  .ort-dep-name {
    flex: 1;
  }

  .ort-dep-detail {
    font-size: 11px;
    color: var(--text-secondary);
  }

  .ort-guide-toggle {
    margin-top: 12px;
    background: none;
    border: none;
    color: #6ac;
    cursor: pointer;
    font-size: 13px;
    padding: 0;
    text-decoration: underline;
  }

  .ort-guide-toggle:hover {
    color: #8ce;
  }

  .ort-guide {
    margin-top: 12px;
    padding: 12px;
    background: rgba(0, 0, 0, 0.2);
    border-radius: 4px;
    font-size: 13px;
    line-height: 1.5;
  }

  .ort-guide p {
    margin: 0 0 8px 0;
  }

  .ort-guide ol {
    margin: 0;
    padding-left: 20px;
  }

  .ort-guide li {
    margin-bottom: 8px;
  }

  .ort-guide a {
    color: #6ac;
  }

  .ort-guide a:hover {
    color: #8ce;
  }

  .ort-guide code {
    background: rgba(0, 0, 0, 0.3);
    padding: 2px 6px;
    border-radius: 3px;
    font-family: monospace;
  }

  .ort-info-note {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    padding: 10px 12px;
    background: rgba(100, 150, 200, 0.1);
    border: 1px solid rgba(100, 150, 200, 0.3);
    border-radius: 6px;
    font-size: 12px;
    color: var(--text-secondary);
  }

  .ort-info-icon {
    font-size: 14px;
    color: #6ac;
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
