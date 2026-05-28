<script lang="ts">
  import "$lib/theme.css";
  import { invoke, convertFileSrc } from "@tauri-apps/api/core";
  import { onMount } from "svelte";

  interface UmapPoint {
    path: string;
    x: number;
    y: number;
  }
  interface UmapResult {
    points: UmapPoint[];
    computed_at: number;
  }
  interface ClusterAssignment {
    path: string;
    cluster_id: number;
  }
  interface ClusterResult {
    assignments: ClusterAssignment[];
    num_clusters: number;
    num_points: number;
    computed_at: number;
  }

  let umap = $state<UmapResult | null>(null);
  let clusters = $state<ClusterResult | null>(null);
  let loading = $state(true);
  let error = $state<string | null>(null);

  let canvasEl = $state<HTMLCanvasElement | null>(null);
  let tooltipPath = $state<string | null>(null);
  let tooltipThumb = $state<string | null>(null);
  let tooltipX = $state(0);
  let tooltipY = $state(0);
  let thumbnails: Record<string, string | null> = {};

  // Computed bounds and color map
  let bounds = $derived.by(() => {
    if (!umap || umap.points.length === 0)
      return { minX: 0, maxX: 1, minY: 0, maxY: 1 };
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const p of umap.points) {
      if (p.x < minX) minX = p.x;
      if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.y > maxY) maxY = p.y;
    }
    return { minX, maxX, minY, maxY };
  });

  let clusterMap = $derived.by(() => {
    const m = new Map<string, number>();
    if (clusters) {
      for (const a of clusters.assignments) m.set(a.path, a.cluster_id);
    }
    return m;
  });

  // Generate a distinct color per cluster id. Noise (-1) = grey.
  function colorForCluster(id: number | undefined): string {
    if (id === undefined || id === -1) return "#6a615c";
    // Golden-ratio hue stepping for visual distinctness
    const hue = (id * 137.508) % 360;
    return `hsl(${hue}, 70%, 60%)`;
  }

  function project(p: UmapPoint, width: number, height: number, padding = 20) {
    const { minX, maxX, minY, maxY } = bounds;
    const w = maxX - minX || 1;
    const h = maxY - minY || 1;
    const x = padding + ((p.x - minX) / w) * (width - 2 * padding);
    const y = padding + ((p.y - minY) / h) * (height - 2 * padding);
    return { x, y };
  }

  function draw() {
    if (!canvasEl || !umap) return;
    const ctx = canvasEl.getContext("2d");
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    const cssW = canvasEl.clientWidth;
    const cssH = canvasEl.clientHeight;
    if (canvasEl.width !== cssW * dpr || canvasEl.height !== cssH * dpr) {
      canvasEl.width = cssW * dpr;
      canvasEl.height = cssH * dpr;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);

    const radius = 3;
    for (const p of umap.points) {
      const { x, y } = project(p, cssW, cssH);
      ctx.fillStyle = colorForCluster(clusterMap.get(p.path));
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  function findNearest(mouseX: number, mouseY: number): UmapPoint | null {
    if (!umap || !canvasEl) return null;
    const cssW = canvasEl.clientWidth;
    const cssH = canvasEl.clientHeight;
    let best: UmapPoint | null = null;
    let bestDist = 144; // max hover distance squared (12px)
    for (const p of umap.points) {
      const { x, y } = project(p, cssW, cssH);
      const dx = x - mouseX;
      const dy = y - mouseY;
      const d = dx * dx + dy * dy;
      if (d < bestDist) {
        bestDist = d;
        best = p;
      }
    }
    return best;
  }

  async function ensureThumb(path: string): Promise<string | null> {
    if (path in thumbnails) return thumbnails[path];
    try {
      const tp: string = await invoke("get_thumbnail_path", { imagePath: path });
      const url = convertFileSrc(tp);
      thumbnails[path] = url;
      return url;
    } catch {
      thumbnails[path] = null;
      return null;
    }
  }

  async function handleMove(e: MouseEvent) {
    if (!canvasEl) return;
    const rect = canvasEl.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const nearest = findNearest(x, y);
    if (!nearest) {
      tooltipPath = null;
      tooltipThumb = null;
      return;
    }
    tooltipPath = nearest.path;
    tooltipX = e.clientX;
    tooltipY = e.clientY;
    tooltipThumb = await ensureThumb(nearest.path);
  }

  function handleLeave() {
    tooltipPath = null;
    tooltipThumb = null;
  }

  async function handleClick(e: MouseEvent) {
    if (!canvasEl) return;
    const rect = canvasEl.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const nearest = findNearest(x, y);
    if (nearest) {
      await invoke("open_image", { path: nearest.path });
    }
  }

  async function load() {
    loading = true;
    error = null;
    try {
      [umap, clusters] = await Promise.all([
        invoke<UmapResult | null>("get_umap_result"),
        invoke<ClusterResult | null>("get_cluster_result"),
      ]);
    } catch (e) {
      error = String(e);
    } finally {
      loading = false;
      requestAnimationFrame(draw);
    }
  }

  async function recompute() {
    loading = true;
    error = null;
    try {
      umap = await invoke<UmapResult>("compute_umap");
    } catch (e) {
      error = String(e);
    } finally {
      loading = false;
      requestAnimationFrame(draw);
    }
  }

  function getFilename(p: string): string {
    return p.split(/[\\/]/).pop() || p;
  }

  function formatTime(ms: number): string {
    return new Date(ms).toLocaleString();
  }

  // Redraw when data changes
  $effect(() => {
    umap;
    clusterMap;
    requestAnimationFrame(draw);
  });

  onMount(() => {
    load();
    const onResize = () => requestAnimationFrame(draw);
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  });
</script>

<div class="page">
  <header class="toolbar">
    <h1 class="title">
      2D Map
      {#if umap}
        <span class="sub">
          ({umap.points.length} images, computed {formatTime(umap.computed_at)})
        </span>
      {/if}
    </h1>
    <button class="btn" onclick={recompute} disabled={loading}>
      {umap ? "Recompute" : "Compute 2D Map"}
    </button>
  </header>

  <main class="content">
    {#if error}
      <div class="message error">{error}</div>
    {:else if loading}
      <div class="message">Working…</div>
    {:else if !umap}
      <div class="message">
        No 2D map computed yet. Click <b>Compute 2D Map</b> to project your images into a 2D layout where similar images sit near each other.
      </div>
    {:else}
      <canvas
        bind:this={canvasEl}
        onmousemove={handleMove}
        onmouseleave={handleLeave}
        onclick={handleClick}
      ></canvas>
    {/if}
  </main>

  {#if tooltipPath}
    <div
      class="tooltip"
      style="left: {tooltipX + 12}px; top: {tooltipY + 12}px;"
    >
      {#if tooltipThumb}
        <img src={tooltipThumb} alt="" class="tooltip-thumb" />
      {/if}
      <div class="tooltip-text">{getFilename(tooltipPath)}</div>
    </div>
  {/if}
</div>

<style>
  .page {
    display: flex;
    flex-direction: column;
    height: 100vh;
    width: 100vw;
  }
  .toolbar {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 12px;
    background: var(--bg-toolbar);
    border-bottom: 1px solid var(--border-color);
    flex-shrink: 0;
  }
  .title {
    margin: 0;
    font-size: 14px;
    font-weight: 500;
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .sub {
    color: var(--text-secondary);
    font-size: 12px;
    font-weight: 400;
    margin-left: 6px;
  }
  .btn {
    background: var(--bg-hover);
    border: 1px solid var(--border-color);
    border-radius: 4px;
    color: var(--text-primary);
    padding: 6px 12px;
    cursor: pointer;
    font-size: 13px;
  }
  .btn:hover:not(:disabled) {
    background: var(--border-color);
  }
  .btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  .content {
    flex: 1;
    position: relative;
    overflow: hidden;
  }
  canvas {
    width: 100%;
    height: 100%;
    display: block;
    cursor: crosshair;
  }
  .message {
    padding: 24px;
    color: var(--text-secondary);
    text-align: center;
  }
  .message.error {
    color: #ff6b6b;
  }
  .tooltip {
    position: fixed;
    z-index: 100;
    background: var(--bg-toolbar);
    border: 1px solid var(--border-color);
    border-radius: 4px;
    padding: 6px;
    pointer-events: none;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    max-width: 200px;
  }
  .tooltip-thumb {
    display: block;
    width: 160px;
    height: 160px;
    object-fit: cover;
    border-radius: 2px;
  }
  .tooltip-text {
    margin-top: 4px;
    font-size: 11px;
    color: var(--text-primary);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
</style>
