<script lang="ts">
  import "$lib/theme.css";
  import { invoke, convertFileSrc } from "@tauri-apps/api/core";
  import { listen } from "@tauri-apps/api/event";
  import { onMount } from "svelte";

  interface ClusterPoint {
    path: string;
    x: number;
    y: number;
    cluster: number;
  }
  interface ClusterResult {
    points: ClusterPoint[];
    num_clusters: number;
    num_noise: number;
  }

  interface PixelPoint {
    px: number;
    py: number;
    path: string;
    cluster: number;
  }

  let result = $state<ClusterResult | null>(null);
  let loading = $state(true);

  let canvasEl = $state<HTMLCanvasElement | null>(null);
  let containerEl = $state<HTMLDivElement | null>(null);
  let pixelPoints: PixelPoint[] = [];

  let hovered = $state<{ path: string; cluster: number; x: number; y: number } | null>(null);
  let hoverThumb = $state<string | null>(null);
  const thumbCache = new Map<string, string | null>();

  const PADDING = 26;
  const DOT_RADIUS = 3;
  const HIT_RADIUS = 9;

  function getFilename(path: string): string {
    return path.split(/[\\/]/).pop() ?? path;
  }

  // Distinct, stable colour per cluster using the golden-angle hue rotation.
  // Noise (-1) is rendered as a muted grey.
  function clusterColor(cluster: number): string {
    if (cluster < 0) return "#6a615c";
    const hue = (cluster * 137.508) % 360;
    return `hsl(${hue}, 65%, 58%)`;
  }

  function draw() {
    const canvas = canvasEl;
    const container = containerEl;
    if (!canvas || !container) return;

    const w = container.clientWidth;
    const h = container.clientHeight;
    canvas.width = w;
    canvas.height = h;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, w, h);

    pixelPoints = [];
    if (!result || result.points.length === 0) return;

    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const p of result.points) {
      if (p.x < minX) minX = p.x;
      if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.y > maxY) maxY = p.y;
    }
    const rangeX = maxX - minX || 1;
    const rangeY = maxY - minY || 1;
    const scaleX = (w - 2 * PADDING) / rangeX;
    const scaleY = (h - 2 * PADDING) / rangeY;

    for (const p of result.points) {
      const px = PADDING + (p.x - minX) * scaleX;
      // Flip Y so the plot reads bottom-up like a normal chart.
      const py = PADDING + (maxY - p.y) * scaleY;
      pixelPoints.push({ px, py, path: p.path, cluster: p.cluster });

      ctx.beginPath();
      ctx.arc(px, py, DOT_RADIUS, 0, Math.PI * 2);
      ctx.fillStyle = clusterColor(p.cluster);
      ctx.fill();
    }
  }

  async function load() {
    loading = true;
    try {
      result = await invoke("get_cluster_result");
    } finally {
      loading = false;
    }
    // Wait for the canvas to exist after the conditional renders.
    requestAnimationFrame(draw);
  }

  function nearestPoint(mx: number, my: number): PixelPoint | null {
    let best: PixelPoint | null = null;
    let bestDist = HIT_RADIUS * HIT_RADIUS;
    for (const p of pixelPoints) {
      const dx = p.px - mx;
      const dy = p.py - my;
      const d = dx * dx + dy * dy;
      if (d <= bestDist) {
        bestDist = d;
        best = p;
      }
    }
    return best;
  }

  async function loadHoverThumb(path: string) {
    if (thumbCache.has(path)) {
      hoverThumb = thumbCache.get(path) ?? null;
      return;
    }
    try {
      const thumbPath: string = await invoke("get_thumbnail_path", { imagePath: path });
      const url = convertFileSrc(thumbPath);
      thumbCache.set(path, url);
      if (hovered?.path === path) hoverThumb = url;
    } catch {
      thumbCache.set(path, null);
      if (hovered?.path === path) hoverThumb = null;
    }
  }

  function handleMouseMove(e: MouseEvent) {
    const canvas = canvasEl;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const p = nearestPoint(mx, my);
    if (p) {
      if (hovered?.path !== p.path) {
        hoverThumb = null;
        loadHoverThumb(p.path);
      }
      hovered = { path: p.path, cluster: p.cluster, x: mx, y: my };
      canvas.style.cursor = "pointer";
    } else {
      hovered = null;
      hoverThumb = null;
      canvas.style.cursor = "default";
    }
  }

  function handleMouseLeave() {
    hovered = null;
    hoverThumb = null;
  }

  async function handleClick(e: MouseEvent) {
    const canvas = canvasEl;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const p = nearestPoint(e.clientX - rect.left, e.clientY - rect.top);
    if (p) await invoke("open_image", { path: p.path });
  }

  onMount(() => {
    load();
    const unlisten = listen<void>("clusters_ready", () => load());
    const onResize = () => draw();
    window.addEventListener("resize", onResize);
    return () => {
      unlisten.then((u) => u());
      window.removeEventListener("resize", onResize);
    };
  });
</script>

<div class="map-page">
  {#if loading}
    <div class="empty">Loading map…</div>
  {:else if !result}
    <div class="empty">
      No clusters yet. In the main window, choose <strong>Clusters → Compute Clusters</strong> first.
    </div>
  {:else}
    <div class="summary">
      {result.num_clusters} clusters · {result.points.length} images · hover to preview, click to open
    </div>
    <div class="canvas-wrap" bind:this={containerEl}>
      <canvas
        bind:this={canvasEl}
        onmousemove={handleMouseMove}
        onmouseleave={handleMouseLeave}
        onclick={handleClick}
      ></canvas>

      {#if hovered}
        <div
          class="tooltip"
          style="left: {hovered.x + 14}px; top: {hovered.y + 14}px;"
        >
          <div class="tooltip-thumb">
            {#if hoverThumb}
              <img src={hoverThumb} alt="" />
            {:else}
              <div class="tooltip-placeholder"></div>
            {/if}
          </div>
          <div class="tooltip-meta">
            <div class="tooltip-name">{getFilename(hovered.path)}</div>
            <div class="tooltip-cluster">
              {hovered.cluster < 0 ? "Unclustered" : `Cluster ${hovered.cluster + 1}`}
            </div>
          </div>
        </div>
      {/if}
    </div>
  {/if}
</div>

<style>
  .map-page {
    display: flex;
    flex-direction: column;
    height: 100vh;
    box-sizing: border-box;
  }

  .empty {
    color: var(--text-secondary);
    text-align: center;
    padding: 40px 20px;
    line-height: 1.6;
  }

  .summary {
    color: var(--text-secondary);
    font-size: 12px;
    padding: 8px 12px;
    border-bottom: 1px solid var(--border-color);
    flex-shrink: 0;
  }

  .canvas-wrap {
    position: relative;
    flex: 1;
    min-height: 0;
    overflow: hidden;
  }

  canvas {
    display: block;
    width: 100%;
    height: 100%;
  }

  .tooltip {
    position: absolute;
    pointer-events: none;
    display: flex;
    gap: 8px;
    padding: 6px;
    background: var(--bg-toolbar);
    border: 1px solid var(--border-color);
    border-radius: 4px;
    max-width: 220px;
    z-index: 10;
  }

  .tooltip-thumb img,
  .tooltip-placeholder {
    width: 64px;
    height: 64px;
    object-fit: cover;
    border-radius: 2px;
    display: block;
  }

  .tooltip-placeholder {
    background: var(--bg-base);
  }

  .tooltip-meta {
    display: flex;
    flex-direction: column;
    justify-content: center;
    min-width: 0;
  }

  .tooltip-name {
    font-size: 12px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .tooltip-cluster {
    font-size: 11px;
    color: var(--text-secondary);
  }
</style>
