<script lang="ts">
  import "$lib/theme.css";
  import { invoke, convertFileSrc } from "@tauri-apps/api/core";
  import { onMount } from "svelte";

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

  let result = $state<ClusterResult | null>(null);
  let loading = $state(true);
  let error = $state<string | null>(null);
  let thumbnails = $state<Record<string, string | null>>({});
  let selectedClusterId = $state<number | null>(null);

  // Group assignments by cluster_id
  let clusters = $derived.by(() => {
    if (!result) return [] as { id: number; paths: string[] }[];
    const map = new Map<number, string[]>();
    for (const a of result.assignments) {
      const arr = map.get(a.cluster_id);
      if (arr) arr.push(a.path);
      else map.set(a.cluster_id, [a.path]);
    }
    // Sort: real clusters by size desc, noise (-1) last
    return Array.from(map.entries())
      .map(([id, paths]) => ({ id, paths }))
      .sort((a, b) => {
        if (a.id === -1) return 1;
        if (b.id === -1) return -1;
        return b.paths.length - a.paths.length;
      });
  });

  let selectedCluster = $derived(
    selectedClusterId === null
      ? null
      : clusters.find((c) => c.id === selectedClusterId) ?? null
  );

  async function loadThumbs(paths: string[]) {
    const missing = paths.filter((p) => !(p in thumbnails));
    if (missing.length === 0) return;
    const next: Record<string, string | null> = { ...thumbnails };
    await Promise.all(
      missing.map(async (p) => {
        try {
          const tp: string = await invoke("get_thumbnail_path", { imagePath: p });
          next[p] = convertFileSrc(tp);
        } catch {
          next[p] = null;
        }
      })
    );
    thumbnails = next;
  }

  async function load() {
    loading = true;
    error = null;
    try {
      result = await invoke<ClusterResult | null>("get_cluster_result");
      if (result) {
        // Prefetch cover thumbnails (one per cluster)
        const covers = clusters.slice(0, 200).map((c) => c.paths[0]);
        await loadThumbs(covers);
      }
    } catch (e) {
      error = String(e);
    } finally {
      loading = false;
    }
  }

  async function recompute() {
    loading = true;
    error = null;
    try {
      result = await invoke<ClusterResult>("compute_clusters");
      selectedClusterId = null;
      thumbnails = {};
      const covers = clusters.slice(0, 200).map((c) => c.paths[0]);
      await loadThumbs(covers);
    } catch (e) {
      error = String(e);
    } finally {
      loading = false;
    }
  }

  async function openCluster(id: number) {
    selectedClusterId = id;
    const cluster = clusters.find((c) => c.id === id);
    if (cluster) {
      await loadThumbs(cluster.paths);
    }
  }

  function backToOverview() {
    selectedClusterId = null;
  }

  async function openImage(path: string) {
    await invoke("open_image", { path });
  }

  function getFilename(p: string): string {
    return p.split(/[\\/]/).pop() || p;
  }

  function formatTime(ms: number): string {
    return new Date(ms).toLocaleString();
  }

  onMount(() => {
    load();
  });
</script>

<div class="page">
  <header class="toolbar">
    {#if selectedCluster !== null}
      <button class="btn" onclick={backToOverview}>← Back</button>
      <h1 class="title">
        Cluster {selectedCluster.id === -1 ? "Noise" : selectedCluster.id}
        <span class="sub">({selectedCluster.paths.length} images)</span>
      </h1>
    {:else}
      <h1 class="title">
        Clusters
        {#if result}
          <span class="sub">
            ({result.num_clusters} clusters, {result.num_points} images,
            computed {formatTime(result.computed_at)})
          </span>
        {/if}
      </h1>
      <button class="btn" onclick={recompute} disabled={loading}>
        {result ? "Recompute" : "Compute Clusters"}
      </button>
    {/if}
  </header>

  <main class="content">
    {#if error}
      <div class="message error">{error}</div>
    {:else if loading}
      <div class="message">Working…</div>
    {:else if !result}
      <div class="message">
        No clusters computed yet. Click <b>Compute Clusters</b> to group your images by visual similarity.
      </div>
    {:else if selectedCluster !== null}
      <div class="grid">
        {#each selectedCluster.paths as p}
          <button class="cell" ondblclick={() => openImage(p)} onclick={() => openImage(p)}>
            {#if thumbnails[p]}
              <img src={thumbnails[p]} alt="" class="thumb" />
            {:else if thumbnails[p] === null}
              <div class="thumb placeholder">!</div>
            {:else}
              <div class="thumb placeholder"></div>
            {/if}
            <span class="caption">{getFilename(p)}</span>
          </button>
        {/each}
      </div>
    {:else}
      <div class="grid">
        {#each clusters as c}
          <button class="cell cluster-cell" onclick={() => openCluster(c.id)}>
            {#if thumbnails[c.paths[0]]}
              <img src={thumbnails[c.paths[0]]} alt="" class="thumb" />
            {:else if thumbnails[c.paths[0]] === null}
              <div class="thumb placeholder">!</div>
            {:else}
              <div class="thumb placeholder"></div>
            {/if}
            <span class="caption">
              {c.id === -1 ? "Noise" : `Cluster ${c.id}`}
              <span class="count">{c.paths.length}</span>
            </span>
          </button>
        {/each}
      </div>
    {/if}
  </main>
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
    overflow-y: auto;
    overflow-x: hidden;
  }
  .message {
    padding: 24px;
    color: var(--text-secondary);
    text-align: center;
  }
  .message.error {
    color: #ff6b6b;
  }
  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
    gap: 4px;
    padding: 4px;
  }
  .cell {
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
  .cell:hover {
    background: var(--bg-hover);
  }
  .thumb {
    width: 100%;
    aspect-ratio: 1;
    object-fit: cover;
    border-radius: 2px;
  }
  .thumb.placeholder {
    background: var(--bg-toolbar);
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--text-secondary);
  }
  .caption {
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
  .cluster-cell .caption {
    color: var(--text-primary);
    font-size: 12px;
  }
  .count {
    margin-left: 6px;
    color: var(--text-secondary);
    font-size: 11px;
  }
</style>
