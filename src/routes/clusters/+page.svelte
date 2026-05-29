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

  interface ClusterGroup {
    id: number;
    label: string;
    points: ClusterPoint[];
  }

  let result = $state<ClusterResult | null>(null);
  let groups = $state<ClusterGroup[]>([]);
  let thumbnails = $state<Record<string, string | null>>({});
  let loading = $state(true);

  function getFilename(path: string): string {
    return path.split(/[\\/]/).pop() ?? path;
  }

  function buildGroups(res: ClusterResult): ClusterGroup[] {
    const byCluster = new Map<number, ClusterPoint[]>();
    for (const p of res.points) {
      const arr = byCluster.get(p.cluster);
      if (arr) arr.push(p);
      else byCluster.set(p.cluster, [p]);
    }
    // Real clusters first (ascending id), then the unclustered bucket last.
    const ids = [...byCluster.keys()].filter((id) => id >= 0).sort((a, b) => a - b);
    const out: ClusterGroup[] = ids.map((id) => ({
      id,
      label: `Cluster ${id + 1}`,
      points: byCluster.get(id)!,
    }));
    const noise = byCluster.get(-1);
    if (noise && noise.length > 0) {
      out.push({ id: -1, label: "Unclustered", points: noise });
    }
    return out;
  }

  async function loadThumbnails(points: ClusterPoint[]) {
    const next: Record<string, string | null> = { ...thumbnails };
    const missing = points.filter((p) => !(p.path in next));
    await Promise.all(
      missing.map(async (p) => {
        try {
          const thumbPath: string = await invoke("get_thumbnail_path", { imagePath: p.path });
          next[p.path] = convertFileSrc(thumbPath);
        } catch {
          next[p.path] = null;
        }
      })
    );
    thumbnails = next;
  }

  async function load() {
    loading = true;
    try {
      result = await invoke("get_cluster_result");
      if (result) {
        groups = buildGroups(result);
        await loadThumbnails(result.points);
      } else {
        groups = [];
      }
    } finally {
      loading = false;
    }
  }

  async function openImage(path: string) {
    await invoke("open_image", { path });
  }

  onMount(() => {
    load();
    const unlisten = listen<void>("clusters_ready", () => load());
    return () => {
      unlisten.then((u) => u());
    };
  });
</script>

<div class="clusters">
  {#if loading}
    <div class="empty">Loading clusters…</div>
  {:else if !result}
    <div class="empty">
      No clusters yet. In the main window, choose <strong>Clusters → Compute Clusters</strong> first.
    </div>
  {:else if groups.length === 0}
    <div class="empty">No clusters found.</div>
  {:else}
    <div class="summary">
      {result.num_clusters} clusters · {result.num_noise} unclustered · {result.points.length} images
    </div>
    {#each groups as group (group.id)}
      <section class="cluster-section">
        <h2 class="cluster-title">
          {group.label}
          <span class="count">{group.points.length}</span>
        </h2>
        <div class="image-grid">
          {#each group.points as p (p.path)}
            <button
              class="image-cell"
              title={p.path}
              ondblclick={() => openImage(p.path)}
            >
              {#if thumbnails[p.path]}
                <img src={thumbnails[p.path]} alt="" class="thumbnail" />
              {:else if thumbnails[p.path] === null}
                <div class="thumbnail-placeholder">!</div>
              {:else}
                <div class="thumbnail-placeholder"></div>
              {/if}
              <span class="filename">{getFilename(p.path)}</span>
            </button>
          {/each}
        </div>
      </section>
    {/each}
  {/if}
</div>

<style>
  .clusters {
    height: 100vh;
    overflow-y: auto;
    padding: 12px 16px;
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
    font-size: 13px;
    margin-bottom: 12px;
  }

  .cluster-section {
    margin-bottom: 20px;
  }

  .cluster-title {
    font-size: 15px;
    margin: 0 0 8px 0;
    padding-bottom: 4px;
    border-bottom: 1px solid var(--border-color);
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .count {
    font-size: 12px;
    color: var(--text-secondary);
    font-weight: normal;
  }

  .image-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(110px, 1fr));
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

  .thumbnail {
    width: 100px;
    height: 100px;
    object-fit: cover;
    border-radius: 2px;
  }

  .thumbnail-placeholder {
    width: 100px;
    height: 100px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: var(--bg-toolbar);
    color: var(--text-secondary);
    border-radius: 2px;
  }

  .filename {
    margin-top: 2px;
    font-size: 11px;
    max-width: 100px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
</style>
