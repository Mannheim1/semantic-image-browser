<script lang="ts">
  import "$lib/theme.css";
  import { invoke, convertFileSrc } from "@tauri-apps/api/core";
  import { emit, listen } from "@tauri-apps/api/event";
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
    count: number;
    // Four images sampled at random to represent the cluster on its card.
    sample: ClusterPoint[];
  }

  let result = $state<ClusterResult | null>(null);
  let groups = $state<ClusterGroup[]>([]);
  let thumbnails = $state<Record<string, string | null>>({});
  let loading = $state(true);

  // Pick up to four random points to feature on a cluster card. Chosen once when
  // the groups are built so the collage doesn't reshuffle on every redraw.
  function sample4(points: ClusterPoint[]): ClusterPoint[] {
    if (points.length <= 4) return points.slice();
    const chosen = new Set<number>();
    while (chosen.size < 4) chosen.add(Math.floor(Math.random() * points.length));
    return [...chosen].map((i) => points[i]);
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
    const out: ClusterGroup[] = ids.map((id) => {
      const points = byCluster.get(id)!;
      return { id, label: `Cluster ${id + 1}`, count: points.length, sample: sample4(points) };
    });
    const noise = byCluster.get(-1);
    if (noise && noise.length > 0) {
      out.push({ id: -1, label: "Unclustered", count: noise.length, sample: sample4(noise) });
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
        // Only the few featured images per card need thumbnails here.
        await loadThumbnails(groups.flatMap((g) => g.sample));
      } else {
        groups = [];
      }
    } finally {
      loading = false;
    }
  }

  // Tell the main window to display this cluster's images. The browser window
  // stays open and focused so the user can hop between clusters.
  async function openCluster(id: number) {
    await emit("show-cluster", { cluster: id });
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
      {result.num_clusters} clusters · {result.num_noise} unclustered · click a cluster to view its images
    </div>
    <div class="card-grid">
      {#each groups as group (group.id)}
        <button class="cluster-card" title={group.label} onclick={() => openCluster(group.id)}>
          <div class="collage">
            {#each group.sample as p (p.path)}
              {#if thumbnails[p.path]}
                <img src={thumbnails[p.path]} alt="" class="cell" />
              {:else if thumbnails[p.path] === null}
                <div class="cell cell-empty">!</div>
              {:else}
                <div class="cell cell-empty"></div>
              {/if}
            {/each}
            {#each Array(Math.max(0, 4 - group.sample.length)) as _unused}
              <div class="cell cell-empty"></div>
            {/each}
          </div>
          <div class="card-label">
            <span class="card-title">{group.label}</span>
            <span class="card-count">{group.count}</span>
          </div>
        </button>
      {/each}
    </div>
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

  .card-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
    gap: 12px;
  }

  .cluster-card {
    display: flex;
    flex-direction: column;
    padding: 0;
    background: var(--bg-toolbar);
    border: 1px solid var(--border-color);
    border-radius: 6px;
    overflow: hidden;
    cursor: pointer;
    color: inherit;
    font: inherit;
    text-align: left;
  }

  .cluster-card:hover {
    border-color: var(--text-secondary);
    background: var(--bg-hover);
  }

  .collage {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1px;
    width: 100%;
  }

  /* Each cell sets its OWN square via aspect-ratio (same pattern as the main
     image grid). The two rows then auto-size to those squares, so the collage
     is square overall — no dependence on the container's height. */
  .cell {
    width: 100%;
    aspect-ratio: 1;
    object-fit: cover;
    display: block;
  }

  .cell-empty {
    display: flex;
    align-items: center;
    justify-content: center;
    background: var(--bg-base);
    color: var(--text-secondary);
  }

  .card-label {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    padding: 6px 8px;
  }

  .card-title {
    font-size: 13px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .card-count {
    font-size: 12px;
    color: var(--text-secondary);
    flex-shrink: 0;
  }
</style>
