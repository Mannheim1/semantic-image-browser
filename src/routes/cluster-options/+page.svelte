<script lang="ts">
  import "$lib/theme.css";
  import { invoke } from "@tauri-apps/api/core";
  import { emit } from "@tauri-apps/api/event";
  import { getCurrentWindow } from "@tauri-apps/api/window";
  import { onMount } from "svelte";

  let minClusterSize = $state("");
  let maxClusterSize = $state("");
  let minSamples = $state("");
  let epsilon = $state("");

  type DefaultParams = {
    min_cluster_size: number;
    min_samples: number;
    epsilon: number;
  };

  // Pre-fill each field with the value the app would actually use for the
  // current library, so the dialog opens showing the real defaults rather than
  // bare placeholders. Max cluster size has no numeric default ("No limit").
  function applyDefaults(def: DefaultParams) {
    minClusterSize = String(def.min_cluster_size);
    minSamples = String(def.min_samples);
    epsilon = String(def.epsilon);
    maxClusterSize = "";
  }

  // Keep the fetched defaults so the Reset button can restore them.
  let defaults: DefaultParams | null = null;

  onMount(async () => {
    try {
      const def = await invoke<DefaultParams>("get_default_cluster_params");
      defaults = def;
      applyDefaults(def);
    } catch {
      // Leave blank; the backend still applies the auto defaults on Compute.
    }
  });

  function reset() {
    if (defaults) applyDefaults(defaults);
  }

  // Snap an integer field to the nearest valid whole number: round to an
  // integer, then clamp up to `min`. A blank field stays blank (= auto). Garbage
  // that isn't a number also clears to blank. Returns the normalised text.
  function snapInt(value: string, min: number): string {
    if (String(value).trim() === "") return "";
    const n = Math.round(Number(value));
    return Number.isFinite(n) ? String(Math.max(n, min)) : "";
  }

  // Snap a rational field to the nearest valid value: clamp up to `min`, keeping
  // the decimal. A blank field stays blank (= auto); garbage clears to blank.
  function snapFloat(value: string, min: number): string {
    if (String(value).trim() === "") return "";
    const n = Number(value);
    return Number.isFinite(n) ? String(Math.max(n, min)) : "";
  }

  // Parse a normalised field to a number, or null when blank (= auto).
  function toNumber(value: string): number | null {
    return value === "" ? null : Number(value);
  }

  async function compute() {
    // Snap every field to its nearest valid value first, so an out-of-range or
    // non-integer entry becomes the closest legal number instead of reverting
    // to auto.
    minClusterSize = snapInt(minClusterSize, 2);
    maxClusterSize = snapInt(maxClusterSize, 2);
    minSamples = snapInt(minSamples, 1);
    epsilon = snapFloat(epsilon, 0);
    await emit("start-clustering", {
      minClusterSize: toNumber(minClusterSize),
      maxClusterSize: toNumber(maxClusterSize),
      minSamples: toNumber(minSamples),
      epsilon: toNumber(epsilon),
    });
    await getCurrentWindow().close();
  }

  async function cancel() {
    await getCurrentWindow().close();
  }

  function onKeydown(e: KeyboardEvent) {
    if (e.key === "Enter") compute();
    else if (e.key === "Escape") cancel();
  }
</script>

<svelte:window onkeydown={onKeydown} />

<div class="dialog">
  <h1>Parameter selection</h1>

  <div class="field">
    <label for="min">Minimum cluster size</label>
    <input
      id="min"
      type="number"
      min="2"
      placeholder="Auto"
      bind:value={minClusterSize}
      onblur={() => (minClusterSize = snapInt(minClusterSize, 2))}
    />
    <span class="hint">
      Smallest group that counts as a cluster. Smaller = more, finer clusters.
    </span>
  </div>

  <div class="field">
    <label for="max">Maximum cluster size</label>
    <input
      id="max"
      type="number"
      min="2"
      placeholder="No limit"
      bind:value={maxClusterSize}
      onblur={() => (maxClusterSize = snapInt(maxClusterSize, 2))}
    />
    <span class="hint">
      Caps a cluster's size, splitting up one dominant group. Blank = no limit.
    </span>
  </div>

  <div class="field">
    <label for="samples">Minimum samples</label>
    <input
      id="samples"
      type="number"
      min="1"
      placeholder="Auto"
      bind:value={minSamples}
      onblur={() => (minSamples = snapInt(minSamples, 1))}
    />
    <span class="hint">
      How dense a point's neighbourhood must be to join a cluster. Lower = fewer
      unclustered images; higher = tighter clusters, more left out.
    </span>
  </div>

  <div class="field">
    <label for="epsilon">Epsilon</label>
    <input
      id="epsilon"
      type="number"
      min="0"
      step="0.1"
      placeholder="0"
      bind:value={epsilon}
      onblur={() => (epsilon = snapFloat(epsilon, 0))}
    />
    <span class="hint">
      Distance threshold that merges nearby clusters and pulls in stray points.
      Higher = fewer, larger clusters; 0 = off.
    </span>
  </div>

  <div class="actions">
    <button class="btn" onclick={cancel}>Cancel</button>
    <button class="btn" onclick={reset}>Reset</button>
    <button class="btn btn-primary" onclick={compute}>Compute</button>
  </div>
</div>

<style>
  .dialog {
    padding: 20px;
    display: flex;
    flex-direction: column;
    gap: 14px;
  }

  h1 {
    font-size: 18px;
    margin: 0;
  }

  .field {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  label {
    font-size: 13px;
    color: var(--text-primary);
  }

  input {
    height: 32px;
    padding: 0 10px;
    background: var(--bg-base);
    border: 1px solid var(--border-color);
    border-radius: 4px;
    color: var(--text-primary);
    font-size: 14px;
    outline: none;
    box-sizing: border-box;
  }

  input:focus {
    border-color: #5a5250;
  }

  .hint {
    font-size: 11px;
    color: var(--text-secondary);
  }

  .actions {
    display: flex;
    justify-content: flex-end;
    gap: 8px;
    margin-top: 4px;
  }

  .btn {
    background: var(--bg-hover);
    border: 1px solid var(--border-color);
    border-radius: 4px;
    color: var(--text-primary);
    padding: 7px 14px;
    cursor: pointer;
    font-size: 13px;
  }

  .btn:hover {
    background: var(--border-color);
  }

  .btn-primary {
    background: #4a6ea9;
    border-color: #4a6ea9;
    color: #fff;
  }

  .btn-primary:hover {
    background: #5a7eb9;
  }
</style>
