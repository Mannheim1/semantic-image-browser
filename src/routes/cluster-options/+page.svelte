<script lang="ts">
  import "$lib/theme.css";
  import { invoke } from "@tauri-apps/api/core";
  import { emit } from "@tauri-apps/api/event";
  import { getCurrentWindow } from "@tauri-apps/api/window";
  import { onMount } from "svelte";

  let minClusterSize = $state("");
  let maxClusterSize = $state("");

  // Pre-fill the minimum cluster size with the value the app would choose for
  // the current library, so the dialog opens showing the real default rather
  // than a bare placeholder. Max cluster size has no numeric default ("No
  // limit"), so it stays blank.
  onMount(async () => {
    try {
      const def: number = await invoke("get_default_min_cluster_size");
      minClusterSize = String(def);
    } catch {
      // Leave blank; the backend still applies the auto default on Compute.
    }
  });

  // Parse a field to a positive integer, or null when blank/invalid (= auto).
  function parseField(value: string): number | null {
    const n = parseInt(value, 10);
    return Number.isFinite(n) && n >= 2 ? n : null;
  }

  async function compute() {
    await emit("start-clustering", {
      minClusterSize: parseField(minClusterSize),
      maxClusterSize: parseField(maxClusterSize),
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
  <h1>Compute Clusters</h1>
  <p class="intro">
    Leave a field blank to let the app choose automatically. Click Compute to run.
  </p>

  <div class="field">
    <label for="min">Minimum cluster size</label>
    <input
      id="min"
      type="number"
      min="2"
      placeholder="Auto"
      bind:value={minClusterSize}
    />
    <span class="hint">Smaller = more, finer clusters.</span>
  </div>

  <div class="field">
    <label for="max">Maximum cluster size</label>
    <input
      id="max"
      type="number"
      min="2"
      placeholder="No limit"
      bind:value={maxClusterSize}
    />
    <span class="hint">Caps a cluster's size, splitting up one dominant group.</span>
  </div>

  <div class="actions">
    <button class="btn" onclick={cancel}>Cancel</button>
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

  .intro {
    margin: 0;
    font-size: 13px;
    color: var(--text-secondary);
    line-height: 1.5;
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
