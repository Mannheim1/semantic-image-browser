<script lang="ts">
  import "$lib/theme.css";
  import { invoke } from "@tauri-apps/api/core";
  import { emit, listen } from "@tauri-apps/api/event";
  import { open } from "@tauri-apps/plugin-dialog";
  import { onMount } from "svelte";

  let watchedDirectories = $state<string[]>([]);
  let isScanning = $state(false);
  let skipNextDirsChanged = false;

  async function loadDirectories() {
    watchedDirectories = await invoke("get_watched_directories");
  }

  async function addDirectory() {
    const selected = await open({
      directory: true,
      multiple: true,
      title: "Select folder(s) to watch",
    });
    if (selected) {
      isScanning = true;
      const paths = Array.isArray(selected) ? selected : [selected];
      for (const path of paths) {
        await invoke("add_watched_directory", { path });
      }
      isScanning = false;
      await loadDirectories();
      skipNextDirsChanged = true;
      await emit("directories-changed");
    }
  }

  async function removeDirectory(path: string) {
    isScanning = true;
    await invoke("remove_watched_directory", { path });
    isScanning = false;
    await loadDirectories();
    skipNextDirsChanged = true;
    await emit("directories-changed");
  }

  onMount(() => {
    loadDirectories();

    const unlistenDirsChanged = listen<void>("directories-changed", () => {
      if (skipNextDirsChanged) {
        skipNextDirsChanged = false;
        return;
      }
      loadDirectories();
    });

    return () => {
      unlistenDirsChanged.then((unlisten) => unlisten());
    };
  });
</script>

<div class="manage-folders">
  <div class="header">
    <h1>Manage Folders</h1>
    <button class="add-btn" onclick={addDirectory} disabled={isScanning}>Add Folder...</button>
  </div>

  {#if watchedDirectories.length === 0}
    <div class="empty">No folders added yet.</div>
  {:else}
    <div class="folders-list">
      {#each watchedDirectories as dir}
        <div class="folder-item">
          <span class="folder-path" title={dir}>{dir}</span>
          <button class="folder-remove" onclick={() => removeDirectory(dir)} disabled={isScanning}>
            ×
          </button>
        </div>
      {/each}
    </div>
  {/if}
</div>

<style>
  .manage-folders {
    padding: 16px;
  }

  .header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
  }

  h1 {
    font-size: 18px;
    margin: 0;
  }

  .add-btn {
    background: var(--bg-hover);
    border: 1px solid var(--border-color);
    border-radius: 4px;
    color: var(--text-primary);
    padding: 6px 12px;
    cursor: pointer;
    font-size: 13px;
  }

  .add-btn:hover:not(:disabled) {
    background: var(--border-color);
  }

  .add-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .empty {
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

  .folder-remove:hover:not(:disabled) {
    color: #ff6b6b;
  }

  .folder-remove:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
</style>
