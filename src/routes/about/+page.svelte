<script lang="ts">
  import "$lib/theme.css";
  import { invoke } from "@tauri-apps/api/core";
  import { getVersion } from "@tauri-apps/api/app";
  import { onMount } from "svelte";

  let appVersion = $state("");
  let buildVariant = $state("");

  onMount(async () => {
    appVersion = await getVersion();
    buildVariant = await invoke("get_build_variant");
  });
</script>

<div class="about">
  <img class="app-icon" src="/app-icon.png" alt="Semantic Image Browser icon" />
  <h1 class="title">Semantic Image Browser</h1>
  <p class="version">
    {appVersion} for {buildVariant}
    (<a href="https://github.com/Mannheim1/semantic-image-browser/releases" target="_blank" rel="noreferrer">release notes</a>)
  </p>

  {#if buildVariant.includes("CUDA")}
    <p class="meta">Includes NVIDIA CUDA and cuDNN libraries</p>
  {/if}

  <p class="meta">
    Licensed under
    <a href="https://github.com/Mannheim1/semantic-image-browser/blob/main/LICENSE" target="_blank" rel="noreferrer">GPL-3.0-only</a>
  </p>
</div>

<style>
  .about {
    padding: 24px;
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    gap: 10px;
  }

  .app-icon {
    width: 64px;
    height: 64px;
    border-radius: 12px;
    margin-bottom: 4px;
  }

  .title {
    color: var(--text-primary);
    font-size: 18px;
    margin: 0;
  }

  .version {
    color: var(--text-primary);
    font-size: 14px;
    margin: 0;
  }

  .about a {
    color: var(--accent, #3b82f6);
    text-decoration: none;
  }

  .about a:hover {
    text-decoration: underline;
  }

  .meta {
    font-size: 13px;
    color: var(--text-secondary);
    margin: 0;
    line-height: 1.4;
  }
</style>
