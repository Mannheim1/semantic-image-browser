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
  <h1>Semantic Image Browser</h1>
  <p class="version">Version {appVersion} — {buildVariant}</p>
  <p>
    Semantic Image Search helps you find images using natural-language queries by indexing local
    folders, generating thumbnails, and ranking results with visual embeddings. The app is built
    with Tauri v2 and a Svelte + TypeScript frontend, with a Rust backend that uses LanceDB for
    vector search, ONNX Runtime for model inference, and Tesseract for OCR.
  </p>
  {#if buildVariant.includes("CUDA")}
    <p class="legal">
      This software includes NVIDIA CUDA and cuDNN libraries. NVIDIA, CUDA, and cuDNN are
      trademarks of NVIDIA Corporation.
    </p>
  {/if}
</div>

<style>
  .about {
    padding: 24px;
  }

  h1 {
    font-size: 18px;
    margin: 0 0 8px 0;
  }

  .version {
    color: var(--text-secondary);
    font-size: 13px;
    margin: 0 0 16px 0;
  }

  p {
    line-height: 1.5;
    font-size: 13px;
    color: var(--text-secondary);
  }

  .legal {
    font-size: 12px;
    margin-top: 16px;
  }
</style>
