<script lang="ts">
  import "$lib/theme.css";
</script>

<div class="controls">
  <h1>View Controls</h1>
  <ul>
    <li>Click a thumbnail to open the detail side panel.</li>
    <li>Double-click a thumbnail to open the image in your default viewer.</li>
    <li>Right-click a thumbnail to open the context menu.</li>
    <li>With the panel open, use Left/Right arrow keys to move to the previous or next image.</li>
  </ul>
</div>

<style>
  .controls {
    padding: 24px;
  }

  h1 {
    font-size: 18px;
    margin: 0 0 16px 0;
  }

  ul {
    padding-left: 20px;
    margin: 0;
  }

  li {
    line-height: 1.6;
    margin-bottom: 8px;
    color: var(--text-secondary);
    font-size: 13px;
  }
</style>
