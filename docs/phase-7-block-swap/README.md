# Phase 7 — Block Swap Performance Overhaul

Replace the vendor's per-layer VRAM management with Kijai-style block-level swapping + async CUDA stream prefetch. Target: ≤2 min per second of video.

## Problem

Phase 6 patched the vendor's overflow forward methods from `copy.deepcopy().to()` to in-place `.to()`, but the fundamental architecture remains: **every individual layer** (Linear, QLinear, Conv3d, LayerNorm) inside each DiT block is wrapped with `AutoWrapped*` wrappers. Overflow layers (~4.4B params across hundreds of small modules) each do synchronous `to(GPU)` → compute → `to(CPU)` on every forward call.

With 40 DiT blocks and ~15-20 wrapped layers per block, overflow blocks produce **hundreds of tiny synchronous CPU↔GPU transfers per denoising step**. Each transfer has fixed PCIe overhead regardless of size, so many small transfers are far worse than fewer large ones.

## Solution: Block-Level Swapping with Async Prefetch

Inspired by [Kijai's ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper), which uses the same technique for Wan2.1 models:

1. **Block-level granularity** — Instead of wrapping individual layers, keep the first N blocks permanently on GPU and swap entire blocks (all layers at once) between CPU and GPU.

2. **Async prefetch via CUDA streams** — While block B computes on GPU, block B+1 is transferred from CPU in the background on a separate CUDA stream. When block B finishes, block B+1 is already on GPU (or nearly so). Transfer latency is hidden behind compute time.

3. **Non-blocking transfers** — All `.to(device)` calls for swapped blocks use `non_blocking=True`, allowing CPU and GPU work to overlap.

### How It Works

```
Step 1: Block 32 computing on GPU
        Block 33 prefetching from CPU → GPU (async, separate CUDA stream)

Step 2: Block 32 done → offload to CPU (async)
        Block 33 ready on GPU → start computing
        Block 34 prefetching from CPU → GPU (async)

...and so on for all swapped blocks
```

## Key Numbers

| Metric | Value |
| --- | --- |
| Total DiT blocks | 40 |
| Block size (FP8) | ~470MB each |
| Total DiT params | ~18.9B |
| GPU VRAM (after VAE/CLIP offload) | ~15.8GB free |
| Blocks fitting on GPU | ~32 |
| Blocks to swap | ~8 |
| PCIe transfer per block | ~0.1-0.2s (hidden behind compute) |
| Block compute time | ~0.5-1.0s |

## Files Modified

| File | Changes |
| --- | --- |
| `vendor/infinitetalk/wan/modules/model.py` | Added `block_swap()` method, modified `forward()` block loop with prefetch |
| `video_engine.py` | Replaced `enable_vram_management` with block swap approach |
| `config.py` | `auto_tune_blocks_to_swap()`, `VIDEO_BLOCKS_TO_SWAP`, `VIDEO_PREFETCH_BLOCKS` |
| `generate.py` | `--blocks-to-swap`, `--prefetch-blocks` CLI flags |
| `pipeline.py` | Pass through new params |
| `README.md` | Updated performance expectations |
| `docs/plan.md` | Added Phase 7 entry |

## Expected Performance

| Metric | Phase 6 (per-layer) | Phase 7 (block swap) |
| --- | --- | --- |
| Transfer granularity | ~15-20 layers per block | 1 whole block |
| Transfer mode | Synchronous | Async prefetch (CUDA stream) |
| Transfers per step | ~300-400 tiny | ~8-16 large (hidden) |
| Time per denoising step | ~38s | ~5-10s (estimated) |
| Time per second of video | ~5+ min | ~1-2 min (target) |
