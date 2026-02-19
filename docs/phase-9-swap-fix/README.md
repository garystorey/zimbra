# Phase 9 — Block Swap VRAM Fix

## Problem

Video generation showed **0/32 steps after 7+ minutes** despite previous
optimisations (Phases 6-8). The double-buffer async rewrite of
`_run_blocks_with_swap` made performance *worse*, not better.

## Root Cause Analysis

### 1. `auto_tune_blocks_to_swap` didn't reserve swap buffer VRAM

The auto-tune function calculated how many DiT blocks could be persistent
on GPU, but it only subtracted:

- `activation_reserve` (2 GiB) — for intermediate tensors during compute
- `non_block_overhead` (500 MB) — for embeddings, head, audio_proj

It did **not** account for the **swap buffer slot** — the one swapped block
that must temporarily reside on GPU while being computed. This block
(~470 MB) ate directly into the activation reserve.

On a 16 GB GPU:
- Old auto-tune: 30 persistent + 10 swap → 1 swap slot leaves **~1.5 GiB**
  for activations (barely sufficient)
- With async double-buffer (2 slots): **~1.0 GiB** for activations → VRAM
  thrashing (5-10 s/block instead of 0.25 s/block)

### 2. Async double-buffer pattern doesn't actually help

The rewrite used separate CUDA streams for load/offload/compute. However:

- **GPU → CPU with non-pinned destination is synchronous.** `cudaMemcpyAsync`
  to pageable host memory blocks the CPU thread until the copy completes.
  Since `nn.Module.to(cpu, non_blocking=True)` allocates *new* (unpinned)
  CPU tensors, every "async" offload was actually synchronous.
- **After the first forward pass, prefetches are also synchronous.** The
  initial `block_swap()` pins swapped blocks, but after offloading
  (GPU → new CPU tensor), the CPU data is no longer pinned. Subsequent
  prefetches from unpinned memory also block the CPU thread.
- **2 swap slots vs 1.** The async pattern temporarily holds two swapped
  blocks on GPU (one being offloaded + one being loaded), doubling the
  swap buffer VRAM cost and further squeezing activations.

Net result: no overlap benefit, higher VRAM pressure.

### 3. Diagnostic logging was invisible

All `[diag]` messages used `logging.info()`, but the default log level is
`WARNING`. The user never saw timing breakdowns unless running `--verbose`.

## Fixes Applied

### Fix 1: Reserve swap buffer slot in `auto_tune_blocks_to_swap`

**File:** `config.py`

Added `swap_buffer_reserve = VIDEO_BLOCK_SIZE_BYTES` (~470 MB) to the VRAM
budget. This reduces persistent blocks from 30 → 29, giving a full 2+ GiB
activation headroom with the swap slot included.

```
activation_reserve = 2.0 GiB
swap_buffer_reserve = 470 MB   ← NEW
non_block_overhead  = 500 MB
```

### Fix 2: Revert to synchronous offload + re-pin pattern

**File:** `vendor/infinitetalk/wan/modules/multitalk_model.py`

Reverted `_run_blocks_with_swap` from the broken double-buffer async
pattern back to the proven synchronous pattern:

1. **Async prefetch** next block on a dedicated CUDA stream (from pinned
   CPU → GPU, truly async because source is pinned)
2. **Compute** current block on the default stream
3. **Synchronous offload** current block to CPU (`non_blocking=False`)
4. **Re-pin** block memory for fast DMA on the next prefetch cycle

This needs only **1 swap buffer slot** on GPU and was previously measured
at 0.22-0.30 s/block, ~31 s/step.

### Fix 3: Visible timing diagnostics

**File:** `vendor/infinitetalk/wan/multitalk.py`

Added `print()` timing at key points (visible at all log levels):
- `[timing] offload_blocks: Xs`
- `[timing] CLIP encode+offload: Xs`
- `[timing] VAE encode: Xs`
- `[timing] restore_blocks: Xs (total preproc)`
- `[timing] step N/M fwd1: Xs`
- `[timing] step N/M fwd2: Xs`

## What Was Tried and Failed

| Attempt | Why it failed |
|---------|--------------|
| Async double-buffer offload (`non_blocking=True`, separate offload stream) | GPU→CPU to non-pinned memory is synchronous regardless of `non_blocking`. No overlap achieved. Required 2 swap slots, eating 470 MB from activation headroom. |
| Removing `_pin_block_memory()` after offload | Without re-pinning, subsequent prefetches use pageable memory (also synchronous). Transfer speed similar but lost the ability to overlap with compute. |
| Relying on "pinned pages survive .to() round-trips" | False. `nn.Module.to(device)` creates *new* tensors; the original pinned tensors are garbage collected. |

## Key Numbers (16 GB RTX 4090 Laptop)

| Metric | Value |
|--------|-------|
| DiT blocks | 40 (each ~470 MB FP8) |
| Persistent on GPU | 29 |
| Swapped CPU ↔ GPU | 11 |
| Non-block params | ~500 MB |
| Swap buffer (1 slot) | ~470 MB |
| Activation headroom | ~2.0+ GiB |
| Expected per-block | 0.22-0.30 s |
| Expected per-step | ~31 s (2 forward passes) |

## Expected Performance

With `text_guide_scale=1.0` (2 forward passes/step) and LoRA (4 steps/chunk):
- Per step: ~31 s
- Per chunk: ~125 s (4 steps + pre/post processing)
- 8 chunks (~10 s audio): ~17 min total video generation
