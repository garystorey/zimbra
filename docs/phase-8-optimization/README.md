# Phase 8: Performance Optimization

Reduces per-chunk overhead and block transfer latency for video generation while staying within 16GB VRAM.

## Changes

### Bug Fix: Progress Bar Step Calculation

- **Files:** `video_engine.py`, `pipeline.py`
- `estimate_total_steps()` and `_estimate_video_eta()` divided by `motion_frame` (9) instead of `frame_num - motion_frame` (32)
- Progress bar showed ~92 steps when only ~32 would fire, making generation appear stuck

### Chunk Caching (CLIP + VAE + Block Offload Skip)

- **File:** `vendor/infinitetalk/wan/multitalk.py`
- For image inputs, `clip_context` and VAE conditioning `y` are identical every chunk
- First chunk: full offload_blocks → CLIP → VAE → restore_blocks cycle, populate caches
- Subsequent chunks: skip block offload/restore entirely (~26GB transfers eliminated per chunk)
- VAE motion-frame encode still runs (~500MB fits alongside DiT blocks within 16GB)
- Also skips redundant `cond_image` re-extraction via `extract_specific_frames` for image inputs
- Video inputs fall back to the original per-chunk encode path

### Pinned CPU Memory for Swapped Blocks

- **File:** `vendor/infinitetalk/wan/modules/multitalk_model.py`
- Added `_pin_block_memory()` static method to WanModel
- Pins parameters/buffers of CPU-resident blocks into page-locked memory
- Enables direct DMA for CPU→GPU prefetch at full PCIe bandwidth (~2x faster than staged pageable copies)
- Called after `block_swap()` init, after `offload_blocks()`, and after each block offload in `_run_blocks_with_swap`
- VRAM impact: zero (pinned memory is host-side only)
- Host RAM cost: 12 blocks × 470MB ≈ 5.6GB pinned (fine for 32GB+ systems)

### inference_mode + GC Cleanup

- **File:** `vendor/infinitetalk/wan/multitalk.py`
- Replaced `torch.no_grad()` with `torch.inference_mode()` (disables version counting, lower overhead)
- Removed 5 unnecessary `torch_gc()` calls that added sync points without benefit
- Replaced remaining `torch_gc()` with `torch.cuda.empty_cache()` (drops useless `ipc_collect()`)
- **Kept** critical `empty_cache()` calls between forward passes (prevents VRAM thrashing)

### Bug Fix: `_scaled_mm` Device Mismatch (cuda:0 vs cpu)

- **Error:** `Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! (when checking argument for argument mat2 in method wrapper_CUDA___scaled_mm)`
- **Root cause:** Two contributing factors in the FP8 block swap path:
  1. **`scale_weight` stored as plain tensor attribute** — `load_fp8_weights_native()` set `parent.scale_weight = tensor` which `nn.Module.to()` does NOT move during block swap / offload / restore (only registered parameters and buffers are moved)
  2. **Missing `restore_blocks()` for cached chunks** — After each chunk's denoising, `offload_blocks()` moves persistent blocks to CPU for VAE decode. The cached path for chunks 1+ skipped `restore_blocks()`, leaving all ~28 persistent blocks on CPU during denoising
- **Fixes applied:**

| Fix                                        | File                  | Detail                                                                                                                                                             |
| ------------------------------------------ | --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Register `scale_weight` as buffer          | `fp8_optimization.py` | `parent.register_buffer("scale_weight", tensor, persistent=False)` — now tracked by `nn.Module.to()`, moved during block swap, and pinned by `_pin_block_memory()` |
| `restore_blocks()` for cached chunks       | `multitalk.py`        | After VAE motion-frame encode in cached path, call `self.model.restore_blocks()` to bring persistent blocks back to GPU before denoising                           |
| Fix `_pin_block_memory` for nested buffers | `multitalk_model.py`  | Iterate `block.modules()` and update each module's `_buffers` dict directly instead of using dotted names on the top-level block                                   |
| Skip `scale_weight` in dtype conversion    | `video_engine.py`     | Buffer conversion loop now skips `scale_weight` (must stay float32 for `_scaled_mm`)                                                                               |
| LoRA uses `register_buffer`                | `video_engine.py`     | `_apply_delta_to_param()` uses `register_buffer()` instead of plain assignment for `scale_weight`                                                                  |

- **Reverted (caused performance regression):** Device guards in `fp8_linear_forward`, `_ensure_block_on_device` in hot loop, `non_blocking=False` for current-block transfer. These added per-forward or per-block overhead that caused ~4x slowdown. With `scale_weight` as a buffer and `restore_blocks()` for cached chunks, `block.to()` correctly moves all params/buffers and no guards are needed.

## Not Implemented

### Batched CFG

- Would merge 2 sequential forward passes into 1 batched pass (~25-35% per-step speedup)
- **Blocked:** Audio batch dimension doubles as `human_num` in `WanModel.forward()`. Concatenating cond + drop_audio along batch would set `human_num=2`, disabling single-person attention optimization and triggering expensive O(n²) multi-person attention path
- Requires deep refactoring of model input preprocessing to separate CFG batch from human count

### Non-blocking Block Offload

- Kept synchronous because `pin_memory()` requires fully-transferred CPU data. Non-blocking offload + immediate pin would copy incomplete data. The pinned prefetch direction (CPU→GPU) provides the higher-impact speedup.

## Expected Impact

| Optimization        | Savings                                              | Scope               |
| ------------------- | ---------------------------------------------------- | ------------------- |
| Chunk caching       | ~10-20s per chunk (skip 26GB block transfers + CLIP) | 7 of 8 chunks       |
| Pinned memory       | ~15-25% faster block prefetch                        | Every swapped block |
| inference_mode + GC | ~3-5% (fewer sync points)                            | Entire denoising    |
| Progress bar fix    | Correct step count display                           | UI only             |

For a 35s video (~8 chunks, ~32 steps): chunk caching alone saves ~70-140s total. Pinned memory reduces per-step block swap time. Combined conservative estimate: **~1.2-1.4x faster**.
