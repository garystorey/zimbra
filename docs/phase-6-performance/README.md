# Phase 6 — Video Performance Fix

Eliminated catastrophic per-layer CPU↔GPU bottlenecks during DiT denoising.

## Root Cause (Original — Phase 6a)

`auto_tune_persistent_params()` ran when VAE (~500MB) + CLIP (~4.7GB) were still on GPU, so it calculated only ~10.9B persistent params. With ~3B overflow, each denoising step took 7+ minutes.

**Fix:** Offload VAE/CLIP to CPU before auto-tune → auto-tune sees ~15.8GB free → ~14.5B persistent.

## Root Cause (Discovered Later — Phase 6b)

The InfiniteTalk FP8 DiT has **~18.9B params** (not 14B). The extra ~4.9B comes from audio cross-attention layers unique to InfiniteTalk vs base Wan2.1. Even with optimal auto-tune (~14.5B persistent), **~4.4B params overflow to CPU**.

The vendor's overflow forward methods were catastrophically slow:

- **`AutoWrappedModule`**: `copy.deepcopy(module).to(GPU)` per forward call
- **`AutoWrappedQLinear`**: `cast_to_device()` temp copy per forward call
- **`AutoWrappedLinear`**: `cast_to()` temp copy per forward call

With ~4.4B overflow params deep-copied CPU→GPU on every forward pass (thousands per step), GPU sat idle while RAM maxed out.

**Fix:** Patched `vendor/infinitetalk/src/vram_management/layers.py` — all three overflow forward methods now use in-place `.to(device)` → compute → `.to(offload_device)`. Standard CPU offloading, no deep copies.

## All Fixes Applied

| Fix                                 | Detail                                                                |
| ----------------------------------- | --------------------------------------------------------------------- |
| `t5_cpu=True`                       | T5 runs on CPU (only used once for prompt encoding)                   |
| VAE/CLIP offloaded before auto-tune | Auto-tune sees ~15.8GB free → ~14.5B persistent                       |
| `memory_allocated` not `reserved`   | Tighter free VRAM estimate                                            |
| Activation reserve 1.0GB            | Sufficient for 480p + frame_num=41                                    |
| `frame_num` 81→41                   | Smaller chunks = less activation VRAM                                 |
| **Vendor overflow patch**           | In-place `.to()` replaces `copy.deepcopy().to()` for overflow modules |

## Files Modified

| File                                                | Changes                                                  |
| --------------------------------------------------- | -------------------------------------------------------- |
| `vendor/infinitetalk/src/vram_management/layers.py` | Overflow forward: deepcopy → in-place `.to()` transfers  |
| `video_engine.py`                                   | `t5_cpu=True`, offload VAE/CLIP, log param split         |
| `config.py`                                         | Auto-tune reflects 18.9B model, 1.0GB activation reserve |
| `README.md`                                         | Updated performance section                              |
| `docs/compatibility.md`                             | Added section 3b (vendor patch)                          |
| `docs/dependency-matrix.md`                         | Added vendor patches table                               |

## Expected Performance

| Metric                  | Phase 6a (broken) | Phase 6a (fixed) | Phase 6b (fixed)                   |
| ----------------------- | ----------------- | ---------------- | ---------------------------------- |
| Persistent DiT params   | ~10.9B / 14B\*    | ~14.5B / 18.9B   | ~14.5B / 18.9B                     |
| Overflow params         | ~3B               | ~4.4B            | ~4.4B                              |
| Overflow forward method | deep copy + move  | deep copy + move | **in-place move** (vendor patch)   |
| Time per denoising step | >7 min (DNF)      | ~30-40s          | TBD (expected significant speedup) |

\*14B was the incorrect assumption; actual model is 18.9B.
