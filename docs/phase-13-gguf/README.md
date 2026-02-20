# Phase 13 — GGUF Migration + Performance Fixes

## Goal

Replace FP8 (~18.9GB) with GGUF Q4_0 base + Q8 adapter (~12.8GB) to match ComfyUI's model format and improve performance on 16GB VRAM.

## Phase 13.0 — GGUF Loading Pipeline

### Changes Applied

- **New file `gguf_loader.py`**: Full GGUF loading pipeline ported from city96/ComfyUI-GGUF (Apache 2.0):
  - Dequantization functions for all GGUF quant types (Q4_0, Q4_K, Q5_K, Q6_K, Q8_0, etc.)
  - `GGMLTensor` subclass carrying quantization metadata
  - `load_gguf_state_dict()` — reads GGUF files via `gguf.GGUFReader`
  - `load_gguf_into_model()` — loads state_dict into WanModel (quantized weights stay packed)
  - `patch_gguf_linears()` — monkey-patches Linear.forward() for on-the-fly dequantization
  - `apply_lora_as_patches()` — attaches LoRA deltas as buffers (applied during forward)
  - `load_gguf_weights()` — high-level orchestrator

- **`config.py`**: New GGUF paths, HF repos, GGUF-specific overrides. Default `VIDEO_QUANT` changed from `"fp8"` to `"gguf"`.

- **`model_manager.py`**: New `ensure_gguf_models()` for auto-download of base (~10.2GB) and adapter (~2.65GB).

- **`video_engine.py`**: GGUF loading path in `load_model()`. LoRA applied as patches.

- **`generate.py`**: New `--quant TYPE` CLI option (`gguf` or `fp8`).

- **`vendor/.../multitalk.py`**: Added `"gguf"` to valid quant types, meta device creation for GGUF.

- **`requirements.txt`**: Added `gguf>=0.6.0`.

### GGUF Models

| Model | Source | Size |
|---|---|---|
| Wan2.1-I2V-14B Q4_0 (base) | `city96/Wan2.1-I2V-14B-480P-gguf` | ~10.2 GB |
| InfiniteTalk Single Q8 (adapter) | `Kijai/WanVideo_comfy_GGUF` | ~2.65 GB |

---

## Phase 13.1 — GGUF Hang Fix

### Root Cause

`offload_blocks()`/`restore_blocks()` moved ~10GB of GGUF blocks off/on GPU for CLIP/VAE encoding. Combined with `frame_num=81` VAE 3D-conv activation spike, this caused OOM/hang on 16GB. ComfyUI keeps GGUF blocks on their assigned device — they never move after placement.

### Fixes Applied

1. **Logging bug**: Root logger was INFO with `--verbose`, silently dropping all `logging.debug()`. Fixed: root always DEBUG, console handler controls visibility.

2. **Skip offload for CLIP/VAE encode**: Set `_skip_block_offload=True` on pipeline for GGUF. GGUF model (~10GB) fits alongside CLIP (~1.2GB) + VAE (~0.5GB) on 16GB without moving DiT blocks during encoding.

3. **frame_num 81→41**: VAE encoder processes all frames in one 3D-conv pass. At 81 frames/480p, Conv3D intermediates too large. 41 frames proven safe.

4. **activation_reserve 4.0→7.0 GiB**: GGUF dequant temporaries need same headroom as FP8.

5. **VAE offload after decode**: Explicit `self.vae.model.cpu()` + `empty_cache()` after every VAE decode.

6. **LoRA buffer device mismatch**: LoRA buffers registered on CPU after block_swap moved blocks to GPU. Fixed: `_gguf_linear_forward` uses `device=weight.device` for all LoRA tensors.

7. **audio_guide_scale 1.0→2.0**: Without audio CFG, lip sync degraded after first chunk. 2 passes/step enables audio guidance.

8. **VAE bfloat16**: Set `self.pipe.vae.dtype = torch.bfloat16` for ~2x faster encode/decode on tensor cores.

---

## Phase 13.2 — VAE Decode Performance + ETA Accuracy

### Problem

VAE decode took **~6 minutes 41 seconds per chunk** — far longer than expected (~30-60s). This was not included in the ETA calculation, causing the total estimated time (10m40s) to be wildly inaccurate vs actual runtime (46+ min).

### Root Cause

`_skip_block_offload=True` was set **globally** for GGUF, preventing DiT block offload before VAE decode. During VAE decode:

- 27 persistent DiT blocks on GPU: ~8.6 GB
- VAE model: ~0.5 GB
- Available for VAE decode intermediates: ~6.4 GB
- **VAE's 3D transposed convolutions exceeded this** → Windows WDDM transparently spilled to system RAM (~40-50x slower PCIe bandwidth)

The `_skip_block_offload` flag was correct for CLIP+VAE **encode** (first chunk — GGUF model fits alongside), but should NOT have applied to VAE **decode** (transposed conv intermediates are much larger than encoding intermediates).

### Fixes Applied

1. **Always offload DiT blocks before VAE decode** (`multitalk.py`):
   - Changed `if _block_swap_active and not _skip_offload:` → `if _block_swap_active:` for block offload before VAE decode
   - Changed same condition for block restore before next chunk's denoising
   - CLIP+VAE encode offload (first chunk) still gated by `_skip_offload` — unchanged

2. **ETA now includes VAE time** (`config.py`, `video_engine.py`, `pipeline.py`):
   - Added `VIDEO_VAE_DECODE_SECS = 60` (per-chunk, conservative estimate)
   - Added `VIDEO_FIRST_CHUNK_EXTRA_SECS = 50` (CLIP + VAE encode, first chunk only)
   - Updated ETA in `video_engine.py` and `_estimate_video_eta()` in `pipeline.py`
   - Also fixed `pipeline.py` to use `VIDEO_GGUF_SECS_PER_STEP` (was using FP8 constant)

3. **ETA accuracy logging** (previous session):
   - Logs estimated vs actual for TTS, video (gen+load), and total pipeline
   - Will calibrate VAE constants from real run data

### Expected Impact

| Metric | Before Fix | After Fix |
|---|---|---|
| VAE decode per chunk | ~401s (WDDM spill) | ~60s (estimated) |
| Block offload/restore overhead | 0s (skipped) | ~8s per chunk |
| Total for 5 chunks (5.3s audio) | 46+ min | ~17 min |

---

## Current Settings (Phase 13.2)

```python
VIDEO_QUANT = "gguf"
VIDEO_GGUF_FRAME_NUM = 41
VIDEO_GGUF_MOTION_FRAME = 13
VIDEO_GGUF_LORA_SAMPLE_STEPS = 4
VIDEO_GGUF_LORA_AUDIO_GUIDE_SCALE = 2.0   # 2 passes/step
VIDEO_GGUF_SECS_PER_STEP = 16             # ~15.9s/pass measured
VIDEO_GGUF_ACTIVATION_RESERVE = 7.0       # GiB
VIDEO_VAE_DECODE_SECS = 60                # Per-chunk (conservative)
VIDEO_FIRST_CHUNK_EXTRA_SECS = 50         # First chunk CLIP+VAE encode
VIDEO_LORA_SOLVER = "dpmpp_sde"
GPU_THERMAL_THRESHOLD = 70
```

## Status

All fixes compile clean. Awaiting test run to:
- Verify VAE decode is ~60s/chunk (was ~401s)
- Calibrate ETA constants from "ETA accuracy" log lines

## Files Modified

- `gguf_loader.py` — New file (Phase 13.0)
- `config.py` — GGUF paths, overrides, VAE time constants
- `model_manager.py` — GGUF auto-download
- `video_engine.py` — GGUF loading path, ETA calculation with VAE time
- `generate.py` — `--quant` CLI option, logging fix
- `pipeline.py` — ETA calculation with VAE time, GGUF secs_per_step
- `vendor/infinitetalk/wan/multitalk.py` — GGUF support, `_skip_block_offload` logic, always offload before VAE decode
- `requirements.txt` — `gguf>=0.6.0`
- `docs/plan.md` — Phase 13 documentation
- `docs/handoff.md` — Updated config, next steps, critical rules
- `README.md` — Updated performance table
