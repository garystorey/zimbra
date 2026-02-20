# Phase 14 — Denoising Performance Optimizations

**Goal:** Reduce denoising loop overhead without quality loss.

## Changes

### 1. Remove per-pass `torch.cuda.empty_cache()` from denoising loop

**File:** `vendor/infinitetalk/wan/multitalk.py`

Previously called after every model forward pass (2-3× per step × 4 steps × 5 chunks = 40+ calls). Each call forces CUDA synchronization + Windows WDDM kernel memory manager round-trip (~200-500ms each).

The CUDA allocator efficiently reuses cached freed blocks between identical forward passes — `empty_cache()` is only needed before VAE decode (different allocation pattern), which already has its own call.

**Estimated savings: 8-20s per generation.**

### 2. Remove dead motion re-noising code

**File:** `vendor/infinitetalk/wan/multitalk.py`

Two blocks of dead code removed:

- **Pre-loop block** (before denoising): `torch.randn_like` + `self.add_noise` to inject noised motion frames into initial latent. Always overwritten by clean motion frame injection at the start of each denoising step (`latent[:, :N] = latent_motion_frames`).

- **Post-solver block** (after each solver update): Same `randn_like` + `add_noise` pattern. Always overwritten by clean motion frame write on the next line.

Both blocks were vestigial from the original vendor code, made dead by Phase 10's direct latent passthrough optimization. The clean injection provides temporal coherence without VAE reconstruction error accumulation.

**Saves ~20 unnecessary GPU tensor operations per generation.**

### 3. DPM++ SDE 2nd-order multistep correction (REVERTED)

**File:** `vendor/infinitetalk/wan/multitalk.py`

Attempted 2nd-order correction that extrapolates the predicted x₀ from current and previous step's denoised estimates:

```
denoised_d = (1 + 1/(2r)) * denoised - (1/(2r)) * old_denoised
```

**Caused severe quality degradation with 2-pass CFG (audio_cfg=2.0).** The extrapolation overshoots when denoised estimates include CFG scaling — CFG amplifies the noise prediction difference, and 2nd-order extrapolation amplifies it further. Only worked with 1-pass mode (audio_cfg=1.0), which itself is not viable for streaming quality.

**Reverted to 1st-order SDE** (proven stable in Phases 12-13 with audio_cfg=2.0). The reference implementation (`fm_solvers.py`) uses a different 2nd-order formulation for SDE (`D0 + 0.5 * D1` terms) that operates on model outputs directly rather than extrapolating denoised, which may be more stable with CFG. Not pursued — 1st-order quality is sufficient.

### 4. GGUF audio_cfg 2.0→1.0→2.0 (reverted)

**File:** `config.py`

Tried `VIDEO_GGUF_LORA_AUDIO_GUIDE_SCALE = 1.0` to skip CFG entirely (1 pass/step instead of 2). This halved denoising time and the `_skip_cfg` optimization fired correctly.

However, **tested output showed noticeable quality degradation** — warped facial features, less defined detail in later chunks, progressive distortion. Even with DPM++ SDE 2nd-order correction, audio_cfg=1.0 is insufficient for multi-chunk streaming. The audio conditioning layers need real CFG guidance.

**Reverted to 2.0** (2 passes/step). User can still use `--audio-cfg 1.0` for max speed if quality is acceptable.

## Calibrated ETA Constants (from run_20260220_160641)

| Constant                       | Original | Interim | Final   | Measured                                        |
| ------------------------------ | -------- | ------- | ------- | ----------------------------------------------- |
| `VIDEO_GGUF_SECS_PER_STEP`     | 16s      | 16s     | **17s** | 16.8s/pass avg (15.8 chunk 0, ~17.1 chunks 1-4) |
| `VIDEO_VAE_DECODE_SECS`        | 60s      | 8s      | **15s** | VAE ~7s + inter-chunk gap ~8s per chunk         |
| `VIDEO_FIRST_CHUNK_EXTRA_SECS` | 50s      | 30s     | **23s** | CLIP ~3s + VAE encode ~20s                      |

ETA accuracy: **778s estimated vs 784s actual = 0.8% error** (was 1030s estimated = 17% overestimate).

**Bug fixed:** `_estimate_video_eta` (pipeline.py) and `estimate_total_steps` (video_engine.py) used `VIDEO_LORA_SAMPLE_STEPS=6` instead of `VIDEO_GGUF_LORA_SAMPLE_STEPS=4` for GGUF mode, adding 320s phantom time. Also moved `_vo_start` in generate.py to after model loading so the accuracy comparison is fair.

## Files Modified

| File                                   | Change                                      |
| -------------------------------------- | ------------------------------------------- |
| `vendor/infinitetalk/wan/multitalk.py` | Optimizations 1-3, 2nd-order SDE reverted   |
| `config.py`                            | audio_cfg revert, calibrated ETA constants  |
| `pipeline.py`                          | Fixed `_estimate_video_eta` GGUF overrides  |
| `video_engine.py`                      | Fixed `estimate_total_steps` GGUF overrides |
| `generate.py`                          | Moved `_vo_start` after model loading       |
| `docs/plan.md`                         | Phase 14 section updated                    |

## Measured Performance (5-chunk, 5.8s video)

**Run with audio_cfg=1.0 (1 pass/step):**

- Denoising: 5 chunks × 4 steps × ~16s/step = ~330s total (67s/chunk)
- VAE decode: ~7s/chunk × 5 = ~35s
- First-chunk overhead: ~28s (CLIP + VAE encode)
- Total generation: 461s (~7.7 min)
- **Quality: degraded** — warped features in later chunks

**Verified with audio_cfg=2.0 (2 passes/step, run_20260220_160641):**

- Denoising: 5 chunks × 4 steps × ~34s/step = ~672s (chunk 0: 126s, chunks 1-4: ~136s)
- Per-chunk overhead: 5 × ~15s = ~75s (VAE decode + inter-chunk gap)
- First-chunk CLIP+VAE encode: ~23s
- Total inference: 784s (~13 min, ~2.2 min/sec)
- **Quality: good** — audio CFG prevents progressive degradation, minor color fluctuation at chunk boundaries

## Key Findings

1. **audio_cfg=1.0 is NOT viable for streaming mode with frame_num=41.** The skip_cfg optimization saves ~50% denoising time but the quality tradeoff is unacceptable. ComfyUI uses audio_cfg=1.0 with frame_num=81 (fewer chunks = less cross-chunk degradation). Until we implement VAE temporal tiling to enable frame_num=81, audio_cfg=2.0 is the minimum for acceptable quality.

2. **DPM++ SDE 2nd-order correction is NOT compatible with CFG.** The extrapolation overshoots when denoised estimates include CFG scaling. 1st-order SDE is the correct choice for multi-pass mode.

## Status

**Verified.** 1st-order SDE + audio_cfg=2.0 produces good quality. 5.8s video in ~13 min (~2.2 min/s). ETA accuracy within 1%. Minor color fluctuation at chunk boundaries is expected (per-chunk color correction re-anchoring).
