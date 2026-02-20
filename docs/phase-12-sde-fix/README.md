# Phase 12 — Fix Progressive Degradation: DPM++ SDE + Color Correction

## Problem

Severe progressive quality degradation: first ~2 seconds of video are good, quality
deteriorates steadily, and the last seconds are unrecognizable.

## Root Cause

Phase 11 implemented **DPM++ 2M** but the ComfyUI workflow actually uses **DPM++ SDE**.
These are fundamentally different algorithms:

### DPM++ 2M (deterministic) — what we had

- Uses 2nd-order multistep correction with `old_denoised` state across steps
- Motion frame injection between steps forcibly replaces latent positions, disrupting
  the trajectory assumption that 2nd-order correction relies on
- Deterministic errors compound across chunks via latent passthrough
- No mechanism to self-correct accumulated errors

### DPM++ SDE (stochastic) — what ComfyUI uses

- Adds stochastic noise at each step (`sde-dpmsolver++`)
- The noise term `sigma_t * sqrt(1 - exp(-2h)) * noise` acts as a regularizer
- Prevents deterministic error accumulation across streaming chunks
- Each chunk gets a "fresh start" via the stochastic component

## Changes Applied

### 1. DPM++ SDE solver (`multitalk.py`)

Implemented `sde-dpmsolver++` 1st-order update formula from `fm_solvers.py`:

```python
# For flow matching: alpha = 1 - sigma, lambda = log(alpha/sigma)
alpha_s = 1.0 - sigma_s
alpha_t = 1.0 - sigma_t
lambda_s = log(alpha_s) - log(sigma_s)
lambda_t = log(alpha_t) - log(sigma_t)
h = lambda_t - lambda_s

# SDE update: deterministic part + stochastic noise
x_t = (sigma_t/sigma_s * exp(-h)) * x_s
    + (alpha_t * (1 - exp(-2h))) * denoised
    + sigma_t * sqrt(1 - exp(-2h)) * noise    ← stochastic term
```

The DPM++ 2M implementation is preserved as a fallback (`solver="dpmpp_2m"`).

### 2. Color correction re-enabled (`video_engine.py`)

`color_correction_strength`: 0.0 → **1.0** (original vendor default)

Applied per-chunk AFTER the latent passthrough is saved, so it only affects visual
output — cannot hurt generation quality. Anchors each chunk's color statistics to the
reference image, preventing color drift.

### 3. Motion frame increased

`motion_frame`: 9 → **13** (0.52s overlap vs 0.36s, same 8 chunk count)

More temporal context between chunks for better continuity.

### 4. Shared timestamps

All output files (log, audio, video) now share `config.RUN_TIMESTAMP`, set once in
`setup_logging()`. Example: `run_20260219_234500.log` ↔ `tts_20260219_234500.wav`
↔ `video_20260219_234500.mp4`.

## What Was Tried and Failed

| Change | Result |
|---|---|
| DPM++ 2M + shift=11 + 6 steps | Progressive degradation (wrong scheduler) |
| audio_cfg=2.5 | Over-guidance artifacts |
| color_correction=0.5 + audio_cfg=2.5 | Made output worse |
| Euler (Phase 10) | Significantly slower, not working — **DO NOT USE** |

## Current Settings (Phase 12)

```python
VIDEO_LORA_SOLVER = "dpmpp_sde"       # Matches ComfyUI
VIDEO_LORA_SAMPLE_STEPS = 6
VIDEO_LORA_SAMPLE_SHIFT = 11
VIDEO_LORA_TEXT_GUIDE_SCALE = 1.0
VIDEO_LORA_AUDIO_GUIDE_SCALE = 2.0
VIDEO_FRAME_NUM = 41
VIDEO_MOTION_FRAME = 13
color_correction_strength = 1.0       # Vendor default
GPU_THERMAL_THRESHOLD = 60
```

## Performance

Same as Phase 11: ~22 min for 9s video. DPM++ SDE has identical cost to DPM++ 2M per step.

## Status

**DPM++ SDE implemented but not yet tested.** Awaiting user test run.

## Next Steps (Phase 13 — GGUF Migration)

The remaining gap vs ComfyUI is the model format. ComfyUI uses Q4_0 GGUF (~7-8GB)
while we use FP8 safetensors (~18.9GB). GGUF would:

1. Free ~12GB VRAM → enable `frame_num=81` (4 chunks instead of 8)
2. Fewer chunks → less cross-chunk degradation → `audio_cfg=1.0` may work
3. Potentially eliminate block swap entirely
4. Match ComfyUI quality exactly

Available GGUF models (`Kijai/WanVideo_comfy_GGUF`):
- Base Wan2.1 I2V 14B: Q4_0 ~7GB (city96 conversions)
- InfiniteTalk adapter: Single_Q4_K_M 1.4GB, Single_Q8 2.65GB
- Requires: `gguf` Python package + on-demand dequantization (city96/ComfyUI-GGUF)

## Files Modified

- `config.py` — `VIDEO_LORA_SOLVER` changed to `"dpmpp_sde"`, shift kept at 11, motion_frame 9→13
- `video_engine.py` — `color_correction_strength` 0.0→1.0
- `vendor/infinitetalk/wan/multitalk.py` — DPM++ SDE implementation added to denoising loop
- `generate.py` — `config.RUN_TIMESTAMP` set in `setup_logging()`
- `tts_engine.py` — Uses `config.RUN_TIMESTAMP` for audio filename
- `pipeline.py` — Uses `config.RUN_TIMESTAMP` for audio filename
- `docs/plan.md` — Phase 12 documentation
