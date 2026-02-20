# Phase 11 — ComfyUI Workflow Alignment

## Goal

Match the user's ComfyUI workflow parameters for speed and quality.

## ComfyUI Reference Workflow

```
Scheduler: DPM++ SDE, steps=6, shift=11, audio_cfg=1.0
frame_window=81, motion_frame=25, colormatch=disabled
Model: Q4_0 GGUF (~7-8GB), blocks_to_swap=20
LoRA: lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16
```

## Changes Applied

- **DPM++ 2M solver** implemented in `multitalk.py` (replaced Euler). 2nd-order multistep
  with r-ratio fallback (r<0.15 → 1st order). Velocity→denoised: `x_0 = x_t - sigma * v`.
  NOTE: This was later found to be the **wrong scheduler** — ComfyUI uses DPM++ SDE, not
  DPM++ 2M. Fixed in Phase 12.
- **Steps**: 8 → 6 (DPM++ converges faster than Euler)
- **Shift**: 2 → 11 (concentrates timesteps at high noise)
- **Color correction**: disabled (`color_correction_strength=0.0`)
- **Solver param**: added `VIDEO_LORA_SOLVER`, plumbed through `video_engine.py` → `multitalk.py`

## What Was Tried and Reverted

| Change | Result | Reverted to |
|---|---|---|
| `audio_cfg=1.0` (ComfyUI default) | Progressive blur — our system has more chunks (frame_num=41 vs 81) so needs real audio CFG | 2.0 |
| `frame_num=81` | OOM during LoRA merge (only 13 persistent blocks with 9GB activation reserve) | 41 |
| `motion_frame=25` | Doubled chunk count (14 vs 8) without fixing blur — blur caused by no audio guidance, not temporal context | 9 |
| `activation_reserve=9.0 GiB` | OOM | 7.0 |
| `audio_cfg=2.5` | Over-guidance artifacts | 2.0 |
| `color_correction=0.5` | Combined with audio_cfg=2.5, made output worse | 0.0 |

## Key Insight

Our system differs from ComfyUI in two critical ways:

1. **FP8 (18.9GB) vs Q4_0 GGUF (~7GB)** — we can't fit `frame_num=81`
2. **Without frame_num=81**, we have more chunks → more cross-chunk degradation → need `audio_cfg > 1.0`

The GGUF model migration (Phase 13) would close this gap entirely.

## Settings After Phase 11

```
DPM++ 2M solver (WRONG — should be SDE, fixed in Phase 12)
6 steps, shift=11
audio_cfg=2.0, text_cfg=1.0
frame_num=41, motion_frame=9
color_correction=disabled
GPU thermal threshold: 60°C
VIDEO_SECS_PER_STEP: 14s/pass
```

## Performance

8 chunks × 6 steps × 2 passes × ~14s = ~1,344s ≈ **22 min for 9s video**

## Files Modified

- `config.py` — Added `VIDEO_LORA_SOLVER`, `VIDEO_LORA_SAMPLE_SHIFT`, updated steps/shift/audio_cfg
- `video_engine.py` — Solver selection logic, solver param passthrough to `generate_infinitetalk()`
- `vendor/infinitetalk/wan/multitalk.py` — DPM++ 2M implementation in denoising loop, `solver` parameter plumbing
- `docs/plan.md` — Phase 11 documentation
