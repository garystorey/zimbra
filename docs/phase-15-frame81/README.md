# Phase 15 — frame_num=81 & Quality Refinement

## Problem

Video quality degraded (blurring) after Phase 14 speed optimizations reduced
GGUF denoising steps from 6 to 4. The "good quality" run (`run_20260220_185832`,
commit `252138e`) used `VIDEO_LORA_SAMPLE_STEPS = 6` because `generate()` had
no GGUF-specific step override at that time. After commit `bee09bf` added the
GGUF override (`VIDEO_GGUF_LORA_SAMPLE_STEPS = 4`), quality dropped.

Increasing steps back to 6 added ~8 minutes — rejected by user.

## Root Cause

1. **4 steps insufficient for Q4_0 GGUF** — the LightX2V LoRA was designed for
   4 steps on full-precision weights. Q4_0 quantization noise requires more
   refinement to converge.
2. **8 chunks at frame_num=41** — each chunk boundary introduces cross-chunk
   degradation (latent passthrough, color correction). 7 boundaries compounded
   quality loss.

## Fix: frame_num=81 (match ComfyUI)

The user's own ComfyUI workflow uses `frame_window=81, motion_frame=25,
blocks_to_swap=20` on the same 16GB GPU. Adopting the same config:

- **Fewer chunks**: 4 instead of 8 for an 8.7s video
- **Fewer cross-chunk boundaries**: 3 instead of 7
- **Better quality**: User reported 90% (was "significantly degraded")

### Changes Applied

| File              | Change                                                  | Why                                     |
| ----------------- | ------------------------------------------------------- | --------------------------------------- |
| `config.py`       | `VIDEO_GGUF_FRAME_NUM` 41→81                            | Match ComfyUI frame_window              |
| `config.py`       | `VIDEO_GGUF_MOTION_FRAME` 13→25                         | Match ComfyUI for 81-frame chunks       |
| `config.py`       | `VIDEO_GGUF_ACTIVATION_RESERVE` 7.0→8.5 GiB             | 81-frame activations need more headroom |
| `multitalk.py`    | Always offload DiT blocks before first-chunk VAE encode | Conv3D at 81 frames needs VRAM headroom |
| `video_engine.py` | `_skip_block_offload = False`                           | No longer skip — offload is required    |

### ETA Constants (calibrated from `run_20260221_004411`)

| Constant                       | Old (41-frame) | New (81-frame) | Measured                                                 |
| ------------------------------ | -------------- | -------------- | -------------------------------------------------------- |
| `VIDEO_SECS_PER_STEP`          | 17s            | **37s**        | 37.3s/pass avg (35.7 chunk 0, ~38 chunks 1-3)            |
| `VIDEO_VAE_DECODE_SECS`        | 15s            | **34s**        | VAE ~13s + block offload/restore ~15s + overhead         |
| `VIDEO_FIRST_CHUNK_EXTRA_SECS` | 23s            | **35s**        | Block offload 7s + CLIP 5s + VAE encode 9s + restore 13s |

ETA accuracy: estimated 22m 35s vs actual 22m 47s = 0.9% error.

## Measured Performance (`run_20260221_004411`)

- **4 chunks** (was 8), **4 steps**, **2 passes/step** (audio_cfg=2.0)
- **22 persistent + 18 swapped** DiT blocks (was 27+13 at frame_num=41)
- **Per-pass: ~37s** (was ~17s at 41 frames — 2.2x from longer sequence)
- **Total inference: 1366s (22m 47s)**
- **Quality: 90%** — some softness in later chunks, slight color shifts
- GPU temp: 64-66°C (well under 70°C threshold)

### Per-Chunk Denoising

| Chunk | Denoising | Per-pass (avg) | diff_norm (step 1→4) |
| ----- | --------- | -------------- | -------------------- |
| 0     | 285.6s    | 35.7s          | 359→228              |
| 1     | 299.2s    | 37.4s          | 217→186              |
| 2     | 305.2s    | 38.2s          | 109→171              |
| 3     | 304.3s    | 38.0s          | 281→209              |

## Phase 15b: audio_cfg=1.0 (attempted, REVERTED)

### Problem

Per-block compute at 81 frames is ~2.2x slower (attention over 33K vs 17K
tokens). With audio_cfg=2.0, total denoising is 4×4×2×37s = 1184s (~20m).
Fewer chunks doesn't compensate — net result ~22m vs ~19m at frame_num=41.

### Attempted Fix

`VIDEO_GGUF_LORA_AUDIO_GUIDE_SCALE` 2.0→1.0. Hypothesis: with frame_num=81
(only 3 chunk boundaries vs 7), audio_cfg=1.0 might be viable.

### Result: FAILED

Contact sheet comparison (`video_20260221_161802` vs ComfyUI reference
`winston_datamine_00036`) revealed three quality regressions:

1. **Bad lip sync** — mouth barely moves. With `_skip_cfg=True` (text=1.0 +
   audio=1.0), zero classifier-free guidance fires. The model gets no audio
   conditioning signal at all.
2. **Progressive blurriness** — later frames increasingly soft. Without CFG,
   output under-converges and errors accumulate across chunks via latent
   passthrough.
3. **3 arms artifact** — body hallucination in mid-video frames. Without
   guidance steering, model hallucinates freely.

### Revert

`VIDEO_GGUF_LORA_AUDIO_GUIDE_SCALE` 1.0→**2.0**. audio_cfg=2.0 is required
regardless of frame_num. The second forward pass (drop_audio) is essential
for the audio CFG formula to provide real audio conditioning.

**Conclusion:** audio_cfg=1.0 is NOT viable for our rank-32 LoRA, even with
frame_num=81. The rank-32 LoRA may have lost audio conditioning fidelity
compared to the rank-64 LoRA used in ComfyUI workflows. audio_cfg≥2.0 is
the minimum for acceptable lip sync and sharpness.

## Phase 15c: motion_frame=9 + color_correction=0.3 (perf + lip sync fix)

### Problem

With audio_cfg=2.0 restored, a 28s video (`run_20260221_173914`) took **68 min**
(13 chunks, 2.42 min/sec — over the ≤2 min/sec guideline). Lip sync was also
lagging: "mouth moves slower than words."

### Root Cause: motion_frame=25 freezes too many frames

The denoising loop freezes the first `motion_frame` frames of every chunk:

```python
latent[:, :cur_motion_frames_latent_num] = latent_motion_frames  # line 897, 1077
```

With motion_frame=25 and 12 chunk transitions:

- **300 frozen frames out of 702 total = 43% audio-unresponsive**
- stride = 81-25 = 56 → 13 chunks for 28s audio

Frozen frames are completely locked to the previous chunk's motion latent.
The model CANNOT change them regardless of audio input. This is the primary
cause of perceived lip sync lag — nearly half the video ignores audio.

### Fix

1. **`VIDEO_GGUF_MOTION_FRAME`** 25→9 — 0.36s overlap (was 1.0s)
   - stride = 72 → **10 chunks** (was 13)
   - Frozen frames: 81/702 = **12%** (was 43%)
   - Direct latent passthrough ensures smooth chunk boundaries

2. **`color_correction_strength`** 1.0→0.3 — mild anchoring
   - 1.0 overcorrected in later chunks, flattening contrast
   - Reduced visibility of subtle mouth/skin color changes
   - 0.3 provides mild color anchoring without overcorrection

### Expected Performance (28s audio)

- 10 chunks × 4 steps × 2 passes × 35s = 2800s denoising
- VAE: 10 × 34 + 35 = 375s
- Total: ~3175s ≈ **53 min → 1.88 min/sec** ✓

## Current Settings (Phase 15 final)

- frame_num=81, motion_frame=9
- audio_cfg=2.0 (2 passes/step — required for lip sync)
- DPM++ SDE solver, 4 steps, shift=11
- color_correction=0.3
- Expected: ~1.9 min per second of audio

## Remaining Issues

### 1. Softness in later chunks

Residual progressive quality loss across 3 chunk boundaries. Likely from:

- Accumulated latent passthrough imprecision
- Q4_0 quantization noise compounding
- color_correction_strength=1.0 may overcorrect in later chunks

### 2. Slight color shifts

color_correction_strength=1.0 anchors to the reference frame. May need tuning
(0.5-0.7) for 81-frame chunks where each chunk covers more temporal range.

## What Was Tried and Failed (cumulative)

- `audio_cfg=1.0` at frame_num=41: progressive blur (7+ chunk boundaries) ❌
- `audio_cfg=1.0` at frame_num=81: bad lip sync, blur, 3 arms (zero CFG) ❌
- `audio_cfg=2.5`: over-guidance artifacts ❌
- DPM++ 2M solver: progressive degradation ❌
- DPM++ SDE 2nd-order: overshoots with CFG ❌
- Euler solver: slower, not working ❌
- `color_correction=0.5 + audio_cfg=2.5`: made output worse ❌
- `frame_num=81` without block offload: OOM during VAE encode ❌
- `motion_frame=25` without frame_num=81: doubled chunks, didn't help ❌
- `motion_frame=25` with frame_num=81 + 28s audio: 13 chunks, 68 min, 43% frozen frames ❌
- 6 steps: good quality but +8 min render time — rejected ❌

## Phase 15d: I2V rank-64 LoRA (match ComfyUI)

### Problem

We were using the **wrong LoRA type** — a T2V (text-to-video) rank-32 variant
for an I2V (image-to-video) pipeline. ComfyUI uses the correct I2V rank-64.

|      | Old (wrong)               | New (ComfyUI)                          |
| ---- | ------------------------- | -------------------------------------- |
| Type | T2V                       | **I2V**                                |
| Rank | 32                        | **64**                                 |
| Size | 317 MB                    | **738 MB**                             |
| File | `Wan21_T2V_14B_...rank32` | `lightx2v_I2V_14B_480p_...rank64_bf16` |

The T2V LoRA was distilled for text-to-video generation, not image-to-video.
This likely weakened audio conditioning, forcing us to use audio_cfg=2.0.
With the correct I2V LoRA, audio_cfg=1.0 may become viable again.

### Changes

- `config.py`: Updated `LIGHTX2V_LORA_FILENAME` and paths
- `model_manager.py`: Updated download description
- New LoRA auto-downloads on next run (~738 MB from `Kijai/WanVideo_comfy`)

### Results (`run_20260221_190632`)

Quality: **"pretty close to spot on"** — consistent face, color, and detail across
all 3 chunks. No artifacts, no progressive blur. I2V rank-64 is a clear
improvement over T2V rank-32.

| Metric               | Value                                  |
| -------------------- | -------------------------------------- |
| Audio                | 8.7s (3 chunks at frame_num=81)        |
| Inference            | 968s (16.1 min)                        |
| **Rate**             | **1.85 min/sec**                       |
| Denoising            | 860s (89% of inference)                |
| 2nd pass (audio CFG) | ~430s — half the total                 |
| VAE decode           | ~40s (3 chunks)                        |
| Per-pass avg         | 35.8s (thermal creep: 0.73→0.91 s/blk) |
| swap_wait            | 0.0s everywhere (prefetch fully hides) |

### ETA Calibration (from `run_20260221_190632`)

| Constant                       | Old | New     | Measured                        |
| ------------------------------ | --- | ------- | ------------------------------- |
| `VIDEO_SECS_PER_STEP`          | 37s | **36s** | 35.8s/pass avg across 3 chunks  |
| `VIDEO_GGUF_SECS_PER_STEP`     | 37s | **36s** | Same measurement                |
| `VIDEO_VAE_DECODE_SECS`        | 34s | **28s** | VAE ~13s + offload/restore ~10s |
| `VIDEO_FIRST_CHUNK_EXTRA_SECS` | 35s | **24s** | CLIP 10s + VAE encode 8s + 6s   |

ETA check: 3×4×2×36 + 3×28 + 24 = **972s** vs actual **968s** = 0.4% error ✓

## Phase 15e: audio_cfg=1.0 with I2V rank-64 LoRA

### Problem

With audio_cfg=2.0, the 2nd forward pass per step (drop_audio for CFG) accounts
for **half the total denoising time** (~430s of ~860s). Eliminating it would
bring the rate from 1.85 min/sec to ~1.0 min/sec.

### Hypothesis

Phase 15b tested audio_cfg=1.0 with the **wrong T2V rank-32 LoRA** and it failed
(bad lip sync, blur, 3 arms). The T2V LoRA was distilled for text-to-video, not
image-to-video — it likely had degraded audio conditioning fidelity, requiring
CFG amplification (audio_cfg≥2.0) to compensate.

The I2V rank-64 LoRA was designed for the correct pipeline type. With stronger
native audio conditioning, the model may produce acceptable output without CFG
amplification (audio_cfg=1.0, skip_cfg=True, 1 pass/step).

### Change

`VIDEO_GGUF_LORA_AUDIO_GUIDE_SCALE` 2.0 → **1.0**

### Expected Performance (8.7s audio)

- 3 chunks × 4 steps × **1 pass** × 36s = 432s denoising (was 860s)
- Overhead: 3 × 28 + 24 = 108s
- Total: ~540s ≈ **9 min → 1.03 min/sec** (was 1.85)

### What to Watch For

1. **Lip sync** — mouth must visibly track speech. Phase 15b failure was "mouth
   barely moves." If same symptom, revert immediately.
2. **Progressive blur** — later chunks should stay sharp. Under-convergence
   without CFG compounds across chunks via latent passthrough.
3. **Body artifacts** — Phase 15b had "3 arms" hallucination. Without guidance
   steering, model may hallucinate freely.

### Results (`run_20260221_223335`)

**audio_cfg=1.0 WORKS with I2V rank-64 LoRA!** Contact sheet shows consistent
face, good color, visible mouth movement, no blur, no artifacts.

| Metric       | audio_cfg=2.0   | audio_cfg=1.0      | Change   |
| ------------ | --------------- | ------------------ | -------- |
| Inference    | 968s (16.1 min) | **521s (8.7 min)** | **-46%** |
| Rate         | 1.85 min/sec    | **1.06 min/sec**   | **-43%** |
| Denoising    | 860s            | 405s               | -53%     |
| Per-step avg | ~72s (2 passes) | **~34s (1 pass)**  | -53%     |
| GPU temp     | 65-67°C         | 65-66°C            | —        |

### Per-Chunk Denoising (1 pass/step)

| Chunk | Denoising | Per-step (avg) | Thermal creep   |
| ----- | --------- | -------------- | --------------- |
| 0     | 130.0s    | 32.5s          | 0.75→0.82 s/blk |
| 1     | 135.7s    | 33.9s          | 0.75→0.85 s/blk |
| 2     | 139.3s    | 34.8s          | 0.75→0.87 s/blk |

### ETA Calibration (from `run_20260221_223335`)

| Constant                   | Old | New     | Measured                      |
| -------------------------- | --- | ------- | ----------------------------- |
| `VIDEO_SECS_PER_STEP`      | 36s | **34s** | 33.7s/pass avg at 1-pass mode |
| `VIDEO_GGUF_SECS_PER_STEP` | 36s | **34s** | Same measurement              |

ETA check: 3×4×1×34 + 3×28 + 24 = **516s** vs actual **521s** = 1.0% error ✓

### Why It Works Now (vs Phase 15b failure)

Phase 15b used the **T2V rank-32 LoRA** (text-to-video, wrong model type). The
T2V LoRA was distilled for text-to-video generation where audio conditioning
doesn't exist. When applied to the I2V pipeline, it degraded the audio
conditioning pathways, requiring CFG amplification (audio_cfg≥2.0) to
compensate.

The **I2V rank-64 LoRA** (image-to-video, correct model type) preserves audio
conditioning fidelity. The model produces properly audio-conditioned output in
a single forward pass, without needing CFG to amplify the signal.

### Revert Plan

If lip sync is insufficient after watching video: `--audio-cfg 2.0` CLI override
or change `VIDEO_GGUF_LORA_AUDIO_GUIDE_SCALE` to 2.0 in config.py.

## Current Settings (Phase 15e final)

- frame_num=81, motion_frame=9
- **audio_cfg=1.0** (1 pass/step — skip CFG)
- DPM++ SDE solver, 4 steps, shift=11
- color_correction=0.3
- I2V rank-64 LoRA (lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16)
- **Expected: ~1.06 min per second of audio**

## What Was Tried and Failed (cumulative)

- `audio_cfg=1.0` at frame_num=41 with T2V rank-32 LoRA: progressive blur ❌
- `audio_cfg=1.0` at frame_num=81 with T2V rank-32 LoRA: bad lip sync, blur, 3 arms ❌
- `audio_cfg=1.0` at frame_num=81 with **I2V rank-64 LoRA: WORKS ✓**
- `audio_cfg=2.5`: over-guidance artifacts ❌
- DPM++ 2M solver: progressive degradation ❌
- DPM++ SDE 2nd-order: overshoots with CFG ❌
- Euler solver: slower, not working ❌
- `color_correction=0.5 + audio_cfg=2.5`: made output worse ❌
- `frame_num=81` without block offload: OOM during VAE encode ❌
- `motion_frame=25` without frame_num=81: doubled chunks, didn't help ❌
- `motion_frame=25` with frame_num=81 + 28s audio: 13 chunks, 68 min, 43% frozen ❌
- 6 steps: good quality but +8 min render time — rejected ❌
- T2V rank-32 LoRA: wrong model type, degraded audio conditioning ❌
