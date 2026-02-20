# Zimbra Project — Full Context Handoff

Use this document to bootstrap a new Cascade window with full project context.

---

## Project Overview

**Talking Head Video Generator** — Python CLI that takes a voice-clone reference audio, a portrait image, and a text script, then generates a "talking head" video locally on an NVIDIA RTX 4090 Laptop GPU (16GB VRAM).

**Workspace:** `C:\Users\GaryS\Desktop\zimbra`
**Venv:** `C:\Users\GaryS\Desktop\zimbra\venv` (Python 3.10.11)
**OS:** Windows 11

---

## Architecture

```
generate.py          CLI entry point (argparse, --quant gguf/fp8)
pipeline.py          Orchestrates TTS → Video → Post-process
tts_engine.py        Chatterbox Turbo TTS wrapper (voice cloning)
video_engine.py      InfiniteTalk Wan2.1-I2V-14B wrapper
config.py            All paths, defaults, GPU settings
model_manager.py     Auto-download logic for all model weights
gguf_loader.py       GGUF dequantization, loading, patched Linear, LoRA-as-patches
fp8_optimization.py  Native FP8 matmul via torch._scaled_mm (legacy)
utils.py             GPU thermal management, helpers
```

**Vendor code:** `vendor/infinitetalk/` — cloned InfiniteTalk repo. Key modified file:

- `vendor/infinitetalk/wan/multitalk.py` — denoising loop with DPM++ SDE/2M/Euler solvers, motion frame injection, latent passthrough, color correction

**Models run sequentially** — TTS fully unloads before video model loads.

---

## Tech Stack

| Component    | Library                        | Notes                                                |
| ------------ | ------------------------------ | ---------------------------------------------------- |
| TTS          | Chatterbox Turbo (350M params) | Voice cloning from ~10s ref clip                     |
| Video        | InfiniteTalk (Wan2.1-I2V-14B)  | Audio-driven, any art style, 480p/720p               |
| Quantization | GGUF Q4_0 + Q8 adapter         | ~12.8GB model on 16GB VRAM (was FP8 ~18.9GB)         |
| VRAM mgmt    | Block swap (Kijai approach)    | Auto-tuned based on free VRAM                        |
| Solver       | DPM++ SDE (sde-dpmsolver++)    | Stochastic — prevents cross-chunk error accumulation |

**Dependencies (key pins):**

- torch==2.4.1+cu121, xformers==0.0.28.post1, flash_attn==2.7.4.post1
- diffusers>=0.31.0, transformers>=4.49.0,<5
- Python 3.10 (required — 3.11+ breaks some deps)
- Chatterbox installed with `--no-deps` to avoid conflicts

---

## Current Configuration (Phase 14 — GGUF)

```python
# config.py — GGUF mode (default)
VIDEO_QUANT = "gguf"                # GGUF Q4_0 base + Q8 adapter (was "fp8")
VIDEO_GGUF_FRAME_NUM = 41           # 4n+1, safe for VAE 3D-conv on 16GB
VIDEO_GGUF_MOTION_FRAME = 13        # Overlap frames between chunks
VIDEO_GGUF_LORA_SAMPLE_STEPS = 4    # LightX2V designed for 4 steps
VIDEO_GGUF_LORA_AUDIO_GUIDE_SCALE = 2.0  # 2 passes/step (audio CFG for lip sync)
VIDEO_GGUF_SECS_PER_STEP = 17      # Measured ~16.8s/pass avg on RTX 4090 Laptop 16GB
VIDEO_GGUF_ACTIVATION_RESERVE = 7.0 # GiB — dequant temporaries need headroom
VIDEO_VAE_DECODE_SECS = 15           # Per-chunk: VAE decode (~7s) + inter-chunk gap (~8s)
VIDEO_FIRST_CHUNK_EXTRA_SECS = 23    # CLIP encode (~3s) + VAE encode (~20s), first chunk only

# LoRA accelerator
VIDEO_LORA_SOLVER = "dpmpp_sde"    # DPM++ SDE
VIDEO_LORA_SAMPLE_STEPS = 6        # LoRA distilled (overridden by GGUF_LORA_SAMPLE_STEPS)
VIDEO_LORA_SAMPLE_SHIFT = 11       # ComfyUI-validated
VIDEO_LORA_TEXT_GUIDE_SCALE = 1.0   # Must be 1.0 for distilled LoRA

GPU_THERMAL_THRESHOLD = 70          # Pause between chunks if GPU > 70°C

# video_engine.py
color_correction_strength = 1.0     # Vendor default, per-chunk, anchored to ref image
config.RUN_TIMESTAMP                # Set in setup_logging(), used for log/audio/video filenames
```

**Measured performance (GGUF):** 5 chunks × 4 steps × 2 passes × ~17s/pass + ~98s overhead ≈ **~13 min for 5.8s video (~2.2 min/s)**

**Previous (FP8):** 8 chunks × 6 steps × 2 passes × ~14s/pass ≈ 22 min (~2.4 min/s)

---

## ComfyUI Reference Workflow (our target)

```
Scheduler: DPM++ SDE, steps=6, shift=11, audio_cfg=1.0
frame_window=81, motion_frame=25, colormatch=disabled
Model: Q4_0 GGUF (~7-8GB), blocks_to_swap=20
LoRA: lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16
```

**Phase 13 closes this gap:** We now use GGUF Q4_0 base + Q8 adapter (~12.8GB), matching ComfyUI's model format. frame_num=41 and audio_cfg=2.0 differ from ComfyUI (frame_num=81 needs VAE temporal tiling; audio_cfg=1.0 causes progressive blur with more chunks).

---

## Phase History (what was tried)

### Phases 1-5: Foundation

- Phase 1: Project setup, venv, dependency resolution
- Phase 2: Chatterbox Turbo TTS engine (9/9 tests passed)
- Phase 3: InfiniteTalk video engine
- Phase 4: Pipeline + CLI
- Phase 5: Error handling, README, docs

### Phases 6-9: Performance

- Phase 6: Eliminated per-step CPU↔GPU transfer bottleneck
- Phase 7: Block swap (Kijai approach) — swap entire DiT blocks with async prefetch
- Phase 8: FP8 device mismatch fix (scale_weight as buffer)
- Phase 9: VRAM spill fix — activation_reserve 4→7 GiB, 23→17 persistent blocks

### Phase 10: Progressive Blur Fix

**Tried and learned:**

- audio_cfg=1.0 → progressive blur (LoRA only distills text, audio layers need real CFG)
- Euler + 4 steps + shift=2 → **significantly slower, not working** ⚠️ DO NOT USE EULER
- motion_frame=25 → doubled chunk count, didn't fix blur
- Moved `_saved_motion_latent` BEFORE VAE decode (eliminates VAE round-trip error)

### Phase 11: ComfyUI Alignment

**Tried and learned:**

- Implemented DPM++ 2M solver (but ComfyUI actually uses DPM++ SDE — wrong scheduler!)
- frame_num=81 → OOM during LoRA merge
- audio_cfg=2.5 → over-guidance artifacts
- color_correction=0.5 + audio_cfg=2.5 → made output worse
- activation_reserve=9.0 → OOM

### Phase 12: Fix Progressive Degradation

**Root cause:** Phase 11 implemented DPM++ 2M but ComfyUI uses DPM++ SDE.

- DPM++ 2M is deterministic — errors compound across chunks
- DPM++ SDE adds stochastic noise per step, preventing error accumulation

### Phase 13: GGUF Migration + Fixes

- Full GGUF loading pipeline (gguf_loader.py) — Q4_0 base + Q8 adapter
- Phase 13.1: Fixed hang (skip block offload for CLIP/VAE encode, frame_num 81→41, activation_reserve 7.0, VAE bfloat16, audio_cfg 1.0→2.0)
- Phase 13.2: Fixed VAE decode perf (always offload DiT blocks before VAE decode). VAE decode: ~7s/chunk (was ~401s).

### Phase 14: Denoising Optimizations (CURRENT)

- Restored `empty_cache()` between forward passes for GGUF (prevents WDDM spill)
- Removed dead motion re-noising code
- DPM++ SDE 2nd-order multistep correction
- **audio_cfg=1.0 tested and reverted to 2.0** — quality degraded (warped features in later chunks)
- ETA constants calibrated from actual run data

**Status:** Verified working. 5.8s video in ~8 min (audio_cfg=1.0) or ~12 min (audio_cfg=2.0).

---

## Immediate Next Steps

### 1. Test with audio_cfg=2.0

Run a full generation to verify quality is restored with audio_cfg=2.0. Expected ~12 min for 5.8s video.

### 2. Future Speed Improvements

- **VAE temporal tiling** — Enable frame_num=81 (fewer chunks → less overhead, potentially audio_cfg=1.0)
- **3-step denoising** — DPM++ SDE 2nd-order may converge well enough in 3 steps (25% faster)
- **SageAttention** — If Windows-compatible wheels become available

To fall back to FP8: `python generate.py --quant fp8`

---

## Key Files and What They Do

| File                       | Purpose                          | Key details                                                                        |
| -------------------------- | -------------------------------- | ---------------------------------------------------------------------------------- |
| `config.py`                | All configuration                | LoRA settings at lines 96-104, block swap at 113-134, thermal at 123-129           |
| `video_engine.py`          | Video generation                 | `load_model()` (282-454), `generate()` (506-787), solver/shift selection (554-573) |
| `generate.py`              | CLI entry                        | `setup_logging()` sets `config.RUN_TIMESTAMP` (line 115-117)                       |
| `pipeline.py`              | TTS→Video orchestration          | Audio output uses `config.RUN_TIMESTAMP` (line 200)                                |
| `tts_engine.py`            | Chatterbox TTS                   | Audio output uses `config.RUN_TIMESTAMP` (line 295)                                |
| `vendor/.../multitalk.py`  | Core denoising loop              | DPM++ SDE at 1010-1054, DPM++ 2M at 1055-1093, Euler at 1094-1099                  |
| `vendor/.../fm_solvers.py` | Reference solver implementations | sde-dpmsolver++ formula used for our implementation                                |
| `docs/plan.md`             | Full phase history               | Phases 1-12 documented with what worked and what failed                            |
| `gguf_loader.py`           | GGUF loading + dequant           | On-demand dequant, LoRA-as-patches, monkey-patched Linear                          |
| `fp8_optimization.py`      | FP8 weight loading (legacy)      | Native torch.\_scaled_mm approach (used with `--quant fp8`)                        |

---

## Critical Rules

1. **NEVER switch to Euler solver** — Phase 10 proved it's significantly slower and doesn't work
2. **NEVER increase audio_cfg above 2.0** — 2.5 caused over-guidance artifacts
3. **frame_num=81 requires GGUF** — OOMs with FP8 (~18.9GB model)
4. **Read docs/plan.md** before making changes — full history of what was tried and failed
5. **All output files** (log, audio, video) must share `config.RUN_TIMESTAMP`
6. **LoRA requires text_cfg=1.0** — it's a distilled model, CFG is baked in
7. **Block swap activation_reserve**: 7.0 GiB for both FP8 and GGUF
8. **`--quant fp8`** falls back to legacy FP8 path if GGUF has issues
9. **Always offload DiT blocks before VAE decode** — even for GGUF (3D transposed conv needs VRAM headroom)
10. **audio_cfg=1.0 degrades quality** with frame_num=41 — use 2.0 minimum (tested and confirmed)
11. **DPM++ SDE 2nd-order correction breaks with CFG** — extrapolation overshoots when denoised includes CFG scaling. Use 1st-order SDE only.
