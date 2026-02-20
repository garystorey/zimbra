# Talking Head Video Generator — Implementation Plan

A Python CLI app that takes a voice-clone reference audio, a portrait image, and a text script, then generates a "talking head" video locally on an NVIDIA RTX 4090 Laptop GPU (16GB VRAM).

---

## Technology Choices

| Component                 | Library                                                       | Why                                                                                                                                                                             |
| ------------------------- | ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **TTS (voice cloning)**   | [Chatterbox Turbo](https://github.com/resemble-ai/chatterbox) | 350M params, instant voice cloning from ~10s reference clip, paralinguistic tags (`[laugh]`, `[cough]`), emotion control via `exaggeration`/`cfg_weight`, ~2-3GB VRAM           |
| **Talking Head Video**    | [InfiniteTalk](https://github.com/MeiGen-AI/InfiniteTalk)     | Wan2.1-I2V-14B base, audio-driven (wav2vec2), cartoon/3D/photo input, arbitrary aspect ratios, 480p/720p, streaming mode for unlimited length, 16GB VRAM with CPU offload + FP8 |
| **Interface**             | Python CLI (`argparse`)                                       | Simplest to implement; easy to wrap in a UI later                                                                                                                               |
| **Video post-processing** | FFmpeg                                                        | Mux audio+video, set frame rate to 25fps, match input image resolution                                                                                                          |

Models run sequentially — TTS is fully unloaded before the video model loads. InfiniteTalk uses CPU offloading + FP8 quantization to fit in 16GB VRAM.

---

## Dependency Conflict Resolution

Chatterbox and InfiniteTalk have **2 soft dependency conflicts**:

| Package       | Chatterbox pins | InfiniteTalk pins | Resolution                                                                                      |
| ------------- | --------------- | ----------------- | ----------------------------------------------------------------------------------------------- |
| **torch**     | `==2.6.0`       | `==2.4.1` (cu121) | Use `torch==2.4.1+cu121` (InfiniteTalk + xformers + flash_attn tied to 2.4.1; Chatterbox works) |
| **diffusers** | `==0.29.0`      | `>=0.31.0`        | Use `diffusers>=0.31.0` (InfiniteTalk needs >=0.31.0; Chatterbox's usage is minimal)            |

**Strategy: Install Chatterbox with `--no-deps`, then pin unified versions.**

This is a proven approach — the [Chatterbox TTS Server](https://github.com/devnen/Chatterbox-TTS-Server) project uses the same `--no-deps` technique to avoid conflicts. The install order will be:

1. Install `torch==2.4.1+cu121`, `torchvision==0.19.1+cu121`, `torchaudio==2.4.1+cu121` from PyTorch cu121 index
2. Install `xformers==0.0.28` from cu121 index
3. Install `flash_attn==2.7.4.post1` from prebuilt Windows wheel
4. Install InfiniteTalk dependencies from `requirements.txt`
5. Install `chatterbox-tts` with `--no-deps` to get only the Chatterbox source code
6. Install Chatterbox's unique deps that don't conflict: `librosa`, `s3tokenizer`, `conformer`, `resemble-perth`, `safetensors==0.5.3`, `spacy-pkuseg`, `pykakasi`, `pyloudnorm`
7. Verify both imports work: `from chatterbox.tts_turbo import ChatterboxTurboTTS` and InfiniteTalk's pipeline

A **verification test** in Phase 1 will confirm both libraries import and initialize without errors.

---

## Auto-Download Strategy

All models, checkpoints, text encoders, and enhancement weights are **automatically downloaded on first run** — zero manual steps required. A `model_manager.py` module handles this:

| Asset                    | Source                                                | Size (approx) | Auto-Download Method                  |
| ------------------------ | ----------------------------------------------------- | ------------- | ------------------------------------- |
| Chatterbox Turbo weights | HuggingFace (via `from_pretrained`)                   | ~700 MB       | Built-in to Chatterbox library        |
| Wan2.1-I2V-14B-480P      | HuggingFace (`Wan-AI/Wan2.1-I2V-14B-480P`)            | ~5.3 GB       | `huggingface_hub.snapshot_download()` |
| chinese-wav2vec2-base    | HuggingFace (`TencentGameMate/chinese-wav2vec2-base`) | ~360 MB       | `huggingface_hub.snapshot_download()` |
| InfiniteTalk FP8 ckpt    | HuggingFace (`MeiGen-AI/InfiniteTalk`)                | ~26 GB        | `huggingface_hub.snapshot_download()` |
| LightX2V LoRA            | HuggingFace (`Kijai/WanVideo_comfy`)                  | ~317 MB       | `huggingface_hub.hf_hub_download()`   |

**Behavior:**

- On first run, each engine checks if its required weights exist in `models/` before loading
- Missing weights are downloaded with a progress bar (`tqdm`)
- Downloads are atomic (temp file → rename) to avoid corrupt partial downloads
- Subsequent runs skip download and load directly from `models/`
- Models are downloaded automatically on first run with rich progress bars

---

## Project Structure (final)

```
zimbra/
├── README.md
├── requirements.txt           # Pinned dependency list
├── install.bat                # One-click venv creation + dependency install (Windows)
├── install.py                 # Cross-platform install script (creates venv, installs deps)
├── config.py                  # Paths, defaults, GPU settings
├── generate.py                # CLI entry point
├── tts_engine.py              # Chatterbox Turbo wrapper
├── video_engine.py            # InfiniteTalk wrapper
├── pipeline.py                # Orchestrates TTS → video → post-process
├── model_manager.py           # Auto-download logic for all model weights
├── utils.py                   # File validation, ffmpeg helpers
├── fp8_optimization.py        # Native FP8 matmul via torch._scaled_mm
├── inputs/                    # Default input assets (CLI falls back here)
│   ├── voice_ref/
│   │   └── default.wav        # Default voice clone reference (.wav or .mp3)
│   ├── images/
│   │   └── default.png        # Default portrait image
│   └── scripts/
│       └── default.txt        # Default script text
├── outputs/                   # Generated assets
│   ├── audio/                 # TTS-generated audio
│   └── video/                 # Final talking-head videos
├── models/                    # Downloaded model weights (gitignored)
│   ├── infinitetalk/          # Wan2.1-I2V-14B, wav2vec2, InfiniteTalk ckpts
│   └── chatterbox/
├── vendor/                    # Cloned repos (gitignored)
│   └── infinitetalk/          # InfiniteTalk source
├── venv/                      # Dedicated virtual environment (gitignored)
└── docs/
    ├── plan.md                # This plan
    ├── phase-1-setup/
    ├── phase-2-tts/
    ├── phase-3-video/
    ├── phase-4-pipeline/
    ├── phase-5-polish/
    ├── phase-6-performance/
    ├── phase-7-block-swap/
    ├── phase-8-optimization/
    ├── phase-9-swap-fix/
    ├── phase-10-speed/
    ├── phase-11-comfyui/
    ├── phase-12-sde-fix/
    └── phase-13-gguf/
```

---

## Implementation Phases

### Phase 1 — Project Setup & Environment

**Goal:** Scaffold the project, install dependencies, verify GPU access.

### Phase 2 — TTS Engine (Chatterbox Turbo)

**Goal:** Generate cloned-voice audio from a text script and reference audio file.

### Phase 3 — Video Engine (InfiniteTalk)

**Goal:** Generate a fully-animated talking-head video (head, face, expressions, upper body) from a portrait image and audio file. Supports cartoon/3D/photo input and arbitrary aspect ratios.

### Phase 4 — Pipeline & CLI

**Goal:** Wire everything together into a single CLI command.

### Phase 5 — Polish & Documentation

**Goal:** Error handling, README, and final cleanup.

### Phase 6 — Video Performance Fix

**Goal:** Eliminate the per-step CPU↔GPU weight transfer bottleneck. T5 runs on CPU, VAE/CLIP offloaded before VRAM auto-tune (~14.5B persistent / 18.9B total). Vendor overflow forward methods patched from `copy.deepcopy().to()` to in-place `.to()` transfers.

### Phase 7 — Block Swap Performance Overhaul

**Goal:** Replace the vendor's per-layer VRAM management with block-level swapping + async CUDA stream prefetch (Kijai's ComfyUI approach). Swap entire DiT blocks instead of individual layers, prefetch next block while current one computes. Target: ≤2 min per second of video.

### Phase 8 — Performance Optimization & FP8 Device Mismatch Fix

**Goal:** Reduce per-chunk overhead (chunk caching, pinned memory, inference_mode) and fix `_scaled_mm` device mismatch crash. Root cause: `scale_weight` stored as plain tensor attribute (invisible to `nn.Module.to()`), combined with `non_blocking` transfers and LoRA replacing pinned data. Fixed by registering `scale_weight` as a buffer, synchronous current-block transfer, and device guards.

### Phase 9 — Block Swap VRAM Spill Fix

**Goal:** Fix 20-26x slowdown in swapped blocks caused by Windows WDDM shared memory spill.

**Root cause:** With 25 persistent blocks (11.75GB) + 1 swap buffer (0.47GB) + non-block params (0.5GB) = 12.72GB on GPU during swap block compute, plus ~3.5GB intermediates = ~16.2GB peak, exceeding 16GB physical VRAM. Windows WDDM transparently spills to system RAM via PCIe, causing progressive per-block slowdown (self_attn 2.5x → ffn 55x).

**Fix:** Increased `activation_reserve` from 4.0 to 5.0 GiB. Result: 23→17 persistent. Swap blocks: 5.5s/blk → 0.27s/blk.

### Phase 10 — Video Quality: Progressive Blur Fix

**Goal:** Fix progressive quality degradation (blur) across streaming chunks.

**Root cause analysis & attempts:**

1. **audio_guide_scale=1.0 (no audio CFG)** — LightX2V LoRA only distills TEXT guidance. InfiniteTalk's audio conditioning layers need real CFG. Without it, the model's per-chunk output under-converges, and errors accumulate across chunks via the motion latent passthrough.

2. **Euler solver with 4 steps + shift=2** — Coarse integration: timestep schedule [1000,800,500,2,0] has one step doing 50% of denoising in a single jump. Insufficiently converged output compounds across chunks.

3. **`_saved_motion_latent` placed after VAE decode** — Motion latent was saved after VAE round-trip, accumulating reconstruction error.

**Fixes applied (Phase 10):**

- `VIDEO_LORA_AUDIO_GUIDE_SCALE`: 1.0 → 2.5 (2 passes/step, real audio guidance)
- `VIDEO_LORA_SAMPLE_STEPS`: 4 → 8 (7 effective steps, max dt=0.25)
- Moved `_saved_motion_latent` save BEFORE VAE decode in multitalk.py
- Suppressed `[diag]` logging to `logging.debug` (only shows with `--verbose`)
- `print(f"diff_norm: ...")` → `logging.debug(...)` in multitalk_utils.py

**Result:** Blur eliminated but performance unacceptable (~39 min for 9s video at 8 steps × 2 passes = 16 passes/chunk).

### Phase 11 — ComfyUI Workflow Alignment

**Goal:** Match the user's ComfyUI workflow parameters for speed and quality.

**ComfyUI reference workflow:**

- Scheduler: DPM++ SDE, steps=6, shift=11, audio_cfg=1.0
- frame_window=81, motion_frame=25, colormatch=disabled
- Model: Q4_0 GGUF, blocks_to_swap=20
- LoRA: lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16

**Changes applied:**

- **DPM++ 2M solver** implemented in multitalk.py (replaces Euler). 2nd-order multistep with r-ratio fallback (r<0.15 → 1st order). Velocity→denoised: `x_0 = x_t - sigma * v`.
- **Steps**: 8 → 6 (DPM++ 2M converges faster than Euler)
- **Shift**: 2 → 11 (concentrates timesteps at high noise where 2nd-order correction helps most)
- **Color correction**: disabled (color_correction_strength=0.0)
- **Solver param**: added `VIDEO_LORA_SOLVER = "dpmpp_2m"`, plumbed through video_engine.py → multitalk.py

**What was tried and reverted:**

- `audio_cfg=1.0` (ComfyUI setting) — causes progressive blur on our system. ComfyUI avoids this because it uses frame_window=81 (fewer chunks). Reverted to 2.0.
- `frame_num=81` — OOMs during LoRA merge on 16GB (only 13 persistent blocks with 9GB activation reserve). Reverted to 41.
- `motion_frame=25` — Doubled chunk count (14 vs 8) without fixing blur. The blur is caused by insufficient audio guidance, not temporal context. Reverted to 9.
- `activation_reserve=9.0 GiB` — Too aggressive, caused OOM. Reverted to 7.0.

**Current settings (Phase 11 final):**

- DPM++ 2M solver, 6 steps, shift=11
- audio_cfg=2.0 (2 passes/step — prevents progressive blur)
- frame_num=41, motion_frame=9
- color_correction=disabled
- GPU thermal threshold: 60°C (was 75°C)
- VIDEO_SECS_PER_STEP: 14s/pass (measured)

**Key insight:** Our system differs from ComfyUI in two critical ways:

1. FP8 (18.9GB) vs Q4_0 GGUF (~7GB) — we can't fit frame_num=81
2. Without frame_num=81, we have more chunks → more cross-chunk degradation → need audio CFG > 1.0

**Expected performance:** 8 chunks × 6 steps × 2 passes × ~14s = ~1344s ≈ **22 min for 9s video**.

### Phase 12 — Fix Progressive Degradation: DPM++ SDE + Color Correction

**Goal:** Fix severe progressive quality degradation (first 2s good, last seconds unrecognizable).

**Root cause:** Phase 11 implemented DPM++ 2M, but the ComfyUI workflow actually uses **DPM++ SDE**. These are fundamentally different algorithms:

- **DPM++ 2M** (deterministic): Uses 2nd-order multistep correction with `old_denoised` state. The motion frame injection between steps disrupts the trajectory assumption, causing the 2nd-order correction to amplify errors. Deterministic errors compound across chunks.
- **DPM++ SDE** (stochastic): Adds noise at each step (`sde-dpmsolver++`). The stochastic noise term acts as a regularizer that prevents deterministic error accumulation across streaming chunks.

**What was tried and failed:**

- DPM++ 2M + shift=11 + 6 steps: progressive degradation (wrong scheduler — ComfyUI uses SDE, not 2M)
- audio_cfg=2.5: over-guidance artifacts
- color_correction=0.5 + audio_cfg=2.5: made output worse
- Euler (Phase 10): significantly slower, not working

**Changes applied (Phase 12):**

- **Solver**: DPM++ 2M → **DPM++ SDE** (matches ComfyUI workflow). Implemented `sde-dpmsolver++` 1st-order update formula from `fm_solvers.py` in the multitalk.py denoising loop.
- **Color correction**: 0.0 → **1.0** (original vendor default, applied per-chunk AFTER latent passthrough is saved)
- **motion_frame**: 9 → **13** (0.52s overlap, same 8 chunk count, better temporal continuity)
- **Timestamps**: Video/audio filenames now share `config.RUN_TIMESTAMP` with log file

**Current settings (Phase 12):**

- DPM++ SDE solver, 6 steps, shift=11
- audio_cfg=2.0, text_cfg=1.0
- frame_num=41, motion_frame=13
- color_correction=1.0 (full, per-chunk, anchored to reference image)
- GPU thermal threshold: 60°C

**Performance:** Same as Phase 11 (~22 min for 9s video). DPM++ SDE has identical cost to DPM++ 2M per step.

### Phase 13 — GGUF Migration

**Goal:** Replace FP8 (~18.9GB) with GGUF Q4_0 base + Q8 adapter (~12.8GB) to match ComfyUI's model format, enable frame_num=81, audio_cfg=1.0, and dramatically improve performance.

**Changes applied:**

- **New file `gguf_loader.py`**: Full GGUF loading pipeline ported from city96/ComfyUI-GGUF (Apache 2.0). Includes:
  - Dequantization functions for all GGUF quant types (Q4_0, Q4_K, Q5_K, Q6_K, Q8_0, etc.)
  - `GGMLTensor` subclass carrying quantization metadata
  - `load_gguf_state_dict()` — reads GGUF files via `gguf.GGUFReader`, wraps tensors as `GGMLTensor`
  - `load_gguf_into_model()` — loads state_dict into WanModel (quantized Linear weights stay packed, others dequantized at load time)
  - `patch_gguf_linears()` — monkey-patches Linear.forward() for on-the-fly dequantization
  - `apply_lora_as_patches()` — attaches LoRA deltas as buffers (applied during monkey-patched forward, avoids dequant→merge→requant)
  - `load_gguf_weights()` — high-level: loads base GGUF + adapter GGUF, merges, loads into model, patches

- **`config.py`**: New GGUF paths (`GGUF_DIR`, `GGUF_BASE_PATH`, `GGUF_ADAPTER_PATH`), HF repos (`GGUF_BASE_HF_REPO`, `GGUF_ADAPTER_HF_REPO`), GGUF-specific overrides (`VIDEO_GGUF_FRAME_NUM=41`, `VIDEO_GGUF_MOTION_FRAME=13`, `VIDEO_GGUF_LORA_AUDIO_GUIDE_SCALE=1.0`, `VIDEO_GGUF_ACTIVATION_RESERVE=7.0`). Default `VIDEO_QUANT` changed from `"fp8"` to `"gguf"`.

- **`model_manager.py`**: New `ensure_gguf_models()` downloads base Wan2.1-I2V-14B Q4_0 GGUF (~10.2GB from city96) and InfiniteTalk Single Q8 adapter (~2.65GB from Kijai).

- **`video_engine.py`**: GGUF loading path in `load_model()` (creates model on meta device, loads via `gguf_loader.load_gguf_weights()`). LoRA applied as patches for GGUF mode. `generate()` uses GGUF-specific frame_num/motion_frame/audio_cfg/ETA.

- **`generate.py`**: New `--quant TYPE` CLI option (`gguf` or `fp8`).

- **`vendor/.../multitalk.py`**: Added `"gguf"` to valid quant types, creates model on meta device for GGUF (same as native_fp8), skips dtype conversion. All `offload_blocks()`/`restore_blocks()` calls gated by `_skip_block_offload` flag. Key generation milestones promoted to INFO logging.

- **`requirements.txt`**: Added `gguf>=0.6.0`.

**GGUF models:**

- Base: `wan2.1-i2v-14b-480p-Q4_0.gguf` (~10.2GB) from `city96/Wan2.1-I2V-14B-480P-gguf`
- Adapter: `Wan2_1-InfiniteTalk_Single_Q8.gguf` (~2.65GB) from `Kijai/WanVideo_comfy_GGUF`

**Bug fixes (Phase 13.1 — GGUF hang diagnosis):**

- **Logging bug**: Root logger was set to INFO with `--verbose`, silently dropping ALL `logging.debug()` calls before they reached the file handler. Fixed: root always at DEBUG, console handler controls visibility.
- **Hang root cause**: `offload_blocks()`/`restore_blocks()` moved ~10GB of GGUF blocks off/on GPU for CLIP/VAE encoding. Combined with `frame_num=81` VAE 3D-conv activation spike, this caused OOM/hang on 16GB. ComfyUI keeps GGUF blocks on their assigned device — they never move after initial placement.
- **Fix: skip offload/restore for CLIP/VAE encode**: Set `_skip_block_offload=True` on pipeline. GGUF model (~10GB) is small enough for CLIP (~1.2GB) + VAE (~0.5GB) to fit alongside on 16GB without moving DiT blocks during encoding. Blocks are still offloaded before VAE decode (heavy 3D transposed conv needs VRAM).
- **Fix: frame_num 81→41**: VAE encoder processes all frames in one 3D-conv pass. At 81 frames/480p, first Conv3D intermediate is ~4.15GB, plus subsequent layers exceeds safe headroom. 41 frames is proven safe (same as FP8).
- **Fix: activation_reserve 4.0→7.0 GiB**: GGUF dequant temporaries need same headroom as FP8 to prevent WDDM spill.
- **Fix: VAE offload after decode**: Explicit `self.vae.model.cpu()` after every VAE decode, freeing ~0.5GB for next chunk's denoising.
- **Fix: LoRA buffer device mismatch**: LoRA buffers registered on CPU after block_swap moved blocks to GPU. Fixed: `_gguf_linear_forward` now uses `device=weight.device` for all LoRA tensors.
- **Fix: audio_guide_scale 1.0→2.0**: Lip sync degraded after first chunk because audio conditioning had no CFG amplification. With `audio_guide_scale=2.0`, a second forward pass (drop_audio) enables audio CFG. Matches ComfyUI/FP8 defaults.
- **Fix: VAE bfloat16**: VAE was running in float32 (default). Set `self.pipe.vae.dtype = torch.bfloat16` for ~2x faster encode/decode on tensor cores. WanVAE.encode/decode already convert output to float32.

**Measured performance (GGUF, RTX 4090 Laptop 16GB):**

- Denoising: ~16s/pass, 2 passes/step (audio CFG), ~32s/step, ~90s/chunk (6 steps)
- VAE decode: ~5.5 min/chunk (float32), expected ~2.5 min with bfloat16
- VAE encode: ~3 min (chunk 0 only, float32), expected ~1.5 min with bfloat16
- Block offload/restore: ~8s round-trip per chunk (for VAE decode)
- Total (6 chunks, 6.3s audio): ~35 min estimated with bfloat16 VAE

**Status:** All fixes implemented, compiles clean. Quality confirmed excellent (no blur, no color drift). Lip sync fix + VAE bfloat16 awaiting test.

**Bug fix (Phase 13.2 — VAE Decode Performance + ETA Accuracy):**

- **Root cause of ~6m41s VAE decode per chunk**: `_skip_block_offload=True` was set globally for GGUF, preventing DiT block offload before VAE decode. With 27 persistent blocks (~8.6GB) on GPU, only ~6.4GB remained for VAE decode intermediates. VAE's 3D transposed convolutions exceeded this, causing Windows WDDM to transparently spill to system RAM (~40-50x slower PCIe bandwidth).
- **Fix: always offload DiT blocks before VAE decode**: Changed offload/restore conditions in multitalk.py to always run for VAE decode, regardless of `_skip_block_offload`. The flag now only gates CLIP+VAE encode offload (first chunk), where GGUF model fits alongside. Expected VAE decode: ~60s/chunk (was ~401s).
- **Fix: ETA now includes VAE time**: Added `VIDEO_VAE_DECODE_SECS=60` and `VIDEO_FIRST_CHUNK_EXTRA_SECS=50` to config.py. Updated ETA calculations in video_engine.py and pipeline.py to include per-chunk VAE decode overhead and first-chunk CLIP+VAE encode overhead. Also fixed pipeline.py to use `VIDEO_GGUF_SECS_PER_STEP` (was using FP8 constant for GGUF mode).
- **ETA accuracy logging**: Added in previous session — logs estimated vs actual times for TTS, video (gen+load), and total pipeline. Will calibrate VAE constants from real run data.

**Revised performance estimate (GGUF, with VAE fix):**

- Denoising: ~32s/step × 4 steps × 5 chunks = ~640s
- VAE decode: ~60s × 5 chunks = ~300s
- First chunk overhead (CLIP+VAE encode): ~50s
- Block offload/restore: ~8s × 5 chunks = ~40s
- Total (5 chunks, 5.3s audio): ~1030s ≈ **~17 min** (was 46+ min before fix)

**Status:** Compiles clean. Awaiting test to measure actual VAE decode time with block offload enabled.

### Phase 14 — Denoising Performance Optimizations

**Goal:** Reduce denoising loop overhead without quality loss. Hit ≤2 min/sec target.

**Changes applied:**

1. **Restore `torch.cuda.empty_cache()` between forward passes for GGUF** (multitalk.py): GGUF on-demand dequantization creates BF16 temporaries across 604 linear layers per pass. These temporaries fragment CUDA's free pool (+0.6G reserved/pass). Without reclaiming, reserved VRAM grows 13.2G → 16.9G over 7 passes, causing Windows WDDM to spill to system RAM (50.7s/block set vs 4.2s nominal — 12x slower). Fix: `empty_cache()` after every `self.model()` call. Cost: ~150ms/call × 20 calls = ~3s total. Prevents catastrophic WDDM spill. Note: FP8 mode doesn't need this (in-place `_scaled_mm`), but the overhead is negligible.

2. **GGUF audio_cfg 2.0→1.0→2.0** (config.py `VIDEO_GGUF_LORA_AUDIO_GUIDE_SCALE`): Tried 1.0 (skip CFG, 1 pass/step) for speed — halved denoising time. However, tested output showed noticeable quality degradation (warped features, less defined detail in later chunks). Even with DPM++ SDE 2nd-order correction, audio_cfg=1.0 is insufficient for multi-chunk streaming. **Reverted to 2.0** (2 passes/step). CLI `--audio-cfg 1.0` available for max speed if quality is acceptable.

3. **Remove dead motion re-noising code** (multitalk.py): The pre-loop and post-solver motion frame noising (`torch.randn_like` + `self.add_noise`) was always overwritten by the clean motion frame injection on the next line (`latent[:, :N] = latent_motion_frames`). Dead code since Phase 10's direct latent passthrough. **Saves ~20 GPU tensor operations per generation.**

4. **DPM++ SDE 2nd-order multistep correction (REVERTED)** (multitalk.py): Attempted 2nd-order correction (`denoised_d = (1 + 1/(2r)) * denoised - (1/(2r)) * old_denoised`) applied to SDE drift. **Caused severe quality degradation with 2-pass CFG (audio_cfg=2.0)** — the extrapolation overshoots when denoised estimates include CFG scaling. Only worked with 1-pass mode (audio_cfg=1.0). **Reverted to 1st-order SDE** which was proven stable in Phases 12-13.

**Performance (GGUF, 5.8s video, measured from run_20260220_160641):**

- Denoising: 5 chunks × 4 steps × 2 passes × ~17s/pass = ~680s
- Per-chunk overhead: 5 × ~15s (VAE decode ~7s + inter-chunk gap ~8s) = ~75s
- First-chunk CLIP+VAE encode: ~23s
- Total inference: 784s ≈ **~13 min → 2.2 min/sec**
- Previous run with audio_cfg=1.0: 461s (~7.7 min) but quality degraded

**Calibrated ETA constants (from run_20260220_160641):**

- VIDEO_GGUF_SECS_PER_STEP: 16→17 (measured ~16.8s/pass avg, 15.8 chunk 0, ~17.1 chunks 1-4)
- VIDEO_VAE_DECODE_SECS: 60→8→15 (VAE ~7s + inter-chunk gap ~8s per chunk)
- VIDEO_FIRST_CHUNK_EXTRA_SECS: 50→30→23 (CLIP ~3s + VAE encode ~20s)
- ETA accuracy: 778s estimated vs 784s actual = 0.8% error

**Bug fixed:** `_estimate_video_eta` in pipeline.py/generate.py was using `VIDEO_LORA_SAMPLE_STEPS=6` instead of `VIDEO_GGUF_LORA_SAMPLE_STEPS=4`, adding 320s phantom denoising time. Also `estimate_total_steps` in video_engine.py had the same GGUF override bug. Both fixed.

**Files modified:** vendor/infinitetalk/wan/multitalk.py, config.py, pipeline.py, video_engine.py, generate.py, docs/plan.md

---

## Key Design Decisions

- **Sequential GPU usage**: TTS runs first, then model is unloaded and video model loads. Peak VRAM is ~10-16GB during InfiniteTalk inference.
- **Chatterbox Turbo over original Chatterbox**: Turbo is 350M params (vs larger original), faster inference, and includes paralinguistic tags natively. Perfect for short ~30s clips.
- **InfiniteTalk**: Handles all art styles (photorealistic, cartoon, 3D, illustrated), follows input image aspect ratio, and needs only image + audio. No pre-computed pose sequences required.
- **InfiniteTalk over Hallo/Hallo2**: SD 1.5-based, square-only (512×512), no arbitrary aspect ratios.
- **InfiniteTalk over HunyuanVideo-Avatar**: Supports everything but minimum 24GB VRAM natively.
- **CPU offloading + FP8 quantization**: The InfiniteTalk DiT is ~18.9B params (~38GB fp16). On 16GB VRAM, FP8 quantization (~18.9GB) + strategic offloading of VAE/CLIP/T5 to CPU keeps ~14.5B params persistent on GPU; ~4.4B overflow stream per-layer via in-place CPU↔GPU transfers.
- **FFmpeg post-processing**: Ensures consistent 25fps and proper audio/video muxing.

---

## Expected Performance

RTX 4090 Laptop 16GB VRAM:

### GGUF mode (default, Phase 14)

- **TTS** (~30s script): ~5-18 seconds
- **Video** (DPM++ SDE 1st-order, 4 steps, shift=11, audio_cfg=2.0): ~2.2 min per second of audio
- **Per-pass**: ~17s (measured 16.8s avg, GGUF dequant + LoRA patch overhead, stable with empty_cache)
- **Per-step**: ~34s (2 passes: audio_cfg=2.0 for quality)
- **Per-chunk denoising**: ~136s (4 steps × ~34s, chunk 0 faster at ~126s)
- **Per-chunk overhead**: ~15s (VAE decode ~7s + inter-chunk gap ~8s)
- **First-chunk extra**: ~23s (CLIP encode ~3s + VAE encode ~20s)
- **5.8s video**: ~13 min (5 chunks, inference 784s measured)
- **Model size**: ~12.8GB (Q4_0 base ~10.2GB + Q8 adapter ~2.65GB)
- **Quality**: Good output — DPM++ SDE stochastic regularization + audio CFG
- **Speed mode**: Use `--audio-cfg 1.0` for ~2x faster denoising (quality degrades in later chunks)
- **Advantage over FP8**: Smaller model (~12.8GB vs ~18.9GB), faster VAE, comparable denoising

### FP8 mode (legacy, `--quant fp8`)

- **Video** (DPM++ SDE, 6 steps, shift=11, audio_cfg=2.0): ~2.4 min per second of audio
- **Per-pass**: ~14s
- **Per-step**: ~28s (2 passes: conditional + drop-audio)
- **Per-chunk**: ~168s (6 steps × 28s)
- **9s video**: ~22 min (8 chunks × 6 steps × 2 passes)
- **VRAM**: 9.2-9.8G allocated / 13.3G reserved during denoising
