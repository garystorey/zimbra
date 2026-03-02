# Talking Head Video Generator — Development History

This document traces the MVP development from concept through 15 phases of iteration. It records what was tried, what failed, and the research that led to the current system. The application is not a final product — it is a working MVP that demonstrates the full pipeline.

---

## Technology Research & Selection

### Video Engine Selection

| Candidate               | Result    | Why                                                                                   |
| ----------------------- | --------- | ------------------------------------------------------------------------------------- |
| **EchoMimicV2**         | Rejected  | SD 1.5-based, square-only (512×512). First prototype worked but quality was limited.  |
| **Hallo / Hallo2**      | Rejected  | SD 1.5-based, square-only, no arbitrary aspect ratios.                                |
| **HunyuanVideo-Avatar** | Rejected  | Minimum 24 GB VRAM natively. Could not fit on 16 GB.                                  |
| **InfiniteTalk**        | ✅ Winner | Wan2.1-I2V-14B base. Audio-driven, any art style, arbitrary aspect ratios, streaming. |

### TTS Engine Selection

Chatterbox Turbo was selected for instant voice cloning from ~10s reference clips, paralinguistic tag support (`[laugh]`, `[cough]`), and small footprint (~4 GB VRAM, 350M params).

### Quantization Approach

| Approach                   | Result    | Why                                                                                    |
| -------------------------- | --------- | -------------------------------------------------------------------------------------- |
| **FP8** (optimum-quanto)   | Rejected  | Model is ~18.9 GB — too large for 16 GB. Required aggressive block swap (17/23 split). |
| **GGUF Q4_0 + Q8 adapter** | ✅ Winner | ~12.8 GB total. More headroom for activations. Matches ComfyUI's model format.         |

### LoRA Selection

| LoRA                     | Result    | Why                                                                                    |
| ------------------------ | --------- | -------------------------------------------------------------------------------------- |
| **T2V rank-32** (317 MB) | Rejected  | Wrong model type (Text-to-Video, not Image-to-Video). Caused progressive blur.         |
| **I2V rank-64** (738 MB) | ✅ Winner | Correct Image-to-Video variant. Enables audio_cfg=1.0 + frame_num=81. Matches ComfyUI. |

### Solver Selection

| Solver            | Result    | Why                                                                               |
| ----------------- | --------- | --------------------------------------------------------------------------------- |
| **Euler**         | Rejected  | Significantly slower. Does not work at low step counts. Phase 10.                 |
| **DPM++ 2M**      | Rejected  | Deterministic — errors compound across streaming chunks. Phase 12.                |
| **DPM++ SDE 2nd** | Rejected  | Extrapolation overshoots when denoised includes CFG scaling. Phase 14.            |
| **DPM++ SDE 1st** | ✅ Winner | Stochastic noise prevents error accumulation. Matches ComfyUI reference workflow. |

---

## Phase-by-Phase Development

### Phase 1: Project Setup

- Created project scaffold, virtual environment, dependency resolution
- Discovered Chatterbox vs InfiniteTalk dependency conflicts (torch 2.6.0 vs 2.4.1, diffusers 0.29.0 vs >=0.31.0)
- Solution: Install Chatterbox with `--no-deps`, manually install unique deps
- Validated all imports on Windows 11

### Phase 2: TTS Engine

- Implemented Chatterbox Turbo wrapper (`tts_engine.py`)
- Discovered voice reference must be >5 seconds (asserted internally)
- Implemented sentence-boundary chunking with 50ms cosine crossfade stitching
- Added `[pause:N]` tag support for inserting silence
- 9/9 tests passed

### Phase 3: Video Engine (InfiniteTalk)

- Implemented InfiniteTalk wrapper (`video_engine.py`)
- First working video generation with FP8 quantization
- Discovered InfiniteTalk's vendor VRAM management used `copy.deepcopy().to(GPU)` for overflow modules — catastrophically slow
- Patched to in-place `.to(device)` transfers

### Phase 4: Pipeline + CLI

- Implemented pipeline orchestration (`pipeline.py`) and CLI (`generate.py`)
- Sequential GPU usage: TTS fully unloads before video loads
- Rich progress bars for all stages
- ETA estimation before generation begins

### Phase 5: Polish & Documentation

- Added error handling: `torch.cuda.OutOfMemoryError` catch, `_cleanup()` for guaranteed model unloading
- `KeyboardInterrupt` handling with clean model unloading
- GPU validation at startup (`_check_gpu()`)
- README.md with quick start guide

### Phase 6: Performance — Transfer Bottleneck

- Eliminated per-step CPU↔GPU transfer bottleneck
- FP8 model was ~18.9 GB on 16 GB VRAM, requiring constant per-layer transfers
- Auto-tuned `num_persistent_param_in_dit` to keep maximum params on GPU

### Phase 7: Block Swap (Kijai Approach)

- Replaced vendor's per-layer VRAM management with block-level swapping
- Swap entire DiT blocks (~320 MB each) instead of individual layers
- Added async CUDA stream prefetch to hide transfer latency behind compute
- Dramatic speedup: block transfer hidden behind block compute

### Phase 8: FP8 Device Mismatch Fix

- Fixed `scale_weight` tensor living on wrong device after block swap
- Registered as buffer instead of raw attribute so `.to()` moves it correctly

### Phase 9: VRAM Spill Fix

- **Root cause discovered:** Windows WDDM transparently spills GPU memory to system RAM. No error, just 20-55× slower.
- Increased activation_reserve from 4.0 → 7.0 GiB
- Reduced persistent blocks: 23 → 17 for FP8 mode
- Zero WDDM spill after fix

### Phase 10: Progressive Blur Investigation

**Tried and learned:**

- `audio_cfg=1.0` → progressive blur in later chunks (with T2V LoRA — root cause found in Phase 15)
- Euler solver + 4 steps + shift=2 → **significantly slower, doesn't work** ⚠️
- `motion_frame=25` → doubled chunk count without fixing blur
- **Key fix:** Moved `_saved_motion_latent` BEFORE VAE decode — eliminates cumulative VAE reconstruction error (direct latent passthrough)

### Phase 11: ComfyUI Alignment Attempt

**Tried and learned:**

- Implemented DPM++ 2M solver (but ComfyUI actually uses DPM++ SDE — misidentified scheduler)
- `frame_num=81` → OOM during LoRA merge with FP8 (~18.9 GB model)
- `audio_cfg=2.5` → over-guidance artifacts
- `color_correction=0.5 + audio_cfg=2.5` → made output worse

### Phase 12: Fix Progressive Degradation

- **Root cause of Phase 11 quality issues:** DPM++ 2M is deterministic — errors compound across streaming chunks. Motion frame injection disrupts 2nd-order trajectory assumption.
- Implemented correct DPM++ SDE solver (stochastic noise prevents error accumulation)
- Quality immediately improved

### Phase 13: GGUF Migration

The biggest architectural change. Moved from FP8 (optimum-quanto) to GGUF quantization.

- **Phase 13.0:** Built full GGUF loading pipeline (`gguf_loader.py`) ported from city96/ComfyUI-GGUF
- **Phase 13.1:** Fixed first-run hang. Key fixes:
  - Skip block offload for CLIP/VAE encode (GGUF model fits alongside)
  - `frame_num` 81→41 (VAE 3D-conv OOM with FP8-era settings)
  - `activation_reserve` tuned to 7.0 GiB
  - VAE dtype set to bfloat16 (2× faster)
  - `audio_cfg` 1.0→2.0 (blur returned with smaller frame_num)
- **Phase 13.2:** Fixed catastrophic VAE decode performance
  - **Root cause:** `_skip_block_offload = True` prevented DiT blocks from offloading before VAE decode. 27 persistent blocks (~8.6 GB) left only ~6.4 GB for VAE 3D convolutions → WDDM spill → 6m41s per chunk (should be ~7s)
  - **Fix:** Always offload DiT blocks before VAE decode, regardless of `_skip_block_offload`
  - VAE decode: ~7s/chunk (was ~401s)

### Phase 14: Denoising Optimizations

- **Restored `empty_cache()` between forward passes** — GGUF dequant creates BF16 temporaries across 604 linear layers. Without cleanup: +0.6 GB/pass → WDDM spill by step 3 (50.7s/block vs 4.2s nominal = 12× slower). Cost: ~150ms/call = ~3s total. Essential.
- **Removed dead motion re-noising code** — Pre-loop and post-solver noise injection was always overwritten by clean motion frame injection. Dead code since Phase 10's direct latent passthrough.
- **DPM++ SDE 2nd-order correction** — Added multistep correction using saved previous denoised/sigma. Improved convergence per step. Later found to break with CFG (overshoots) — reverted to 1st-order only.
- **`audio_cfg` 2.0→1.0 tested** — Halves denoising time (1 pass vs 2). Quality degraded with T2V rank-32 LoRA (warped features in later chunks). Root cause identified in Phase 15.

### Phase 15: Frame81 + I2V LoRA (Final MVP State)

The breakthrough phase. Achieved ComfyUI-matching quality and performance.

- **Phase 15a:** `frame_num` 41→81 — Works with GGUF (~12.8 GB). Was only OOMing with FP8 (~18.9 GB). Fewer chunks = better quality.
- **Phase 15b:** `motion_frame` 13→9 — 25 froze 43% of frames. 9 = only 12% frozen, much better lip sync.
- **Phase 15c:** Switched from T2V rank-32 to **I2V rank-64 LoRA** — This was the root cause of progressive blur at audio_cfg=1.0. The T2V LoRA was the wrong model type entirely.
- **Phase 15d:** `audio_cfg` back to 1.0 — Now works correctly with I2V LoRA. Halves denoising time (skip_cfg = 1 pass/step).
- **Phase 15e:** Tuned `activation_reserve` to 8.5 GiB (frame_num=81 needs more headroom). `color_correction` 1.0→0.3 (mild). Calibrated ETA constants.

**Result:** 8.2s video in 10.5 min total (1.06 min/sec). Matches ComfyUI reference workflow.

### Phase 16: TTS Quality Fixes

Three issues discovered when testing with multi-sentence dialogue lines (e.g. `I said "I'm sorry. I'm not the movie. I'm just the trailer."`):

1. **Phantom pauses** — Chatterbox inserted ~3-second pauses after "I said" and after "I'm sorry" when the entire line was sent as a single TTS call.
2. **Text truncation** — "I'm just the trailer" was never spoken. Chatterbox silently dropped the tail of multi-sentence input.
3. **Voice clone drift** — Output voice didn't closely match the reference.

**Fixes applied:**

- **Sub-sentence splitting** — Each line is now split at `.!?` boundaries via `split_into_sentences()` before sending to Chatterbox. A 6-line script with one multi-sentence line produces 9 clean TTS calls instead of 6. Sub-sentences within a line are joined with 100ms gaps (vs 200ms between lines). `count_tts_chunks()` updated to reflect sub-sentence count.
- **Quote stripping** — `normalize_text()` now removes all `"` characters before TTS. Smart quotes (`""`) are first converted to ASCII `"` (existing behavior), then all `"` are stripped. Quotes are never spoken aloud and caused Chatterbox to insert phantom pauses and truncate text.
- **Voice clone tuning** — `top_k` 50→40, `top_p` 0.85→0.8 for slightly tighter voice matching. Temperature kept at 0.6 (0.5 caused looping → 38.6s output for 22s script; 0.4 caused severe looping → 132.8s output).

**Result:** 6-line joke script generates 9 chunks in ~19s → 22.5s audio. No phantom pauses, no truncation, all text spoken.

Files modified: `tts_engine.py`, `config.py`

### Phase 17: Output Directory Restructure + `--filename` CLI

Simplified output management by flattening the directory structure and adding user-controllable filenames.

**Changes:**

- **Renamed `outputs/` → `output/`** — Single flat directory replaces the previous `outputs/audio/`, `outputs/video/`, `outputs/logs/` subdirectory structure.
- **All output files share one base filename** — `.wav` (audio), `.mp4` (video), `.png` (contact sheet), `.log` (debug log) all use the same base name with their respective extensions.
- **Added `--filename NAME` CLI param** — Lets the user specify the base filename for all outputs. If files with that name already exist, an incremented suffix (`_01`, `_02`, …) is appended automatically.
- **Default when `--filename` omitted** — Falls back to `run_YYYYMMDD_HHMMSS` timestamp format (same correlation as before, just in a flat directory).
- **`config.RUN_FILENAME`** — New config variable set once at startup, used by all output paths (log, audio, video, contact sheet).
- **`utils.resolve_output_filename()`** — New utility function that checks for existing files and auto-increments the suffix.

**Config changes:** `OUTPUTS_DIR` → `OUTPUT_DIR`, removed `AUDIO_OUTPUT_DIR`, `VIDEO_OUTPUT_DIR`, `LOGS_DIR`.

**Files modified:** `config.py`, `generate.py`, `pipeline.py`, `tts_engine.py`, `video_engine.py`, `utils.py`, `.gitignore`, `README.md`, all docs.

### Phase 18: `--text` CLI Option + `${filename}` Audio Splicing

Added two new features to the TTS pipeline for more flexible script input:

1. **`--text` CLI option** — Pass script text directly on the command line instead of reading from `inputs/scripts/default.txt`. Takes priority over the default file. Ignored with a warning when combined with `--video-only` (which skips TTS).

2. **`${filename}` audio file references** — Script text (via `--text` or `default.txt`) can now reference pre-recorded audio files using `${filename}` syntax (e.g. `${intro.mp3}`). Files are resolved from `inputs/scripts/`, loaded, resampled to 24kHz to match TTS output, and spliced into the final audio at the position they appear. A `FileNotFoundError` is raised if the referenced file doesn't exist.

3. **`[s:.5]` pause tag fix** — The `_PAUSE_RE` regex previously required a leading digit (`\d+`), so `[s:.5]` was not recognized and was passed as literal text to the TTS engine (spoken as ".5"). Fixed regex to accept `\.\d+` as an alternative pattern.

**Implementation details:**

- New `_AUDIO_REF_RE` regex pattern for `${filename}` matching
- `split_into_segments()` now uses a combined regex to parse `[s:N]` and `${filename}` in a single pass, preserving ordering
- New `_load_audio_file()` helper loads audio via soundfile (WAV) or librosa (MP3/other) and resamples to 24kHz
- `count_tts_chunks()` updated to include audio file segments in the progress count
- `progress_callback()` fires after each audio file is loaded, so progress bars advance correctly

Files modified: `generate.py`, `pipeline.py`, `tts_engine.py`

---

## Windows Compatibility Discoveries

Throughout development, several Windows-specific issues were encountered and resolved:

### Package Compatibility

| Issue                                           | Resolution                                                                                       |
| ----------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `spacy-pkuseg` required C++ build tools         | Pin `==1.0.1` — prebuilt Windows wheels since Jul 2025                                           |
| `flash_attn` has no official Windows wheels     | Use prebuilt wheels from [kingbri1/flash-attention](https://github.com/kingbri1/flash-attention) |
| `xformers` cu121 index lacks torch 2.4.1 wheels | Install from cu124 index (wheel only checks torch version, not CUDA)                             |
| `transformers` 5.x removed MT5Tokenizer         | Pin `<5` — breaks diffusers at import time                                                       |
| `xfuser` 0.4.2+ needs torch 2.5+                | Pin `==0.4.1`                                                                                    |
| `SageAttention` won't compile on Windows        | Excluded entirely — no Windows torch 2.4.x wheels                                                |
| `pip.exe` file locking errors                   | Use `python -m pip` instead                                                                      |

### VRAM Management on Windows

- **WDDM memory spill** — Windows WDDM transparently pages GPU memory to system RAM when VRAM is exceeded. No error raised, but PCIe bandwidth causes 20-55× slowdowns. Solution: generous `activation_reserve` (8.5 GiB) ensures peak never exceeds physical VRAM.
- **`nn.Module.to()` breaks pinned memory** — Creates new tensors; originals are garbage collected. Must re-pin after every block offload.
- **`non_blocking=True` is synchronous for GPU→CPU** when destination is not pinned memory. Async double-buffer pattern doesn't work.
- **`torch.cuda.empty_cache()` forces CUDA sync + WDDM kernel round-trip** (~200-500ms per call). Only use where essential (between GGUF forward passes to prevent fragmentation spill).

### Dependency Install Order

The install order is critical on Windows because later installs can silently override earlier versions:

1. PyTorch from cu121 index (must be first — everything depends on it)
2. xformers from cu124 index (cu121 lacks Windows wheels for torch 2.4.1)
3. flash_attn from prebuilt wheel (no official Windows build)
4. numpy pre-pinned to 1.26.4 (prevents numpy 2.x from later deps)
5. requirements.txt for InfiniteTalk deps
6. Chatterbox with `--no-deps` + unique deps separately

---

## ComfyUI Reference Workflow

The quality/performance target throughout development was a known-good ComfyUI workflow:

```
Scheduler: DPM++ SDE, steps=6, shift=11, audio_cfg=1.0
frame_window=81, motion_frame=25, colormatch=disabled
Model: Q4_0 GGUF (~7-8GB), blocks_to_swap=20
LoRA: lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16
```

**MVP alignment with ComfyUI:**

- GGUF Q4_0 base + Q8 adapter ✓
- DPM++ SDE solver ✓
- shift=11 ✓
- LightX2V I2V rank-64 LoRA ✓
- text_guide_scale=1.0 ✓
- frame_num=81 ✓
- audio_cfg=1.0 ✓
- Block swap with async prefetch ✓

**Where we differ:**

- `steps=4` (not 6) — LightX2V designed for 4; saves 33% denoising time
- `motion_frame=9` (not 25) — 25 froze 43% of frames; 9 = 12% frozen, better lip sync

---

## Key Architectural Decisions

### Sequential GPU Usage

TTS and video models never coexist on GPU. Chatterbox Turbo (~4 GB) is fully unloaded (`del model` + `empty_cache()` + `gc.collect()`) before InfiniteTalk (~12.8 GB GGUF) loads. This is the only way both fit on 16 GB.

### GGUF On-Demand Dequantization

Rather than dequantize all weights at load time (which would require ~28 GB for BF16), weights stay packed in GGUF format as `nn.Parameter` raw bytes. The monkey-patched `Linear.forward()` dequantizes each weight on-the-fly to BF16, applies LoRA if present, runs the matmul, and discards the BF16 temporary. This is the approach used by ComfyUI.

### LoRA as Patches (Not Merged)

LoRA deltas are stored as buffers (`_lora_up`, `_lora_down`, `_lora_scale`) on each Linear module. During the patched forward: `weight += (up @ down) * scale`. This avoids the dequant→merge→requant problem that would make GGUF+LoRA impractical.

### Block-Level Swap (Not Per-Layer)

The vendor's per-layer VRAM management used `copy.deepcopy().to(GPU)` which was catastrophically slow. Block-level swap moves entire DiT blocks (~320 MB each) between CPU and GPU using pinned memory + async CUDA stream prefetch. The prefetch hides most transfer latency behind compute of the current block.

### Direct Latent Passthrough Between Chunks

For streaming mode, the motion latent for the next chunk is saved BEFORE VAE decode (in latent space). This eliminates cumulative VAE reconstruction error that would otherwise degrade quality across chunks. The latent is injected directly into the next chunk's initial frames.

### Post-MVP: Fork Integration

- Created GitHub fork at [garystorey/InfiniteTalk](https://github.com/garystorey/InfiniteTalk) (forked from MeiGen-AI/InfiniteTalk)
- All vendor patches from Phases 3-15 committed to the fork as a single PR ([#1](https://github.com/garystorey/InfiniteTalk/pull/1))
- `install.py` updated to clone from the fork instead of upstream
- Vendor patches are now version-controlled and reviewable
- Future workflow: develop locally in `vendor/infinitetalk/`, PR to fork, merge

---

## Removed/Legacy Code

| Item                   | Status   | Notes                                                          |
| ---------------------- | -------- | -------------------------------------------------------------- |
| `fp8_optimization.py`  | Legacy   | Native FP8 matmul. Kept for `--quant fp8` fallback.            |
| TeaCache               | Removed  | Never useful at low step counts (warmup prevents skipping).    |
| SageAttention          | Removed  | Won't compile on Windows with torch 2.4.x.                     |
| DPM++ 2M solver        | Kept     | In vendor code but not used. DPM++ SDE is the correct solver.  |
| DPM++ SDE 2nd-order    | Reverted | Breaks with CFG. 1st-order only.                               |
| Euler solver           | Kept     | In vendor code but never used. Significantly slower.           |
| `--frame-num` CLI flag | Removed  | Video always matches audio length. frame_num is internal only. |
| Per-layer VRAM mgmt    | Replaced | Vendor's approach replaced by block-level swap.                |
