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
    └── phase-8-optimization/
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

**Diagnostics added:** Per-sub-operation timing in `WanAttentionBlock.forward()`, FP8 path counters, weight dtype/device/contiguity inspection, VRAM allocated/reserved snapshots. Confirmed FP8 fast path works correctly (fast=15, fallback=0) and the issue is purely memory pressure.

**Fix:** Increased `activation_reserve` from 4.0 to 5.0 GiB in `auto_tune_blocks_to_swap()`. Result: 23 persistent / 17 swapped (was 25/15). Swap blocks now have 4.22 GiB headroom (was 3.28 GiB), safely above the 3.75 GiB threshold proven by persistent blocks.

**Result:** Swap blocks: 5.5s/blk → 0.27s/blk. Per-pass: ~90s → ~12s. Per-chunk denoising: ~720s → ~117s. Overall: ~1.15 min per second of video (4 LoRA steps).

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

- **TTS** (~30s script): ~5-18 seconds on RTX 4090 Laptop
- **Video**: ~1.2 min per second of audio on RTX 4090 Laptop (16 GB VRAM) with FP8 + block swap
- **Total**: ~14 minutes for a 10-second clip (4 LoRA steps, 23/40 blocks persistent)
- **Per-step**: ~35s (including VAE decode and chunk overhead)
- **Block speed**: persistent 0.25s/blk, swap 0.27s/blk (VRAM: 12.8G/15.0G peak during swap)
