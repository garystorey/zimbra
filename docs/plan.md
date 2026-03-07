# Talking Head Video Generator — Plan

The single reference document for understanding, installing, and operating this application (MVP).

---

## 1. Overview

A Python CLI app that takes a voice-clone reference audio, a portrait image, and a text script, then generates a "talking head" video locally on an NVIDIA GPU with 16 GB VRAM.

**Pipeline:** Text → Voice-Cloned Speech (Chatterbox Turbo) → Talking-Head Video (InfiniteTalk/GGUF) → FFmpeg Mux → MP4

Models run **sequentially** — TTS is fully unloaded (0 GB VRAM) before the video model loads. Peak VRAM never exceeds ~13.5 GB during inference.

---

## 2. Technology Stack

| Component            | Library                                                                                                                                          | Why                                                                                                                                 |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------- |
| **TTS**              | [Chatterbox Turbo](https://github.com/resemble-ai/chatterbox) (350 M params)                                                                     | Instant voice cloning from ~10 s reference clip. Paralinguistic tags (`[laugh]`, `[cough]`). ~4 GB VRAM. 24 kHz output.             |
| **Video**            | [InfiniteTalk](https://github.com/garystorey/InfiniteTalk) (Wan2.1-I2V-14B base, fork of [MeiGen-AI](https://github.com/MeiGen-AI/InfiniteTalk)) | Audio-driven (wav2vec2). Any art style (photo/cartoon/3D). Arbitrary aspect ratios. 480p/720p. Streaming mode for unlimited length. |
| **Quantization**     | GGUF Q4_0 base + Q8 adapter (~12.8 GB)                                                                                                           | On-demand dequantization. Ported from city96/ComfyUI-GGUF (Apache 2.0). Matches ComfyUI workflow.                                   |
| **LoRA Accelerator** | LightX2V I2V rank-64 distilled LoRA (738 MB)                                                                                                     | Reduces denoising from 20 steps to 6. Applied as patches during GGUF dequant forward — no dequant→merge→requant needed.             |
| **Solver**           | DPM++ SDE (sde-dpmsolver++) 1st-order                                                                                                            | Stochastic noise prevents deterministic error accumulation across streaming chunks. Matches ComfyUI reference workflow.             |
| **VRAM Management**  | Block-level swap with async CUDA stream prefetch (Kijai approach)                                                                                | Swap entire DiT blocks (~320 MB each GGUF). Async prefetch hides transfer latency behind compute. Auto-tuned based on free VRAM.    |
| **CLI**              | Python `argparse`                                                                                                                                | Simplest interface. Easy to wrap in a UI later.                                                                                     |
| **Post-processing**  | FFmpeg                                                                                                                                           | Mux audio+video at 25 FPS. Contact sheet generation for quality review.                                                             |

---

## 3. Target Environment

| Requirement                | Value                                                   |
| -------------------------- | ------------------------------------------------------- |
| **OS**                     | Windows 11 (native, no WSL)                             |
| **Python**                 | 3.10.x (64-bit) — intersection of all package support   |
| **GPU**                    | NVIDIA with 16 GB+ VRAM (tested on RTX 4090 Laptop GPU) |
| **CUDA**                   | 12.1 (bundled with PyTorch cu121 wheels)                |
| **FFmpeg**                 | In PATH or auto-detected via `imageio-ffmpeg`           |
| **HuggingFace account**    | Required for Chatterbox model download                  |
| **Visual C++ Build Tools** | NOT needed — all packages have prebuilt wheels          |
| **Git**                    | Recommended (for cloning InfiniteTalk fork)             |

---

## 4. Installation

### 4.1 Quick Start

```bash
git clone <repo-url> zimbra && cd zimbra
install.bat
.\venv\Scripts\activate
huggingface-cli login
generate.bat                    # ~20 GB model download on first run
```

### 4.2 Install Order (Critical)

The order matters — later installs can override earlier versions. `install.py` automates this:

| Step | Command                                                                         | Why This Order                                                                             |
| ---- | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| 1    | Create `venv/`, upgrade pip, setuptools, wheel                                  | Use `python -m pip` on Windows (not `pip.exe`) to avoid file locking                       |
| 2    | `torch==2.4.1+cu121`, `torchvision`, `torchaudio` from cu121 index              | PyTorch must be first — xformers, flash_attn depend on it                                  |
| 3a   | `xformers==0.0.28.post1` from **cu124** index                                   | cu121 index lacks Windows wheels for torch 2.4.1; cu124 wheel checks torch version only    |
| 3b   | `flash_attn==2.7.4.post1` from prebuilt wheel                                   | Windows wheel from [kingbri1/flash-attention](https://github.com/kingbri1/flash-attention) |
| 4a   | `numpy==1.26.4`                                                                 | **Pre-pin** before requirements.txt — prevents PyTorch pulling numpy 2.x                   |
| 4b   | `pip install -r requirements.txt`                                               | InfiniteTalk deps (excludes gradio, mlflow, dashscope)                                     |
| 5a   | `chatterbox-tts --no-deps`                                                      | Gets Chatterbox without touching any deps (avoids torch/diffusers conflicts)               |
| 5b   | Chatterbox unique deps: `librosa spacy-pkuseg==1.0.1 s3tokenizer conformer ...` | Only non-conflicting deps installed separately                                             |
| 6    | `git clone garystorey/InfiniteTalk → vendor/infinitetalk`                       | Patched fork — all vendor patches pre-applied                                              |
| 7    | Create runtime directories, verify all imports                                  | Catch install failures immediately                                                         |

### 4.3 Dependency Conflict Resolution

Chatterbox and InfiniteTalk have 2 soft conflicts resolved by installing Chatterbox with `--no-deps`:

| Package     | Chatterbox Pins | InfiniteTalk Pins | Our Resolution  | Rationale                                                |
| ----------- | --------------- | ----------------- | --------------- | -------------------------------------------------------- |
| `torch`     | `==2.6.0`       | `==2.4.1` (cu121) | `==2.4.1+cu121` | InfiniteTalk + xformers + flash_attn tied to 2.4.1       |
| `diffusers` | `==0.29.0`      | `>=0.31.0`        | `>=0.31.0`      | InfiniteTalk needs >=0.31.0; Chatterbox usage is minimal |

### 4.4 Key Dependency Pins

| Package        | Version      | Pin Reason                                                                    |
| -------------- | ------------ | ----------------------------------------------------------------------------- |
| `torch`        | 2.4.1+cu121  | xformers + flash_attn + InfiniteTalk all tied to this version                 |
| `xformers`     | 0.0.28.post1 | From cu124 index (cu121 lacks Windows wheels for torch 2.4.1)                 |
| `flash_attn`   | 2.7.4.post1  | Prebuilt Windows wheel (cu124torch2.4.0 variant)                              |
| `numpy`        | 1.26.4       | Pre-pinned to prevent numpy 2.x (breaks PyTorch internals)                    |
| `transformers` | >=4.49.0,<5  | 5.x removed MT5Tokenizer, breaks diffusers at import time                     |
| `xfuser`       | 0.4.1        | 0.4.2+ requires `torch.distributed.tensor.experimental` (torch 2.5+ only)     |
| `spacy-pkuseg` | 1.0.1        | Prebuilt Windows wheels since Jul 2025; earlier versions need C++ build tools |
| `safetensors`  | 0.5.3        | Chatterbox model loading compatibility                                        |

### 4.5 Excluded Dependencies

| Package     | Why Excluded                                    |
| ----------- | ----------------------------------------------- |
| `gradio`    | CLI-only interface, no web UI                   |
| `mlflow`    | Experiment tracking, not needed for inference   |
| `dashscope` | InfiniteTalk's built-in TTS — we use Chatterbox |
| `kokoro`    | InfiniteTalk's built-in TTS — we use Chatterbox |

---

## 5. Model Weights

All auto-download on first run via `huggingface_hub`. Pre-flight check downloads any missing files with a progress bar before the pipeline starts.

| Model                     | HuggingFace Source                      | Size     | Download Method               | Purpose                                                      |
| ------------------------- | --------------------------------------- | -------- | ----------------------------- | ------------------------------------------------------------ |
| Chatterbox Turbo          | `ResembleAI/chatterbox-turbo`           | ~700 MB  | `from_pretrained(token=True)` | TTS voice cloning                                            |
| Wan2.1-I2V-14B (VAE+CLIP) | `Wan-AI/Wan2.1-I2V-14B-480P`            | ~5.3 GB  | `snapshot_download()`         | VAE, CLIP, tokenizers, config                                |
| chinese-wav2vec2-base     | `TencentGameMate/chinese-wav2vec2-base` | ~360 MB  | `snapshot_download()`         | Audio features. `model.safetensors` from `refs/pr/1` branch. |
| GGUF base (Q4_0)          | `city96/Wan2.1-I2V-14B-480P-gguf`       | ~10.2 GB | `hf_hub_download()`           | DiT base model (quantized)                                   |
| GGUF adapter (Q8)         | `Kijai/WanVideo_comfy_GGUF`             | ~2.65 GB | `hf_hub_download()`           | InfiniteTalk audio cross-attention adapter                   |
| LightX2V I2V LoRA rank-64 | `Kijai/WanVideo_comfy`                  | ~738 MB  | `hf_hub_download()`           | I2V step distillation (20→6 steps)                           |

**Total first-run download: ~20 GB**

```
models/
├── infinitetalk/
│   ├── Wan2.1-I2V-14B-480P/     # VAE, CLIP, tokenizers
│   ├── chinese-wav2vec2-base/    # wav2vec2
│   ├── gguf/
│   │   ├── wan2.1-i2v-14b-480p-Q4_0.gguf        # GGUF base
│   │   └── Wan2_1-InfiniteTalk_Single_Q8.gguf    # GGUF adapter
│   └── loras/
│       └── Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors
└── chatterbox/
```

---

## 6. Project Structure

```
zimbra/
├── README.md                   # Quick start, CLI reference, troubleshooting
├── requirements.txt            # Pinned dependencies (DO NOT pip install directly)
├── install.bat / install.py    # One-click installer (creates venv, deps, vendor)
├── generate.bat / generate.py  # CLI entry point
├── config.py                   # All paths, defaults, GPU settings, auto-tune
├── pipeline.py                 # Orchestrates TTS → Video → Post-process
├── tts_engine.py               # Chatterbox Turbo wrapper
├── video_engine.py             # InfiniteTalk wrapper
├── model_manager.py            # Auto-download + pre-flight check for all model weights
├── gguf_loader.py              # GGUF dequantization, loading, LoRA-as-patches
├── utils.py                    # GPU thermal, file validation, FFmpeg helpers
├── inputs/                     # Default inputs (committed to git)
│   ├── images/default.png
│   ├── voice_ref/default.mp3
│   ├── scripts/default.txt
│   └── animation/default.txt
├── output/                     # All generated files (gitignored)
├── models/                     # Downloaded model weights (gitignored, ~20 GB)
├── vendor/infinitetalk/        # Cloned from garystorey/InfiniteTalk fork (gitignored)
├── venv/                       # Virtual environment (gitignored)
└── docs/
    ├── plan.md                 # This document — single system reference
    ├── history.md              # Development phases, research, rejected approaches
    ├── prd.md                  # Product Requirements Document
    ├── tdd.md                  # Technical Design Document
    └── future-enhancements.md  # Backlog of known improvement opportunities
```

---

## 7. Configuration

All values in `config.py`. These are the **correct final values (Phase 15e)** — do not use any intermediate values from development phases.

### 7.1 GGUF Model Settings

```python
VIDEO_QUANT = "gguf"
VIDEO_GGUF_FRAME_NUM = 81                   # 4n+1, matches ComfyUI. Fewer chunks = better quality.
VIDEO_GGUF_MOTION_FRAME = 9                 # 0.36s overlap, 11% frozen — best lip sync (Phase 15b + 22). VAE re-encode from corrected frames handles drift.
VIDEO_GGUF_LORA_SAMPLE_STEPS = 6            # 6 steps for better quality (LightX2V supports 4-6)
VIDEO_GGUF_LORA_AUDIO_GUIDE_SCALE = 1.0     # 1 pass/step (skip_cfg). Keeps render time within benchmarks.
VIDEO_GGUF_SECS_PER_STEP = 34              # ~34s/pass at 1-pass, ~36s at 2-pass
VIDEO_GGUF_ACTIVATION_RESERVE = 8.5         # GiB — 81-frame activations need more headroom
VIDEO_BLOCK_SIZE_BYTES = 320_000_000        # ~320 MB per DiT block in GGUF Q4_0+Q8
```

### 7.2 Shared Settings

```python
VIDEO_LORA_SOLVER = "dpmpp_sde"             # DPM++ SDE 1st-order
VIDEO_LORA_SAMPLE_SHIFT = 11                # ComfyUI-validated
VIDEO_LORA_TEXT_GUIDE_SCALE = 1.0           # Must be 1.0 for distilled LoRA
VIDEO_USE_LORA = True                       # LightX2V I2V rank-64 LoRA
VIDEO_NUM_DIT_BLOCKS = 40                   # InfiniteTalk has 40 DiT blocks
VIDEO_PREFETCH_BLOCKS = 1                   # Async prefetch 1 block ahead
GPU_THERMAL_THRESHOLD = 70                  # °C — pause between chunks if exceeded
color_correction_strength = 0.5             # per-frame Lab correction anchored to reference image. L channel: mean-only shift. a/b channels: full mean+std transfer. Feeds corrected overlap frames forward via VAE re-encode to prevent cumulative drift.
VIDEO_VAE_DECODE_SECS = 28                  # Per-chunk: VAE decode + block offload/restore
VIDEO_FIRST_CHUNK_EXTRA_SECS = 24           # CLIP (~10s) + VAE encode (~8s) + setup (~6s)
VIDEO_FPS = 25                              # InfiniteTalk output frame rate
VIDEO_MODE = "streaming"                    # Unlimited-length video in overlapping chunks
```

### 7.3 Key `config.py` Functions

- **`auto_tune_blocks_to_swap()`** — Calculates optimal block swap count based on free VRAM, accounting for activation reserve + swap buffer + non-block overhead
- **`ensure_directories()`** — Creates all runtime dirs (output/, models/, etc.)
- **`get_gpu_info()`** — Returns GPU name, CUDA version, VRAM
- **FFmpeg discovery** — env var → PATH → `imageio-ffmpeg` fallback

---

## 8. VRAM Budget (16 GB, GGUF Mode)

Every number here was measured on RTX 4090 Laptop 16 GB:

```
GGUF model total: ~12.8 GB (10.2 GB base + 2.65 GB adapter)
  - 40 DiT blocks × ~320 MB = ~12.8 GB
  - Non-block params (embeddings, head, audio_proj, img_emb): ~500 MB

With auto_tune_blocks_to_swap (activation_reserve=8.5 GiB):
  - ~22 persistent blocks on GPU: ~7.0 GB
  - ~18 swapped blocks on CPU (pinned memory)
  - 1 swap buffer slot: ~320 MB
  - Non-block params: ~500 MB
  - Total weights on GPU: ~7.8 GB
  - Activation headroom: ~8.2 GB

During denoising (frame_num=81, 1-pass audio_cfg=1.0):
  - Allocated: ~8.5 GB
  - Reserved: ~13.5 GB
  - No WDDM spill (peak well under 16 GB)

During VAE decode (blocks offloaded):
  - DiT blocks moved to CPU → ~500 MB weights on GPU
  - VAE model: ~500 MB
  - VAE 3D transposed conv intermediates: several GB
  - Headroom: sufficient (no WDDM spill)

TTS (Chatterbox Turbo): ~4.16 GB (fully unloaded before video)
```

**Why activation_reserve = 8.5 GiB:** Windows WDDM transparently spills GPU memory to system RAM via PCIe when VRAM is exceeded. This is invisible (no error) but causes 20-55× slowdowns. 8.5 GiB is the minimum that prevents spill with frame_num=81.

---

## 9. Vendor Patches

Patches are maintained in the [garystorey/InfiniteTalk](https://github.com/garystorey/InfiniteTalk) fork (forked from MeiGen-AI/InfiniteTalk). `install.py` clones this fork into `vendor/infinitetalk/`. All patches are version-controlled in the fork — no post-clone patching needed.

### 9.0 How to PR Changes to the Fork

When you modify files inside `vendor/infinitetalk/`, use this workflow to push them to the fork permanently:

```powershell
# 1. Clone the fork to a temp working directory (next to zimbra)
git clone https://github.com/garystorey/InfiniteTalk.git C:\Users\GaryS\Desktop\InfiniteTalk-fork

# 2. Create a feature branch
cd C:\Users\GaryS\Desktop\InfiniteTalk-fork
git checkout -b my-patch-name

# 3. Copy changed files from vendor into the clone
#    (repeat for each modified file)
copy C:\Users\GaryS\Desktop\zimbra\vendor\infinitetalk\wan\multitalk.py .\wan\multitalk.py

# 4. Review, commit, push
git diff                                          # verify changes look right
git add -A
git commit -m "Description of what changed and why"
git push origin my-patch-name

# 5. Create the PR
gh pr create --repo garystorey/InfiniteTalk --base main --head my-patch-name --title "Short title" --body "Detailed description"

# 6. Review & merge the PR on GitHub
#    https://github.com/garystorey/InfiniteTalk/pulls

# 7. Cleanup — delete the temp clone (safe after merge)
Remove-Item -Recurse -Force C:\Users\GaryS\Desktop\InfiniteTalk-fork
```

After merging, any fresh `install.py` run will clone the updated fork with your changes included.

**Tips:**

- Use `git diff --stat` to see which files differ before copying
- One PR per logical change (don't bundle unrelated fixes)
- The temp clone can be anywhere — it's disposable after the PR is merged
- `gh` CLI must be installed and authenticated (`gh auth status` to check)

Patches that are **essential** for correct operation on 16 GB VRAM:

### 9.1 Block Swap (`wan/modules/multitalk_model.py`)

- **`block_swap()`** — Moves last N blocks to CPU, pins their memory for DMA
- **`_run_blocks_with_swap()`** — Async prefetch next block → compute current → sync offload → re-pin
- **Single-person optimization** — `human_num` param skips O(n²) attention map when ≤1 person

### 9.2 Denoising Loop (`wan/multitalk.py`)

- **DPM++ SDE solver** — sde-dpmsolver++ 1st-order update formula
- **`_skip_cfg` optimization** — 1 forward pass/step when both guide scales ≈ 1.0
- **Direct latent passthrough** — Save motion latent BEFORE VAE decode (no cumulative VAE error)
- **`torch.cuda.empty_cache()` between forward passes** — Prevents GGUF dequant VRAM fragmentation (+0.6 GB/pass → WDDM spill)
- **Block offload before VAE decode** — Free VRAM for 3D transposed convolutions
- **VAE offload after decode** — Free VAE VRAM for next chunk's denoising
- **Bilinear resize** (was nearest) in `resize_and_centercrop()`

### 9.3 Other Patches

- **FFmpeg CRF 10→1** in `wan/utils/multitalk_utils.py` — Near-lossless intermediate encoding
- **Remove `torch_gc()` + `@torch.compile`** from `wan/utils/multitalk_utils.py` hot path

---

## 10. Component Details

### 10.1 GGUF Loader (`gguf_loader.py`)

Ported from [city96/ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) (Apache 2.0):

- **`load_gguf_weights()`** — Loads base + adapter GGUF, merges state dicts, loads into model, patches forwards
- **`patch_gguf_linears()`** — Monkey-patches `Linear.forward()`: dequantize packed weight → bf16 → apply LoRA if present → `F.linear()`
- **`apply_lora_as_patches()`** — Attaches LoRA deltas as buffers on Linear modules (avoids dequant→merge→requant)
- **LoRA buffer device alignment** — After `block_swap()`, patched forward uses `device=weight.device` to avoid mismatch

### 10.2 TTS Engine (`tts_engine.py`)

- **Model**: `ChatterboxTurboTTS.from_pretrained(device="cuda", token=True)` — HF auth required
- **Voice ref**: Must be >5 seconds. Auto-converted from MP3 to WAV if needed.
- **Text normalization**: `normalize_text()` replaces Unicode smart quotes/dashes with ASCII, strips all `"` (quotes cause phantom pauses and truncation in Chatterbox)
- **`--text` CLI option**: Pass script text directly instead of reading from `inputs/scripts/default.txt`. Priority: `--text` > file.
- **`[s:N]` pause tags**: Insert N seconds of silence (e.g. `[s:1.5]`, `[s:.5]`)
- **`${filename}` audio references**: Splice pre-recorded audio files from `inputs/scripts/` into the output (e.g. `${intro.mp3}`). Files are loaded and resampled to 24kHz.
- **Line-based splitting**: `_split_script_to_segments()` splits script by lines. Each line = one logical unit. `[s:N]` tags become explicit silence segments. `${filename}` tags become audio file splice segments.
- **Sub-sentence splitting**: Within each line, `split_into_sentences()` splits at `.!?` boundaries. Each sub-sentence is a separate TTS call. This prevents Chatterbox from inserting phantom pauses or truncating multi-sentence lines.
- **Gaps**: 200ms silence between lines, 100ms between sub-sentences within a line
- **Silence trimming**: `trim_silence()` trims leading (50ms margin) and trailing (200ms margin) silence from each chunk at -40dB threshold
- **Output**: 24 kHz mono WAV, padded 0.25s front + 0.5s end
- **Seed control**: `torch.manual_seed(config.VIDEO_SEED + chunk_counter)` for deterministic but unique output per chunk

### 10.3 Video Engine (`video_engine.py`)

**`load_model()` flow:**

1. Ensure all weights downloaded (model_manager)
2. Create InfiniteTalkPipeline with WanModel on meta device
3. Load GGUF weights (base + adapter), patch Linear forwards
4. Pre-encode T5 text prompt on GPU (before block swap fills VRAM)
5. Offload VAE/CLIP/T5 to CPU for VRAM headroom
6. Auto-tune block swap based on free VRAM
7. Apply LoRA as patches on Linear modules

**`generate()` flow:**

1. Normalize audio to -23 LUFS
2. Extract wav2vec2 embeddings
3. Call `pipe.generate_infinitetalk()` (streaming denoising loop)
4. FFmpeg mux video + audio → MP4

---

## 11. Performance (RTX 4090 Laptop 16 GB VRAM)

| Metric                                | Value                               |
| ------------------------------------- | ----------------------------------- |
| Per-pass denoising                    | ~34s (I2V rank-64 LoRA, 22+18 swap) |
| Per-step (1 pass, audio_cfg=1.0)      | ~34s                                |
| Per-chunk denoising (6 steps)         | ~204s                               |
| Per-chunk overhead (VAE + gap)        | ~28s                                |
| First-chunk extra (CLIP + VAE encode) | ~24s                                |
| **8.2s video (estimated, 6 steps)**   | **~720s inference = ~1.46 min/sec** |
| Total pipeline (8.2s video)           | ~14 min                             |
| VRAM allocated                        | ~8.5 GB                             |
| VRAM reserved                         | ~13.5 GB                            |
| GPU temp                              | 67-68°C                             |
| ETA accuracy                          | ~1% error                           |

**CLI override:** `--audio-cfg 2.0` for stronger lip sync at ~2× render time.

---

## 12. CLI Reference

| Flag                  | Description                                       | Default                        |
| --------------------- | ------------------------------------------------- | ------------------------------ |
| `--text STRING`       | Script text for TTS (instead of default.txt)      |                                |
| `--audio-only`        | Generate audio only (skip video)                  |                                |
| `--video-only AUDIO`  | Generate video from existing audio file           |                                |
| `--animation TEXT`    | Animation description (text or path to .txt file) | `inputs/animation/default.txt` |
| `--filename NAME`     | Base filename for all outputs (auto-increments)   | `run_YYYYMMDD_HHMMSS`          |
| `--audio-cfg SCALE`   | Audio guidance scale                              | 1.0                            |
| `--blocks-to-swap N`  | DiT blocks to swap CPU↔GPU (0-40)                 | auto-tune                      |
| `--prefetch-blocks N` | Async prefetch N blocks ahead                     | 1                              |

---

## 13. Critical Rules

These rules were each learned through painful debugging. Violating any causes significant regressions. See `docs/history.md` for the full story behind each.

1. **NEVER use Euler solver** — Significantly slower and doesn't work.
2. **NEVER increase audio_cfg above 2.0** — 2.5 caused over-guidance artifacts.
3. **audio_cfg=1.0 requires I2V rank-64 LoRA** — T2V rank-32 LoRA causes progressive blur.
4. **DPM++ SDE 2nd-order breaks with CFG** — Extrapolation overshoots. Use 1st-order only.
5. **frame_num=81 requires GGUF** — OOMs with FP8 (~18.9 GB model).
6. **LoRA requires text_cfg=1.0** — Distilled model, text CFG is baked in.
7. **activation_reserve must be 8.5 GiB** for frame_num=81 — prevents WDDM spill.
8. **Always offload DiT blocks before VAE decode** — 3D transposed conv needs VRAM headroom.
9. **Skip block offload for CLIP+VAE encode** — GGUF model fits alongside.
10. **`torch.cuda.empty_cache()` between forward passes** — Essential for GGUF dequant temporaries.
11. **All output files share `config.RUN_FILENAME`** — Resolved once at startup from `--filename` or timestamp. All outputs (`.wav`, `.mp4`, `.png`, `.log`) use the same base name in `output/`.
12. **Root logger must be DEBUG level** — Console handler controls visibility.
13. **Re-pin block memory after each offload** — `nn.Module.to()` creates new tensors.
14. **`python -m pip` on Windows** — Not `pip.exe`, to avoid file locking.
15. **`transformers>=5.0` breaks diffusers** — Removed MT5Tokenizer. Pin `<5`.
16. **`xfuser>=0.4.2` breaks on torch 2.4.1** — Needs torch 2.5+.
17. **TTS temperature must stay ≥0.6** — 0.5 causes looping (38.6s for 22s script); 0.4 causes severe looping (132.8s). Chatterbox Turbo only.
18. **Strip `"` before TTS** — Quotes cause phantom pauses and text truncation in Chatterbox. `normalize_text()` handles this.

---

## 14. Future Improvements

See `docs/future-enhancements.md`.
