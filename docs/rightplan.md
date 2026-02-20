# Talking Head Video Generator — Reimplementation Plan

A definitive guide to rebuilding this project from scratch, incorporating every lesson learned across 14 phases of development. This document reflects the **final working state** — not the exploratory path we took to get here.

---

## What This Project Does

A Python CLI app that takes a voice-clone reference audio, a portrait image, and a text script, then generates a "talking head" video locally on an NVIDIA RTX 4090 Laptop GPU (16 GB VRAM).

**Pipeline:** Text → Voice-Cloned Speech (Chatterbox Turbo) → Talking-Head Video (InfiniteTalk) → FFmpeg Mux → MP4

Models run **sequentially** — TTS is fully unloaded (0 GB VRAM) before the video model loads. Peak VRAM never exceeds ~13.5 GB during InfiniteTalk inference.

---

## Technology Choices (Final)

| Component            | Library                                                                         | Why It Won                                                                                                                                                                         |
| -------------------- | ------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **TTS**              | [Chatterbox Turbo](https://github.com/resemble-ai/chatterbox) (350M params)     | Instant voice cloning from ~10s reference clip. Paralinguistic tags (`[laugh]`, `[cough]`). ~4 GB VRAM. 24 kHz output.                                                             |
| **Video**            | [InfiniteTalk](https://github.com/MeiGen-AI/InfiniteTalk) (Wan2.1-I2V-14B base) | Audio-driven (wav2vec2). Any art style (photo/cartoon/3D/illustrated). Arbitrary aspect ratios. 480p/720p. Streaming mode for unlimited length. 16 GB VRAM with GGUF + block swap. |
| **Quantization**     | GGUF Q4_0 base + Q8 adapter (~12.8 GB total)                                    | On-demand dequantization. ~12.8 GB fits on 16 GB with block swap. Matches ComfyUI's model format. Ported from city96/ComfyUI-GGUF.                                                 |
| **LoRA Accelerator** | LightX2V distilled LoRA (317 MB)                                                | Reduces denoising from 20 steps to 4. `text_guide_scale=1.0` (no text CFG). Applied as patches during GGUF dequant forward — no dequant→merge→requant needed.                      |
| **Solver**           | DPM++ SDE (sde-dpmsolver++) 1st-order                                           | Stochastic noise prevents deterministic error accumulation across streaming chunks. Matches ComfyUI reference workflow.                                                            |
| **VRAM Management**  | Block-level swap with async CUDA stream prefetch (Kijai approach)               | Swap entire DiT blocks (~320 MB each GGUF) instead of per-layer transfers. Async prefetch hides transfer latency behind compute. Auto-tuned based on free VRAM.                    |
| **CLI**              | Python `argparse`                                                               | Simplest interface. Easy to wrap in a UI later.                                                                                                                                    |
| **Post-processing**  | FFmpeg                                                                          | Mux audio+video at 25 FPS.                                                                                                                                                         |

### What Was Tried and Rejected

| Technology                            | Why Rejected                                                                                                                                                     |
| ------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **EchoMimicV2**                       | Original video engine. SD 1.5-based, square-only (512×512). Replaced by InfiniteTalk.                                                                            |
| **Hallo/Hallo2**                      | SD 1.5-based, square-only, no arbitrary aspect ratios.                                                                                                           |
| **HunyuanVideo-Avatar**               | Minimum 24 GB VRAM natively.                                                                                                                                     |
| **FP8 quantization** (optimum-quanto) | Model is ~18.9 GB — too large for 16 GB VRAM. Requires aggressive block swap (17 persistent / 23 swap). GGUF is smaller (~12.8 GB) and faster. Removed entirely. |
| **TeaCache**                          | Skips redundant DiT forward passes. Never useful at low step counts (warmup period prevents any skipping). Removed entirely.                                     |
| **SageAttention**                     | Won't compile on Windows with torch 2.4.x. Removed.                                                                                                              |
| **Euler solver**                      | Significantly slower than DPM++ variants. Do NOT use.                                                                                                            |
| **DPM++ 2M solver**                   | Deterministic — errors compound across streaming chunks. Motion frame injection disrupts 2nd-order trajectory assumption.                                        |
| **DPM++ SDE 2nd-order**               | Extrapolation overshoots when denoised estimates include CFG scaling. Only works with audio_cfg=1.0 which itself causes progressive blur.                        |
| **audio_cfg=1.0**                     | Skips second forward pass (halves denoising time), but causes progressive blur/warp in later chunks. Only viable with frame_num=81 (fewer chunks).               |
| **frame_num=81**                      | OOMs during VAE 3D-conv on 16 GB. Would need VAE temporal tiling (not implemented).                                                                              |

---

## Target Environment

| Requirement                | Value                                                   |
| -------------------------- | ------------------------------------------------------- |
| **OS**                     | Windows 11 (native, no WSL)                             |
| **Python**                 | 3.10.x (64-bit) — intersection of all package support   |
| **GPU**                    | NVIDIA with 16 GB+ VRAM (tested on RTX 4090 Laptop GPU) |
| **CUDA**                   | 12.1 (bundled with PyTorch cu121 wheels)                |
| **FFmpeg**                 | In PATH or auto-detected via `imageio-ffmpeg`           |
| **HuggingFace account**    | Required for Chatterbox model download                  |
| **Visual C++ Build Tools** | NOT needed — all packages have prebuilt wheels          |
| **Git**                    | Recommended (for cloning InfiniteTalk repo)             |

### Why Python 3.10

| Package                    | Constraint     |
| -------------------------- | -------------- |
| InfiniteTalk               | Tested on 3.10 |
| flash_attn prebuilt wheels | 3.10 - 3.12    |
| xformers cu121 wheels      | 3.10 - 3.12    |
| Chatterbox                 | >= 3.10        |
| spacy-pkuseg               | 3.9 - 3.13     |

---

## Dependency Stack (Exact Versions)

### Conflict Resolution

Chatterbox and InfiniteTalk have 2 soft dependency conflicts:

| Package     | Chatterbox Pins | InfiniteTalk Pins | Resolution      | Rationale                                                                       |
| ----------- | --------------- | ----------------- | --------------- | ------------------------------------------------------------------------------- |
| `torch`     | `==2.6.0`       | `==2.4.1` (cu121) | `==2.4.1+cu121` | InfiniteTalk + xformers + flash_attn tied to 2.4.1; Chatterbox works fine on it |
| `diffusers` | `==0.29.0`      | `>=0.31.0`        | `>=0.31.0`      | InfiniteTalk needs >=0.31.0; Chatterbox's usage is minimal                      |

**Strategy:** Install Chatterbox with `--no-deps`, then install its unique non-conflicting deps separately. Proven approach used by [Chatterbox TTS Server](https://github.com/devnen/Chatterbox-TTS-Server).

### PyTorch Stack

| Package       | Version      | Source                                                                  | Notes                                                                                                                           |
| ------------- | ------------ | ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `torch`       | 2.4.1+cu121  | `download.pytorch.org/whl/cu121`                                        | Must be installed first                                                                                                         |
| `torchvision` | 0.19.1+cu121 | `download.pytorch.org/whl/cu121`                                        |                                                                                                                                 |
| `torchaudio`  | 2.4.1+cu121  | `download.pytorch.org/whl/cu121`                                        |                                                                                                                                 |
| `xformers`    | 0.0.28.post1 | `download.pytorch.org/whl/cu124`                                        | cu121 index lacks Windows wheels for torch 2.4.1. cu124 wheel only checks torch version, not CUDA version of the torch install. |
| `flash_attn`  | 2.7.4.post1  | [kingbri1/flash-attention](https://github.com/kingbri1/flash-attention) | Prebuilt Windows wheel. cu124torch2.4.0 variant works with torch 2.4.1+cu121.                                                   |

### InfiniteTalk Dependencies (from PyPI via requirements.txt)

| Package           | Version     | Notes                                                                               |
| ----------------- | ----------- | ----------------------------------------------------------------------------------- |
| `diffusers`       | >=0.31.0    |                                                                                     |
| `transformers`    | >=4.49.0,<5 | **transformers 5.x removed MT5Tokenizer, breaks diffusers**                         |
| `accelerate`      | >=1.1.1     |                                                                                     |
| `tokenizers`      | >=0.20.3    |                                                                                     |
| `numpy`           | 1.26.4      | **Must be pre-pinned** before requirements.txt to prevent PyTorch pulling numpy 2.x |
| `optimum-quanto`  | 0.2.6       | Required by InfiniteTalk vendor code (T5 quantization)                              |
| `xfuser`          | 0.4.1       | **0.4.2+ requires `torch.distributed.tensor.experimental` which is torch 2.5+**     |
| `gguf`            | >=0.6.0     | GGUF model loading                                                                  |
| `opencv-python`   | >=4.9.0.80  |                                                                                     |
| `moviepy`         | 1.0.3       |                                                                                     |
| `imageio`         | latest      |                                                                                     |
| `imageio-ffmpeg`  | latest      | FFmpeg binary bundled                                                               |
| `scikit-image`    | latest      |                                                                                     |
| `loguru`          | latest      |                                                                                     |
| `easydict`        | latest      |                                                                                     |
| `ftfy`            | latest      |                                                                                     |
| `pyloudnorm`      | latest      | Audio loudness normalization                                                        |
| `scenedetect`     | latest      |                                                                                     |
| `decord`          | latest      | Video decoding                                                                      |
| `einops`          | latest      |                                                                                     |
| `soundfile`       | latest      |                                                                                     |
| `huggingface_hub` | latest      | Model downloads                                                                     |
| `rich`            | latest      | Progress bars, console output                                                       |
| `tqdm`            | latest      | HuggingFace progress                                                                |

### Chatterbox Unique Dependencies (installed after `--no-deps`)

| Package          | Version | Notes                                                                                      |
| ---------------- | ------- | ------------------------------------------------------------------------------------------ |
| `librosa`        | latest  | Audio analysis                                                                             |
| `spacy-pkuseg`   | 1.0.1   | **Must pin** — prebuilt Windows wheels since Jul 2025. Earlier versions require C++ build. |
| `s3tokenizer`    | latest  | Tokenizer                                                                                  |
| `conformer`      | latest  | Conformer model                                                                            |
| `resemble-perth` | latest  | Audio processing                                                                           |
| `safetensors`    | 0.5.3   | Model loading                                                                              |
| `pykakasi`       | latest  | Japanese transliteration                                                                   |
| `omegaconf`      | latest  | Config management (**required** by Chatterbox, not InfiniteTalk)                           |

### Excluded Dependencies (do NOT install)

| Package           | Why Excluded                                    |
| ----------------- | ----------------------------------------------- |
| `gradio`          | CLI-only interface, no web UI                   |
| `mlflow`          | Experiment tracking, not needed for inference   |
| `dashscope`       | InfiniteTalk's built-in TTS — we use Chatterbox |
| `kokoro`          | InfiniteTalk's built-in TTS — we use Chatterbox |
| `controlnet-aux`  | Not used by InfiniteTalk                        |
| `mediapipe`       | Not used by InfiniteTalk (was EchoMimicV2)      |
| `onnxruntime-gpu` | Not used by InfiniteTalk                        |

---

## Model Weights

All auto-download on first run via `huggingface_hub`:

| Model                     | HuggingFace Source                      | Size     | Download Method               | Purpose                                                                                                              |
| ------------------------- | --------------------------------------- | -------- | ----------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| Chatterbox Turbo          | `ResembleAI/chatterbox-turbo`           | ~700 MB  | `from_pretrained(token=True)` | TTS voice cloning                                                                                                    |
| Wan2.1-I2V-14B (VAE+CLIP) | `Wan-AI/Wan2.1-I2V-14B-480P`            | ~5.3 GB  | `snapshot_download()`         | VAE, CLIP, tokenizers, config                                                                                        |
| chinese-wav2vec2-base     | `TencentGameMate/chinese-wav2vec2-base` | ~360 MB  | `snapshot_download()`         | Audio feature extraction. **Note:** `model.safetensors` must be fetched from `refs/pr/1` branch (missing from main). |
| GGUF base (Q4_0)          | `city96/Wan2.1-I2V-14B-480P-gguf`       | ~10.2 GB | `hf_hub_download()`           | DiT base model (quantized)                                                                                           |
| GGUF adapter (Q8)         | `Kijai/WanVideo_comfy_GGUF`             | ~2.65 GB | `hf_hub_download()`           | InfiniteTalk audio cross-attention adapter                                                                           |
| LightX2V LoRA             | `Kijai/WanVideo_comfy`                  | ~317 MB  | `hf_hub_download()`           | Step distillation (20→4 steps)                                                                                       |

**Download locations:**

```
models/
├── infinitetalk/
│   ├── Wan2.1-I2V-14B-480P/     # VAE, CLIP, tokenizers
│   ├── chinese-wav2vec2-base/    # wav2vec2
│   ├── gguf/
│   │   ├── wan2.1-i2v-14b-480p-Q4_0.gguf        # GGUF base
│   │   └── Wan2_1-InfiniteTalk_Single_Q8.gguf    # GGUF adapter
│   └── loras/
│       └── Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors
└── chatterbox/
```

---

## Project Structure (Final)

```
zimbra/
├── README.md                   # Quick start, CLI reference, troubleshooting
├── requirements.txt            # Pinned dependencies (DO NOT pip install directly)
├── install.bat                 # Windows one-click installer → calls install.py
├── install.py                  # Creates venv, installs deps in correct order, clones vendor
├── generate.bat                # Convenience wrapper: activates venv + runs generate.py
├── config.py                   # All paths, defaults, GPU settings, auto-tune functions
├── generate.py                 # CLI entry point (argparse)
├── pipeline.py                 # Orchestrates TTS → Video → Post-process
├── tts_engine.py               # Chatterbox Turbo wrapper
├── video_engine.py             # InfiniteTalk wrapper
├── model_manager.py            # Auto-download logic for all model weights
├── gguf_loader.py              # GGUF dequantization, loading, patched Linear, LoRA-as-patches
├── utils.py                    # GPU thermal management, file validation, FFmpeg helpers
├── inputs/
│   ├── images/default.png         # Default portrait image
│   ├── voice_ref/default.mp3      # Default voice reference (>5 seconds)
│   ├── scripts/default.txt        # Default script text
│   └── animation/default.txt      # Default animation description
├── outputs/                    # Generated files (gitignored)
│   ├── audio/                  # TTS audio (WAV)
│   ├── video/                  # Final videos (MP4)
│   └── logs/                   # Run logs
├── models/                     # Downloaded model weights (gitignored, ~20 GB for GGUF)
├── vendor/                     # Cloned repos (gitignored)
│   └── infinitetalk/           # InfiniteTalk source (with our patches)
├── venv/                       # Virtual environment (gitignored)
└── docs/
    └── rightplan.md            # This document
```

---

## Step-by-Step Reimplementation Plan

### Step 1: Project Scaffold

Create the directory structure, `.gitignore`, `requirements.txt`, `install.bat`, and `install.py`.

**`.gitignore`:** Exclude `venv/`, `models/`, `outputs/`, `vendor/`, `__pycache__/`, IDE files, OS files.

**`requirements.txt`:** Pin all InfiniteTalk deps (see dependency list above). Add header warning not to `pip install -r` directly. List Chatterbox unique deps as comments.

**`install.py`** — The install script must follow this **exact order** (order matters):

```
1.  Create venv/ (auto-detect Python 3.10)
2.  Upgrade pip, setuptools, wheel (use `python -m pip`, NOT `pip.exe` — avoids Windows file locking)
3a. Install torch==2.4.1+cu121, torchvision==0.19.1+cu121, torchaudio==2.4.1+cu121 from cu121 index
3b. Install xformers==0.0.28.post1 from cu124 index (cu121 lacks Windows wheels for torch 2.4.1)
3c. Install flash_attn==2.7.4.post1 from prebuilt Windows wheel URL (kingbri1/flash-attention)
4a. Pre-pin numpy==1.26.4 (prevents PyTorch from pulling numpy 2.x)
4b. Install requirements.txt (InfiniteTalk deps)
5a. Install chatterbox-tts --no-deps
5b. Install Chatterbox unique deps: librosa spacy-pkuseg==1.0.1 s3tokenizer conformer resemble-perth safetensors==0.5.3 pykakasi omegaconf
6.  Clone InfiniteTalk into vendor/ (git clone https://github.com/MeiGen-AI/InfiniteTalk vendor/infinitetalk)
7.  Create runtime directories (outputs/, models/)
8.  Verify: torch CUDA, xformers, flash_attn, Chatterbox import, wan import
```

**`install.bat`:** Calls `python install.py`, auto-detects Python 3.10 if system default is different.

**Validation checks after install:**

- `import torch; assert torch.cuda.is_available()`
- `import xformers`
- `import flash_attn`
- `from chatterbox.tts_turbo import ChatterboxTurboTTS`
- `import wan` (with `vendor/infinitetalk` on sys.path)
- Triton warning on Windows is expected and harmless

---

### Step 2: Configuration (`config.py`)

Central configuration module. All defaults overridable via CLI.

**Key sections:**

- **Project paths** — PROJECT_ROOT, INPUTS_DIR, OUTPUTS_DIR, MODELS_DIR, VENDOR_DIR, etc.
- **Default input file resolution** — Auto-find `default.wav`/`.mp3`, `default.png`/`.jpg`, `default.txt` in inputs/
- **CLI defaults** — exaggeration, cfg_weight, audio_format, FPS
- **RUN_TIMESTAMP** — Set once at startup by `setup_logging()`, shared across all output filenames (log, audio, video) so they correlate
- **Video engine defaults** — All the InfiniteTalk settings (see Final Configuration section below)
- **Model paths** — HuggingFace repo IDs, local checkpoint paths, GGUF paths, LoRA paths
- **GPU detection** — `get_gpu_info()` returns device name, CUDA version, VRAM
- **FFmpeg discovery** — env var → PATH → imageio-ffmpeg fallback
- **`auto_tune_blocks_to_swap()`** — Calculates optimal block swap count based on free VRAM, accounting for activation reserve + swap buffer slot + non-block overhead
- **`ensure_directories()`** — Creates all runtime dirs

---

### Step 3: Model Manager (`model_manager.py`)

Auto-download logic for all model weights:

- `ensure_wan_models()` — Downloads VAE, CLIP, tokenizers from `Wan-AI/Wan2.1-I2V-14B-480P`
- `ensure_wav2vec_model()` — Downloads wav2vec2 from `TencentGameMate/chinese-wav2vec2-base`. **Important:** `model.safetensors` must be from `refs/pr/1` branch.
- `ensure_lora_model()` — Downloads LightX2V LoRA from `Kijai/WanVideo_comfy`
- `ensure_gguf_models()` — Downloads GGUF base (~10.2 GB from `city96`) and adapter (~2.65 GB from `Kijai`)

All downloads are atomic (temp file → rename), with progress bars via `tqdm`/`rich`. Subsequent runs skip download if files exist.

---

### Step 4: Utilities (`utils.py`)

- File validation (audio/image format checks)
- MP3↔WAV conversion (librosa/soundfile for MP3→WAV, FFmpeg for WAV→MP3)
- FFmpeg helpers (mux audio+video, resize)
- Image dimension reading
- GPU thermal management (check temp, pause if above threshold)

---

### Step 5: TTS Engine (`tts_engine.py`)

Wrapper around Chatterbox Turbo:

```python
class TTSEngine:
    def load_model(self)      # ~4 GB VRAM
    def generate(self, text, voice_ref_path=None, output_path=None, temperature=0.8) -> str
    def unload_model(self)    # Frees to ~0 GB
```

**Key implementation details:**

- `ChatterboxTurboTTS.from_pretrained(device="cuda", token=True)` — HuggingFace auth required
- HF repo is `ResembleAI/chatterbox-turbo` (NOT `ResembleAI/chatterbox`)
- Output sample rate: 24000 Hz (`S3GEN_SR` constant)
- **Voice reference must be >5 seconds** (asserted in `prepare_conditionals`)
- Built-in voice (bundled `conds.pt`) works when no voice ref provided
- **Sentence-boundary chunking** — Split long text at `.!?` boundaries, max ~250 chars/chunk
- **50ms cosine crossfade stitching** between chunks eliminates audible seams
- Turbo version **ignores** CFG, exaggeration, min_p params (logs warnings, doesn't error)
- `unload_model()` must do `del self.model`, `torch.cuda.empty_cache()`, `gc.collect()`

---

### Step 6: GGUF Loader (`gguf_loader.py`)

Full GGUF loading pipeline, ported from [city96/ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) (Apache 2.0).

**Components:**

- **Dequantization functions** for all GGUF quant types (Q4_0, Q4_K, Q5_K, Q6_K, Q8_0, etc.)
- **`GGMLTensor`** subclass carrying quantization metadata (used during loading only, NOT stored in model)
- **`load_gguf_state_dict()`** — Reads GGUF files via `gguf.GGUFReader`, wraps tensors as `GGMLTensor`
- **`load_gguf_into_model()`** — Loads state_dict into WanModel. Quantized Linear weights stay packed as raw bytes in `nn.Parameter`. Non-Linear weights (LayerNorm, embeddings) are dequantized at load time. GGUF metadata (`_gguf_qtype`, `_gguf_shape`) stored on the **module**, not the parameter.
- **`patch_gguf_linears()`** — Monkey-patches `Linear.forward()` for on-the-fly dequantization. The patched forward:
  1. Reads `_gguf_qtype` and `_gguf_shape` from the module
  2. Dequantizes packed weight → bfloat16
  3. If LoRA patches exist (buffers named `_lora_up`, `_lora_down`), applies them: `weight += (up @ down) * scale`
  4. Runs `F.linear(input, weight, bias)`
- **`apply_lora_as_patches()`** — Attaches LoRA deltas as buffers on the corresponding Linear modules. Avoids the dequant→merge→requant problem by applying LoRA during the forward pass.
- **`load_gguf_weights()`** — High-level orchestrator: loads base GGUF + adapter GGUF, merges state dicts, loads into model, patches forwards.

**Critical: LoRA buffer device alignment.** After `block_swap()` moves blocks to GPU, LoRA buffers must use `device=weight.device` in the patched forward to avoid device mismatch.

---

### Step 7: Video Engine (`video_engine.py`)

Wrapper around InfiniteTalk:

```python
class VideoEngine:
    def load_model(self)
    def generate(self, image_path, audio_path, output_path=None, size="infinitetalk-480",
                 sample_steps=None, seed=42, animation_prompt=None) -> str
    def unload_model(self)
```

**`load_model()` flow:**

1. Call `model_manager` to ensure all weights are downloaded
2. Add `vendor/infinitetalk` to `sys.path`
3. Import InfiniteTalk pipeline classes
4. Create WanModel on **meta device** (no memory until weights loaded)
5. Call `gguf_loader.load_gguf_weights()` to load base + adapter GGUF, merge state dicts, patch Linear forwards
6. Call `gguf_loader.apply_lora_as_patches()` to attach LoRA deltas as buffers on Linear modules
7. Call `model.block_swap(blocks_to_swap)` for VRAM management
8. Set `pipe.vram_management = False` — bypass vendor's per-layer wrapping entirely
9. Set `pipe.t5_cpu = True` — T5 only encodes prompt once, doesn't need GPU
10. Set `pipe.vae.dtype = torch.bfloat16` — 2x faster encode/decode on tensor cores
11. Set `pipe._skip_block_offload = True` — GGUF blocks don't move during CLIP/VAE encode (model fits alongside)

**`generate()` flow:**

1. Normalize audio to -23 LUFS
2. Select solver, shift, steps, guide scales based on LoRA config
3. Call `pipe.generate_infinitetalk()` which runs the denoising loop
4. Mux result with audio via FFmpeg

**Solver selection (when LoRA enabled):**

- Solver: `dpmpp_sde` (DPM++ SDE 1st-order)
- Steps: 4
- Shift: 11
- text_guide_scale: 1.0
- audio_guide_scale: 2.0
- color_correction_strength: 1.0 (per-chunk, anchored to reference image)

---

### Step 8: Vendor Patches (applied to `vendor/infinitetalk/`)

These patches to the cloned InfiniteTalk source are **essential** for correct operation on 16 GB VRAM:

#### 8a. Block Swap (`wan/modules/multitalk_model.py`)

Add block-level swapping with async prefetch (replaces vendor's per-layer VRAM management):

- **`block_swap(blocks_to_swap, ...)`** — Moves last N blocks to CPU, pins their memory for DMA. Keeps first (40-N) blocks on GPU permanently.
- **`_run_blocks_with_swap(blocks, x, ...)`** — Main block execution loop:
  1. Async prefetch next block from pinned CPU → GPU on dedicated CUDA stream
  2. Compute current block on default stream
  3. Synchronous offload current block to CPU (`non_blocking=False`)
  4. Re-pin block memory for next prefetch cycle
- **`_pin_block_memory(block)`** — Pins parameters and buffers of CPU-resident blocks into page-locked memory. Must iterate `block.modules()` and update each module's `_buffers` dict directly.
- **Single-person attention optimization** — Pass `human_num` to `WanSelfAttention.forward()`. When `human_num <= 1`, skip the expensive O(n²) `get_attn_map_with_target` computation (only used for multi-person videos).

#### 8b. Denoising Loop (`wan/multitalk.py`)

The core generation function `generate_infinitetalk()` needs these changes:

- **DPM++ SDE solver implementation** — `sde-dpmsolver++` 1st-order update formula:
  ```python
  alpha_s, alpha_t = 1.0 - sigma_s, 1.0 - sigma_t
  h = log(alpha_t/sigma_t) - log(alpha_s/sigma_s)
  x_t = (sigma_t/sigma_s * exp(-h)) * x_s + (alpha_t * (1 - exp(-2h))) * denoised + sigma_t * sqrt(1 - exp(-2h)) * noise
  ```
- **`_skip_cfg` optimization** — When both `text_guide_scale ≈ 1.0` AND `audio_guide_scale ≈ 1.0`, skip all extra forward passes (CFG reduces to identity)
- **Direct latent passthrough** — Save denoised motion latent (`x0[0][:, -N:]`) BEFORE VAE decode and pass directly to next chunk. Eliminates cumulative VAE reconstruction error.
- **`torch.inference_mode()`** instead of `torch.no_grad()` (disables version counting, lower overhead)
- **`torch.cuda.empty_cache()` between forward passes** — Essential for GGUF mode (on-demand dequant creates BF16 temporaries across 604 linear layers, fragmenting CUDA free pool). Without it, VRAM grows +0.6 GB/pass → WDDM spill. Cost: ~150ms/call, prevents catastrophic 12x slowdown.
- **Block offload before VAE decode** — Always offload DiT blocks before VAE decode (3D transposed convolutions need VRAM headroom), even for GGUF. But skip offload for CLIP+VAE encode (GGUF model fits alongside).
- **VAE offload after decode** — Explicit `self.vae.model.cpu()` + `empty_cache()` after every VAE decode
- **Remove dead motion re-noising code** — Pre-loop and post-solver `torch.randn_like + add_noise` is always overwritten by clean motion frame injection
- **Tensor resize: bilinear** instead of nearest (`align_corners=False`) in `resize_and_centercrop()`

#### 8c. FFmpeg CRF (`wan/utils/multitalk_utils.py`)

In `cache_video()`: Change intermediate FFmpeg CRF from 10 → 1 (near-lossless intermediate encoding between chunks).

---

### Step 9: Pipeline Orchestration (`pipeline.py`)

```python
class Pipeline:
    def run(self, script, voice_ref, image, animation_prompt, ...)
    def _cleanup(self)  # Guaranteed model unloading
```

**Flow:**

1. Validate inputs (script text, voice reference, portrait image, animation prompt)
2. Load Chatterbox Turbo → generate audio → save WAV → unload TTS (0 GB VRAM)
3. Load InfiniteTalk → generate video → save MP4 → unload video engine
4. Report timings via rich panels (H:M:S format)

**Error handling:**

- `torch.cuda.OutOfMemoryError` caught in both TTS and video steps
- `_cleanup()` ensures models are always unloaded on any exception
- MP3 conversion failure is a warning, not fatal (WAV output still succeeds)

**ETA estimation:**

- TTS: ~5-18s depending on script length
- Video: `(num_chunks × steps × passes_per_step × secs_per_step) + (num_chunks × vae_decode_secs) + first_chunk_extra_secs`
- Use `VIDEO_SECS_PER_STEP` (17s), `VIDEO_VAE_DECODE_SECS` (15s), `VIDEO_FIRST_CHUNK_EXTRA_SECS` (23s)

---

### Step 10: CLI Entry Point (`generate.py`)

Full `argparse`-based CLI with grouped options:

| Flag                   | Description                              | Default                        |
| ---------------------- | ---------------------------------------- | ------------------------------ |
| `--script TEXT`        | Script text (inline)                     | `inputs/scripts/default.txt`   |
| `--script-file PATH`   | Script from file                         |                                |
| `--voice-ref PATH`     | Voice reference audio (>5s)              | `inputs/voice_ref/default.mp3` |
| `--image PATH`         | Portrait image                           | `inputs/images/default.png`    |
| `--animation TEXT`     | Animation description                    | `inputs/animation/default.txt` |
| `--audio-only`         | Generate audio only                      |                                |
| `--video-only AUDIO`   | Generate video from existing audio       |                                |
| `--video-size SIZE`    | `infinitetalk-480` or `infinitetalk-720` | `infinitetalk-480`             |
| `--sample-steps N`     | Denoising steps                          | 4                              |
| `--seed N`             | Random seed                              | 42                             |
| `--audio-cfg SCALE`    | Audio guidance scale                     | 2.0                            |
| `--blocks-to-swap N`   | DiT blocks to swap (0-40)                | auto-tune                      |
| `--prefetch-blocks N`  | Async prefetch blocks                    | 1                              |
| `--lora` / `--no-lora` | Enable/disable LoRA                      | enabled                        |
| `--verbose`            | Debug logging                            |                                |

**Startup flow:**

1. `setup_logging()` — Set `config.RUN_TIMESTAMP`, configure root logger at DEBUG (console handler controls visibility), file handler always at DEBUG
2. `_check_gpu()` — Verify CUDA availability and minimum VRAM
3. Parse args → run pipeline or utility mode
4. `try/finally` in all modes for guaranteed cleanup
5. `KeyboardInterrupt` → `pipeline._cleanup()`

**Logging architecture (critical to get right):**

- Root logger: always DEBUG level
- Console handler: INFO (or DEBUG with `--verbose`)
- File handler: always DEBUG, writes to `outputs/logs/run_TIMESTAMP.log`
- This ensures `logging.debug()` calls reach the file handler even without `--verbose`

---

## Final Configuration Values

These are the **correct final values** — do not use any intermediate values from the development phases:

### GGUF Configuration

```python
VIDEO_QUANT = "gguf"
VIDEO_GGUF_FRAME_NUM = 41                   # 4n+1, safe for VAE 3D-conv on 16GB
VIDEO_GGUF_MOTION_FRAME = 13                # Overlap frames between chunks (0.52s)
VIDEO_GGUF_LORA_SAMPLE_STEPS = 4            # LightX2V designed for 4 steps
VIDEO_GGUF_LORA_AUDIO_GUIDE_SCALE = 2.0     # 2 passes/step (audio CFG for lip sync)
VIDEO_SECS_PER_STEP = 17                    # ~16.8s/pass measured on RTX 4090 Laptop 16GB
VIDEO_GGUF_ACTIVATION_RESERVE = 7.0         # GiB — dequant temporaries need headroom
VIDEO_BLOCK_SIZE_BYTES = 320_000_000        # ~320 MB per DiT block in GGUF Q4_0+Q8
```

### Shared Settings

```python
VIDEO_LORA_SOLVER = "dpmpp_sde"             # DPM++ SDE 1st-order
VIDEO_LORA_SAMPLE_SHIFT = 11                # ComfyUI-validated
VIDEO_LORA_TEXT_GUIDE_SCALE = 1.0           # Must be 1.0 for distilled LoRA
VIDEO_USE_LORA = True                       # LightX2V LoRA enabled by default
VIDEO_NUM_DIT_BLOCKS = 40                   # InfiniteTalk has 40 DiT blocks
VIDEO_PREFETCH_BLOCKS = 1                   # Async prefetch 1 block ahead
GPU_THERMAL_THRESHOLD = 70                  # °C — pause between chunks if exceeded
color_correction_strength = 1.0             # Per-chunk, anchored to reference image
VIDEO_VAE_DECODE_SECS = 15                  # Per-chunk: VAE decode (~7s) + inter-chunk gap (~8s)
VIDEO_FIRST_CHUNK_EXTRA_SECS = 23           # CLIP encode (~3s) + VAE encode (~20s)
VIDEO_FPS = 25                              # InfiniteTalk output frame rate
VIDEO_MODE = "streaming"                    # Unlimited-length video in overlapping chunks
```

---

## VRAM Budget (16 GB, GGUF Mode)

Understanding the VRAM budget is critical. Every number here was measured:

```
GGUF model total: ~12.8 GB (10.2 GB base + 2.65 GB adapter)
  - 40 DiT blocks × ~320 MB = ~12.8 GB
  - Non-block params (embeddings, head, audio_proj, img_emb): ~500 MB

With auto_tune_blocks_to_swap (activation_reserve=7.0 GiB):
  - ~27 persistent blocks on GPU: ~8.6 GB
  - ~13 swapped blocks on CPU (pinned memory)
  - 1 swap buffer slot: ~320 MB
  - Non-block params: ~500 MB
  - Total weights on GPU: ~9.4 GB
  - Activation headroom: ~6.6 GB

During denoising:
  - Allocated: 8.7-8.9 GB
  - Reserved: 13.2-13.5 GB
  - No WDDM spill (peak well under 16 GB)

During VAE decode (blocks offloaded):
  - DiT blocks moved to CPU → ~500 MB weights on GPU
  - VAE model: ~500 MB
  - VAE 3D transposed conv intermediates: several GB
  - Headroom: sufficient (no WDDM spill)

TTS (Chatterbox Turbo): ~4.16 GB (fully unloaded before video)
```

### Why activation_reserve = 7.0 GiB

This was the most critical tuning parameter. The history:

- 1.0-2.0 GiB: Worked for persistent blocks but swap blocks caused WDDM spill (55x slowdown)
- 4.0-5.0 GiB: Still caused occasional spill with swap buffer overhead
- 7.0 GiB: Gives ~27 persistent + 13 swap for GGUF. Peak with intermediates stays well under 16 GB. Zero WDDM spill.

**The root issue:** Windows WDDM transparently spills GPU memory to system RAM via PCIe when physical VRAM is exceeded. This is invisible (no error) but causes 20-55x slowdowns. You must leave enough headroom to prevent this entirely.

---

## ComfyUI Reference Workflow (Our Target)

The quality/performance target was a known-good ComfyUI workflow:

```
Scheduler: DPM++ SDE, steps=6, shift=11, audio_cfg=1.0
frame_window=81, motion_frame=25, colormatch=disabled
Model: Q4_0 GGUF (~7-8GB), blocks_to_swap=20
LoRA: lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16
```

**Where we match ComfyUI:**

- GGUF Q4_0 base + Q8 adapter ✓
- DPM++ SDE solver ✓
- shift=11 ✓
- LightX2V LoRA ✓
- text_guide_scale=1.0 ✓
- Block swap with async prefetch ✓

**Where we differ (and why):**

- `frame_num=41` (not 81) — VAE 3D-conv OOMs at 81 on 16 GB without temporal tiling
- `audio_cfg=2.0` (not 1.0) — With frame_num=41, we have more chunks → more cross-chunk degradation → need audio CFG to prevent progressive blur
- `steps=4` for GGUF (not 6) — LightX2V designed for 4 steps; GGUF denoising is slightly slower per step so we save time
- `motion_frame=13` (not 25) — 25 doubled chunk count without fixing quality; 13 gives good overlap

---

## Measured Performance (RTX 4090 Laptop 16 GB)

| Metric                                | Value                      |
| ------------------------------------- | -------------------------- |
| Per-pass denoising                    | ~17s (16.8s avg)           |
| Per-step (2 passes, audio CFG)        | ~34s                       |
| Per-chunk denoising (4 steps)         | ~136s                      |
| Per-chunk overhead (VAE + gap)        | ~15s                       |
| First-chunk extra (CLIP + VAE encode) | ~23s                       |
| **5.8s video (5 chunks)**             | **~13 min (~2.2 min/sec)** |
| VRAM allocated                        | 8.7-8.9 GB                 |
| VRAM reserved                         | 13.2-13.5 GB               |
| GPU temp                              | 67-68°C                    |
| ETA accuracy                          | 0.8% error                 |

---

## Critical Rules (Hard-Won Lessons)

These rules were each learned through painful debugging. Violating any of them causes significant regressions:

1. **NEVER use Euler solver** — Significantly slower and doesn't work. Tested in Phase 10.
2. **NEVER increase audio_cfg above 2.0** — 2.5 caused over-guidance artifacts.
3. **audio_cfg=1.0 causes progressive blur** with frame_num=41. Only viable with frame_num=81 (fewer chunks).
4. **DPM++ SDE 2nd-order breaks with CFG** — Extrapolation overshoots when denoised includes CFG scaling. Use 1st-order only.
5. **frame_num=81 requires VAE temporal tiling** — OOMs during 3D-conv on 16 GB without it.
6. **LoRA requires text_cfg=1.0** — It's a distilled model, text CFG is baked in.
7. **activation_reserve must be 7.0 GiB** — prevents WDDM spill.
8. **Always offload DiT blocks before VAE decode** — Even for GGUF. 3D transposed conv needs VRAM headroom.
9. **Skip block offload for CLIP+VAE encode** — GGUF model fits alongside.
10. **`torch.cuda.empty_cache()` between forward passes** is essential for GGUF — dequant temporaries fragment CUDA pool.
11. **All output files share `config.RUN_TIMESTAMP`** — set once in `setup_logging()`.
12. **Root logger must be DEBUG level** — Console handler controls visibility. If root is INFO, `logging.debug()` calls are silently dropped before reaching file handler.
13. **Re-pin block memory after each offload** — `nn.Module.to()` creates new tensors; original pinned tensors are garbage collected.
14. **GPU→CPU with non-pinned destination is synchronous** regardless of `non_blocking=True` — async double-buffer pattern doesn't work.
15. **Use `python -m pip` on Windows** (not `pip.exe`) to avoid file locking errors.
16. **`transformers>=5.0` breaks diffusers** — removed MT5Tokenizer. Pin `<5`.
17. **`xfuser>=0.4.2` breaks on torch 2.4.1** — requires `torch.distributed.tensor.experimental` from torch 2.5+.

---

## Future Improvements

These are known opportunities not yet implemented:

| Improvement             | Expected Impact                                                        | Blocker                                                                                                     |
| ----------------------- | ---------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| **VAE temporal tiling** | Enable frame_num=81 → fewer chunks → audio_cfg=1.0 viable → ~2x faster | Implementation complexity                                                                                   |
| **3-step denoising**    | 25% faster denoising                                                   | Quality validation needed                                                                                   |
| **SageAttention**       | Faster attention computation                                           | No Windows-compatible wheels for torch 2.4.x                                                                |
| **Batched CFG**         | ~25-35% per-step speedup (merge 2 forward passes into 1)               | Audio batch dimension doubles as `human_num` in model — would trigger expensive multi-person attention path |
| **FusioniX LoRA**       | Alternative 8-step LoRA (different quality tradeoffs)                  | Testing needed                                                                                              |
