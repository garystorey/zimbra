# Windows 11 Compatibility

Full validation of all dependencies for native Windows 11 support without WSL.

---

## Target Environment

- **OS**: Windows 11
- **Python**: 3.10.x (64-bit)
- **GPU**: NVIDIA RTX 4090 Laptop GPU (16GB VRAM)
- **CUDA**: 12.1 (bundled with PyTorch cu121 wheels)

---

## Compatibility Matrix

| Package           | Windows wheel?  | Version      | Notes                                                                              |
| ----------------- | --------------- | ------------ | ---------------------------------------------------------------------------------- |
| `torch`           | ✅ win_amd64    | 2.4.1+cu121  | From `download.pytorch.org/whl/cu121`                                              |
| `torchaudio`      | ✅ win_amd64    | 2.4.1+cu121  | From `download.pytorch.org/whl/cu121`                                              |
| `torchvision`     | ✅ win_amd64    | 0.19.1+cu121 | From `download.pytorch.org/whl/cu121`                                              |
| `xformers`        | ✅ win_amd64    | 0.0.28.post1 | From `download.pytorch.org/whl/cu124` (cu121 lacks Windows wheels for torch 2.4.1) |
| `flash_attn`      | ✅ win_amd64    | 2.7.4.post1  | Prebuilt Windows wheel from `github.com/kingbri1/flash-attention`                  |
| `diffusers`       | ✅ pure Python  | >=0.31.0     | No platform issues                                                                 |
| `transformers`    | ✅ pure Python  | >=4.49.0,<5  | transformers 5.x breaks diffusers (removed MT5Tokenizer)                           |
| `accelerate`      | ✅ pure Python  | >=1.1.1      | No platform issues                                                                 |
| `optimum-quanto`  | ✅ pure Python  | 0.2.6        | FP8 quantization for InfiniteTalk                                                  |
| `spacy-pkuseg`    | ✅ win_amd64    | 1.0.1        | **Prebuilt wheels since Jul 2025** — no C++ build needed                           |
| `decord`          | ✅ win_amd64    | latest       | Prebuilt Windows wheel                                                             |
| `opencv-python`   | ✅ win_amd64    | >=4.9.0.80   | Standard Windows wheel                                                             |
| `numpy`           | ✅ win_amd64    | 1.26.4       | Standard Windows wheel                                                             |
| `librosa`         | ✅ pure Python  | latest       | No platform issues                                                                 |
| `soundfile`       | ✅ win_amd64    | latest       | Bundles libsndfile for Windows                                                     |
| `moviepy`         | ✅ pure Python  | 1.0.3        | No platform issues                                                                 |
| `pyloudnorm`      | ✅ pure Python  | latest       | Audio loudness normalization                                                       |
| `scikit-image`    | ✅ win_amd64    | latest       | Image processing                                                                   |
| `loguru`          | ✅ pure Python  | latest       | Logging                                                                            |
| `easydict`        | ✅ pure Python  | latest       | Config handling                                                                    |
| `ftfy`            | ✅ pure Python  | latest       | Text processing                                                                    |
| `rich`            | ✅ pure Python  | latest       | Progress bars; transitive dep of transformers/accelerate                           |
| `safetensors`     | ✅ win_amd64    | 0.5.3        | Chatterbox model loading                                                           |
| `imageio`         | ✅ pure Python  | latest       | Image I/O                                                                          |
| `imageio-ffmpeg`  | ✅ pure Python  | latest       | FFmpeg binary bundled                                                              |
| `scenedetect`     | ✅ pure Python  | latest       | Scene detection                                                                    |
| `huggingface_hub` | ✅ pure Python  | latest       | Model downloads                                                                    |
| **`gradio`**      | ❌ **Excluded** | —            | **Not needed** — CLI-only interface, no web UI                                     |
| **`mlflow`**      | ❌ **Excluded** | —            | **Not needed** — experiment tracking, not used for inference                       |
| **`dashscope`**   | ❌ **Excluded** | —            | **Not needed** — InfiniteTalk TTS feature, we use Chatterbox instead               |

---

## Prerequisites on Windows 11

| Prerequisite           | Required?      | Notes                                                      |
| ---------------------- | -------------- | ---------------------------------------------------------- |
| Python 3.10.x (64-bit) | ✅ Yes         | Required by flash_attn prebuilt wheels and InfiniteTalk    |
| NVIDIA GPU driver      | ✅ Yes         | Latest Game Ready or Studio driver                         |
| CUDA 12.1 runtime      | ✅ Bundled     | Included with PyTorch cu121 wheels — no separate install   |
| FFmpeg binary          | ✅ Yes         | In PATH, or auto-detected via `imageio-ffmpeg`             |
| WSL                    | ❌ Not needed  | Everything runs natively on Windows                        |
| Visual C++ Build Tools | ❌ Not needed  | All packages have prebuilt wheels                          |
| Git                    | ⚠️ Recommended | For cloning InfiniteTalk repo; install.py warns if missing |

---

## Python Version Rationale

Python 3.10 was chosen as the target because it is the intersection of all package support:

| Package                    | Python support |
| -------------------------- | -------------- |
| InfiniteTalk               | Tested on 3.10 |
| flash_attn prebuilt wheels | 3.10 - 3.12    |
| xformers cu121 wheels      | 3.10 - 3.12    |
| Chatterbox                 | >= 3.10        |
| spacy-pkuseg               | 3.9 - 3.13     |

---

## Known Gotchas & Mitigations

### 1. `spacy-pkuseg` version (Chatterbox) — LOW RISK

**Problem:** Chatterbox's `pyproject.toml` depends on `spacy-pkuseg` which historically required C++ compilation on Windows. This was the #1 reported install issue.
**Mitigation:** `spacy-pkuseg==1.0.1` (released Jul 2025) now ships prebuilt Windows wheels for cp310-cp313. We pin this version explicitly. Since we install Chatterbox with `--no-deps`, we control which version gets installed. No C++ build tools needed.

### 2. `flash_attn` on Windows — MEDIUM RISK

**Problem:** `flash_attn` does not provide official Windows wheels. Building from source requires CUDA toolkit + MSVC.
**Mitigation:** Use prebuilt Windows wheels from [kingbri1/flash-attention](https://github.com/kingbri1/flash-attention/releases) (bdashore3 redirects here). The `cu124torch2.4.0` variant works with our `torch==2.4.1+cu121`. Install via direct URL in `install.py`.

### 2b. `xformers` Windows wheels — LOW RISK

**Problem:** The cu121 PyTorch index only has xformers up to 0.0.27.post2 (for torch 2.4.0). There is no xformers 0.0.28 wheel for torch 2.4.1 on cu121.
**Mitigation:** Install `xformers==0.0.28.post1` from the cu124 index instead. The xformers wheel only checks the torch version (2.4.1), not the CUDA version of the torch install. Works correctly with torch 2.4.1+cu121.

### 2c. `transformers` 5.x breaks diffusers — LOW RISK

**Problem:** `transformers>=5.0` removed `MT5Tokenizer`, which `diffusers` imports at module load time (via HunyuanDiT pipeline). This causes `RuntimeError` on `import xfuser`.
**Mitigation:** Pin `transformers>=4.49.0,<5` in requirements.txt.

### 2d. `xfuser` 0.4.2+ incompatible with torch 2.4.1 — LOW RISK

**Problem:** `xfuser>=0.4.2` imports `torch.distributed.tensor.experimental._attention` which was added in torch 2.5. Fails at import time with torch 2.4.1.
**Mitigation:** Pin `xfuser==0.4.1` in requirements.txt.

### 3. InfiniteTalk CPU offloading for 16GB VRAM — EXPECTED

**Problem:** The InfiniteTalk FP8 DiT has ~18.9B params (~18.9GB at 1 byte/param), which exceeds 16GB VRAM. The extra ~4.9B over the base Wan 14B comes from audio cross-attention layers.
**Mitigation:** Auto-tuned VRAM management keeps ~14.5B params persistent on GPU; ~4.4B overflow params stream per-layer via in-place CPU↔GPU transfers during each forward pass. FP8 quantization (`optimum-quanto`) halves the base model from ~28GB fp16. LightX2V distilled LoRA reduces denoising from 20 steps to 4 steps with `text_guide_scale=1.0` (2-pass CFG). Vendor `layers.py` patched to replace catastrophic `copy.deepcopy().to(GPU)` overflow path with efficient in-place `.to()` transfers.

### 3b. Vendor VRAM management patch — CRITICAL

**Problem:** The original InfiniteTalk `src/vram_management/layers.py` overflow forward methods used `copy.deepcopy(module).to(GPU)` per forward call for `AutoWrappedModule`, and `cast_to_device()` temp copies for `AutoWrappedQLinear`/`AutoWrappedLinear`. With ~4.4B overflow params, this caused GPU to sit idle while RAM maxed out on synchronous deep copies.
**Mitigation:** Patched all three overflow forward methods to use in-place `.to(device)` → compute → `.to(offload_device)`. Standard CPU offloading pattern, no deep copies.

### 4. Model download size (~30GB+) — LOW RISK

**Problem:** First-time setup requires downloading ~30GB of model weights (Wan2.1-I2V-14B-480P alone is ~28GB).
**Mitigation:** One-time download with progress bar via `huggingface_hub.snapshot_download()`. Atomic downloads prevent corruption. A `--download-only` CLI flag allows pre-fetching.

### 5. Long audio chunking for TTS — LOW RISK

**Problem:** Chatterbox Turbo works best with short text segments. Poor chunking causes audible seams.
**Mitigation:** Chunk at sentence boundaries, crossfade-stitch with 50ms overlap.

### 6. VRAM fragmentation between models — LOW RISK

**Problem:** Loading/unloading models can leave fragmented GPU memory.
**Mitigation:** Explicit `del model`, `torch.cuda.empty_cache()`, `gc.collect()` between model loads.

### 7. `decord` on Windows — LOW RISK

**Problem:** `decord` may not have prebuilt wheels for all Python versions on Windows.
**Mitigation:** Prebuilt wheels are available for cp310. Fallback to `imageio` if needed.

### 8. Chatterbox on torch 2.4.1 — MEDIUM RISK

**Problem:** Chatterbox pins `torch==2.6.0` but we use `2.4.1+cu121` for InfiniteTalk compatibility.
**Mitigation:** Install Chatterbox with `--no-deps`. Chatterbox's torch usage is standard and works on 2.4.x (tested in A1111/ComfyUI ecosystems). Verify on fresh install.
