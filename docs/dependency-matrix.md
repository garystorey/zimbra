# Dependency Matrix

Full table of pinned versions, conflict resolutions, and install order rationale.

---

## Soft Conflicts Between Chatterbox and InfiniteTalk

| Package     | Chatterbox pins | InfiniteTalk pins | Our resolution  | Rationale                                                                        |
| ----------- | --------------- | ----------------- | --------------- | -------------------------------------------------------------------------------- |
| `torch`     | `==2.6.0`       | `==2.4.1` (cu121) | `==2.4.1+cu121` | InfiniteTalk + xformers + flash_attn tied to torch 2.4.1; Chatterbox works on it |
| `diffusers` | `==0.29.0`      | `>=0.31.0`        | `>=0.31.0`      | InfiniteTalk needs >=0.31.0; Chatterbox's diffusers usage is minimal             |

## Resolution Strategy

Install Chatterbox with `--no-deps` to prevent it from pulling in conflicting versions, then manually install only its unique non-conflicting dependencies. This is a proven approach used by the [Chatterbox TTS Server](https://github.com/devnen/Chatterbox-TTS-Server) project.

## Install Order

The order matters — later installs can override earlier ones if not careful.

| Step | Command                                                                                                              | Why this order                                                                                           |
| ---- | -------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| 1    | `python -m pip install --upgrade pip setuptools wheel`                                                               | Ensure latest build tools. **Must use `python -m pip`** on Windows (not `pip.exe`) to avoid file locking |
| 2    | `pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121 --index-url .../cu121`             | PyTorch must be installed first — xformers, flash_attn, and many deps depend on it                       |
| 3a   | `pip install xformers==0.0.28.post1 --index-url .../cu124`                                                           | cu124 index has Windows wheels for torch 2.4.1; cu121 only goes up to 0.0.27.post2                       |
| 3b   | `pip install flash_attn==2.7.4.post1` (from prebuilt wheel URL)                                                      | Prebuilt Windows wheel for Python 3.10 + torch 2.4.0 + CUDA 12.4 (kingbri1/flash-attention)              |
| 4a   | `pip install numpy==1.26.4`                                                                                          | **Pre-pin** before requirements.txt — prevents PyTorch from pulling numpy 2.x                            |
| 4b   | `pip install -r requirements.txt`                                                                                    | InfiniteTalk deps (excludes gradio, mlflow, dashscope)                                                   |
| 5a   | `pip install chatterbox-tts --no-deps`                                                                               | Gets Chatterbox source code without touching any deps                                                    |
| 5b   | `pip install librosa spacy-pkuseg==1.0.1 s3tokenizer conformer resemble-perth safetensors==0.5.3 pykakasi omegaconf` | Chatterbox's unique deps that don't conflict                                                             |

## Full Pinned Dependency List

### PyTorch Stack (from cu121 index)

| Package       | Version      | Source                                                          |
| ------------- | ------------ | --------------------------------------------------------------- |
| `torch`       | 2.4.1+cu121  | download.pytorch.org/whl/cu121                                  |
| `torchvision` | 0.19.1+cu121 | download.pytorch.org/whl/cu121                                  |
| `torchaudio`  | 2.4.1+cu121  | download.pytorch.org/whl/cu121                                  |
| `xformers`    | 0.0.28.post1 | download.pytorch.org/whl/cu124 (Windows wheels for torch 2.4.1) |
| `flash_attn`  | 2.7.4.post1  | Prebuilt Windows wheel (github.com/kingbri1/flash-attention)    |

### InfiniteTalk Dependencies (from PyPI)

| Package           | Version     | Notes                                                              |
| ----------------- | ----------- | ------------------------------------------------------------------ |
| `diffusers`       | >=0.31.0    | Flexible pin, use latest compatible                                |
| `transformers`    | >=4.49.0,<5 | transformers 5.x removed MT5Tokenizer, breaks diffusers            |
| `accelerate`      | >=1.1.1     | Flexible pin                                                       |
| `tokenizers`      | >=0.20.3    | Flexible pin                                                       |
| `numpy`           | 1.26.4      | Pre-pinned to avoid numpy 2.x                                      |
| `opencv-python`   | >=4.9.0.80  | Computer vision                                                    |
| `moviepy`         | 1.0.3       | Video editing                                                      |
| `imageio`         | latest      | Image I/O                                                          |
| `imageio-ffmpeg`  | latest      | FFmpeg binary bundled                                              |
| `scikit-image`    | latest      | Image processing                                                   |
| `loguru`          | latest      | Logging                                                            |
| `easydict`        | latest      | Config handling                                                    |
| `ftfy`            | latest      | Text processing                                                    |
| `pyloudnorm`      | latest      | Audio normalization                                                |
| `optimum-quanto`  | 0.2.6       | FP8 quantization for InfiniteTalk                                  |
| `xfuser`          | 0.4.1       | 0.4.2+ requires torch.distributed.tensor.experimental (torch 2.5+) |
| `scenedetect`     | latest      | Scene detection                                                    |
| `decord`          | latest      | Video decoding                                                     |
| `tqdm`            | latest      | Progress bars (used by HF)                                         |
| `einops`          | latest      | Tensor operations                                                  |
| `soundfile`       | latest      | Audio I/O                                                          |
| `huggingface_hub` | latest      | Model downloads                                                    |
| `rich`            | latest      | Progress bars                                                      |
| `librosa`         | latest      | Audio analysis (shared with Chatterbox)                            |

### Chatterbox Unique Dependencies (from PyPI)

| Package          | Version | Notes                                         |
| ---------------- | ------- | --------------------------------------------- |
| `librosa`        | latest  | Audio analysis                                |
| `spacy-pkuseg`   | 1.0.1   | Chinese segmentation (prebuilt Windows wheel) |
| `s3tokenizer`    | latest  | Tokenizer                                     |
| `conformer`      | latest  | Conformer model                               |
| `resemble-perth` | latest  | Audio processing                              |
| `safetensors`    | 0.5.3   | Model loading                                 |
| `pykakasi`       | latest  | Japanese transliteration                      |
| `omegaconf`      | latest  | Config management (required by Chatterbox)    |

### LightX2V LoRA Accelerator

| File                                                              | Source                    | Size   | Notes                                                      |
| ----------------------------------------------------------------- | ------------------------- | ------ | ---------------------------------------------------------- |
| `Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors` | `Kijai/WanVideo_comfy` HF | 317 MB | Distilled LoRA: 20 steps → 4 steps, `text_guide_scale=1.0` |

Downloaded to `models/infinitetalk/loras/`. Applied via custom FP8-aware `_apply_lora_to_model()` in `video_engine.py` (vendor code blocks LoRA+FP8; our code dequantizes → merges → requantizes).

### Vendor Patches (applied to `vendor/infinitetalk/`)

| File                            | Change                                                                                           |
| ------------------------------- | ------------------------------------------------------------------------------------------------ |
| `src/vram_management/layers.py` | Overflow forward methods: `copy.deepcopy().to(GPU)` → in-place `.to(GPU)` → compute → `.to(CPU)` |

### Removed Dependencies (no longer needed)

| Package                 | Why removed                                                   |
| ----------------------- | ------------------------------------------------------------- |
| `controlnet-aux`        | ControlNet preprocessors — not used by InfiniteTalk           |
| `open-clip-torch`       | OpenCLIP — not used by InfiniteTalk                           |
| `clip` (GitHub)         | OpenAI CLIP — not used by InfiniteTalk                        |
| `torchdiffeq`           | Differential equations — not used by InfiniteTalk             |
| `torchsde`              | Stochastic differential equations — not used by InfiniteTalk  |
| `torchmetrics`          | Metrics — not used by InfiniteTalk                            |
| `torchtyping`           | Type annotations — not used by InfiniteTalk                   |
| `mediapipe`             | Pose/face mesh — not used by InfiniteTalk                     |
| `onnxruntime-gpu`       | ONNX inference — not used by InfiniteTalk                     |
| `av` (PyAV)             | Video I/O — InfiniteTalk uses ffmpeg directly                 |
| ~~`omegaconf`~~         | ~~Config — InfiniteTalk uses easydict~~ (kept for Chatterbox) |
| `Pillow==10.2.0`        | Narrow pin no longer needed — InfiniteTalk has no Pillow pin  |
| `opencv-contrib-python` | Superset no longer needed — InfiniteTalk uses opencv-python   |

### Excluded Dependencies

| Package     | Why excluded                                                |
| ----------- | ----------------------------------------------------------- |
| `gradio`    | CLI-only interface, no web UI needed                        |
| `mlflow`    | Experiment tracking, not used for inference                 |
| `dashscope` | InfiniteTalk's TTS feature — we use Chatterbox TTS instead  |
| `kokoro`    | InfiniteTalk's built-in TTS — we use Chatterbox TTS instead |
