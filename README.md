# Talking Head Video Generator

Generate talking-head videos from a text script, voice reference, and portrait image — all locally on an NVIDIA GPU.

**Pipeline:** Text → Voice-Cloned Speech (Chatterbox Turbo) → Talking-Head Video (InfiniteTalk)

---

## Features

- **Voice cloning** — Clone any voice from a ~10s reference clip
- **Any art style** — Photorealistic, cartoon, 3D, illustrated portraits all work
- **Arbitrary aspect ratios** — Follows input image dimensions (not locked to square)
- **Streaming mode** — Generate unlimited-length video in overlapping chunks
- **Sequential GPU usage** — TTS fully unloads before video model loads; fits in 16 GB VRAM
- **One-click install** — `install.bat` handles venv creation, dependency resolution, and vendor cloning
- **Auto-download** — All model weights (~32 GB) download automatically on first run
- **CLI-driven** — Full `argparse` interface with sensible defaults

## Requirements

- **OS:** Windows 10/11 (Linux should work but untested)
- **GPU:** NVIDIA GPU with 16 GB+ VRAM (tested on RTX 4090 Laptop 16 GB)
- **Python:** 3.10.x (auto-detected by installer)
- **CUDA:** 12.1+ (bundled with PyTorch)
- **FFmpeg:** Required for audio/video muxing (auto-detected from PATH or `imageio-ffmpeg`)
- **HuggingFace account:** Required for Chatterbox model download (`huggingface-cli login`)

## Quick Start

```bash
# 1. Install (one-click on Windows)
install.bat

# 2. Activate the virtual environment
.\venv\Scripts\activate

# 3. Log in to HuggingFace (first time only)
huggingface-cli login

# 4. Place your inputs
#    - Portrait image       → inputs/images/default.png
#    - Voice reference      → inputs/voice_ref/default.mp3  (>5 seconds)
#    - Script text          → inputs/scripts/default.txt
#    - Animation description → inputs/animation/default.txt  (optional)

# 5. Generate! (models download automatically on first run)
python generate.py
```

## Usage

```bash
# Full pipeline (TTS + video)
python generate.py

# TTS only (skip video)
python generate.py --audio-only

# Video only from existing audio
python generate.py --video-only outputs/audio/tts_20250216.wav
```

### Options

| Flag                  | Description                                       |
| --------------------- | ------------------------------------------------- |
| `--audio-only`        | Generate audio only (skip video)                  |
| `--video-only AUDIO`  | Generate video from existing audio file           |
| `--animation TEXT`    | Animation description (text or path to .txt file) |
| `--verbose`           | Debug logging                                     |
| `--blocks-to-swap N`  | DiT blocks to swap CPU↔GPU (default: auto-tune)   |
| `--prefetch-blocks N` | Async prefetch N blocks ahead (default: 1)        |

Run `python generate.py --help` for full details.

## Project Structure

```
zimbra/
├── README.md                 # This file
├── requirements.txt          # Pinned dependencies
├── install.bat               # Windows one-click installer
├── install.py                # Cross-platform install script
├── config.py                 # Paths, defaults, GPU settings
├── generate.py               # CLI entry point
├── pipeline.py               # Orchestrates TTS → Video
├── tts_engine.py             # Chatterbox Turbo wrapper
├── video_engine.py           # InfiniteTalk wrapper
├── model_manager.py          # Auto-download logic
├── utils.py                  # File validation, FFmpeg helpers
├── inputs/
│   ├── images/default.png       # Default portrait image
│   ├── voice_ref/default.mp3    # Default voice reference
│   ├── scripts/default.txt      # Default script text
│   └── animation/default.txt    # Default animation description
├── outputs/                  # Generated files (gitignored)
│   ├── audio/                # TTS audio
│   └── video/                # Final videos
├── models/                   # Model weights (gitignored, ~32 GB)
├── vendor/                   # Cloned repos (gitignored)
│   └── infinitetalk/         # InfiniteTalk source
├── venv/                     # Virtual environment (gitignored)
└── docs/                     # Implementation documentation
```

## Technology Stack

| Component           | Library                                                       | Notes                                          |
| ------------------- | ------------------------------------------------------------- | ---------------------------------------------- |
| **TTS**             | [Chatterbox Turbo](https://github.com/resemble-ai/chatterbox) | 350M params, instant voice cloning, ~4 GB VRAM |
| **Video**           | [InfiniteTalk](https://github.com/MeiGen-AI/InfiniteTalk)     | Wan2.1-I2V-14B, audio-driven, any art style    |
| **CLI**             | Python `argparse`                                             | Grouped options, sensible defaults             |
| **Post-processing** | FFmpeg                                                        | Audio/video muxing at 25 FPS                   |

## How It Works

1. **TTS** — Chatterbox Turbo loads (~4 GB VRAM), generates voice-cloned audio from script text. Long text is chunked at sentence boundaries and crossfade-stitched.
2. **Unload** — TTS model fully unloaded, GPU memory freed to ~0 GB.
3. **Video** — InfiniteTalk loads with FP8 quantization (~18.9B params, 40 DiT blocks). VAE/CLIP/T5 offloaded to CPU. Block-level swapping keeps 23 blocks on GPU and swaps 17 blocks with async CUDA stream prefetch. Generates talking-head video driven by wav2vec2 audio embeddings.
4. **Mux** — FFmpeg combines video with audio into final MP4.

## Performance

On RTX 4090 Laptop (16 GB VRAM):

| Stage | Time          | Notes                                             |
| ----- | ------------- | ------------------------------------------------- |
| TTS   | ~5-18s        | Chatterbox Turbo, depends on script length        |
| Video | ~1.2 min/s    | 4 steps (LoRA), block swap + async prefetch       |
| Total | ~14 min / 10s | Including CLIP/VAE encode, decode, chunk overhead |

Video length always matches audio length. Video resolution matches input image aspect ratio.

### Performance Optimizations

1. **Block-level swapping with async prefetch** — Instead of per-layer CPU↔GPU transfers,
   entire DiT blocks are swapped. While block N computes on GPU, block N+1 prefetches from
   CPU via a dedicated CUDA stream. Transfer latency is hidden behind compute time.
   Auto-tuned: 23/40 blocks on GPU, 17 blocks swap (16 GB VRAM). VRAM headroom is carefully
   tuned to prevent Windows WDDM shared memory spill (~0.27s/block swap vs 0.25s/block persistent).

2. **FP8 quantization** — The DiT uses FP8 weights (~1 byte/param), reducing the 18.9B
   model from ~38 GB (fp16) to ~18.9 GB. Combined with CPU offloading, fits on 16 GB.

3. **LightX2V distilled LoRA** — Reduces denoising from 20 steps to 4 steps. With
   `text_guide_scale=1.0`, only 2 DiT forward passes per step (no text CFG).

4. **Smaller chunks** — `frame_num=41` (down from 81) reduces per-chunk activation VRAM.

5. **Strategic model offloading** — VAE, CLIP, and T5 are offloaded to CPU before the
   denoising loop. T5 runs on CPU (`t5_cpu=True`) since it only encodes the prompt once.

## Model Weights

All models auto-download on first run:

| Model                     | Source                                  | Size    |
| ------------------------- | --------------------------------------- | ------- |
| Chatterbox Turbo          | `ResembleAI/chatterbox-turbo`           | ~700 MB |
| Wan2.1-I2V-14B (VAE+CLIP) | `Wan-AI/Wan2.1-I2V-14B-480P`            | ~5.3 GB |
| chinese-wav2vec2-base     | `TencentGameMate/chinese-wav2vec2-base` | ~360 MB |
| InfiniteTalk FP8          | `MeiGen-AI/InfiniteTalk`                | ~26 GB  |
| LightX2V LoRA             | `Kijai/WanVideo_comfy`                  | ~317 MB |

## Troubleshooting

- **"No CUDA GPU detected"** — Ensure NVIDIA drivers are installed and `nvidia-smi` works.
- **"GPU out of memory"** — Close other GPU applications. Use `--video-size infinitetalk-480`. The 18.9B FP8 DiT needs ~15 GB VRAM (23 persistent blocks + swap buffer + activation headroom).
- **HuggingFace auth errors** — Run `huggingface-cli login` and accept the Chatterbox model license.
- **FFmpeg not found** — Install FFmpeg and add to PATH, or `pip install imageio-ffmpeg`.
- **Triton warning on Windows** — Expected and harmless (no Windows wheels available).

## License

This project integrates open-source models. See individual model repositories for their licenses:

- [Chatterbox](https://github.com/resemble-ai/chatterbox) — MIT
- [InfiniteTalk](https://github.com/MeiGen-AI/InfiniteTalk) — Apache 2.0
- [Wan2.1](https://github.com/Wan-Video/Wan2.1) — Apache 2.0
