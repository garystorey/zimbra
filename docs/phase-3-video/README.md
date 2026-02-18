# Phase 3 — Video Engine (InfiniteTalk)

**Status:** ✅ Complete

---

## What Was Built

`video_engine.py` — A wrapper around InfiniteTalk's Wan2.1-I2V-14B pipeline that provides:

- **Model lifecycle**: `load_model()` / `unload_model()` with explicit GPU memory cleanup
- **Auto-download**: All model weights (~32 GB total) downloaded on first call via `model_manager.py`
- **Audio-driven animation**: wav2vec2 audio embeddings drive facial movements and gestures
- **Any art style**: Works with photorealistic, cartoon, 3D, and illustrated portraits
- **Arbitrary aspect ratios**: Follows input image dimensions (not locked to square)
- **Streaming mode**: Generates unlimited-length video in overlapping chunks
- **Low-VRAM support**: CPU offloading + FP8 quantization fits in 16 GB VRAM
- **Audio muxing**: Automatically combines generated video with source audio via FFmpeg
- **Auto-save**: Generates timestamped MP4 files in `outputs/video/`

## API

```python
from video_engine import VideoEngine

engine = VideoEngine()
engine.load_model()

# Generate with defaults
video_path = engine.generate(
    image_path="inputs/images/portrait.png",
    audio_path="outputs/audio/speech.wav",
)

# Generate with custom settings
video_path = engine.generate(
    image_path="inputs/images/portrait.png",
    audio_path="outputs/audio/speech.wav",
    size="infinitetalk-720",
    sample_steps=40,
    mode="streaming",
    seed=42,
    output_path="outputs/video/custom.mp4",
)

engine.unload_model()  # Frees GPU memory
```

## Key Parameters

| Parameter           | Default              | Notes                                                        |
| ------------------- | -------------------- | ------------------------------------------------------------ |
| `image_path`        | required             | Portrait image (any art style, any aspect ratio)             |
| `audio_path`        | required             | WAV audio file                                               |
| `output_path`       | auto                 | Auto-generated timestamp path if not specified               |
| `prompt`            | auto                 | Text prompt for video generation                             |
| `size`              | `"infinitetalk-480"` | Resolution bucket (`infinitetalk-480` or `infinitetalk-720`) |
| `sample_steps`      | `4` (LoRA) / `20`    | Denoising steps (4 with LoRA, 20 without)                    |
| `mode`              | `"streaming"`        | `"streaming"` for long video, `"clip"` for single chunk      |
| `motion_frame`      | `9`                  | Driven frame length for streaming mode                       |
| `seed`              | `42`                 | Random seed for reproducibility                              |
| `text_guide_scale`  | `1.0`                | Text CFG scale                                               |
| `audio_guide_scale` | `4.0`                | Audio CFG scale                                              |

## Model Weights

All weights auto-download into `models/infinitetalk/`:

| Source                                      | Size    | Contents                              |
| ------------------------------------------- | ------- | ------------------------------------- |
| `Wan-AI/Wan2.1-I2V-14B-480P`                | ~5.3 GB | VAE, CLIP encoder, tokenizers, config |
| `TencentGameMate/chinese-wav2vec2-base`     | ~360 MB | Audio feature extractor (wav2vec2)    |
| `MeiGen-AI/InfiniteTalk` (FP8 quant subset) | ~26 GB  | FP8 quantized DiT + T5 text encoder   |

Only the files needed for FP8 inference are downloaded — the full repo (~160+ GB) is not required.

## Discoveries During Implementation

1. **Art style flexibility** — InfiniteTalk handles all art styles (photorealistic, cartoon, 3D, illustrated), follows input image aspect ratio, and needs only image + audio.
2. **Strategic offloading** — The InfiniteTalk DiT is ~18.9B params (~38 GB fp16). With FP8 (~18.9 GB), offloading VAE/CLIP/T5 to CPU before auto-tune keeps ~14.5B params persistent on GPU; ~4.4B overflow stream per-layer via in-place CPU↔GPU transfers. T5 runs on CPU (`t5_cpu=True`) since it only encodes the prompt once.
3. **FP8 quantization** — Reduces model size significantly; FP8 checkpoint replaces the 7 diffusion shards (~66 GB) and full T5 (~11.4 GB).
4. **wav2vec2 model.safetensors from PR branch** — The main branch of `chinese-wav2vec2-base` lacks `model.safetensors`; it must be fetched from `refs/pr/1`.
5. **Audio loudness normalization** — Input audio is normalized to -23 LUFS before embedding extraction for consistent results.
6. **25 FPS output** — InfiniteTalk generates at 25 FPS.
7. **Streaming mode** — Generates video in overlapping chunks (`frame_num` frames each, `motion_frame` overlap), enabling unlimited-length output.

## Files

| File                           | Purpose                                                        |
| ------------------------------ | -------------------------------------------------------------- |
| `video_engine.py`              | VideoEngine class, audio embedding, generation                 |
| `config.py`                    | Updated with InfiniteTalk settings and weight paths            |
| `model_manager.py`             | Updated with selective download for InfiniteTalk model weights |
| `docs/phase-3-video/README.md` | This document                                                  |
