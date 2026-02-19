# Phase 4 — Pipeline & CLI

## Status: Complete

## What was built

### `pipeline.py` — Orchestration module

Wires TTS → Video into a single sequential pipeline with proper GPU memory management:

1. **Validate inputs** — script text, voice reference, portrait image (falls back to defaults)
2. **TTS generation** — loads Chatterbox Turbo, generates audio, saves WAV (optional MP3 copy)
3. **GPU cleanup** — fully unloads TTS model before video model loads
4. **Video generation** — loads InfiniteTalk, generates talking-head video from image + audio
5. **Final cleanup** — unloads video model, reports timings

Key design: models run **sequentially** — TTS is fully unloaded (0 GB VRAM) before InfiniteTalk loads. Peak VRAM never exceeds ~14 GB.

### `generate.py` — CLI entry point

Full `argparse`-based CLI:

| Flag                  | Description                                       |
| --------------------- | ------------------------------------------------- |
| `--audio-only`        | Generate audio only (skip video)                  |
| `--video-only AUDIO`  | Generate video from existing audio file           |
| `--animation TEXT`    | Animation description (text or path to .txt file) |
| `--verbose`           | Debug logging                                     |
| `--blocks-to-swap N`  | DiT blocks to swap CPU↔GPU (default: auto-tune)   |
| `--prefetch-blocks N` | Async prefetch N blocks ahead (default: 1)        |

### Utility modes

- **`--audio-only`** — Generate audio only (skip video)
- **`--video-only AUDIO_PATH`** — Generate video from existing audio (skip TTS)
- **`--verbose`** — Debug-level logging

### Default behavior

Running `python generate.py` with no arguments uses:

- `inputs/scripts/default.txt` for the script
- `inputs/voice_ref/default.mp3` (or `.wav`) for voice cloning
- `inputs/images/default.png` for the portrait image
- `inputs/animation/default.txt` for the animation description
- Auto-generated timestamped output paths in `outputs/audio/` and `outputs/video/`

## Files

| File          | Purpose                              |
| ------------- | ------------------------------------ |
| `pipeline.py` | Pipeline orchestration (TTS → Video) |
| `generate.py` | CLI entry point with argparse        |

## Verification

- `python generate.py --help` — exits 0, all options displayed correctly
- All CLI defaults sourced from `config.py`
- Input validation via `utils.py` (file existence, format checks)
- Error handling with user-friendly messages for missing files, bad formats, keyboard interrupt
