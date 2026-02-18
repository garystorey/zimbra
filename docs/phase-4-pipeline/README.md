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

Full `argparse`-based CLI with grouped options:

| Group             | Key flags                                                    |
| ----------------- | ------------------------------------------------------------ |
| **Input sources** | `--script`, `--script-file`, `--voice-ref`, `--image`        |
| **Utility**       | `--download-only`, `--tts-only`, `--video-only`, `--verbose` |

### Utility modes

- **`--download-only`** — Pre-fetch all model weights without running generation
- **`--tts-only`** — Generate audio only (skip video)
- **`--video-only AUDIO_PATH`** — Generate video from existing audio (skip TTS)
- **`--verbose`** — Debug-level logging

### Default behavior

Running `python generate.py` with no arguments uses:

- `inputs/scripts/default.txt` for the script
- `inputs/voice_ref/default.mp3` (or `.wav`) for voice cloning
- `inputs/images/default.png` for the portrait image
- Auto-generated timestamped output paths in `outputs/audio/` and `outputs/video/`

## Files

| File          | Purpose                              |
| ------------- | ------------------------------------ |
| `pipeline.py` | Pipeline orchestration (TTS → Video) |
| `generate.py` | CLI entry point with argparse        |

## Verification

- `python generate.py --help` — exits 0, all options displayed correctly
- All CLI defaults sourced from `config.py`
- `--script` and `--script-file` are mutually exclusive
- Input validation via `utils.py` (file existence, format checks)
- Error handling with user-friendly messages for missing files, bad formats, keyboard interrupt
