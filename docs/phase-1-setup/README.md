# Phase 1 — Project Setup & Environment

## Summary

Scaffolded the project directory structure, created the install pipeline, and implemented core infrastructure modules (`config.py`, `model_manager.py`, `utils.py`).

---

## Files Added

| File                           | Purpose                                                                                                                                                                      |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `.gitignore`                   | Excludes venv/, models/, outputs/, vendor/, **pycache**/, IDE files, OS files, etc.                                                                                          |
| `requirements.txt`             | Pinned dependency list for InfiniteTalk deps (excludes gradio, mlflow, dashscope, kokoro). Chatterbox deps listed as comments — installed separately via `--no-deps`.        |
| `install.bat`                  | Windows one-click installer. Calls `install.py`.                                                                                                                             |
| `install.py`                   | Cross-platform install script. Creates venv, installs deps in conflict-free order, clones InfiniteTalk, verifies imports.                                                    |
| `config.py`                    | Central configuration: project paths, default input file resolution, GPU detection, FFmpeg discovery (env var → PATH → imageio-ffmpeg fallback), model HuggingFace repo IDs. |
| `model_manager.py`             | Auto-download logic for all model weights. Uses `huggingface_hub.snapshot_download()` for InfiniteTalk/Wan2.1/wav2vec2. Atomic downloads with rich progress bars.            |
| `utils.py`                     | File validation (audio/image format checks), MP3↔WAV conversion (librosa/soundfile for MP3→WAV, FFmpeg for WAV→MP3), FFmpeg helpers (mux, resize), image dimension reading.  |
| `docs/plan.md`                 | Copy of the full implementation plan.                                                                                                                                        |
| `inputs/scripts/default.txt`   | Default script text for CLI fallback.                                                                                                                                        |
| `inputs/animation/default.txt` | Default animation description for video model.                                                                                                                               |
| `inputs/voice_ref/.gitkeep`    | Placeholder — user places their `default.wav` or `default.mp3` here.                                                                                                         |
| `inputs/images/.gitkeep`       | Placeholder — user places their `default.png` here.                                                                                                                          |

## Directories Created

| Directory              | Purpose                           | Tracked in git?                       |
| ---------------------- | --------------------------------- | ------------------------------------- |
| `inputs/voice_ref/`    | Voice clone reference audio files | Yes (via .gitkeep)                    |
| `inputs/images/`       | Portrait images                   | Yes (via .gitkeep)                    |
| `inputs/scripts/`      | Text scripts                      | Yes (default.txt tracked)             |
| `inputs/animation/`    | Animation descriptions            | Yes (default.txt tracked)             |
| `outputs/audio/`       | TTS-generated audio               | No (gitignored, created at runtime)   |
| `outputs/video/`       | Final talking-head videos         | No (gitignored, created at runtime)   |
| `models/infinitetalk/` | InfiniteTalk model weights        | No (gitignored, auto-downloaded)      |
| `models/chatterbox/`   | Chatterbox model weights          | No (gitignored, auto-downloaded)      |
| `vendor/infinitetalk/` | Cloned InfiniteTalk source        | No (gitignored, cloned by install.py) |
| `venv/`                | Dedicated virtual environment     | No (gitignored)                       |
| `docs/phase-1-setup/`  | This phase's documentation        | Yes                                   |

## Install Script Flow

```
install.py / install.bat
  │
  ├─ 1.  Create venv/ (auto-detects Python 3.10-3.12)
  ├─ 2.  Upgrade pip, setuptools, wheel (uses python -m pip)
  ├─ 3a. Install PyTorch 2.4.1+cu121 from cu121 index
  ├─ 3b. Install xformers 0.0.28.post1 from cu124 index
  ├─ 3c. Install flash_attn 2.7.4.post1 from prebuilt wheel
  ├─ 4a. Pre-pin numpy==1.26.4
  ├─ 4b. Install InfiniteTalk deps from requirements.txt
  ├─ 5a. Install chatterbox-tts --no-deps
  ├─ 5b. Install Chatterbox unique deps
  ├─ 6.  Clone InfiniteTalk into vendor/
  ├─ 7.  Create runtime directories (outputs/, models/)
  └─ 8.  Verify: torch CUDA, xformers, flash_attn, Chatterbox import, wan import
```

## Related Docs (project-wide)

- [`docs/compatibility.md`](../compatibility.md) — Windows 11 compatibility matrix, prerequisites, known gotchas
- [`docs/dependency-matrix.md`](../dependency-matrix.md) — pinned versions, conflict resolutions, install order rationale

## How to Use

```bash
# Windows (one-click):
install.bat

# Or manually:
python install.py

# After install, activate the venv:
.\venv\Scripts\activate    # Windows
source venv/bin/activate   # Linux/macOS

# Verify:
python -c "import torch; print(torch.cuda.is_available())"
```
