# Phase 2 — TTS Engine (Chatterbox Turbo)

**Status:** ✅ Complete (9/9 tests passed)

---

## What Was Built

`tts_engine.py` — A wrapper around Chatterbox Turbo that provides:

- **Model lifecycle**: `load_model()` / `unload_model()` with explicit GPU memory cleanup
- **Voice cloning**: Pass a reference WAV (>5s) to clone any voice
- **Built-in voice**: Works without a voice reference using bundled `conds.pt`
- **Sentence-boundary chunking**: Long text is split at `.!?` boundaries, not mid-sentence
- **Crossfade stitching**: 50ms cosine crossfade between chunks eliminates audible seams
- **Auto-save**: Generates timestamped WAV files in `outputs/audio/`

## API

```python
from tts_engine import TTSEngine

engine = TTSEngine()
engine.load_model()

# Generate with built-in voice
audio_path = engine.generate(text="Hello world.")

# Generate with voice cloning
audio_path = engine.generate(
    text="Hello world.",
    voice_ref_path="inputs/voice_ref/reference.wav",
    output_path="outputs/audio/custom.wav",
)

engine.unload_model()  # Frees ~4 GB GPU memory
```

## Key Parameters

| Parameter         | Default | Notes                                                  |
| ----------------- | ------- | ------------------------------------------------------ |
| `voice_ref_path`  | `None`  | Path to reference WAV/MP3 (>5s). None = built-in voice |
| `output_path`     | auto    | Auto-generated timestamp path if not specified         |
| `exaggeration`    | `None`  | Logged as warning in Turbo (not supported)             |
| `cfg_weight`      | `None`  | Logged as warning in Turbo (not supported)             |
| `temperature`     | `0.8`   | Sampling temperature                                   |
| `max_chunk_chars` | `250`   | Max characters per chunk for long text                 |

## Discoveries During Implementation

1. **HuggingFace auth required** — `ChatterboxTurboTTS.from_pretrained()` passes `token=True`, requiring `huggingface-cli login` before first use
2. **Repo ID is `ResembleAI/chatterbox-turbo`** — not `ResembleAI/chatterbox` (that's the non-Turbo model). Fixed in `config.py`
3. **Output sample rate is 24000 Hz** — hardcoded in Chatterbox as `S3GEN_SR`
4. **Turbo ignores CFG/exaggeration** — these params are accepted but logged as warnings; they only work in the full (non-Turbo) model
5. **Voice ref must be >5 seconds** — `prepare_conditionals()` asserts this
6. **GPU memory: ~4.16 GB** — freed to 0.01 GB after `unload_model()`

## Test Results

```
TEST 1: Import tts_engine module              ✓ PASSED
TEST 2: Sentence splitting                    ✓ PASSED
TEST 3: Text chunking                         ✓ PASSED
TEST 4: Crossfade stitching                   ✓ PASSED
TEST 5: TTSEngine instantiation               ✓ PASSED
TEST 6: Model loading                         ✓ PASSED
TEST 7: Generate audio (built-in voice)       ✓ PASSED
TEST 8: Generate audio (long text, 3 chunks)  ✓ PASSED
TEST 9: GPU memory cleanup                    ✓ PASSED

RESULTS: 9/9 passed, 0 failed
```

## Files

| File                         | Purpose                              |
| ---------------------------- | ------------------------------------ |
| `tts_engine.py`              | TTSEngine class, chunking, crossfade |
| `docs/phase-2-tts/README.md` | This document                        |
