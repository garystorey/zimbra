# Talking Head Video Generator — Future Enhancements

Known opportunities not yet implemented in this MVP, ordered by priority.

---

## 1. ~~Fork InfiniteTalk to Own Repository~~ ✅ Done

Completed. Fork at [garystorey/InfiniteTalk](https://github.com/garystorey/InfiniteTalk) contains all vendor patches. `install.py` now clones from the fork. Future changes: develop locally in `vendor/infinitetalk/`, then PR to the fork.

---

## 2. ~~Multiline Default File Handling~~ ✅ Done

Completed. Multiline `default.txt` files are read and handled correctly through the pipeline. `normalize_text()` handles smart quotes, em/en dashes, ellipsis, non-breaking spaces. Line-based TTS splitting preserves dialogue integrity. Windows-style line endings handled.

---

## 3. ~~Audio Quality Improvements~~ ✅ Done

Completed. TTS pipeline produces good voice cloning with sub-sentence splitting, quote stripping, tuned sampling (temp=0.6, top_k=40, top_p=0.8), silence trimming, seed control, and unicode normalization. Line-based splitting preserves dialogue integrity with 200ms inter-line gaps and 100ms sub-sentence gaps.

---

## 4. ~~Default Audio File for `--video-only` Mode~~ ✅ Done

Completed. `--video-only` now accepts an optional audio path. When omitted, resolves `inputs/scripts/default.wav` then `default.mp3` via `config.DEFAULT_AUDIO_FILE`. Clear error message if neither exists.

---

## 5. Animation Quality Overhaul (Next Task)

**Priority: High — Core output quality.**

Current output quality is noticeably below the ComfyUI reference in three visible ways:

- Overall facial/body motion looks too weak ("slow motion" / under-animated)
- Lip sync is less accurate
- During silence, the character can look puppet-like or unnaturally idle

The next task is to improve animation dynamics so the character appears alive and expressive across both speech and silence segments, while preserving stability and identity.

**Progress update (Phase 32 implemented):**

- Added a minimal speech-onset conditioning tweak (`VIDEO_AUDIO_SPEECH_ONSET_GAIN`) so the first voiced frame after silence gets a small extra embedding boost.
- Tuned default to `1.10` (from `1.12`) to reduce occasional overshoot while keeping faster mouth pickup after pauses.
- Status: fixed in config and active by default (set `1.0` to disable).

**Progress update (Phase 33 implemented):**

- Added a short post-speech release tail in `_boost_speech_embedding_energy()` so the first silent frames after speech keep a tiny decaying conditioning gain.
- New knobs: `VIDEO_AUDIO_SPEECH_RELEASE_FRAMES = 2` and `VIDEO_AUDIO_SPEECH_RELEASE_GAIN = 1.06`.
- Status: active by default; set `VIDEO_AUDIO_SPEECH_RELEASE_FRAMES = 0` (or `VIDEO_AUDIO_SPEECH_RELEASE_GAIN = 1.0`) to disable.

**Target outcomes:**

- Stronger, more natural overall motion (head, face, micro-expression timing)
- Tighter mouth-to-phoneme alignment during speech
- Natural idle behavior during pauses (subtle breathing/settling motion, no frozen puppet effect)
- Better parity with ComfyUI reference contact sheets

**Implementation focus areas:**

- Revisit audio-to-motion conditioning strength and temporal responsiveness
- Tune overlap/streaming settings that affect perceived motion continuity
- Evaluate motion-frame and guidance interactions with current solver settings
- Add an A/B evaluation pass against ComfyUI outputs using matching prompts/audio

**Done criteria:**

- Side-by-side contact sheet and video comparisons show clear qualitative improvement over current baseline
- No regression in color consistency, identity consistency, or runtime stability

---

## 6. Performance Improvements

| Improvement          | Expected Impact                                       | Blocker                                         |
| -------------------- | ----------------------------------------------------- | ----------------------------------------------- |
| **SageAttention**    | Faster attention computation                          | No Windows-compatible wheels for torch 2.4.x    |
| **Batched CFG**      | ~25-35% per-step speedup                              | Audio batch dim doubles as `human_num` in model |
| **FusioniX LoRA**    | Alternative 8-step LoRA (different quality tradeoffs) | Testing needed                                  |
| **4-step denoising** | 33% faster denoising (vs current 6-step default)      | Quality validation needed                       |

---

## 7. Web Interface for CLI Parameters

**Priority: Medium — Usability.**

Create a simple local web UI that exposes the CLI parameters in a browser form, making it easier for non-technical users to configure and run generations without memorizing flags.

**Approach:**

- Lightweight Python web framework (e.g. Gradio or Flask + vanilla HTML)
- Single page with form fields mapping to CLI params:
  - Script text (textarea)
  - Voice reference file (file picker)
  - Portrait image (file picker / preview)
  - Animation prompt (text input)
  - Audio guidance scale (slider, 1.0–2.0)
  - Audio-only / Video-only toggle
- "Generate" button calls `pipeline.run()` or shells out to `generate.py`
- Real-time progress display (stream Rich output or poll a status file)
- Output preview: play audio, show video + contact sheet inline
- Runs on `localhost` only — no auth needed

**Notes:**

- Gradio was excluded from InfiniteTalk deps but could be added as an optional dependency
- Flask is lighter and avoids potential conflicts
- Keep it optional — CLI remains the primary interface
