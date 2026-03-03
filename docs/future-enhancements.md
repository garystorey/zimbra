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

## 5. Web Interface for CLI Parameters

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

---

## 6. Performance Improvements

| Improvement          | Expected Impact                                       | Blocker                                         |
| -------------------- | ----------------------------------------------------- | ----------------------------------------------- |
| **4-step denoising** | 33% faster denoising (vs current 6-step default)      | Quality validation needed                       |
| **SageAttention**    | Faster attention computation                          | No Windows-compatible wheels for torch 2.4.x    |
| **Batched CFG**      | ~25-35% per-step speedup                              | Audio batch dim doubles as `human_num` in model |
| **FusioniX LoRA**    | Alternative 8-step LoRA (different quality tradeoffs) | Testing needed                                  |
