# Talking Head Video Generator — Future Enhancements

Known opportunities not yet implemented in this MVP, ordered by priority.

---

## 1. ~~Fork InfiniteTalk to Own Repository~~ ✅ Done

Completed. Fork at [garystorey/InfiniteTalk](https://github.com/garystorey/InfiniteTalk) contains all vendor patches. `install.py` now clones from the fork. Future changes: develop locally in `vendor/infinitetalk/`, then PR to the fork.

---

## 2. Multiline Default File Handling

**Priority: High — Correctness.**

Verify and harden that multiline `default.txt` files (scripts, animation prompts) are read and passed through the pipeline correctly.

**Current state:** `validate_script()` and `_resolve_animation()` both use `f.read().strip()` which reads entire files including newlines. The TTS chunker (`split_into_sentences`) splits on `.!?` boundaries regardless of line breaks, so basic multiline works today.

**Items to verify / fix:**

- Newlines within a paragraph are collapsed to spaces before TTS chunking (avoid mid-word breaks)
- Blank lines between paragraphs are preserved as natural pauses (or converted to `[pause:N]`)
- Very long files (multiple paragraphs, 1000+ words) chunk correctly without truncation
- Unicode characters (em dashes, curly quotes, accented characters) pass through cleanly — **Partially done (Phase 16):** `normalize_text()` handles smart quotes, em/en dashes, ellipsis, non-breaking spaces. All `"` stripped before TTS.
- Animation prompt multiline files are joined into a single string for the video engine
- Windows-style line endings (`\r\n`) are handled identically to Unix (`\n`)

---

## 3. Audio Quality Improvements

**Priority: High — Voice clone fidelity.**

The current TTS pipeline produces good voice cloning with Phase 16 fixes (sub-sentence splitting, quote stripping, tuned sampling). Further improvements possible:

| Improvement                       | Expected Impact                  | Status / Notes                                                                                                                                        |
| --------------------------------- | -------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Longer voice reference**        | Better timbre matching           | Current minimum is 5s. 15-30s of clean speech gives Chatterbox more speaker embedding signal.                                                         |
| **Reference audio preprocessing** | Cleaner speaker embedding        | Normalize to -23 LUFS, trim silence, remove background noise before passing to Chatterbox. Could use `noisereduce` or `demucs` for source separation. |
| **Multi-reference blending**      | More robust speaker embedding    | Average embeddings from multiple clips of the same speaker to reduce per-clip variance.                                                               |
| ~~**Temperature tuning**~~        | ~~Reduce artifacts~~             | ✅ **Done (Phase 16).** Swept 0.4–0.8. temp=0.4 → 132.8s looping; 0.5 → 38.6s looping; 0.6 = sweet spot. Also tuned top_k 50→40, top_p 0.85→0.8.      |
| **Chunk overlap increase**        | Smoother crossfades              | Current 50ms cosine crossfade works well. Could experiment with 100-200ms overlap for longer utterances.                                              |
| **Post-processing EQ matching**   | Tonal consistency with reference | Apply spectral envelope matching between generated audio and reference using `pyloudnorm` + FFT-based EQ to reduce tonal drift.                       |
| **Alternative TTS models**        | Better quality at cost of speed  | Evaluate newer voice cloning models (e.g. F5-TTS, XTTS-v2) if Chatterbox quality plateaus. Would need `--no-deps` approach for dependency isolation.  |

---

## 3. Web Interface for CLI Parameters

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
  - Verbose checkbox
- "Generate" button calls `pipeline.run()` or shells out to `generate.py`
- Real-time progress display (stream Rich output or poll a status file)
- Output preview: play audio, show video + contact sheet inline
- Runs on `localhost` only — no auth needed

**Notes:**

- Gradio was excluded from InfiniteTalk deps but could be added as an optional dependency
- Flask is lighter and avoids potential conflicts
- Keep it optional — CLI remains the primary interface

---

## 4. Performance Improvements

| Improvement          | Expected Impact                                       | Blocker                                         |
| -------------------- | ----------------------------------------------------- | ----------------------------------------------- |
| **3-step denoising** | 25% faster denoising                                  | Quality validation needed                       |
| **SageAttention**    | Faster attention computation                          | No Windows-compatible wheels for torch 2.4.x    |
| **Batched CFG**      | ~25-35% per-step speedup                              | Audio batch dim doubles as `human_num` in model |
| **FusioniX LoRA**    | Alternative 8-step LoRA (different quality tradeoffs) | Testing needed                                  |
