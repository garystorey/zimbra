"""
Talking Head Video Generator — TTS Engine (Chatterbox Turbo)

Wraps Chatterbox Turbo for voice-cloned text-to-speech generation.
Handles model loading, text chunking at sentence boundaries, audio
stitching with crossfade, and GPU memory cleanup.
"""

import gc
import os
import re
import time
import logging

import numpy as np
import torch
import soundfile as sf

import config
import utils

logger = logging.getLogger(__name__)

# Chatterbox Turbo output sample rate (24 kHz)
SAMPLE_RATE = 24000

# Crossfade duration in seconds for stitching chunks
CROSSFADE_SECONDS = 0.05  # 50ms overlap


# ---------------------------------------------------------------------------
# Pause tag parsing
# ---------------------------------------------------------------------------

# Matches [s:1.5], [s:2], [s:0.5], [s:.5], etc.
_PAUSE_RE = re.compile(r"\[s:(\d+(?:\.\d+)?|\.\d+)\]", re.IGNORECASE)

# Matches ${filename.mp3}, ${filename.wav}, etc.
_AUDIO_REF_RE = re.compile(r"\$\{([^}]+)\}")

# Unicode → ASCII replacements that confuse TTS tokenizers
_UNICODE_REPLACEMENTS = {
    "\u2018": "'",  # left single quote
    "\u2019": "'",  # right single quote (smart apostrophe)
    "\u201c": '"',  # left double quote
    "\u201d": '"',  # right double quote
    "\u201a": "'",  # single low-9 quote
    "\u201e": '"',  # double low-9 quote
    "\u2013": "-",  # en dash
    "\u2014": "-",  # em dash
    "\u2026": "...",  # horizontal ellipsis
    "\u00a0": " ",  # non-breaking space
}


def normalize_text(text):
    """Replace Unicode smart quotes/dashes with ASCII equivalents for TTS."""
    for unicode_char, ascii_char in _UNICODE_REPLACEMENTS.items():
        text = text.replace(unicode_char, ascii_char)
    # Collapse multiple spaces (but preserve newlines for line-based splitting)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def _split_script_to_segments(text):
    """Split script text into TTS segments preserving line-based structure.

    Each line in the script is one natural sentence/thought. [s:N] tags
    insert explicit silence. Lines without silence tags get a controlled
    inter-sentence gap instead.

    Returns list of dicts:
        {'type': 'text', 'value': 'sentence text'}
        {'type': 'pause', 'duration': 1.5}
    """
    text = normalize_text(text)
    segments = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Use inline-aware splitting to handle multiple [s:N] tags
        # and preserve their position within the line
        line_segments = split_into_segments(line)
        segments.extend(line_segments)
    return segments


def trim_silence(audio, sr=24000, threshold_db=-42, tail_ms=200, lead_ms=100):
    """Trim leading and trailing silence from audio array.

    Removes excess silence/artifacts that Chatterbox pads onto chunks,
    especially important for catching hallucinated outputs that contain
    real speech buried in long silence.
    """
    if len(audio) == 0:
        return audio
    # Convert threshold from dB to linear amplitude
    threshold = 10 ** (threshold_db / 20.0)
    abs_audio = np.abs(audio)
    above = np.where(abs_audio > threshold)[0]
    if len(above) == 0:
        return audio  # all silence, return as-is
    first_loud = above[0]
    last_loud = above[-1]
    lead_samples = int(lead_ms / 1000.0 * sr)
    tail_samples = int(tail_ms / 1000.0 * sr)
    start = max(first_loud - lead_samples, 0)
    end = min(last_loud + tail_samples, len(audio))
    return audio[start:end]


def split_into_segments(text):
    """
    Split text into segments of speech, pauses, and audio file references.

    Recognizes [s:N] tags where N is seconds (int or float).
    Recognizes ${filename} tags referencing audio files in inputs/scripts/.
    Returns a list of dicts:
        {'type': 'text', 'value': 'Hello world.'}
        {'type': 'pause', 'duration': 1.5}
        {'type': 'audio', 'path': '/absolute/path/to/file.mp3'}
    """
    # Combine both patterns into one pass so ordering is preserved
    _COMBINED_RE = re.compile(
        r"(?:" + _PAUSE_RE.pattern + r")" r"|(?:" + _AUDIO_REF_RE.pattern + r")",
        re.IGNORECASE,
    )
    segments = []
    last_end = 0

    for match in _COMBINED_RE.finditer(text):
        # Text before this tag
        before = text[last_end : match.start()].strip()
        if before:
            segments.append({"type": "text", "value": before})

        if match.group(1) is not None:
            # Pause tag — group(1) is the duration from [s:N]
            segments.append({"type": "pause", "duration": float(match.group(1))})
        else:
            # Audio file reference — group(2) is the filename from ${filename}
            filename = match.group(2).strip()
            audio_path = os.path.join(config.SCRIPTS_DIR, filename)
            if not os.path.isfile(audio_path):
                raise FileNotFoundError(
                    f"Audio file referenced in script not found: {audio_path}"
                )
            segments.append({"type": "audio", "path": audio_path})

        last_end = match.end()

    # Remaining text after last tag
    remaining = text[last_end:].strip()
    if remaining:
        segments.append({"type": "text", "value": remaining})

    # If no tags found, return whole text as single segment
    if not segments:
        segments.append({"type": "text", "value": text.strip()})

    return segments


# ---------------------------------------------------------------------------
# Sentence-boundary chunking
# ---------------------------------------------------------------------------


def split_into_sentences(text):
    """
    Split text into sentences at natural boundaries.

    Splits on sentence-ending punctuation (.!?) followed by whitespace,
    while preserving abbreviations and decimal numbers.
    Returns a list of non-empty stripped sentences.
    """
    # Split on sentence-ending punctuation followed by whitespace or end-of-string
    # Negative lookbehind avoids splitting on common abbreviations (Mr. Dr. etc.)
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in parts if s.strip()]
    return sentences


def chunk_text(text, max_chars=250):
    """
    Chunk text into segments suitable for TTS generation.

    Splits at sentence boundaries first, then merges short sentences
    into chunks up to max_chars. This avoids mid-sentence breaks
    that cause audible seams.

    Args:
        text: Input text to chunk.
        max_chars: Maximum characters per chunk.

    Returns:
        List of text chunks.
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return [text] if text.strip() else []

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # If adding this sentence would exceed the limit, finalize current chunk
        if current_chunk and len(current_chunk) + len(sentence) + 1 > max_chars:
            chunks.append(current_chunk)
            current_chunk = sentence
        else:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def _split_for_fallback(text):
    """Split text at ; or . boundaries for fallback TTS when whole-line fails."""
    # Try semicolons first
    if ";" in text:
        parts = [p.strip() for p in text.split(";") if p.strip()]
        if len(parts) > 1:
            return parts
    # Try sentence boundaries (. followed by space and uppercase)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip())
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) > 1:
        return parts
    # Try comma before clause
    if ", " in text:
        parts = [p.strip() for p in text.split(", ", 1) if p.strip()]
        if len(parts) > 1:
            return parts
    return [text]


def _load_audio_file(path, target_sr=SAMPLE_RATE):
    """Load an audio file and resample to the target sample rate.

    Supports WAV, MP3, and other formats via soundfile/librosa.
    Returns a 1D float32 numpy array at target_sr.
    """
    try:
        data, sr = sf.read(path, dtype="float32")
    except Exception:
        import librosa

        data, sr = librosa.load(path, sr=None, mono=True)

    # Convert stereo to mono
    if data.ndim > 1:
        data = data.mean(axis=1)

    # Resample if needed
    if sr != target_sr:
        import librosa

        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)

    return data.astype(np.float32)


def count_tts_chunks(text):
    """Count the number of actionable chunks that generate() will produce.

    Counts text segments (TTS calls) and audio file references.
    Pauses are not counted as they are instantaneous.
    """
    segments = _split_script_to_segments(text)
    return sum(1 for s in segments if s["type"] in ("text", "audio"))


# ---------------------------------------------------------------------------
# Audio crossfade stitching
# ---------------------------------------------------------------------------


def crossfade_stitch(audio_arrays, crossfade_samples):
    """
    Stitch multiple audio arrays with a cosine crossfade.

    Args:
        audio_arrays: List of 1D numpy arrays (mono audio).
        crossfade_samples: Number of samples to crossfade between chunks.

    Returns:
        Single stitched 1D numpy array.
    """
    if not audio_arrays:
        return np.array([], dtype=np.float32)
    if len(audio_arrays) == 1:
        return audio_arrays[0]

    # Build crossfade window
    fade_out = np.linspace(1.0, 0.0, crossfade_samples, dtype=np.float32)
    fade_in = np.linspace(0.0, 1.0, crossfade_samples, dtype=np.float32)

    result = audio_arrays[0]

    for i in range(1, len(audio_arrays)):
        next_audio = audio_arrays[i]

        # If either segment is too short for crossfade, just concatenate
        if len(result) < crossfade_samples or len(next_audio) < crossfade_samples:
            result = np.concatenate([result, next_audio])
            continue

        # Apply crossfade
        overlap_region = (
            result[-crossfade_samples:] * fade_out
            + next_audio[:crossfade_samples] * fade_in
        )

        result = np.concatenate(
            [
                result[:-crossfade_samples],
                overlap_region,
                next_audio[crossfade_samples:],
            ]
        )

    return result


# ---------------------------------------------------------------------------
# TTS Engine class
# ---------------------------------------------------------------------------


class TTSEngine:
    """
    Chatterbox Turbo TTS engine.

    Usage:
        engine = TTSEngine()
        engine.load_model()
        audio_path = engine.generate(
            text="Hello world.",
            voice_ref_path="path/to/reference.wav",
            output_path="output.wav",
        )
        engine.unload_model()
    """

    def __init__(self, device=None):
        self.device = device or config.DEVICE
        self.model = None
        self._loaded = False

    @property
    def is_loaded(self):
        return self._loaded and self.model is not None

    def load_model(self):
        """Load Chatterbox Turbo model. Downloads weights on first call."""
        if self.is_loaded:
            logger.info("Chatterbox Turbo already loaded")
            return

        logger.info("Loading Chatterbox Turbo model...")
        start = time.time()

        # Monkey-patch perth watermarker if native binary unavailable
        import perth

        if perth.PerthImplicitWatermarker is None:
            perth.PerthImplicitWatermarker = perth.DummyWatermarker
            logger.info("perth native watermarker unavailable, using DummyWatermarker")

        from chatterbox.tts_turbo import ChatterboxTurboTTS

        self.model = ChatterboxTurboTTS.from_pretrained(device=self.device)

        elapsed = time.time() - start
        logger.info(f"Chatterbox Turbo loaded in {elapsed:.1f}s (device={self.device})")
        self._loaded = True

    def unload_model(self):
        """Unload model and free GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        self._loaded = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("Chatterbox Turbo unloaded, GPU memory freed")

    def generate_chunk(
        self,
        text,
        voice_ref_path=None,
        temperature=None,
        top_k=None,
        top_p=None,
        repetition_penalty=None,
        seed=None,
        _is_fallback=False,
    ):
        """
        Generate audio for a single text chunk.

        Args:
            text: Text to synthesize.
            voice_ref_path: Path to voice reference WAV (>5s). None uses built-in voice.
            temperature: Sampling temperature (lower = tighter voice match).
            top_k: Top-K sampling (lower = fewer candidates, tighter match).
            top_p: Nucleus sampling threshold (lower = narrower, tighter match).
            repetition_penalty: Token repetition penalty.
            seed: Random seed for reproducible generation.

        Returns:
            1D numpy array of audio samples at 24kHz.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        MAX_RETRIES = 3
        word_count = len(text.split())
        # Only catch severe truncation (<0.18s/word) and hallucination (>1.0s/word)
        min_duration = max(word_count * 0.18, 0.5)
        max_duration = max(word_count * 1.0, 3.0)

        kwargs = {
            "text": text,
            "temperature": (
                temperature if temperature is not None else config.TTS_TEMPERATURE
            ),
            "top_k": top_k if top_k is not None else config.TTS_TOP_K,
            "top_p": top_p if top_p is not None else config.TTS_TOP_P,
            "repetition_penalty": (
                repetition_penalty
                if repetition_penalty is not None
                else config.TTS_REPETITION_PENALTY
            ),
        }

        if voice_ref_path is not None:
            kwargs["audio_prompt_path"] = voice_ref_path

        best_audio = None
        best_diff = float("inf")

        for attempt in range(MAX_RETRIES + 1):
            actual_seed = (seed + attempt) if seed is not None else None
            if actual_seed is not None:
                torch.manual_seed(actual_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(actual_seed)

            with torch.inference_mode():
                wav_tensor = self.model.generate(**kwargs)

            # wav_tensor shape: (1, samples) — squeeze to 1D numpy
            audio = wav_tensor.squeeze(0).cpu().numpy()
            audio = trim_silence(audio, sr=SAMPLE_RATE)
            duration = len(audio) / SAMPLE_RATE

            # Track best attempt (closest to expected duration)
            expected_mid = word_count * 0.4  # ~2.5 words/sec
            diff = abs(duration - expected_mid)
            if diff < best_diff:
                best_audio = audio
                best_diff = diff

            in_range = min_duration <= duration <= max_duration
            if in_range or attempt == MAX_RETRIES:
                if not in_range and not _is_fallback:
                    # All retries failed — try splitting at ; or . boundaries
                    parts = _split_for_fallback(text)
                    if len(parts) > 1:
                        logger.warning(
                            f"    All attempts failed — splitting into {len(parts)} parts"
                        )
                        gap = np.zeros(
                            int(0.1 * SAMPLE_RATE), dtype=np.float32
                        )  # 100ms gap
                        audio_parts = []
                        for pi, part in enumerate(parts):
                            part_audio = self.generate_chunk(
                                text=part,
                                voice_ref_path=voice_ref_path,
                                temperature=temperature,
                                top_k=top_k,
                                top_p=top_p,
                                repetition_penalty=repetition_penalty,
                                seed=(seed + pi * 10) if seed is not None else None,
                                _is_fallback=True,
                            )
                            audio_parts.append(part_audio)
                            if pi < len(parts) - 1:
                                audio_parts.append(gap)
                        audio = np.concatenate(audio_parts)
                        duration = len(audio) / SAMPLE_RATE
                        logger.info(f"    Fallback split produced {duration:.1f}s")
                        return audio
                    # Can't split further — use best attempt
                    audio = best_audio
                    duration = len(audio) / SAMPLE_RATE
                    logger.warning(
                        f"    All attempts out of range "
                        f"[{min_duration:.1f}s–{max_duration:.1f}s] for {word_count} words, "
                        f"using best ({duration:.1f}s)"
                    )
                elif not in_range:
                    audio = best_audio
                    duration = len(audio) / SAMPLE_RATE
                elif attempt > 0:
                    logger.info(
                        f"    Retry {attempt} OK: {duration:.1f}s "
                        f"(range [{min_duration:.1f}s–{max_duration:.1f}s])"
                    )
                logger.debug(f"    Chunk audio: {duration:.1f}s for {word_count} words")
                return audio

            reason = (
                "too short" if duration < min_duration else "too long (hallucination)"
            )
            logger.warning(
                f"    Audio {reason} ({duration:.1f}s for {word_count} words, "
                f"range [{min_duration:.1f}s–{max_duration:.1f}s]) "
                f"— retrying with seed+{attempt + 1}"
            )

    def generate(
        self,
        text,
        voice_ref_path=None,
        output_path=None,
        temperature=None,
        top_k=None,
        top_p=None,
        repetition_penalty=None,
        max_chunk_chars=500,
        progress_callback=None,
    ):
        """
        Generate voice-cloned audio from text.

        Handles long text by chunking at sentence boundaries and
        crossfade-stitching the results.

        Args:
            text: Text to synthesize.
            voice_ref_path: Path to voice reference audio (>5s WAV/MP3).
                            None uses the built-in default voice.
            output_path: Where to save the output WAV. Auto-generated if None.
            temperature: Sampling temperature (lower = tighter voice match).
            top_k: Top-K sampling (lower = fewer candidates, tighter match).
            top_p: Nucleus sampling threshold (lower = narrower, tighter match).
            repetition_penalty: Token repetition penalty.
            max_chunk_chars: Maximum characters per chunk for long text.

        Returns:
            Path to the generated WAV file.
        """
        if not self.is_loaded:
            self.load_model()

        # Ensure output directory exists
        config.ensure_directories()

        # Convert voice ref to WAV if needed
        wav_ref = None
        if voice_ref_path is not None:
            wav_ref = utils.ensure_wav_format(voice_ref_path)

        # Auto-generate output path if not provided
        if output_path is None:
            run_filename = getattr(config, "RUN_FILENAME", None)
            if run_filename is None:
                timestamp = getattr(config, "RUN_TIMESTAMP", None) or time.strftime(
                    "%Y%m%d_%H%M%S"
                )
                run_filename = f"run_{timestamp}"
            output_path = os.path.join(config.OUTPUT_DIR, f"{run_filename}.wav")

        # Split script into segments using line-based boundaries.
        # Each line = one TTS call. [s:N] tags become explicit silence.
        # ${filename} tags splice in pre-recorded audio files.
        segments = _split_script_to_segments(text)
        has_pauses = any(s["type"] == "pause" for s in segments)
        has_audio_refs = any(s["type"] == "audio" for s in segments)
        text_segments = [s for s in segments if s["type"] == "text"]
        audio_segments = [s for s in segments if s["type"] == "audio"]
        total_text_chunks = len(text_segments)
        if has_pauses or has_audio_refs:
            parts = [f"{total_text_chunks} speech"]
            if has_pauses:
                pause_total = sum(
                    s["duration"] for s in segments if s["type"] == "pause"
                )
                parts.append(
                    f"{sum(1 for s in segments if s['type'] == 'pause')} pause = {pause_total:.1f}s silence"
                )
            if has_audio_refs:
                parts.append(f"{len(audio_segments)} audio file(s)")
            logger.info(
                f"Text split into {len(segments)} segment(s) ({', '.join(parts)})"
            )
        else:
            logger.info(
                f"Text: {total_text_chunks} line(s), no explicit pauses "
                f"(200ms gap between lines)"
            )

        # Generate audio — one TTS call per line.
        # Lines are joined with a short controlled gap (SENTENCE_GAP_MS)
        # instead of relying on Chatterbox's unpredictable inter-sentence pauses.
        SENTENCE_GAP_MS = 200  # ms of silence between lines
        sentence_gap = np.zeros(
            int(SENTENCE_GAP_MS / 1000.0 * SAMPLE_RATE), dtype=np.float32
        )
        output_parts = []
        chunk_counter = 0

        for seg_idx, seg in enumerate(segments):
            if seg["type"] == "pause":
                silence = np.zeros(int(seg["duration"] * SAMPLE_RATE), dtype=np.float32)
                output_parts.append(silence)
                logger.info(f"  Inserted {seg['duration']:.1f}s pause")
            elif seg["type"] == "audio":
                audio_data = _load_audio_file(seg["path"])
                duration = len(audio_data) / SAMPLE_RATE
                output_parts.append(audio_data)
                logger.info(
                    f"  Inserted audio file: {os.path.basename(seg['path'])} "
                    f"({duration:.1f}s)"
                )
                # Insert controlled gap after audio file (same as text segments)
                if seg_idx < len(segments) - 1:
                    next_seg = segments[seg_idx + 1]
                    if next_seg["type"] not in ("pause", "audio"):
                        output_parts.append(sentence_gap)
                if progress_callback is not None:
                    progress_callback()
            else:
                line_text = seg["value"]
                chunk_counter += 1
                logger.info(
                    f"  Generating chunk {chunk_counter}/{total_text_chunks}: "
                    f"{line_text[:80]}{'...' if len(line_text) > 80 else ''}"
                )
                chunk_seed = config.VIDEO_SEED + chunk_counter
                audio = self.generate_chunk(
                    text=line_text,
                    voice_ref_path=wav_ref,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    seed=chunk_seed,
                )
                output_parts.append(audio)
                # Insert controlled gap between lines
                # (not after last segment, not before an explicit pause)
                if seg_idx < len(segments) - 1:
                    next_seg = segments[seg_idx + 1]
                    if next_seg["type"] != "pause":
                        output_parts.append(sentence_gap)
                # Advance progress once per LINE (not per sub-sentence)
                if progress_callback is not None:
                    progress_callback()

        # Concatenate all parts (speech segments + silence)
        final_audio = (
            np.concatenate(output_parts)
            if output_parts
            else np.array([], dtype=np.float32)
        )

        # Pad with silence: 0.25s front, 0.5s end
        front_pad = np.zeros(int(0.25 * SAMPLE_RATE), dtype=np.float32)
        end_pad = np.zeros(int(0.5 * SAMPLE_RATE), dtype=np.float32)
        final_audio = np.concatenate([front_pad, final_audio, end_pad])

        # Save output
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, final_audio, SAMPLE_RATE)
        logger.info(
            f"Audio saved: {output_path} "
            f"({len(final_audio) / SAMPLE_RATE:.1f}s, {SAMPLE_RATE}Hz)"
        )

        return output_path
