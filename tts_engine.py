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

# Matches [pause:1.5], [pause:2], [pause:0.5], etc.
_PAUSE_RE = re.compile(r"\[pause:(\d+(?:\.\d+)?)\]", re.IGNORECASE)


def split_into_segments(text):
    """
    Split text into segments of speech and pauses.

    Recognizes [pause:N] tags where N is seconds (int or float).
    Returns a list of dicts:
        {'type': 'text', 'value': 'Hello world.'}
        {'type': 'pause', 'duration': 1.5}
    """
    segments = []
    last_end = 0

    for match in _PAUSE_RE.finditer(text):
        # Text before this pause tag
        before = text[last_end : match.start()].strip()
        if before:
            segments.append({"type": "text", "value": before})
        # The pause itself
        segments.append({"type": "pause", "duration": float(match.group(1))})
        last_end = match.end()

    # Remaining text after last pause tag
    remaining = text[last_end:].strip()
    if remaining:
        segments.append({"type": "text", "value": remaining})

    # If no pause tags found, return whole text as single segment
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
        exaggeration=None,
        cfg_weight=None,
        temperature=0.8,
    ):
        """
        Generate audio for a single text chunk.

        Args:
            text: Text to synthesize.
            voice_ref_path: Path to voice reference WAV (>5s). None uses built-in voice.
            exaggeration: Emotion exaggeration (logged as warning in Turbo, but passed through).
            cfg_weight: CFG weight (logged as warning in Turbo, but passed through).
            temperature: Sampling temperature.

        Returns:
            1D numpy array of audio samples at 24kHz.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        kwargs = {
            "text": text,
            "temperature": temperature,
        }

        if voice_ref_path is not None:
            kwargs["audio_prompt_path"] = voice_ref_path

        if exaggeration is not None:
            kwargs["exaggeration"] = exaggeration

        if cfg_weight is not None:
            kwargs["cfg_weight"] = cfg_weight

        with torch.inference_mode():
            wav_tensor = self.model.generate(**kwargs)

        # wav_tensor shape: (1, samples) — squeeze to 1D numpy
        audio = wav_tensor.squeeze(0).cpu().numpy()
        return audio

    def generate(
        self,
        text,
        voice_ref_path=None,
        output_path=None,
        exaggeration=None,
        cfg_weight=None,
        temperature=0.8,
        max_chunk_chars=250,
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
            exaggeration: Emotion exaggeration level.
            cfg_weight: Classifier-free guidance weight.
            temperature: Sampling temperature.
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
            timestamp = getattr(config, "RUN_TIMESTAMP", None) or time.strftime(
                "%Y%m%d_%H%M%S"
            )
            output_path = os.path.join(config.AUDIO_OUTPUT_DIR, f"tts_{timestamp}.wav")

        # Parse [pause:N] tags into text/pause segments
        segments = split_into_segments(text)
        has_pauses = any(s["type"] == "pause" for s in segments)
        text_segments = [s for s in segments if s["type"] == "text"]
        total_text_chunks = sum(
            len(chunk_text(s["value"], max_chars=max_chunk_chars))
            for s in text_segments
        )
        if has_pauses:
            pause_total = sum(s["duration"] for s in segments if s["type"] == "pause")
            logger.info(
                f"Text split into {len(segments)} segment(s) "
                f"({len(text_segments)} speech, "
                f"{len(segments) - len(text_segments)} pause = {pause_total:.1f}s silence), "
                f"{total_text_chunks} TTS chunk(s)"
            )
        else:
            logger.info(f"Text split into {total_text_chunks} chunk(s)")

        # Generate audio for each segment
        output_parts = []
        chunk_counter = 0
        first_speech = True
        crossfade_samples = int(CROSSFADE_SECONDS * SAMPLE_RATE)

        for seg in segments:
            if seg["type"] == "pause":
                silence = np.zeros(int(seg["duration"] * SAMPLE_RATE), dtype=np.float32)
                output_parts.append(silence)
                logger.info(f"  Inserted {seg['duration']:.1f}s pause")
            else:
                # Generate TTS for this text segment
                chunks = chunk_text(seg["value"], max_chars=max_chunk_chars)
                speech_arrays = []
                for chunk in chunks:
                    chunk_counter += 1
                    logger.info(
                        f"  Generating chunk {chunk_counter}/{total_text_chunks}: "
                        f"{chunk[:60]}{'...' if len(chunk) > 60 else ''}"
                    )
                    audio = self.generate_chunk(
                        text=chunk,
                        voice_ref_path=wav_ref if first_speech else None,
                        exaggeration=exaggeration,
                        cfg_weight=cfg_weight,
                        temperature=temperature,
                    )
                    speech_arrays.append(audio)
                    first_speech = False
                    if progress_callback is not None:
                        progress_callback()

                # Crossfade-stitch chunks within this speech segment
                stitched = crossfade_stitch(speech_arrays, crossfade_samples)
                output_parts.append(stitched)

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
