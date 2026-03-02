"""
Talking Head Video Generator — Pipeline

Orchestrates the full TTS → Video generation pipeline.
Models run sequentially — TTS is fully unloaded before the video model loads
to stay within 16GB VRAM.
"""

import math
import os
import sys
import time
import logging

import torch
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
    TaskProgressColumn,
)

import config
import utils
import model_manager
from tts_engine import TTSEngine, count_tts_chunks
from video_engine import VideoEngine

logger = logging.getLogger(__name__)


def _get_console():
    """Get a Console that writes to the real stderr (survives stdout redirect)."""
    _stderr = getattr(config, "_REAL_STDERR", sys.stderr)
    return Console(file=_stderr)


def _fmt(secs):
    s = int(secs)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m}m {s}s" if h else (f"{m}m {s}s" if m else f"{s}s")


def _make_progress(console):
    """Create a rich Progress bar with a consistent style."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
        redirect_stdout=False,
        redirect_stderr=False,
    )


def _estimate_video_eta(audio_duration):
    """Estimate video generation time in seconds from audio duration."""
    frame_num = config.VIDEO_GGUF_FRAME_NUM
    motion_frame = config.VIDEO_GGUF_MOTION_FRAME
    if config.VIDEO_USE_LORA:
        sample_steps = config.VIDEO_GGUF_LORA_SAMPLE_STEPS
        text_cfg = config.VIDEO_LORA_TEXT_GUIDE_SCALE
        audio_cfg = config.VIDEO_GGUF_LORA_AUDIO_GUIDE_SCALE
    else:
        sample_steps = config.VIDEO_SAMPLE_STEPS
        text_cfg = config.VIDEO_TEXT_GUIDE_SCALE
        audio_cfg = config.VIDEO_AUDIO_GUIDE_SCALE
    audio_frames = int(audio_duration * config.VIDEO_FPS) + 1
    effective_max = max(audio_frames, frame_num)
    if config.VIDEO_MODE == "streaming" and effective_max > frame_num:
        num_chunks = 1 + math.ceil(
            (effective_max - frame_num) / (frame_num - motion_frame)
        )
    else:
        num_chunks = 1
    total_steps = num_chunks * sample_steps
    # VIDEO_SECS_PER_STEP is calibrated for 1 pass/step; scale by passes
    skip_cfg = math.isclose(text_cfg, 1.0) and math.isclose(audio_cfg, 1.0)
    passes = 1 if skip_cfg else (2 if math.isclose(text_cfg, 1.0) else 3)
    base_secs = config.VIDEO_SECS_PER_STEP
    denoise_secs = total_steps * base_secs * passes
    vae_secs = (
        num_chunks * config.VIDEO_VAE_DECODE_SECS + config.VIDEO_FIRST_CHUNK_EXTRA_SECS
    )
    return denoise_secs + vae_secs


def _estimate_audio_duration_from_script(script_text):
    """Rough estimate: ~2.5 words/sec typical speech rate."""
    words = len(script_text.split())
    return max(words / 2.5, 1.0)


class Pipeline:
    """
    End-to-end talking-head video generation pipeline.

    Steps:
        1. Validate inputs (script, voice ref, portrait image)
        2. Run TTS to generate audio from script + voice reference
        3. Unload TTS model, free GPU memory
        4. Run video engine to generate talking-head video from image + audio
        5. Unload video model, free GPU memory
        6. Return path to final video

    Usage:
        pipeline = Pipeline()
        result = pipeline.run(
            script="Hello world.",
            voice_ref="voice.wav",
            image="portrait.png",
        )
        print(result["video_path"])
    """

    def __init__(self):
        self.tts_engine = None
        self.video_engine = None

    def _cleanup(self):
        """Ensure all models are unloaded and GPU memory is freed."""
        if self.tts_engine is not None:
            try:
                self.tts_engine.unload_model()
            except Exception:
                pass
            self.tts_engine = None
        if self.video_engine is not None:
            try:
                self.video_engine.unload_model()
            except Exception:
                pass
            self.video_engine = None

    def run(self, animation_prompt=None, verbose=False, script_text=None):
        """
        Run the full pipeline: TTS → Video.

        Reads all inputs from the inputs/ folder:
            inputs/scripts/default.txt    Script text
            inputs/images/default.png     Portrait image
            inputs/voice_ref/default.mp3  Voice reference audio

        Args:
            animation_prompt: Text describing the animation style. If None,
                the video engine uses its default prompt.

        Returns:
            Dict with keys: script_text, audio_path, video_path, timings
        """
        console = _get_console()
        pipeline_start = time.time()
        timings = {}

        config.ensure_directories()

        # -----------------------------------------------------------------
        # Title
        # -----------------------------------------------------------------
        console.print()
        console.print("[bold]Talking Head Video Generator[/bold]")
        console.print()

        # -----------------------------------------------------------------
        # Pre-flight: download any missing model files
        # -----------------------------------------------------------------
        if not verbose:
            model_manager.preflight_downloads(console=console)
        else:
            model_manager.preflight_downloads()

        # -----------------------------------------------------------------
        # Validate inputs
        # -----------------------------------------------------------------
        script_text = utils.validate_script(text=script_text)

        # -----------------------------------------------------------------
        # ETA estimate (before any model loading)
        # -----------------------------------------------------------------
        est_audio_dur = _estimate_audio_duration_from_script(script_text)
        est_video_secs = _estimate_video_eta(est_audio_dur)
        est_tts_secs = 30 + count_tts_chunks(script_text) * 8  # ~30s load + ~8s/chunk
        est_video_load_secs = 60
        est_total = (
            est_tts_secs + est_video_load_secs + est_video_secs
        )  # +60s for model load
        console.print(
            f"[dim]Estimated total time: ~{_fmt(est_total)} "
            f"(TTS ~{_fmt(est_tts_secs)}, "
            f"video load ~1m, "
            f"video gen ~{_fmt(est_video_secs)})[/dim]"
        )
        console.print()

        voice_ref = config.DEFAULT_VOICE_REF
        if voice_ref is not None:
            utils.validate_audio_file(voice_ref)

        image = config.DEFAULT_IMAGE
        if image is None:
            raise FileNotFoundError(
                "No default image found. " f"Place a default.png in {config.IMAGES_DIR}"
            )
        utils.validate_image_file(image)

        # -----------------------------------------------------------------
        # TTS — Generate audio
        # -----------------------------------------------------------------
        tts_start = time.time()
        run_filename = getattr(config, "RUN_FILENAME", None)
        if run_filename is None:
            timestamp = getattr(config, "RUN_TIMESTAMP", None) or time.strftime(
                "%Y%m%d_%H%M%S"
            )
            run_filename = f"run_{timestamp}"
        audio_output = os.path.join(config.OUTPUT_DIR, f"{run_filename}.wav")
        video_output = os.path.join(config.OUTPUT_DIR, f"{run_filename}.mp4")

        try:
            self.tts_engine = TTSEngine()
            num_chunks = count_tts_chunks(script_text)

            if verbose:
                self.tts_engine.load_model()
                audio_path = self.tts_engine.generate(
                    text=script_text,
                    voice_ref_path=voice_ref,
                    output_path=audio_output,
                )
            else:
                with _make_progress(console) as progress:
                    load_task = progress.add_task("Loading TTS model", total=None)
                    self.tts_engine.load_model()
                    progress.update(load_task, total=1, completed=1)

                    gen_task = progress.add_task("Generating audio", total=num_chunks)

                    audio_path = self.tts_engine.generate(
                        text=script_text,
                        voice_ref_path=voice_ref,
                        output_path=audio_output,
                        progress_callback=lambda: progress.advance(gen_task),
                    )

        except torch.cuda.OutOfMemoryError:
            self._cleanup()
            raise RuntimeError(
                "GPU out of memory during TTS generation. "
                "Close other GPU applications and try again."
            )
        except Exception:
            self._cleanup()
            raise

        timings["tts"] = time.time() - tts_start
        logger.info(
            f"ETA accuracy — TTS: estimated {est_tts_secs:.0f}s, "
            f"actual {timings['tts']:.1f}s, "
            f"diff {timings['tts'] - est_tts_secs:+.1f}s"
        )

        # Unload TTS
        self.tts_engine.unload_model()
        self.tts_engine = None

        # -----------------------------------------------------------------
        # Video — Generate talking-head video
        # -----------------------------------------------------------------
        video_start = time.time()

        try:
            self.video_engine = VideoEngine()

            if verbose:
                self.video_engine.load_model()
                audio_duration = self.video_engine.get_audio_duration(audio_path)
                total_steps = self.video_engine.estimate_total_steps(audio_duration)
                video_path = self.video_engine.generate(
                    image_path=image,
                    audio_path=audio_path,
                    output_path=video_output,
                    animation_prompt=animation_prompt,
                )
            else:
                with _make_progress(console) as progress:
                    load_task = progress.add_task("Loading video model", total=6)

                    def _load_progress(step, total, desc):
                        progress.update(
                            load_task,
                            completed=step,
                            total=total,
                        )

                    self.video_engine.load_model(progress_callback=_load_progress)
                    progress.update(
                        load_task,
                        completed=6,
                        total=6,
                        description="Loading video model",
                    )

                    audio_duration = self.video_engine.get_audio_duration(audio_path)
                    total_steps = self.video_engine.estimate_total_steps(audio_duration)

                    # Refined ETA now that we know actual audio duration
                    refined_video_eta = _estimate_video_eta(audio_duration)
                    est_video_secs = refined_video_eta  # update for final comparison
                    logger.info(
                        f"Video ETA (refined): ~{_fmt(refined_video_eta)} "
                        f"({total_steps} steps × ~{int(refined_video_eta / max(total_steps, 1))}s/step)"
                    )

                    gen_task = progress.add_task("Generating video", total=total_steps)

                    video_path = self.video_engine.generate(
                        image_path=image,
                        audio_path=audio_path,
                        output_path=video_output,
                        animation_prompt=animation_prompt,
                        step_callback=lambda: progress.advance(gen_task),
                    )

                    # Ensure bar completes even if estimate was slightly off
                    progress.update(gen_task, completed=total_steps)

        except torch.cuda.OutOfMemoryError:
            self._cleanup()
            raise RuntimeError(
                "GPU out of memory during video generation. "
                "Try --video-size infinitetalk-480 or close other GPU applications."
            )
        except Exception:
            self._cleanup()
            raise

        timings["video"] = time.time() - video_start
        logger.info(
            f"ETA accuracy — Video (gen+load): estimated {est_video_load_secs + est_video_secs:.0f}s, "
            f"actual {timings['video']:.1f}s, "
            f"diff {timings['video'] - (est_video_load_secs + est_video_secs):+.1f}s"
        )

        # Unload video model
        self.video_engine.unload_model()
        self.video_engine = None

        # -----------------------------------------------------------------
        # Contact sheet (for quality review)
        # -----------------------------------------------------------------
        contact_sheet = utils.generate_contact_sheet(video_path)

        # -----------------------------------------------------------------
        # Summary
        # -----------------------------------------------------------------
        timings["total"] = time.time() - pipeline_start
        logger.info(
            f"ETA accuracy — Total: estimated {est_total:.0f}s, "
            f"actual {timings['total']:.1f}s, "
            f"diff {timings['total'] - est_total:+.1f}s"
        )

        console.print()
        console.print(f"[bold green]Done![/bold green]  {_fmt(timings['total'])}")
        console.print(f"  Audio: {audio_path}")
        console.print(f"  Video: {video_path}")
        if contact_sheet:
            console.print(f"  Contact sheet: {contact_sheet}")
        console.print()

        return {
            "script_text": script_text,
            "audio_path": audio_path,
            "video_path": video_path,
            "contact_sheet": contact_sheet,
            "timings": timings,
        }
