"""
Talking Head Video Generator — Pipeline

Orchestrates the full TTS → Video generation pipeline.
Models run sequentially — TTS is fully unloaded before the video model loads
to stay within 16GB VRAM.
"""

import os
import time
import logging

import torch
from rich.console import Console
from rich.panel import Panel

import config
import utils
from tts_engine import TTSEngine
from video_engine import VideoEngine

logger = logging.getLogger(__name__)
console = Console()


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

    def run(self):
        """
        Run the full pipeline: TTS → Video.

        Reads all inputs from the inputs/ folder:
            inputs/scripts/default.txt    Script text
            inputs/images/default.png     Portrait image
            inputs/voice_ref/default.mp3  Voice reference audio

        Returns:
            Dict with keys: script_text, audio_path, video_path, timings
        """
        pipeline_start = time.time()
        timings = {}

        config.ensure_directories()

        # -----------------------------------------------------------------
        # Step 1: Validate inputs
        # -----------------------------------------------------------------
        console.print(Panel("[bold]Step 1/4: Validating inputs[/bold]"))

        # Script
        script_text = utils.validate_script()
        console.print(
            f"  Script: {len(script_text)} chars, " f"{len(script_text.split())} words"
        )

        # Voice reference
        voice_ref = config.DEFAULT_VOICE_REF
        if voice_ref is not None:
            utils.validate_audio_file(voice_ref)
            console.print(f"  Voice ref: {voice_ref}")
        else:
            console.print("  Voice ref: [dim]built-in default[/dim]")

        # Portrait image
        image = config.DEFAULT_IMAGE
        if image is None:
            raise FileNotFoundError(
                "No default image found. " f"Place a default.png in {config.IMAGES_DIR}"
            )
        utils.validate_image_file(image)
        console.print(f"  Image: {image}")

        # -----------------------------------------------------------------
        # Step 2: TTS — Generate audio
        # -----------------------------------------------------------------
        console.print(
            Panel("[bold]Step 2/4: Generating audio (Chatterbox Turbo)[/bold]")
        )

        tts_start = time.time()

        # Build audio output path
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        audio_output = os.path.join(config.AUDIO_OUTPUT_DIR, f"tts_{timestamp}.wav")

        try:
            self.tts_engine = TTSEngine()
            self.tts_engine.load_model()

            audio_path = self.tts_engine.generate(
                text=script_text,
                voice_ref_path=voice_ref,
                output_path=audio_output,
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
        console.print(f"  [green]Audio generated in {timings['tts']:.1f}s[/green]")

        # -----------------------------------------------------------------
        # Step 3: Unload TTS, free GPU
        # -----------------------------------------------------------------
        console.print(Panel("[bold]Step 3/4: Freeing GPU memory[/bold]"))

        self.tts_engine.unload_model()
        self.tts_engine = None
        console.print("  [green]TTS model unloaded[/green]")

        # -----------------------------------------------------------------
        # Step 4: Video — Generate talking-head video
        # -----------------------------------------------------------------
        console.print(Panel("[bold]Step 4/4: Generating video (InfiniteTalk)[/bold]"))

        video_start = time.time()

        try:
            self.video_engine = VideoEngine()
            self.video_engine.load_model()

            video_path = self.video_engine.generate(
                image_path=image,
                audio_path=audio_path,
            )
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
        console.print(f"  [green]Video generated in {timings['video']:.1f}s[/green]")

        # Unload video model
        self.video_engine.unload_model()
        self.video_engine = None

        # -----------------------------------------------------------------
        # Summary
        # -----------------------------------------------------------------
        timings["total"] = time.time() - pipeline_start

        def _fmt(secs):
            s = int(secs)
            h, rem = divmod(s, 3600)
            m, s = divmod(rem, 60)
            return f"{h}h {m}m {s}s" if h else (f"{m}m {s}s" if m else f"{s}s")

        console.print()
        console.print(
            Panel(
                f"[bold green]Pipeline complete![/bold green]\n\n"
                f"  Audio: {audio_path}\n"
                f"  Video: {video_path}\n\n"
                f"  TTS time:   {_fmt(timings['tts'])}\n"
                f"  Video time: {_fmt(timings['video'])}\n"
                f"  Total time: {_fmt(timings['total'])}",
                title="Results",
            )
        )

        return {
            "script_text": script_text,
            "audio_path": audio_path,
            "video_path": video_path,
            "timings": timings,
        }
