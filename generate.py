"""
Talking Head Video Generator — CLI Entry Point

Usage:
    python generate.py              Full pipeline (TTS + video)
    python generate.py --audio-only  Generate audio only
    python generate.py --video-only path/to/audio.wav   Video from existing audio
    python generate.py --verbose    Debug logging

Inputs are read from the inputs/ folder:
    inputs/scripts/default.txt      Script text
    inputs/images/default.png       Portrait image
    inputs/voice_ref/default.mp3    Voice reference audio
    inputs/animation/default.txt    Animation description
"""

import os
import sys
import traceback

# Triton is unavailable on Windows — disable torch.compile globally before
# any torch imports so @torch.compile decorators in vendor code become no-ops.
os.environ["TORCHDYNAMO_DISABLE"] = "1"

# Suppress noisy triton traceback from xformers on Windows (no triton wheels).
os.environ["XFORMERS_FORCE_DISABLE_TRITON"] = "1"

import argparse
import logging

# Suppress torio FFmpeg extension warnings (soundfile backend is used instead).
logging.getLogger("torio").setLevel(logging.WARNING)

import config


def build_parser():
    """Build the argparse argument parser."""
    parser = argparse.ArgumentParser(
        prog="generate",
        description=(
            "Talking Head Video Generator — Generate a talking-head video "
            "from a text script, voice reference, and portrait image.\n\n"
            "Inputs are read from the inputs/ folder:\n"
            "  inputs/scripts/default.txt    Script text\n"
            "  inputs/images/default.png     Portrait image\n"
            "  inputs/voice_ref/default.mp3  Voice reference audio\n"
            "  inputs/animation/default.txt  Animation description"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--audio-only",
        action="store_true",
        help="Generate audio only (skip video)",
    )
    parser.add_argument(
        "--video-only",
        type=str,
        default=None,
        metavar="AUDIO_PATH",
        help="Generate video from existing audio file (skip TTS)",
    )
    parser.add_argument(
        "--animation",
        type=str,
        default=None,
        metavar="TEXT_OR_FILE",
        help=(
            "Animation description for the video model. "
            "Can be a text string or path to a .txt file. "
            "Defaults to inputs/animation/default.txt"
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose/debug logging",
    )

    return parser


def setup_logging(verbose=False):
    """Configure logging for the application.

    A timestamped log file is ALWAYS created in outputs/logs/ with full
    verbose/debug logging, regardless of the --verbose flag.  This allows
    post-run diagnosis by reading the log file.

    When verbose=False (default), console output is suppressed so the CLI
    shows only the title and progress bars.  When verbose=True, debug
    output also appears on the console.

    Returns:
        str: Absolute path to the log file for this run.
    """
    import time as _time

    # Ensure logs directory exists (early, before config.ensure_directories)
    os.makedirs(config.LOGS_DIR, exist_ok=True)

    timestamp = _time.strftime("%Y%m%d_%H%M%S")
    config.RUN_TIMESTAMP = timestamp
    log_file = os.path.join(config.LOGS_DIR, f"run_{timestamp}.log")

    # --- File handler: always verbose ---
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )

    root = logging.getLogger()
    # Clear any existing handlers (e.g. from basicConfig)
    root.handlers.clear()
    root.addHandler(file_handler)

    # Our project loggers always get DEBUG (written to file).
    for name in (
        "pipeline",
        "video_engine",
        "tts_engine",
        "model_manager",
        "gguf_loader",
        "config",
        "utils",
        "__main__",
    ):
        logging.getLogger(name).setLevel(logging.DEBUG)

    # Suppress noisy third-party loggers that flood the log file.
    for name in ("numba", "PIL", "torio"):
        logging.getLogger(name).setLevel(logging.WARNING)

    if verbose:
        # Also add a console handler so debug output appears on screen.
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        root.addHandler(console_handler)
        # Root at DEBUG so vendor [diag] messages reach the file handler.
        # Console handler is at DEBUG too with --verbose so user sees everything.
        root.setLevel(logging.DEBUG)
    else:
        # Root at DEBUG so everything goes to the file, but nothing to console.
        root.setLevel(logging.DEBUG)

        # Redirect vendor print() / tqdm to devnull
        import tqdm as _tqdm_mod

        _devnull = open(os.devnull, "w")
        _tqdm_mod.tqdm.__init_original__ = _tqdm_mod.tqdm.__init__

        _orig_init = _tqdm_mod.tqdm.__init__

        def _silent_tqdm_init(self, *a, **kw):
            kw["disable"] = True
            _orig_init(self, *a, **kw)

        _tqdm_mod.tqdm.__init__ = _silent_tqdm_init

        # Suppress Python warnings (FutureWarning, DeprecationWarning, etc.)
        import warnings

        warnings.filterwarnings("ignore")

        # Capture stray print()/stderr from vendor code (e.g. xformers triton warning).
        # Store real stderr FIRST so rich Console progress bars still work.
        config._REAL_STDERR = sys.stderr
        sys.stdout = _devnull
        sys.stderr = _devnull

    return log_file


def _check_gpu():
    """Verify CUDA GPU is available. Returns True if OK, prints error if not."""
    try:
        import torch

        if not torch.cuda.is_available():
            return (
                False,
                "No CUDA GPU detected. This tool requires an NVIDIA GPU with CUDA support.",
            )
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if vram_gb < 8:
            return False, (
                f"GPU has only {vram_gb:.1f} GB VRAM. "
                "Minimum 16 GB recommended (8 GB absolute minimum for TTS-only)."
            )
        return True, None
    except ImportError:
        return False, "PyTorch not installed. Run install.bat first."


def _resolve_animation(animation_arg):
    """Resolve animation description from CLI arg or default file.

    Args:
        animation_arg: None (use default file), a path to a .txt file, or inline text.

    Returns:
        str: The animation description text.
    """
    if animation_arg is None:
        # Fall back to default file
        path = config.DEFAULT_ANIMATION_FILE
        if os.path.isfile(path):
            return open(path, "r", encoding="utf-8").read().strip()
        # Hardcoded fallback if file is missing
        return "A person is talking naturally with expressive facial movements and gestures."

    # If it looks like a file path and exists, read it
    if os.path.isfile(animation_arg):
        return open(animation_arg, "r", encoding="utf-8").read().strip()

    # Otherwise treat as inline text
    return animation_arg.strip()


def main():
    parser = build_parser()
    args = parser.parse_args()

    config._VERBOSE = args.verbose
    log_file = setup_logging(verbose=args.verbose)

    from rich.console import Console
    from pipeline import Pipeline

    # Always write user-facing output to stderr (stdout may be silenced)
    _stderr = getattr(config, "_REAL_STDERR", sys.stderr)
    console = Console(file=_stderr)
    console.print(f"[dim]Log file: {log_file}[/dim]")

    # Check GPU availability
    gpu_ok, gpu_msg = _check_gpu()
    if not gpu_ok:
        console.print(f"[red]ERROR: {gpu_msg}[/red]")
        return 1

    # Resolve animation prompt
    animation_prompt = _resolve_animation(args.animation)

    # --video-only: skip TTS, go straight to video
    if args.video_only:
        import utils
        import time as _time
        from pipeline import _make_progress, _fmt, _estimate_video_eta
        from video_engine import VideoEngine

        audio_path = args.video_only
        utils.validate_audio_file(audio_path)

        image = config.DEFAULT_IMAGE
        if image is None:
            console.print(
                "[red]ERROR: No default image found. Place a default.png in inputs/images/[/red]"
            )
            return 1
        utils.validate_image_file(image)

        console.print()
        console.print("[bold]Video-only mode[/bold]")
        console.print()

        engine = VideoEngine()
        try:
            eta_secs = None

            if args.verbose:
                engine.load_model()
                _vo_start = _time.time()
                audio_duration = engine.get_audio_duration(audio_path)
                total_steps = engine.estimate_total_steps(audio_duration)
                video_path = engine.generate(
                    image_path=image,
                    audio_path=audio_path,
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

                    engine.load_model(progress_callback=_load_progress)
                    progress.update(
                        load_task,
                        completed=6,
                        total=6,
                        description="Loading video model",
                    )
                    _vo_start = _time.time()

                    audio_duration = engine.get_audio_duration(audio_path)
                    total_steps = engine.estimate_total_steps(audio_duration)

                    eta_secs = _estimate_video_eta(audio_duration)
                    console.print(
                        f"[dim]Video ETA: ~{_fmt(eta_secs)} "
                        f"({total_steps} steps)[/dim]"
                    )

                    gen_task = progress.add_task("Generating video", total=total_steps)

                    video_path = engine.generate(
                        image_path=image,
                        audio_path=audio_path,
                        animation_prompt=animation_prompt,
                        step_callback=lambda: progress.advance(gen_task),
                    )
                    progress.update(gen_task, completed=total_steps)

            _vo_elapsed = _time.time() - _vo_start
            if not args.verbose and eta_secs:
                logging.getLogger(__name__).info(
                    f"ETA accuracy — Video: estimated {eta_secs:.0f}s, "
                    f"actual {_vo_elapsed:.1f}s, "
                    f"diff {_vo_elapsed - eta_secs:+.1f}s"
                )
            console.print()
            console.print(f"[bold green]Done![/bold green]  {_fmt(_vo_elapsed)}")
            console.print(f"  Video: {video_path}")
            console.print()
            return 0
        except Exception as e:
            logging.getLogger(__name__).exception("Video generation failed")
            console.print(f"\n[red]ERROR: {e}[/red]")
            return 1
        finally:
            engine.unload_model()

    # --audio-only: generate audio only
    if args.audio_only:
        import utils
        import time as _time
        from pipeline import _make_progress, _fmt
        from tts_engine import TTSEngine, chunk_text

        script_text = utils.validate_script()
        voice_ref = config.DEFAULT_VOICE_REF
        chunks = chunk_text(script_text)
        est_secs = 30 + len(chunks) * 8

        console.print()
        console.print("[bold]Audio-only mode[/bold]")
        console.print(
            f"[dim]Estimated time: ~{_fmt(est_secs)} ({len(chunks)} chunk{'s' if len(chunks) != 1 else ''})[/dim]"
        )
        console.print()

        engine = TTSEngine()
        try:
            _ao_start = _time.time()

            if args.verbose:
                engine.load_model()
                audio_path = engine.generate(
                    text=script_text,
                    voice_ref_path=voice_ref,
                )
            else:
                with _make_progress(console) as progress:
                    load_task = progress.add_task("Loading TTS model", total=None)
                    engine.load_model()
                    progress.update(load_task, total=1, completed=1)

                    gen_task = progress.add_task("Generating audio", total=len(chunks))

                    audio_path = engine.generate(
                        text=script_text,
                        voice_ref_path=voice_ref,
                        progress_callback=lambda: progress.advance(gen_task),
                    )

            _ao_elapsed = _time.time() - _ao_start
            logging.getLogger(__name__).info(
                f"ETA accuracy — TTS: estimated {est_secs:.0f}s, "
                f"actual {_ao_elapsed:.1f}s, "
                f"diff {_ao_elapsed - est_secs:+.1f}s"
            )
            console.print()
            console.print(f"[bold green]Done![/bold green]  {_fmt(_ao_elapsed)}")
            console.print(f"  Audio: {audio_path}")
            console.print()
            return 0
        except Exception as e:
            traceback.print_exc()
            console.print(f"\n[red]ERROR: {e}[/red]")
            return 1
        finally:
            engine.unload_model()

    # Full pipeline
    pipeline = Pipeline()

    try:
        result = pipeline.run(animation_prompt=animation_prompt, verbose=args.verbose)
        return 0

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/yellow]")
        pipeline._cleanup()
        return 130
    except Exception as e:
        traceback.print_exc()
        console.print(f"\n[red]FATAL: {e}[/red]")
        return 1


if __name__ == "__main__":
    sys.exit(main())
