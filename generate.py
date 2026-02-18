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
"""

import os

# Triton is unavailable on Windows — disable torch.compile globally before
# any torch imports so @torch.compile decorators in vendor code become no-ops.
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import argparse
import logging
import sys

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
            "  inputs/voice_ref/default.mp3  Voice reference audio"
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
        "--verbose",
        action="store_true",
        help="Enable verbose/debug logging",
    )

    # Block swap options
    parser.add_argument(
        "--blocks-to-swap",
        type=int,
        default=None,
        metavar="N",
        help="Number of DiT blocks to swap CPU↔GPU (default: auto-tune)",
    )
    parser.add_argument(
        "--prefetch-blocks",
        type=int,
        default=None,
        metavar="N",
        help=f"Async prefetch N blocks ahead (default: {config.VIDEO_PREFETCH_BLOCKS})",
    )

    return parser


def setup_logging(verbose=False):
    """Configure logging for the application."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Suppress noisy third-party loggers
    for name in ("urllib3", "filelock", "huggingface_hub"):
        logging.getLogger(name).setLevel(logging.WARNING)


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


def main():
    parser = build_parser()
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)

    # Apply CLI overrides to config before any engine loads
    if args.blocks_to_swap is not None:
        config.VIDEO_BLOCKS_TO_SWAP = args.blocks_to_swap
    if args.prefetch_blocks is not None:
        config.VIDEO_PREFETCH_BLOCKS = args.prefetch_blocks

    from rich.console import Console
    from pipeline import Pipeline

    console = Console()

    # Check GPU availability
    gpu_ok, gpu_msg = _check_gpu()
    if not gpu_ok:
        console.print(f"[red]ERROR: {gpu_msg}[/red]")
        return 1

    # --video-only: skip TTS, go straight to video
    if args.video_only:
        import utils

        audio_path = args.video_only
        utils.validate_audio_file(audio_path)

        image = config.DEFAULT_IMAGE
        if image is None:
            console.print(
                "[red]ERROR: No default image found. Place a default.png in inputs/images/[/red]"
            )
            return 1
        utils.validate_image_file(image)

        console.print(f"[bold]Video-only mode[/bold]: {audio_path}")
        from video_engine import VideoEngine
        import time as _time

        engine = VideoEngine()
        try:
            _vo_start = _time.time()
            engine.load_model()

            video_path = engine.generate(
                image_path=image,
                audio_path=audio_path,
            )
            _vo_elapsed = _time.time() - _vo_start
            _vo_secs = int(_vo_elapsed)
            _h, _rem = divmod(_vo_secs, 3600)
            _m, _s = divmod(_rem, 60)
            _hms = f"{_h}h {_m}m {_s}s" if _h else (f"{_m}m {_s}s" if _m else f"{_s}s")
            console.print(f"\n[bold green]Video saved:[/bold green] {video_path}")
            console.print(f"  Total time: {_hms} ({_vo_elapsed:.1f}s)")
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

        script_text = utils.validate_script()

        voice_ref = config.DEFAULT_VOICE_REF

        console.print(f"[bold]Audio-only mode[/bold]: {len(script_text)} chars")
        from tts_engine import TTSEngine

        engine = TTSEngine()
        try:
            engine.load_model()
            audio_path = engine.generate(
                text=script_text,
                voice_ref_path=voice_ref,
            )

            console.print(f"\n[bold green]Audio saved:[/bold green] {audio_path}")
            return 0
        except Exception as e:
            logging.getLogger(__name__).exception("TTS generation failed")
            console.print(f"\n[red]ERROR: {e}[/red]")
            return 1
        finally:
            engine.unload_model()

    # Full pipeline
    pipeline = Pipeline()

    try:
        result = pipeline.run()
        return 0

    except FileNotFoundError as e:
        console.print(f"\n[red]ERROR: {e}[/red]")
        return 1
    except ValueError as e:
        console.print(f"\n[red]ERROR: {e}[/red]")
        return 1
    except RuntimeError as e:
        console.print(f"\n[red]ERROR: {e}[/red]")
        return 1
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/yellow]")
        pipeline._cleanup()
        return 130
    except Exception as e:
        logging.getLogger(__name__).exception("Pipeline failed")
        console.print(f"\n[red]FATAL: {e}[/red]")
        return 1


if __name__ == "__main__":
    sys.exit(main())
