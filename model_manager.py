"""
Talking Head Video Generator — Model Manager

Handles automatic downloading of all model weights on first run.
Downloads are atomic (temp file → rename) to avoid corrupt partial files.
"""

import os
import sys
import shutil
import hashlib
import tempfile
import urllib.request

from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    DownloadColumn,
    TransferSpeedColumn,
    TimeRemainingColumn,
)
from rich.console import Console

import config


_console = None


def _get_console():
    """Get a Console that writes to the real stderr (survives stdout redirect)."""
    global _console
    if _console is None:
        import sys

        _stderr = getattr(config, "_REAL_STDERR", sys.stderr)
        _console = Console(file=_stderr)
    return _console


def _is_verbose():
    """True when --verbose is active."""
    return getattr(config, "_VERBOSE", False)


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def _download_file(url, dest_path, desc=None):
    """Download a file with a rich progress bar. Atomic: writes to temp then renames."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    if os.path.isfile(dest_path):
        return dest_path

    if desc is None:
        desc = os.path.basename(dest_path)

    _get_console().print(f"  Downloading: [cyan]{desc}[/cyan]")

    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(dest_path))
    os.close(fd)

    try:
        req = urllib.request.urlopen(url)
        total_size = int(req.headers.get("Content-Length", 0))

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            console=_get_console(),
        ) as progress:
            task = progress.add_task(desc, total=total_size if total_size > 0 else None)

            with open(tmp_path, "wb") as f:
                while True:
                    chunk = req.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
                    progress.update(task, advance=len(chunk))

        # Atomic rename
        shutil.move(tmp_path, dest_path)
        return dest_path

    except Exception:
        # Clean up temp file on failure
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def _hf_snapshot_download(
    repo_id,
    local_dir,
    desc=None,
    allow_patterns=None,
    ignore_patterns=None,
    marker_files=None,
):
    """
    Download a HuggingFace repo snapshot to a local directory.

    Args:
        repo_id: HuggingFace repo ID.
        local_dir: Local directory to download into.
        desc: Human-readable description for progress output.
        allow_patterns: List of glob patterns — only matching files are downloaded.
        ignore_patterns: List of glob patterns — matching files are skipped.
        marker_files: List of file paths (relative to local_dir) that must exist
                      for the download to be considered complete. If None, falls
                      back to checking if the directory is non-empty.
    """
    # Check if already downloaded using marker files
    if marker_files:
        all_present = all(
            os.path.isfile(os.path.join(local_dir, f)) for f in marker_files
        )
        if all_present:
            if _is_verbose():
                _get_console().print(f"  [green]Already downloaded:[/green] {repo_id}")
            return local_dir
    elif os.path.isdir(local_dir) and any(os.scandir(local_dir)):
        if _is_verbose():
            _get_console().print(f"  [green]Already downloaded:[/green] {repo_id}")
        return local_dir

    if desc:
        _get_console().print(f"  Downloading: [cyan]{desc}[/cyan] from {repo_id}")
    else:
        _get_console().print(f"  Downloading: [cyan]{repo_id}[/cyan]")

    try:
        from huggingface_hub import snapshot_download

        kwargs = dict(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
        )
        if allow_patterns:
            kwargs["allow_patterns"] = allow_patterns
        if ignore_patterns:
            kwargs["ignore_patterns"] = ignore_patterns

        snapshot_download(**kwargs)
        return local_dir
    except ImportError:
        _get_console().print(
            "[red]ERROR: huggingface_hub not installed. Run install.py first.[/red]"
        )
        raise


# ---------------------------------------------------------------------------
# InfiniteTalk models
# ---------------------------------------------------------------------------


def ensure_infinitetalk_models():
    """
    Download all InfiniteTalk model weights if not already present.

    Only the files actually used by the pipeline are downloaded:

    From Wan-AI/Wan2.1-I2V-14B-480P (~5.3 GB with FP8 quant):
        - config.json                                          (250 B)  — WanModel init
        - Wan2.1_VAE.pth                                       (508 MB) — VAE
        - models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth (4.77 GB) — CLIP
        - xlm-roberta-large/                                   — CLIP tokenizer
        - google/                                              — T5 tokenizer
        SKIPPED (not needed with FP8 quant):
        - diffusion_pytorch_model-*.safetensors  (7 shards, ~66 GB)
        - models_t5_umt5-xxl-enc-bf16.pth        (~11.4 GB)
        - assets/, examples/

    From TencentGameMate/chinese-wav2vec2-base (~360 MB):
        - All files (small repo)
        - model.safetensors from PR branch

    From MeiGen-AI/InfiniteTalk (~26 GB for FP8 single):
        - quant_models/infinitetalk_single_fp8.safetensors  (19.5 GB)
        - quant_models/infinitetalk_single_fp8.json         (49 KB)
        - quant_models/t5_fp8.safetensors                   (6.73 GB)
        - quant_models/t5_map_fp8.json                      (12.5 KB)
        SKIPPED:
        - single/ (~10 GB, only for non-quant mode)
        - multi/ (multi-person checkpoints)
        - comfyui/ (ComfyUI nodes)
        - assets/ (demo images)
        - quant_models/*_multi_*, *_int8_*, *_lora* (~120+ GB)
    """
    if _is_verbose():
        _get_console().print("\n[bold]Checking InfiniteTalk models...[/bold]")

    os.makedirs(config.INFINITETALK_MODELS_DIR, exist_ok=True)

    # 1. Wan2.1-I2V-14B-480P — only VAE, CLIP, tokenizers, config
    #    With FP8 quant, the 7 diffusion shards (~66 GB) and T5 pth (~11.4 GB)
    #    are NOT needed — the FP8 checkpoint and quantized T5 replace them.
    _hf_snapshot_download(
        repo_id=config.WAN_HF_REPO,
        local_dir=config.WAN_CKPT_DIR,
        desc="Wan2.1-I2V-14B base (VAE + CLIP + tokenizers, ~5.3 GB)",
        allow_patterns=[
            "config.json",
            "Wan2.1_VAE.pth",
            "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
            "xlm-roberta-large/**",
            "google/**",
        ],
        marker_files=[
            "config.json",
            "Wan2.1_VAE.pth",
            "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        ],
    )

    # 2. chinese-wav2vec2-base audio encoder (~360 MB)
    _hf_snapshot_download(
        repo_id=config.WAV2VEC_HF_REPO,
        local_dir=config.WAV2VEC_DIR,
        desc="chinese-wav2vec2-base (~360 MB)",
        ignore_patterns=["*.md", ".gitattributes"],
        marker_files=["config.json", "preprocessor_config.json"],
    )

    # 2b. wav2vec2 model.safetensors from PR branch (required by InfiniteTalk)
    try:
        from huggingface_hub import hf_hub_download

        safetensors_path = os.path.join(config.WAV2VEC_DIR, "model.safetensors")
        if not os.path.isfile(safetensors_path):
            _get_console().print(
                "  Downloading: [cyan]wav2vec2 model.safetensors (PR branch)[/cyan]"
            )
            hf_hub_download(
                repo_id=config.WAV2VEC_HF_REPO,
                filename="model.safetensors",
                revision="refs/pr/1",
                local_dir=config.WAV2VEC_DIR,
            )
        else:
            if _is_verbose():
                _get_console().print(
                    "  [green]Already downloaded:[/green] wav2vec2 model.safetensors"
                )
    except Exception as e:
        _get_console().print(
            f"  [yellow]Warning: Could not download wav2vec2 model.safetensors: {e}[/yellow]"
        )

    # 3. InfiniteTalk — only FP8 single-person checkpoint + quantized T5
    #    The full repo is ~160+ GB. We only need ~26 GB.
    _hf_snapshot_download(
        repo_id=config.INFINITETALK_HF_REPO,
        local_dir=config.INFINITETALK_CKPT_DIR,
        desc="InfiniteTalk FP8 checkpoint + quantized T5 (~26 GB)",
        allow_patterns=[
            "quant_models/infinitetalk_single_fp8.safetensors",
            "quant_models/infinitetalk_single_fp8.json",
            "quant_models/t5_fp8.safetensors",
            "quant_models/t5_map_fp8.json",
        ],
        marker_files=[
            "quant_models/infinitetalk_single_fp8.safetensors",
            "quant_models/t5_fp8.safetensors",
        ],
    )

    if _is_verbose():
        _get_console().print("[green]InfiniteTalk models: OK[/green]")
    return {
        "wan_ckpt_dir": config.WAN_CKPT_DIR,
        "wav2vec_dir": config.WAV2VEC_DIR,
        "infinitetalk_ckpt_dir": config.INFINITETALK_CKPT_DIR,
    }


def ensure_lora_models():
    """
    Download the LightX2V accelerator LoRA if not already present.

    Source: Kijai/WanVideo_comfy on HuggingFace (~200 MB)
    This LoRA enables 4-step inference (down from 20) with no CFG overhead.
    """
    lora_path = config.LIGHTX2V_LORA_PATH
    if os.path.isfile(lora_path):
        if _is_verbose():
            _get_console().print(f"  [green]Already downloaded:[/green] LightX2V LoRA")
        return lora_path

    _get_console().print(
        "  Downloading: [cyan]LightX2V accelerator LoRA (~200 MB)[/cyan]"
    )
    os.makedirs(config.LORA_DIR, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download

        hf_hub_download(
            repo_id=config.LIGHTX2V_LORA_HF_REPO,
            filename=config.LIGHTX2V_LORA_HF_FILE,
            local_dir=config.LORA_DIR,
        )
        _get_console().print(
            f"  [green]Downloaded:[/green] {config.LIGHTX2V_LORA_FILENAME}"
        )
        return lora_path
    except Exception as e:
        _get_console().print(
            f"  [red]ERROR: Failed to download LightX2V LoRA: {e}[/red]"
        )
        raise


# ---------------------------------------------------------------------------
# Chatterbox models
# ---------------------------------------------------------------------------


def ensure_chatterbox_models():
    """
    Verify Chatterbox model availability.

    Chatterbox uses `from_pretrained()` which handles its own HuggingFace
    download internally. We just verify the cache or trigger the download.
    """
    if _is_verbose():
        _get_console().print("\n[bold]Checking Chatterbox models...[/bold]")

    # Chatterbox handles its own download via from_pretrained().
    # We check if the HF cache already has the model to avoid
    # a redundant download message.
    try:
        from huggingface_hub import scan_cache_dir

        cache_info = scan_cache_dir()
        chatterbox_cached = any(
            repo.repo_id == config.CHATTERBOX_HF_REPO for repo in cache_info.repos
        )
        if chatterbox_cached:
            if _is_verbose():
                _get_console().print(
                    "  [green]Already cached:[/green] Chatterbox Turbo"
                )
        else:
            if _is_verbose():
                _get_console().print(
                    "  [yellow]Chatterbox weights will be downloaded on first TTS call[/yellow]"
                )
    except Exception:
        if _is_verbose():
            _get_console().print(
                "  [yellow]Chatterbox weights will be downloaded on first TTS call[/yellow]"
            )

    if _is_verbose():
        _get_console().print("[green]Chatterbox models: OK[/green]")


# ---------------------------------------------------------------------------
# Download all models
# ---------------------------------------------------------------------------


def ensure_all_models():
    """Download all required models. Called by --download-only flag."""
    config.ensure_directories()
    ensure_infinitetalk_models()
    ensure_chatterbox_models()
    if _is_verbose():
        _get_console().print("\n[bold green]All models ready.[/bold green]")
