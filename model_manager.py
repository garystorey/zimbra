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

import logging

import config

logger = logging.getLogger(__name__)


_console = None


def _get_console():
    """Get a Console that writes to the real stderr (survives stdout redirect)."""
    global _console
    if _console is None:
        import sys

        _stderr = getattr(config, "_REAL_STDERR", sys.stderr)
        _console = Console(file=_stderr)
    return _console


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
            logger.info("Already downloaded: %s", repo_id)
            return local_dir
    elif os.path.isdir(local_dir) and any(os.scandir(local_dir)):
        logger.info("Already downloaded: %s", repo_id)
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
    Download shared InfiniteTalk model weights if not already present.

    Only the files actually used by the pipeline are downloaded:

    From Wan-AI/Wan2.1-I2V-14B-480P (~5.3 GB):
        - config.json                                          (250 B)  — WanModel init
        - Wan2.1_VAE.pth                                       (508 MB) — VAE
        - models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth (4.77 GB) — CLIP
        - xlm-roberta-large/                                   — CLIP tokenizer
        - google/                                              — T5 tokenizer
        SKIPPED (DiT weights come from GGUF files instead):
        - diffusion_pytorch_model-*.safetensors  (7 shards, ~66 GB)
        - models_t5_umt5-xxl-enc-bf16.pth        (~11.4 GB)
        - assets/, examples/

    From TencentGameMate/chinese-wav2vec2-base (~360 MB):
        - All files (small repo)
        - model.safetensors from PR branch
    """
    logger.info("Checking InfiniteTalk models...")

    os.makedirs(config.INFINITETALK_MODELS_DIR, exist_ok=True)

    # 1. Wan2.1-I2V-14B-480P — only VAE, CLIP, tokenizers, config
    #    The 7 diffusion shards (~66 GB) and T5 pth (~11.4 GB)
    #    are NOT needed — the GGUF files and quantized T5 replace them.
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
            logger.info("Already downloaded: wav2vec2 model.safetensors")
    except Exception as e:
        _get_console().print(
            f"  [yellow]Warning: Could not download wav2vec2 model.safetensors: {e}[/yellow]"
        )

    logger.info("InfiniteTalk models: OK")
    return {
        "wan_ckpt_dir": config.WAN_CKPT_DIR,
        "wav2vec_dir": config.WAV2VEC_DIR,
        "infinitetalk_ckpt_dir": config.INFINITETALK_CKPT_DIR,
    }


def ensure_gguf_models():
    """
    Download GGUF model files if not already present.

    Downloads:
    - Base Wan2.1-I2V-14B Q4_0 GGUF (~10.2 GB) from city96/Wan2.1-I2V-14B-480P-gguf
    - InfiniteTalk Single Q8 adapter GGUF (~2.65 GB) from Kijai/WanVideo_comfy_GGUF
    """
    os.makedirs(config.GGUF_DIR, exist_ok=True)

    # 1. Base Wan2.1 I2V 14B GGUF (Q4_0)
    base_path = config.GGUF_BASE_PATH
    if not os.path.isfile(base_path):
        _get_console().print(
            "  Downloading: [cyan]Wan2.1-I2V-14B Q4_0 GGUF (~10.2 GB)[/cyan]"
        )
        try:
            from huggingface_hub import hf_hub_download

            hf_hub_download(
                repo_id=config.GGUF_BASE_HF_REPO,
                filename=config.GGUF_BASE_FILENAME,
                local_dir=config.GGUF_DIR,
            )
            _get_console().print(
                f"  [green]Downloaded:[/green] {config.GGUF_BASE_FILENAME}"
            )
        except Exception as e:
            _get_console().print(
                f"  [red]ERROR: Failed to download base GGUF: {e}[/red]"
            )
            raise
    else:
        logger.info("Already downloaded: %s", config.GGUF_BASE_FILENAME)

    # 2. InfiniteTalk Single Q8 adapter GGUF
    adapter_path = config.GGUF_ADAPTER_PATH
    if not os.path.isfile(adapter_path):
        _get_console().print(
            "  Downloading: [cyan]InfiniteTalk Single Q8 GGUF (~2.65 GB)[/cyan]"
        )
        try:
            from huggingface_hub import hf_hub_download

            hf_hub_download(
                repo_id=config.GGUF_ADAPTER_HF_REPO,
                filename=f"InfiniteTalk/{config.GGUF_ADAPTER_FILENAME}",
                local_dir=config.GGUF_DIR,
            )
            # hf_hub_download preserves subdir structure — move if needed
            downloaded = os.path.join(
                config.GGUF_DIR, "InfiniteTalk", config.GGUF_ADAPTER_FILENAME
            )
            if os.path.isfile(downloaded) and not os.path.isfile(adapter_path):
                shutil.move(downloaded, adapter_path)
                # Clean up empty subdir
                inf_dir = os.path.join(config.GGUF_DIR, "InfiniteTalk")
                if os.path.isdir(inf_dir) and not os.listdir(inf_dir):
                    os.rmdir(inf_dir)
            _get_console().print(
                f"  [green]Downloaded:[/green] {config.GGUF_ADAPTER_FILENAME}"
            )
        except Exception as e:
            _get_console().print(
                f"  [red]ERROR: Failed to download adapter GGUF: {e}[/red]"
            )
            raise
    else:
        logger.info("Already downloaded: %s", config.GGUF_ADAPTER_FILENAME)

    logger.info("GGUF models: OK")


def ensure_lora_models():
    """
    Download the LightX2V I2V accelerator LoRA if not already present.

    Source: Kijai/WanVideo_comfy on HuggingFace (~738 MB)
    I2V rank-64 variant — matches ComfyUI workflow. Enables 4-step inference.
    """
    lora_path = config.LIGHTX2V_LORA_PATH
    if os.path.isfile(lora_path):
        logger.info("Already downloaded: LightX2V LoRA")
        return lora_path

    _get_console().print(
        "  Downloading: [cyan]LightX2V I2V LoRA rank-64 (~738 MB)[/cyan]"
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
    logger.info("Checking Chatterbox models...")

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
            logger.info("Already cached: Chatterbox Turbo")
        else:
            logger.info("Chatterbox weights will be downloaded on first TTS call")
    except Exception:
        logger.info("Chatterbox weights will be downloaded on first TTS call")

    logger.info("Chatterbox models: OK")


# ---------------------------------------------------------------------------
# Pre-flight download check
# ---------------------------------------------------------------------------

# All required model files with human-readable names and approximate sizes (bytes).
# This list is the single source of truth for what must be present before pipeline runs.
_REQUIRED_FILES = [
    # (local_path, description, approx_size_bytes)
    (
        lambda: os.path.join(config.WAN_CKPT_DIR, "config.json"),
        "Wan2.1 config.json",
        250,
    ),
    (
        lambda: os.path.join(config.WAN_CKPT_DIR, "Wan2.1_VAE.pth"),
        "Wan2.1 VAE",
        508_000_000,
    ),
    (
        lambda: os.path.join(
            config.WAN_CKPT_DIR,
            "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        ),
        "Wan2.1 CLIP",
        4_770_000_000,
    ),
    (lambda: os.path.join(config.WAV2VEC_DIR, "config.json"), "wav2vec2 config", 2_000),
    (
        lambda: os.path.join(config.WAV2VEC_DIR, "preprocessor_config.json"),
        "wav2vec2 preprocessor",
        300,
    ),
    (
        lambda: os.path.join(config.WAV2VEC_DIR, "model.safetensors"),
        "wav2vec2 model",
        360_000_000,
    ),
    (lambda: config.GGUF_BASE_PATH, "GGUF base Q4_0", 10_200_000_000),
    (lambda: config.GGUF_ADAPTER_PATH, "GGUF adapter Q8", 2_650_000_000),
    (lambda: config.LIGHTX2V_LORA_PATH, "LightX2V I2V LoRA rank-64", 738_000_000),
]


def _check_missing_files():
    """Return list of (path, description, size) for files that need downloading."""
    missing = []
    for path_fn, desc, size in _REQUIRED_FILES:
        path = path_fn()
        if not os.path.isfile(path):
            missing.append((path, desc, size))
    return missing


def preflight_downloads(console=None):
    """Check all required model files and download any that are missing.

    Shows a unified progress bar with total file count and size.
    Must be called BEFORE the pipeline starts so that load_model()
    never triggers downloads that corrupt the pipeline progress bars.

    Args:
        console: Rich Console instance for output. Uses default if None.

    Returns:
        True if any files were downloaded, False if all were already present.
    """
    config.ensure_directories()
    missing = _check_missing_files()

    if not missing:
        return False

    con = console or _get_console()
    total_size = sum(s for _, _, s in missing)
    total_size_gb = total_size / (1024**3)
    con.print()
    con.print(
        f"[bold]Downloading {len(missing)} model file{'s' if len(missing) != 1 else ''} "
        f"({total_size_gb:.1f} GB total)[/bold]"
    )
    con.print()

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TextColumn("{task.completed}/{task.total} files"),
        TextColumn("[dim]{task.fields[current]}[/dim]"),
        TimeElapsedColumn(),
        console=con,
        transient=False,
        redirect_stdout=False,
        redirect_stderr=False,
    ) as progress:
        dl_task = progress.add_task(
            "Downloading models",
            total=len(missing),
            current="",
        )

        for _i, (path, desc, _size) in enumerate(missing):
            progress.update(dl_task, current=desc)
            _download_one_model(path, desc)
            progress.advance(dl_task)

        progress.update(dl_task, current="done")

    con.print("[bold green]All model files ready.[/bold green]")
    con.print()
    return True


def _download_one_model(path, desc):
    """Download a single model file identified by its expected local path."""
    import logging

    _logger = logging.getLogger(__name__)

    # Wan2.1 files (VAE, CLIP, config, tokenizers)
    wan_dir = config.WAN_CKPT_DIR
    if path.startswith(wan_dir):
        _logger.info(f"  Downloading {desc} from {config.WAN_HF_REPO}")
        _hf_snapshot_download(
            repo_id=config.WAN_HF_REPO,
            local_dir=wan_dir,
            desc=f"Wan2.1-I2V-14B base (VAE + CLIP + tokenizers)",
            allow_patterns=[
                "config.json",
                "Wan2.1_VAE.pth",
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
                "xlm-roberta-large/**",
                "google/**",
            ],
        )
        return

    # wav2vec2 files
    wav2vec_dir = config.WAV2VEC_DIR
    if path.startswith(wav2vec_dir):
        basename = os.path.basename(path)
        if basename == "model.safetensors":
            _logger.info(
                f"  Downloading {desc} from {config.WAV2VEC_HF_REPO} (PR branch)"
            )
            from huggingface_hub import hf_hub_download

            hf_hub_download(
                repo_id=config.WAV2VEC_HF_REPO,
                filename="model.safetensors",
                revision="refs/pr/1",
                local_dir=wav2vec_dir,
            )
        else:
            _logger.info(f"  Downloading {desc} from {config.WAV2VEC_HF_REPO}")
            _hf_snapshot_download(
                repo_id=config.WAV2VEC_HF_REPO,
                local_dir=wav2vec_dir,
                desc="chinese-wav2vec2-base",
                ignore_patterns=["*.md", ".gitattributes"],
            )
        return

    # GGUF base
    if path == config.GGUF_BASE_PATH:
        _logger.info(f"  Downloading {desc} from {config.GGUF_BASE_HF_REPO}")
        from huggingface_hub import hf_hub_download

        hf_hub_download(
            repo_id=config.GGUF_BASE_HF_REPO,
            filename=config.GGUF_BASE_FILENAME,
            local_dir=config.GGUF_DIR,
        )
        return

    # GGUF adapter
    if path == config.GGUF_ADAPTER_PATH:
        _logger.info(f"  Downloading {desc} from {config.GGUF_ADAPTER_HF_REPO}")
        from huggingface_hub import hf_hub_download

        hf_hub_download(
            repo_id=config.GGUF_ADAPTER_HF_REPO,
            filename=f"InfiniteTalk/{config.GGUF_ADAPTER_FILENAME}",
            local_dir=config.GGUF_DIR,
        )
        # Move from subdir if needed
        downloaded = os.path.join(
            config.GGUF_DIR, "InfiniteTalk", config.GGUF_ADAPTER_FILENAME
        )
        if os.path.isfile(downloaded) and not os.path.isfile(path):
            shutil.move(downloaded, path)
            inf_dir = os.path.join(config.GGUF_DIR, "InfiniteTalk")
            if os.path.isdir(inf_dir) and not os.listdir(inf_dir):
                os.rmdir(inf_dir)
        return

    # LoRA
    if path == config.LIGHTX2V_LORA_PATH:
        _logger.info(f"  Downloading {desc} from {config.LIGHTX2V_LORA_HF_REPO}")
        os.makedirs(config.LORA_DIR, exist_ok=True)
        from huggingface_hub import hf_hub_download

        hf_hub_download(
            repo_id=config.LIGHTX2V_LORA_HF_REPO,
            filename=config.LIGHTX2V_LORA_HF_FILE,
            local_dir=config.LORA_DIR,
        )
        return

    _logger.warning(f"  Unknown model file, cannot download: {path}")


# ---------------------------------------------------------------------------
# Download all models (legacy — prefer preflight_downloads)
# ---------------------------------------------------------------------------


def ensure_all_models():
    """Download all required models. Called by --download-only flag."""
    config.ensure_directories()
    ensure_infinitetalk_models()
    ensure_gguf_models()
    ensure_lora_models()
    ensure_chatterbox_models()
    logger.info("All models ready.")
