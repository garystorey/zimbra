"""
Talking Head Video Generator — Configuration

Central configuration for paths, defaults, GPU settings, and model URLs.
All defaults can be overridden via CLI arguments.
"""

import os
import sys
import platform


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

INPUTS_DIR = os.path.join(PROJECT_ROOT, "inputs")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
VENDOR_DIR = os.path.join(PROJECT_ROOT, "vendor")

VOICE_REF_DIR = os.path.join(INPUTS_DIR, "voice_ref")
IMAGES_DIR = os.path.join(INPUTS_DIR, "images")
SCRIPTS_DIR = os.path.join(INPUTS_DIR, "scripts")

AUDIO_OUTPUT_DIR = os.path.join(OUTPUTS_DIR, "audio")
VIDEO_OUTPUT_DIR = os.path.join(OUTPUTS_DIR, "video")

INFINITETALK_MODELS_DIR = os.path.join(MODELS_DIR, "infinitetalk")
CHATTERBOX_MODELS_DIR = os.path.join(MODELS_DIR, "chatterbox")

INFINITETALK_VENDOR_DIR = os.path.join(VENDOR_DIR, "infinitetalk")


# ---------------------------------------------------------------------------
# Default input files (CLI falls back to these when args are omitted)
# ---------------------------------------------------------------------------

DEFAULT_VOICE_REF_EXTENSIONS = (".wav", ".mp3")
DEFAULT_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def _find_default_file(directory, basename, extensions):
    """Find a default file by basename with any of the given extensions."""
    for ext in extensions:
        path = os.path.join(directory, basename + ext)
        if os.path.isfile(path):
            return path
    return None


DEFAULT_VOICE_REF = _find_default_file(
    VOICE_REF_DIR, "default", DEFAULT_VOICE_REF_EXTENSIONS
)
DEFAULT_IMAGE = _find_default_file(IMAGES_DIR, "default", DEFAULT_IMAGE_EXTENSIONS)
DEFAULT_SCRIPT_FILE = os.path.join(SCRIPTS_DIR, "default.txt")


# ---------------------------------------------------------------------------
# CLI defaults
# ---------------------------------------------------------------------------

DEFAULT_EXAGGERATION = 0.5
DEFAULT_CFG_WEIGHT = 0.5
DEFAULT_ACCELERATED = True
DEFAULT_AUDIO_FORMAT = "wav"
DEFAULT_FPS = 24


# ---------------------------------------------------------------------------
# InfiniteTalk video engine defaults
# ---------------------------------------------------------------------------

VIDEO_SIZE = "infinitetalk-480"  # "infinitetalk-480" or "infinitetalk-720"
VIDEO_SAMPLE_STEPS = 20  # Denoising steps (default 20)
VIDEO_MODE = "streaming"  # "streaming" for long video, "clip" for single chunk
VIDEO_MOTION_FRAME = 9  # Driven frame length for streaming mode
VIDEO_FRAME_NUM = 41  # Frames per chunk (must be 4n+1, internal only)
VIDEO_SEED = 42
VIDEO_SAMPLE_RATE = 16000  # Audio sample rate for wav2vec2
VIDEO_FPS = 25  # InfiniteTalk generates at 25 FPS
VIDEO_TEXT_GUIDE_SCALE = 1.0  # Text CFG (1.0 = 2 passes/step, 5.0 = 3 passes/step)
VIDEO_AUDIO_GUIDE_SCALE = 4.0  # Audio CFG (4.0 default, 2.0 with LoRA)

# LoRA accelerator settings
VIDEO_USE_LORA = True  # Enable LightX2V distilled LoRA by default
VIDEO_LORA_SCALE = 1.0  # LoRA merge strength
# When LoRA is enabled, these override the defaults above:
VIDEO_LORA_SAMPLE_STEPS = 4  # LightX2V distilled: only 4 steps needed
VIDEO_LORA_SAMPLE_SHIFT = 2  # Lower shift for distilled model
VIDEO_LORA_TEXT_GUIDE_SCALE = 1.0  # Must be 1.0 (no CFG) for distilled LoRA
VIDEO_LORA_AUDIO_GUIDE_SCALE = 2.0  # Reduced audio CFG with LoRA

# Time estimation (seconds per denoising step per chunk, for ETA display)
# Calibrate this to your GPU. Default ~10s based on RTX 4090 Laptop 16GB w/ FP8 + block swap.
VIDEO_SECS_PER_STEP = 10

# Block swap settings (for 16GB GPUs)
# None = auto-tune based on available VRAM. With native FP8 (torch._scaled_mm),
# blocks stay at ~470MB during compute (no dequant doubling), so many more
# blocks can remain persistent on GPU vs the old quanto approach.
VIDEO_BLOCKS_TO_SWAP = None  # None = auto-tune, 0 = all on GPU, 40 = swap all
VIDEO_PREFETCH_BLOCKS = (
    1  # Async prefetch N blocks ahead (hides CPU→GPU transfer latency)
)
VIDEO_QUANT = "fp8"  # FP8 quantization (None to disable)

# Number of DiT blocks in the InfiniteTalk model (40 for 14B)
VIDEO_NUM_DIT_BLOCKS = 40
# Approximate size of one DiT block in bytes (FP8: ~470MB)
VIDEO_BLOCK_SIZE_BYTES = 470_000_000

# Model weight directories (resolved after download)
WAN_CKPT_DIR = os.path.join(INFINITETALK_MODELS_DIR, "Wan2.1-I2V-14B-480P")
WAV2VEC_DIR = os.path.join(INFINITETALK_MODELS_DIR, "chinese-wav2vec2-base")
INFINITETALK_CKPT_DIR = os.path.join(INFINITETALK_MODELS_DIR, "InfiniteTalk")
INFINITETALK_SINGLE_CKPT = os.path.join(
    INFINITETALK_CKPT_DIR, "single", "infinitetalk.safetensors"
)
INFINITETALK_FP8_CKPT = os.path.join(
    INFINITETALK_CKPT_DIR, "quant_models", "infinitetalk_single_fp8.safetensors"
)

# LightX2V accelerator LoRA
LORA_DIR = os.path.join(INFINITETALK_MODELS_DIR, "loras")
LIGHTX2V_LORA_FILENAME = (
    "Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors"
)
LIGHTX2V_LORA_PATH = os.path.join(LORA_DIR, LIGHTX2V_LORA_FILENAME)
LIGHTX2V_LORA_HF_REPO = "Kijai/WanVideo_comfy"
LIGHTX2V_LORA_HF_FILE = LIGHTX2V_LORA_FILENAME


# ---------------------------------------------------------------------------
# Model download URLs and HuggingFace repo IDs
# ---------------------------------------------------------------------------

WAN_HF_REPO = "Wan-AI/Wan2.1-I2V-14B-480P"
WAV2VEC_HF_REPO = "TencentGameMate/chinese-wav2vec2-base"
INFINITETALK_HF_REPO = "MeiGen-AI/InfiniteTalk"

CHATTERBOX_HF_REPO = "ResembleAI/chatterbox-turbo"


# ---------------------------------------------------------------------------
# GPU settings
# ---------------------------------------------------------------------------


def get_gpu_info():
    """Detect GPU and return info dict. Returns None if no CUDA GPU available."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return {
            "device_name": torch.cuda.get_device_name(0),
            "cuda_version": torch.version.cuda,
            "vram_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
            "device_count": torch.cuda.device_count(),
        }
    except ImportError:
        return None


TORCH_DTYPE = "float16"
DEVICE = "cuda"


# ---------------------------------------------------------------------------
# FFmpeg configuration
# ---------------------------------------------------------------------------


def _find_ffmpeg():
    """
    Locate ffmpeg binary. Priority:
    1. FFMPEG_PATH environment variable
    2. ffmpeg in system PATH
    3. imageio-ffmpeg bundled binary
    """
    # Check env var
    env_path = os.environ.get("FFMPEG_PATH")
    if env_path:
        if os.path.isfile(env_path):
            return env_path
        # Maybe it's a directory containing ffmpeg
        candidate = os.path.join(
            env_path, "ffmpeg.exe" if platform.system() == "Windows" else "ffmpeg"
        )
        if os.path.isfile(candidate):
            return candidate

    # Check system PATH
    import shutil

    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg

    # Try imageio-ffmpeg
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        pass

    return None


def auto_tune_blocks_to_swap():
    """Auto-tune how many DiT blocks to swap based on available VRAM.

    The native FP8 InfiniteTalk DiT has 40 blocks, each ~470MB in FP8.
    With native FP8 (torch._scaled_mm), blocks stay at ~470MB during
    compute — no dequantization doubling. This means we can keep many
    more blocks persistent on GPU compared to the old quanto approach.

    IMPORTANT: Call this AFTER offloading VAE/CLIP/T5 to CPU so the free
    VRAM calculation sees the maximum available memory.

    Returns:
        int: Number of blocks to swap to CPU. 0 = all blocks fit on GPU.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return VIDEO_NUM_DIT_BLOCKS  # swap all
        free_vram = torch.cuda.get_device_properties(
            0
        ).total_memory - torch.cuda.memory_allocated(0)
        # Reserve VRAM for denoising activations + CUDA allocator overhead.
        # 41 frames at 480p needs ~5-6GB for attention/cross-attention/
        # intermediates, plus ~2-3GB CUDA allocator overhead and fragmentation.
        # Empirically: 5 persistent blocks (2.4GB) + 8.5GB reserve works well
        # on 16GB. Too little reserve causes catastrophic VRAM thrashing.
        activation_reserve = 8.5 * (1024**3)
        usable_vram = max(0, free_vram - activation_reserve)
        # Non-block params (embeddings, head, img_emb, audio_proj) ~500MB
        non_block_overhead = 500_000_000
        vram_for_blocks = max(0, usable_vram - non_block_overhead)
        blocks_that_fit = int(vram_for_blocks / VIDEO_BLOCK_SIZE_BYTES)
        blocks_that_fit = min(blocks_that_fit, VIDEO_NUM_DIT_BLOCKS)
        blocks_to_swap = VIDEO_NUM_DIT_BLOCKS - blocks_that_fit
        return blocks_to_swap
    except Exception:
        return VIDEO_NUM_DIT_BLOCKS  # swap all as fallback


FFMPEG_PATH = _find_ffmpeg()

# Set FFMPEG_PATH env var and add to PATH for InfiniteTalk/ffmpeg compatibility
if FFMPEG_PATH:
    ffmpeg_dir = os.path.dirname(FFMPEG_PATH)
    os.environ["FFMPEG_PATH"] = ffmpeg_dir
    if ffmpeg_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")


# ---------------------------------------------------------------------------
# Ensure runtime directories exist
# ---------------------------------------------------------------------------


def ensure_directories():
    """Create all runtime directories if they don't exist."""
    dirs = [
        AUDIO_OUTPUT_DIR,
        VIDEO_OUTPUT_DIR,
        INFINITETALK_MODELS_DIR,
        CHATTERBOX_MODELS_DIR,
        LORA_DIR,
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
