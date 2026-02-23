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
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
VENDOR_DIR = os.path.join(PROJECT_ROOT, "vendor")

VOICE_REF_DIR = os.path.join(INPUTS_DIR, "voice_ref")
IMAGES_DIR = os.path.join(INPUTS_DIR, "images")
SCRIPTS_DIR = os.path.join(INPUTS_DIR, "scripts")
ANIMATION_DIR = os.path.join(INPUTS_DIR, "animation")


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
DEFAULT_ANIMATION_FILE = os.path.join(ANIMATION_DIR, "default.txt")
DEFAULT_AUDIO_EXTENSIONS = (".wav", ".mp3")
DEFAULT_AUDIO_FILE = _find_default_file(
    SCRIPTS_DIR, "default", DEFAULT_AUDIO_EXTENSIONS
)


# ---------------------------------------------------------------------------
# CLI defaults
# ---------------------------------------------------------------------------

DEFAULT_ACCELERATED = True
DEFAULT_AUDIO_FORMAT = "wav"
DEFAULT_FPS = 24

# TTS sampling parameters — balanced between voice fidelity and generation quality.
# Chatterbox Turbo defaults: temp=0.8, top_k=1000, top_p=0.95, rep_penalty=1.2.
# Too-restrictive settings (low top_k/top_p) cause truncation and hallucination
# because the model can't find valid next tokens and hits early EOS or loops.
TTS_TEMPERATURE = 0.7
TTS_TOP_K = 150
TTS_TOP_P = 0.92
TTS_REPETITION_PENALTY = 1.2

# Run timestamp — set once at startup by setup_logging(), shared across all
# output filenames (log, audio, video) so they can be correlated easily.
RUN_TIMESTAMP = None  # Will be set to e.g. "20260219_201603"
RUN_FILENAME = None  # Will be set to the resolved output base filename


# ---------------------------------------------------------------------------
# InfiniteTalk video engine defaults
# ---------------------------------------------------------------------------

VIDEO_SIZE = "infinitetalk-480"  # "infinitetalk-480" or "infinitetalk-720"
VIDEO_SAMPLE_STEPS = 20  # Denoising steps (default 20)
VIDEO_MODE = "streaming"  # "streaming" for long video, "clip" for single chunk
VIDEO_MOTION_FRAME = (
    13  # Driven frame length for streaming mode (0.52s overlap, same chunk count as 9)
)
VIDEO_FRAME_NUM = 41  # Frames per chunk (must be 4n+1, internal only). 81 OOMs on 16GB during LoRA merge.
VIDEO_SEED = 42
VIDEO_SAMPLE_RATE = 16000  # Audio sample rate for wav2vec2
VIDEO_FPS = 25  # InfiniteTalk generates at 25 FPS
VIDEO_TEXT_GUIDE_SCALE = 1.0  # Text CFG (1.0 = 2 passes/step, 5.0 = 3 passes/step)
VIDEO_AUDIO_GUIDE_SCALE = 4.0  # Audio CFG (4.0 default, 2.0 with LoRA)

# LoRA accelerator settings
VIDEO_USE_LORA = True  # Enable LightX2V distilled LoRA by default
VIDEO_LORA_SCALE = 1.0  # LoRA merge strength
# When LoRA is enabled, these override the defaults above:
VIDEO_LORA_SAMPLE_STEPS = 6  # LoRA distilled for fewer steps.
VIDEO_LORA_SAMPLE_SHIFT = 11  # ComfyUI-validated shift for LoRA + DPM++
VIDEO_LORA_TEXT_GUIDE_SCALE = 1.0  # Must be 1.0 (no CFG) for distilled LoRA
VIDEO_LORA_AUDIO_GUIDE_SCALE = 2.0  # Audio CFG (2 passes/step). 1.0 causes progressive blur. 2.5 too aggressive. 2.0 is the sweet spot.
VIDEO_LORA_SOLVER = "dpmpp_sde"  # DPM++ SDE 1st-order. SDE noise prevents error accumulation across chunks. 2nd-order tried and reverted (overshoots with CFG).

# Time estimation (seconds per denoising step per SINGLE PASS, for ETA display)
# The ETA code in video_engine.py multiplies this by passes_per_step (1, 2, or 3).
# Calibrate this to your GPU. Measured on RTX 4090 Laptop 16GB.
VIDEO_SECS_PER_STEP = 34  # ~34s/pass at 1-pass (audio_cfg=1.0), ~36s/pass at 2-pass (audio_cfg=2.0). I2V rank-64 LoRA, frame_num=81.

# GGUF model settings.
# GGUF model is ~12.8GB.  With block swap most blocks stay on GPU.
# frame_num=81 matches ComfyUI (fewer chunks = better quality + faster).
# DiT blocks are offloaded before VAE encode/decode to give Conv3D headroom.
VIDEO_GGUF_FRAME_NUM = (
    81  # 4n+1, matches ComfyUI frame_window=81. Fewer chunks = better quality.
)
VIDEO_GGUF_MOTION_FRAME = 9  # Reduced from 25. motion_frame=25 froze 43% of frames (audio-unresponsive) and created too many chunks for long videos. 9 = 0.36s overlap, sufficient with direct latent passthrough.
VIDEO_GGUF_LORA_SAMPLE_STEPS = (
    4  # LightX2V designed for 4 steps; saves 33% denoising vs 6
)
VIDEO_GGUF_LORA_AUDIO_GUIDE_SCALE = 1.0  # 1.0 = 1 pass/step (skip CFG). Works with I2V rank-64 LoRA. Previous failure (Phase 15b) was with wrong T2V rank-32 LoRA. CLI --audio-cfg 2.0 for stronger lip sync at 2x render time.
VIDEO_GGUF_SECS_PER_STEP = 34  # Measured ~33.7s/pass at 1-pass, ~35.8s at 2-pass. I2V rank-64 LoRA, 22+18 swapped (run_20260221_223335)
VIDEO_GGUF_ACTIVATION_RESERVE = 8.5  # GiB — 81-frame activations need more headroom than 41-frame. Auto-tunes to ~22 persistent + ~18 swapped.

# Per-chunk overhead (seconds) — VAE decode + block offload/restore round-trip.
# These are NOT included in SECS_PER_STEP; added separately in ETA calculations.
# Conservative estimates — calibrate from "ETA accuracy" log lines after real runs.
VIDEO_VAE_DECODE_SECS = 28  # Per-chunk transition: VAE decode (~13s) + block offload/restore (~10s) + thermal + motion latent. Measured from run_20260221_190632.
VIDEO_FIRST_CHUNK_EXTRA_SECS = 24  # CLIP (~10s) + VAE encode (~8s) + setup (~6s). No block offload/restore during encode with I2V LoRA. Measured from run_20260221_190632.

# Block swap settings (for 16GB GPUs)
# None = auto-tune based on available VRAM. GGUF blocks are ~320MB each;
# on-demand dequantization creates BF16 temporaries during forward passes.
VIDEO_BLOCKS_TO_SWAP = None  # None = auto-tune, 0 = all on GPU, 40 = swap all
VIDEO_PREFETCH_BLOCKS = (
    1  # Async prefetch N blocks ahead (hides CPU→GPU transfer latency)
)
VIDEO_QUANT = "gguf"  # GGUF is the only supported quantization mode

# GPU thermal management — pause between chunks if GPU is too hot.
# Laptop GPUs aggressively throttle clocks at high temps, causing 2-3x
# slower block compute. A brief cooldown pause is faster than running
# at throttled speeds for the entire next chunk.
GPU_THERMAL_THRESHOLD = 70  # °C — pause if above this
GPU_THERMAL_MAX_WAIT = 30  # Max seconds to wait for cooldown
GPU_THERMAL_POLL_INTERVAL = 3  # Seconds between temp checks

# Number of DiT blocks in the InfiniteTalk model (40 for 14B)
VIDEO_NUM_DIT_BLOCKS = 40
# Approximate size of one DiT block in bytes (GGUF Q4_0 + Q8 adapter).
VIDEO_BLOCK_SIZE_BYTES = 320_000_000

# Model weight directories (resolved after download)
WAN_CKPT_DIR = os.path.join(INFINITETALK_MODELS_DIR, "Wan2.1-I2V-14B-480P")
WAV2VEC_DIR = os.path.join(INFINITETALK_MODELS_DIR, "chinese-wav2vec2-base")
INFINITETALK_CKPT_DIR = os.path.join(INFINITETALK_MODELS_DIR, "InfiniteTalk")
INFINITETALK_SINGLE_CKPT = os.path.join(
    INFINITETALK_CKPT_DIR, "single", "infinitetalk.safetensors"
)

# GGUF model paths
GGUF_DIR = os.path.join(INFINITETALK_MODELS_DIR, "gguf")
GGUF_BASE_FILENAME = "wan2.1-i2v-14b-480p-Q4_0.gguf"
GGUF_ADAPTER_FILENAME = "Wan2_1-InfiniteTalk_Single_Q8.gguf"
GGUF_BASE_PATH = os.path.join(GGUF_DIR, GGUF_BASE_FILENAME)
GGUF_ADAPTER_PATH = os.path.join(GGUF_DIR, GGUF_ADAPTER_FILENAME)

# LightX2V accelerator LoRA — I2V rank-64 (matches ComfyUI workflow)
# Previous rank-32 T2V LoRA was wrong model type (text-to-video, not image-to-video)
LORA_DIR = os.path.join(INFINITETALK_MODELS_DIR, "loras")
LIGHTX2V_LORA_FILENAME = (
    "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"
)
LIGHTX2V_LORA_HF_REPO = "Kijai/WanVideo_comfy"
LIGHTX2V_LORA_HF_FILE = "Lightx2v/" + LIGHTX2V_LORA_FILENAME
LIGHTX2V_LORA_PATH = os.path.join(LORA_DIR, "Lightx2v", LIGHTX2V_LORA_FILENAME)


# ---------------------------------------------------------------------------
# Model download URLs and HuggingFace repo IDs
# ---------------------------------------------------------------------------

WAN_HF_REPO = "Wan-AI/Wan2.1-I2V-14B-480P"
WAV2VEC_HF_REPO = "TencentGameMate/chinese-wav2vec2-base"
INFINITETALK_HF_REPO = "MeiGen-AI/InfiniteTalk"

# GGUF model repos
GGUF_BASE_HF_REPO = "city96/Wan2.1-I2V-14B-480P-gguf"
GGUF_ADAPTER_HF_REPO = "Kijai/WanVideo_comfy_GGUF"

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

    The InfiniteTalk DiT has 40 blocks, each ~320MB in GGUF Q4_0+Q8.
    On-demand dequantization creates BF16 temporaries during forward
    passes, so activation_reserve must account for this overhead.

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
        # During swap block compute, persistent blocks + 1 swap buffer + non-block
        # params are on GPU simultaneously. Intermediates (qkv, attention, ffn) need
        # ~3.5-4.0 GiB headroom. If total exceeds physical VRAM, Windows WDDM spills
        # to shared GPU memory (system RAM via PCIe), causing 20-55x slowdowns
        # (progressive within a single block: self_attn 2.5x → ffn 55x).
        # With 7.0 GiB reserve: ~17 persistent + 1 swap = 8.96 GiB weights,
        # leaving ~7.0 GiB headroom. Peak with intermediates ~11.5 GiB,
        # well under 16 GiB — no WDDM spill even with allocator fragmentation.
        activation_reserve = VIDEO_GGUF_ACTIVATION_RESERVE * (1024**3)
        # Reserve 1 swap buffer slot: during prefetch, one swapped block
        # occupies GPU alongside the persistent blocks. Without this
        # reservation, the swap slot eats into activation_reserve, causing
        # VRAM thrashing (5-10s/block instead of 0.25s/block).
        swap_buffer_reserve = VIDEO_BLOCK_SIZE_BYTES
        usable_vram = max(0, free_vram - activation_reserve - swap_buffer_reserve)
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
        OUTPUT_DIR,
        INFINITETALK_MODELS_DIR,
        CHATTERBOX_MODELS_DIR,
        LORA_DIR,
        GGUF_DIR,
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
