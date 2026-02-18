"""
Talking Head Video Generator — Video Engine (InfiniteTalk)

Wraps InfiniteTalk's Wan2.1-I2V-14B pipeline to generate talking-head
videos from a portrait image and audio file.

Handles model loading/unloading, audio embedding extraction via wav2vec2,
video generation with streaming mode, and audio muxing.
Supports CPU offloading + FP8 quantization for 16GB VRAM GPUs.
"""

import os

# Triton is unavailable on Windows — disable torch.compile entirely before
# any torch imports so @torch.compile decorators in vendor code become no-ops.
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import gc
import json
import math
import sys
import time
import logging
import tempfile
import warnings

import numpy as np

import torch
import soundfile as sf

import config
import model_manager
import fp8_optimization

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Vendor path setup — add InfiniteTalk source to sys.path
# ---------------------------------------------------------------------------


def _ensure_vendor_path():
    """Add InfiniteTalk vendor directory to sys.path for imports."""
    vendor_dir = config.INFINITETALK_VENDOR_DIR
    if vendor_dir not in sys.path:
        sys.path.insert(0, vendor_dir)


# ---------------------------------------------------------------------------
# Audio helpers (adapted from InfiniteTalk's generate_infinitetalk.py)
# ---------------------------------------------------------------------------


def _loudness_norm(audio_array, sr=16000, lufs=-23):
    """Normalize audio loudness to target LUFS."""
    import pyloudnorm as pyln

    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio_array)
    if abs(loudness) > 100:
        return audio_array
    normalized_audio = pyln.normalize.loudness(audio_array, loudness, lufs)
    return normalized_audio


def _prepare_audio(audio_path, sample_rate=16000):
    """Load and loudness-normalize an audio file."""
    import librosa

    speech_array, sr = librosa.load(audio_path, sr=sample_rate)
    speech_array = _loudness_norm(speech_array, sr)
    return speech_array


def _get_audio_embedding(
    speech_array, wav2vec_feature_extractor, audio_encoder, sr=16000, device="cpu"
):
    """Extract wav2vec2 audio embeddings for InfiniteTalk."""
    from einops import rearrange

    audio_duration = len(speech_array) / sr
    video_length = audio_duration * 25  # InfiniteTalk assumes 25 FPS

    audio_feature = np.squeeze(
        wav2vec_feature_extractor(speech_array, sampling_rate=sr).input_values
    )
    audio_feature = torch.from_numpy(audio_feature).float().to(device=device)
    audio_feature = audio_feature.unsqueeze(0)

    with torch.no_grad():
        embeddings = audio_encoder(
            audio_feature, seq_len=int(video_length), output_hidden_states=True
        )

    if len(embeddings) == 0:
        raise RuntimeError("Failed to extract audio embedding")

    audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
    audio_emb = rearrange(audio_emb, "b s d -> s b d")
    audio_emb = audio_emb.cpu().detach()
    return audio_emb


# ---------------------------------------------------------------------------
# LoRA helpers (FP8-aware weight merging)
# ---------------------------------------------------------------------------


def _apply_lora_to_model(model, lora_path, lora_scale=1.0):
    """
    Apply a LoRA to a model, handling native FP8 (float8_e4m3fn) weights.

    For FP8 weights: dequantize using scale_weight, apply LoRA delta,
    re-quantize back to FP8. This works with both native FP8 tensors
    (from fp8_optimization.py) and regular float16/float32 weights.
    """
    from safetensors import safe_open
    from tqdm import tqdm

    # Load LoRA weights
    with safe_open(lora_path, framework="pt") as f:
        lora_weights = {key: f.get_tensor(key) for key in f.keys()}

    # Parse LoRA pairs (same logic as WanLoraWrapper)
    prefix = "diffusion_model."
    lora_pairs = {}
    for key in lora_weights:
        if key.endswith("lora_down.weight") and key.startswith(prefix):
            base_name = key[len(prefix) :].replace("lora_down.weight", "weight")
            up_key = key.replace("lora_down.weight", "lora_up.weight")
            if up_key in lora_weights:
                lora_pairs[base_name] = (key, up_key)
        elif key.endswith("diff_b") and key.startswith(prefix):
            base_name = key[len(prefix) :].replace("diff_b", "bias")
            lora_pairs[base_name] = (key,)
        elif key.endswith("diff") and key.startswith(prefix):
            base_name = key[len(prefix) :].replace("diff", "weight")
            lora_pairs[base_name] = (key,)

    def _apply_delta_to_param(parent, attr_name, delta):
        """Apply a LoRA delta to a parameter, handling native FP8."""
        param = getattr(parent, attr_name)
        is_native_fp8 = param.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)

        if is_native_fp8:
            # Native FP8: dequantize via per-tensor scale, add delta, re-quantize
            scale = getattr(parent, "scale_weight", None)
            if scale is not None:
                scale = scale.to(param.device)
                fp_weight = param.data.float() * scale
            else:
                fp_weight = param.data.float()
            delta = delta.to(fp_weight.device, fp_weight.dtype)
            fp_weight.add_(delta)
            # Re-quantize with new per-tensor scale (max may have changed)
            amax = fp_weight.abs().max().clamp(min=1e-12)
            new_scale = (amax / 448.0).float()
            new_data = (fp_weight / new_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
            param.data = new_data
            parent.scale_weight = new_scale
        else:
            # Regular float weight
            delta = delta.to(param.device, param.dtype)
            param.data.add_(delta)

    applied = 0
    skipped = 0
    for name in tqdm(lora_pairs, desc="Applying LoRA weights"):
        # Navigate to the parameter: split into parent path + final attr (weight/bias)
        parts = name.split(".")
        parent = model
        try:
            for part in parts[:-1]:
                parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
            param = getattr(parent, parts[-1])
        except (AttributeError, IndexError, KeyError) as e:
            logger.debug(f"  Skipping LoRA target {name}: {e}")
            skipped += 1
            continue

        if isinstance(lora_pairs[name], tuple) and len(lora_pairs[name]) == 2:
            # Standard LoRA: delta = matmul(up, down) * scale
            down_key, up_key = lora_pairs[name]
            lora_down = lora_weights[down_key].float()
            lora_up = lora_weights[up_key].float()
            delta = torch.matmul(lora_up, lora_down) * lora_scale
            _apply_delta_to_param(parent, parts[-1], delta)
        else:
            # Direct diff weight
            diff_key = lora_pairs[name][0]
            delta = lora_weights[diff_key].float() * lora_scale
            _apply_delta_to_param(parent, parts[-1], delta)
        applied += 1

    logger.info(f"Applied {applied} LoRA weight adjustments (native FP8)")
    if skipped > 0:
        logger.info(f"Skipped {skipped} LoRA targets (no matching model param)")
    if applied == 0:
        logger.warning("No LoRA weights were applied. Check LoRA file compatibility.")
    return applied


# ---------------------------------------------------------------------------
# Video Engine class
# ---------------------------------------------------------------------------


class VideoEngine:
    """
    InfiniteTalk video generation engine.

    Supports cartoon/3D/photo input, arbitrary aspect ratios (follows input
    image dimensions), streaming mode for unlimited-length video, and
    CPU offloading + FP8 quantization for 16GB VRAM GPUs.

    Usage:
        engine = VideoEngine()
        engine.load_model()
        video_path = engine.generate(
            image_path="portrait.png",
            audio_path="speech.wav",
            output_path="output.mp4",
        )
        engine.unload_model()
    """

    # Fixed prompt used for all generations
    _PROMPT = (
        "A person is talking naturally with expressive facial movements and gestures."
    )

    def __init__(self, device=None):
        self.device = device or config.DEVICE
        self.use_lora = config.VIDEO_USE_LORA
        self.pipe = None
        self.wav2vec_feature_extractor = None
        self.audio_encoder = None
        self._loaded = False
        self._lora_applied = False
        self._cached_text_embeddings = None

    @property
    def is_loaded(self):
        return self._loaded and self.pipe is not None

    def _pre_encode_text(self):
        """Pre-encode the text prompt on GPU while VRAM is free.

        FP8-quantized T5 on CPU is catastrophically slow (no FP8 hardware
        support — every op dequantizes in software). By encoding on GPU
        BEFORE block_swap fills VRAM, we get fast native FP8 inference.
        The cached embeddings are stored on CPU and passed to
        generate_infinitetalk() to skip T5 entirely during generation.
        """
        prompt = self._PROMPT
        n_prompt = self.pipe.sample_neg_prompt

        # Move T5 to GPU, encode both prompts together (matches vendor format),
        # then unpack. T5.__call__([a, b], device) returns [tensor_a, tensor_b].
        self.pipe.text_encoder.model.to(self.device)
        with torch.no_grad():
            context, context_null = self.pipe.text_encoder(
                [prompt, n_prompt], self.device
            )
        self.pipe.text_encoder.model.cpu()
        torch.cuda.empty_cache()

        # Cache on CPU to avoid holding GPU memory.
        # context/context_null are individual tensors; store as-is.
        self._cached_text_embeddings = (
            context.cpu(),
            context_null.cpu(),
        )
        logger.info("  T5 text embeddings pre-encoded and cached")

    def load_model(self):
        """Load InfiniteTalk pipeline. Downloads weights on first call."""
        if self.is_loaded:
            logger.info("InfiniteTalk already loaded")
            return

        # Ensure models are downloaded
        model_manager.ensure_infinitetalk_models()

        # Ensure vendor source is importable
        _ensure_vendor_path()

        logger.info("Loading InfiniteTalk pipeline...")
        start = time.time()

        import wan
        from wan.configs import WAN_CONFIGS
        from transformers import Wav2Vec2FeatureExtractor
        from src.audio_analysis.wav2vec2 import Wav2Vec2Model

        # --- Load wav2vec2 audio encoder ---
        logger.info("  Loading wav2vec2 audio encoder...")
        wav2vec_dir = config.WAV2VEC_DIR
        self.audio_encoder = Wav2Vec2Model.from_pretrained(
            wav2vec_dir, local_files_only=True, attn_implementation="eager"
        ).to("cpu")
        self.wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            wav2vec_dir, local_files_only=True
        )

        # --- Load InfiniteTalk pipeline ---
        logger.info("  Loading InfiniteTalk Wan2.1-I2V-14B pipeline...")
        task = "infinitetalk-14B"
        cfg = WAN_CONFIGS[task]

        # Use native FP8 (torch._scaled_mm) instead of quanto for the DiT.
        # This gives hardware-accelerated FP8 matmul on Ada/Hopper GPUs
        # instead of quanto's software dequantize-per-forward approach.
        use_native_fp8 = config.VIDEO_QUANT == "fp8" and os.path.isfile(
            config.INFINITETALK_FP8_CKPT
        )
        quant = "native_fp8" if use_native_fp8 else config.VIDEO_QUANT
        quant_dir = config.INFINITETALK_FP8_CKPT if use_native_fp8 else None
        if not use_native_fp8 and config.VIDEO_QUANT == "fp8":
            quant_dir = config.INFINITETALK_FP8_CKPT

        self.pipe = wan.InfiniteTalkPipeline(
            config=cfg,
            checkpoint_dir=config.WAN_CKPT_DIR,
            quant_dir=quant_dir,
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_usp=False,
            t5_cpu=True,
            lora_dir=None,
            lora_scales=[1.0],
            quant=quant,
            dit_path=None,
            infinitetalk_dir=config.INFINITETALK_SINGLE_CKPT,
        )

        # Load DiT weights natively as FP8 tensors with scaled_mm
        if use_native_fp8:
            logger.info("  Loading DiT weights as native FP8 (torch._scaled_mm)...")
            fp8_start = time.time()
            fp8_optimization.load_fp8_weights_native(
                self.pipe.model,
                config.INFINITETALK_FP8_CKPT,
                base_dtype=self.pipe.param_dtype,
            )
            self.pipe.model.eval().requires_grad_(False)
            # Convert non-FP8 float32 params (biases, norms, Conv3d) to param_dtype.
            # Must happen AFTER FP8 weight loading (model was on meta before).
            # Skip scale_weight attributes — _scaled_mm requires float32 scales.
            for module in self.pipe.model.modules():
                for name, param in module.named_parameters(recurse=False):
                    if param.dtype == torch.float32 and name != "scale_weight":
                        param.data = param.data.to(self.pipe.param_dtype)
                for name, buf in module.named_buffers(recurse=False):
                    if buf.dtype == torch.float32:
                        module._buffers[name] = buf.to(self.pipe.param_dtype)
            fp8_elapsed = time.time() - fp8_start
            logger.info(f"  Native FP8 weights loaded in {fp8_elapsed:.1f}s")

        # Pre-encode text prompt on GPU BEFORE block_swap fills VRAM.
        # FP8-quantized T5 on CPU is catastrophically slow (no FP8 HW support).
        # At this point only T5+VAE+CLIP are loaded; plenty of GPU room for T5.
        logger.info("  Pre-encoding text prompt on GPU (avoids slow FP8 on CPU)...")
        self._pre_encode_text()

        # Block swap: offload VAE/CLIP/T5 to CPU first so auto-tune sees max free VRAM.
        # CLIP is loaded back to GPU on-demand by the vendor code (explicit
        # .to(device) in generate_infinitetalk). VAE must be restored here
        # because the vendor code assumes it stays on GPU.
        logger.info("  Offloading VAE/CLIP to CPU for VRAM headroom...")
        self.pipe.vae.model.cpu()
        self.pipe.clip.model.cpu()
        self.pipe.text_encoder.model.cpu()
        torch.cuda.empty_cache()

        blocks_to_swap = config.VIDEO_BLOCKS_TO_SWAP
        if blocks_to_swap is None:
            blocks_to_swap = config.auto_tune_blocks_to_swap()
        prefetch = config.VIDEO_PREFETCH_BLOCKS
        num_blocks = len(self.pipe.model.blocks)
        persistent = num_blocks - blocks_to_swap
        logger.info(
            f"  DiT blocks: {num_blocks} total, "
            f"{persistent} on GPU, {blocks_to_swap} swapped "
            f"(prefetch={prefetch})"
        )

        if blocks_to_swap > 0:
            # Use block-level swapping with async prefetch (Kijai approach).
            # This replaces the vendor's per-layer enable_vram_management().
            self.pipe.model.block_swap(
                blocks_to_swap=blocks_to_swap,
                main_device=torch.device(self.device),
                prefetch_blocks=prefetch,
            )
            # Tell vendor code we handle VRAM ourselves — skip its per-layer wrapping
            self.pipe.vram_management = False
        else:
            # All blocks fit on GPU — move entire model there
            self.pipe.model.to(self.device)
            self.pipe.vram_management = False

        # VAE stays on CPU — the generation loop moves it to GPU on-demand
        # for encode/decode and offloads it during denoising to free VRAM.

        # Apply LoRA accelerator if enabled
        if self.use_lora:
            lora_path = config.LIGHTX2V_LORA_PATH
            if not os.path.isfile(lora_path):
                logger.info("  Downloading LightX2V LoRA...")
                model_manager.ensure_lora_models()
            if os.path.isfile(lora_path):
                logger.info(
                    f"  Applying LightX2V LoRA (scale={config.VIDEO_LORA_SCALE})..."
                )
                lora_start = time.time()
                count = _apply_lora_to_model(
                    self.pipe.model, lora_path, lora_scale=config.VIDEO_LORA_SCALE
                )
                lora_elapsed = time.time() - lora_start
                logger.info(f"  LoRA applied ({count} weights) in {lora_elapsed:.1f}s")
                self._lora_applied = True
            else:
                logger.warning("  LightX2V LoRA file not found, running without LoRA")
                self._lora_applied = False

        elapsed = time.time() - start
        logger.info(f"InfiniteTalk loaded in {elapsed:.1f}s")
        self._loaded = True

    def unload_model(self):
        """Unload all models and free GPU memory."""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        if self.audio_encoder is not None:
            del self.audio_encoder
            self.audio_encoder = None
        if self.wav2vec_feature_extractor is not None:
            del self.wav2vec_feature_extractor
            self.wav2vec_feature_extractor = None
        self._loaded = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("InfiniteTalk unloaded, GPU memory freed")

    def generate(
        self,
        image_path,
        audio_path,
        output_path=None,
        size=None,
        sample_steps=None,
        seed=None,
    ):
        """
        Generate a talking-head video from a portrait image and audio file.

        Video length matches audio length. Video resolution matches image aspect ratio.

        Args:
            image_path: Path to the portrait image. Aspect ratio is preserved.
            audio_path: Path to the audio WAV file.
            output_path: Where to save the final MP4. Auto-generated if None.
            size: Resolution bucket ("infinitetalk-480" or "infinitetalk-720").
            sample_steps: Number of denoising steps (default 20).
            seed: Random seed for reproducibility.

        Returns:
            Path to the generated MP4 video file.
        """
        if not self.is_loaded:
            self.load_model()

        # Ensure output directory exists
        config.ensure_directories()

        # Apply defaults — override with LoRA params when active
        size = size or config.VIDEO_SIZE
        seed = seed if seed is not None else config.VIDEO_SEED
        mode = config.VIDEO_MODE
        motion_frame = config.VIDEO_MOTION_FRAME
        frame_num = config.VIDEO_FRAME_NUM
        prompt = self._PROMPT

        if self._lora_applied:
            sample_steps = sample_steps or config.VIDEO_LORA_SAMPLE_STEPS
            text_guide_scale = config.VIDEO_LORA_TEXT_GUIDE_SCALE
            audio_guide_scale = config.VIDEO_LORA_AUDIO_GUIDE_SCALE
        else:
            sample_steps = sample_steps or config.VIDEO_SAMPLE_STEPS
            text_guide_scale = config.VIDEO_TEXT_GUIDE_SCALE
            audio_guide_scale = config.VIDEO_AUDIO_GUIDE_SCALE

        # Compute sample_shift based on size and LoRA
        if self._lora_applied:
            sample_shift = config.VIDEO_LORA_SAMPLE_SHIFT
        elif size == "infinitetalk-480":
            sample_shift = 7
        elif size == "infinitetalk-720":
            sample_shift = 11
        else:
            raise ValueError(f"Unsupported size: {size}")

        # Auto-generate output path
        if output_path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(
                config.VIDEO_OUTPUT_DIR, f"video_{timestamp}.mp4"
            )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # --- Prepare audio ---
        logger.info(f"Preparing audio: {audio_path}")
        speech_array = _prepare_audio(audio_path, sample_rate=config.VIDEO_SAMPLE_RATE)

        # Save normalized audio for embedding extraction (16kHz mono)
        tmp_dir = tempfile.mkdtemp()
        sum_audio_path = os.path.join(tmp_dir, "sum_all.wav")
        sf.write(sum_audio_path, speech_array, config.VIDEO_SAMPLE_RATE)

        # Extract audio embedding
        logger.info("Extracting wav2vec2 audio embedding...")
        audio_embedding = _get_audio_embedding(
            speech_array,
            self.wav2vec_feature_extractor,
            self.audio_encoder,
            sr=config.VIDEO_SAMPLE_RATE,
            device="cpu",
        )
        emb_path = os.path.join(tmp_dir, "audio_emb.pt")
        torch.save(audio_embedding, emb_path)

        audio_frames = audio_embedding.shape[0]
        audio_duration = len(speech_array) / config.VIDEO_SAMPLE_RATE
        logger.info(f"Audio: {audio_duration:.1f}s, {audio_frames} embedding frames")

        # --- Build input clip dict (InfiniteTalk's expected format) ---
        input_clip = {
            "prompt": prompt,
            "cond_video": image_path,
            "cond_audio": {"person1": emb_path},
            "video_audio": sum_audio_path,
        }

        # --- Build extra_args namespace ---
        class ExtraArgs:
            pass

        extra_args = ExtraArgs()
        extra_args.use_teacache = False
        extra_args.teacache_thresh = 0.2
        extra_args.size = size
        extra_args.use_apg = False
        extra_args.apg_momentum = -0.75
        extra_args.apg_norm_threshold = 55
        extra_args.scene_seg = False

        # --- Compute video length from audio duration ---
        # Video always matches audio length. frame_num is only the chunk size.
        audio_driven_frames = int(audio_duration * config.VIDEO_FPS) + 1
        effective_max = audio_driven_frames

        # The vendor code requires audio_embedding.shape[0] > frame_num,
        # so shrink frame_num to fit short audio (must stay 4n+1).
        if audio_frames <= frame_num:
            frame_num = max(((audio_frames - 2) // 4) * 4 + 1, 5)  # 4n+1, min 5
            logger.info(
                f"  Audio shorter than chunk size → frame_num reduced to {frame_num}"
            )

        # Ensure at least frame_num frames so the first chunk can run
        effective_max = max(effective_max, frame_num)
        logger.info(
            f"  Video: {audio_driven_frames} frames "
            f"({audio_duration:.1f}s × {config.VIDEO_FPS}fps)"
        )
        # --- ETA estimate ---
        if mode == "streaming" and effective_max > frame_num:
            num_chunks = 1 + math.ceil((effective_max - frame_num) / motion_frame)
        else:
            num_chunks = 1
        total_denoise_steps = num_chunks * sample_steps
        eta_secs = total_denoise_steps * config.VIDEO_SECS_PER_STEP
        eta_m, eta_s = divmod(int(eta_secs), 60)
        eta_h, eta_m = divmod(eta_m, 60)
        eta_str = f"{eta_h}h {eta_m}m" if eta_h else f"{eta_m}m {eta_s}s"
        logger.info(
            f"Running InfiniteTalk: size={size}, "
            f"steps={sample_steps}, mode={mode}, seed={seed}, "
            f"text_cfg={text_guide_scale}, audio_cfg={audio_guide_scale}"
        )
        logger.info(
            f"  ETA ≈ {eta_str} "
            f"({num_chunks} chunk{'s' if num_chunks != 1 else ''} "
            f"× {sample_steps} steps, "
            f"~{config.VIDEO_SECS_PER_STEP}s/step)"
        )
        gen_start = time.time()

        video = self.pipe.generate_infinitetalk(
            input_clip,
            size_buckget=size,
            motion_frame=motion_frame,
            frame_num=frame_num,
            shift=sample_shift,
            sampling_steps=sample_steps,
            text_guide_scale=text_guide_scale,
            audio_guide_scale=audio_guide_scale,
            seed=seed,
            offload_model=True,
            max_frames_num=effective_max,
            color_correction_strength=1.0,
            extra_args=extra_args,
            pre_encoded_text=self._cached_text_embeddings,
        )

        gen_elapsed = time.time() - gen_start
        logger.info(f"Inference completed in {gen_elapsed:.1f}s")

        # --- Save video with audio ---
        _ensure_vendor_path()
        from wan.utils.multitalk_utils import save_video_ffmpeg

        save_file = output_path.replace(".mp4", "")
        save_video_ffmpeg(video, save_file, [audio_path], high_quality_save=False)

        # save_video_ffmpeg appends .mp4 to save_file
        generated_path = save_file + ".mp4"
        if generated_path != output_path and os.path.isfile(generated_path):
            import shutil

            shutil.move(generated_path, output_path)

        # Clean up temp files
        try:
            import shutil as _shutil

            _shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass

        num_frames = video.shape[1] if video is not None else 0
        fps = config.VIDEO_FPS
        duration = num_frames / fps if fps > 0 else 0
        logger.info(
            f"Video saved: {output_path} "
            f"({num_frames} frames, {duration:.1f}s @ {fps} FPS)"
        )

        return output_path
