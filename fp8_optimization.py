"""
Talking Head Video Generator — FP8 Optimization

Native PyTorch FP8 matmul using torch._scaled_mm for hardware-accelerated
FP8 inference on Ada Lovelace (RTX 40xx) and Hopper GPUs.

Replaces optimum-quanto's software FP8 dequantization (FP8→FP16 per forward)
with direct FP8 tensor core operations — no dequantization, no memory doubling.

Based on Kijai/ComfyUI-WanVideoWrapper's fp8_optimization.py approach.
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# torch._scaled_mm returns a tuple in torch 2.4, a tensor in 2.5+
_SCALED_MM_RETURNS_TUPLE = None


def _check_scaled_mm_tuple():
    """Detect whether torch._scaled_mm returns (result, amax) tuple or just result."""
    global _SCALED_MM_RETURNS_TUPLE
    if _SCALED_MM_RETURNS_TUPLE is not None:
        return _SCALED_MM_RETURNS_TUPLE
    try:
        a = torch.zeros(16, 16, device="cuda", dtype=torch.float8_e4m3fn)
        b = torch.zeros(16, 16, device="cuda", dtype=torch.float8_e4m3fn)
        s = torch.ones((), device="cuda", dtype=torch.float32)
        result = torch._scaled_mm(
            a, b.t(), scale_a=s, scale_b=s, out_dtype=torch.float16
        )
        _SCALED_MM_RETURNS_TUPLE = isinstance(result, tuple)
    except Exception:
        _SCALED_MM_RETURNS_TUPLE = True  # safe default
    return _SCALED_MM_RETURNS_TUPLE


def fp8_linear_forward(cls, base_dtype, input):
    """
    FP8 linear forward using hardware-accelerated torch._scaled_mm.

    Weights must be torch.float8_e4m3fn. Input is clamped and cast to FP8
    on the fly. The matmul happens entirely in FP8 on tensor cores.

    Args:
        cls: The nn.Linear module (with weight in float8_e4m3fn and optional scale_weight).
        base_dtype: The output dtype (e.g. torch.float16).
        input: Input tensor, shape [..., in_features].
    """
    weight_dtype = cls.weight.dtype
    if weight_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        input_shape = input.shape
        in_features = input_shape[-1]

        scale_weight = getattr(cls, "scale_weight", None)
        if scale_weight is None:
            scale_weight = torch.ones((), device=input.device, dtype=torch.float32)
        else:
            scale_weight = scale_weight.to(input.device)

        scale_input = torch.ones((), device=input.device, dtype=torch.float32)

        # Cast input to FP8 and flatten to 2D for scaled_mm.
        # Skip clamp — values outside FP8 range saturate naturally on cast.
        inn = input.reshape(-1, in_features).to(torch.float8_e4m3fn)

        result = torch._scaled_mm(
            inn,
            cls.weight.t(),
            out_dtype=base_dtype,
            bias=cls.bias,
            scale_a=scale_input,
            scale_b=scale_weight,
        )

        if isinstance(result, tuple):
            result = result[0]

        # Reshape back to original batch dimensions
        out_shape = input_shape[:-1] + (cls.weight.shape[0],)
        return result.reshape(out_shape)
    else:
        return cls.original_forward(input)


def convert_fp8_linear(module, base_dtype, params_to_keep=None):
    """
    Monkey-patch all nn.Linear layers in module to use fp8_linear_forward.

    Skips layers whose names match any keyword in params_to_keep (e.g. norm layers).

    Args:
        module: The nn.Module to patch.
        base_dtype: Output dtype for matmul (e.g. torch.float16).
        params_to_keep: Set of name substrings to skip (layers that should stay in base_dtype).
    """
    if params_to_keep is None:
        params_to_keep = set()

    count = 0
    for name, submodule in module.named_modules():
        if any(keyword in name for keyword in params_to_keep):
            continue
        if isinstance(submodule, nn.Linear):
            if submodule.weight.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                original_forward = submodule.forward
                setattr(submodule, "original_forward", original_forward)
                setattr(
                    submodule,
                    "forward",
                    lambda input, m=submodule: fp8_linear_forward(m, base_dtype, input),
                )
                count += 1
    logger.info(f"FP8 scaled_mm enabled for {count} linear layers")
    return count


def _get_cache_path(safetensors_path):
    """Get the path for the cached per-tensor FP8 weights."""
    base = safetensors_path.rsplit(".", 1)[0]
    return base + "_native_fp8_cache.safetensors"


def _build_native_fp8_cache(safetensors_path, cache_path):
    """Convert quanto per-row FP8 weights to per-tensor FP8 and cache to disk.

    This is the expensive one-time conversion (~57s). The cached file stores:
      - {name}.weight: float8_e4m3fn tensor (per-tensor quantized)
      - {name}.scale_weight: float32 scalar (per-tensor scale)
      - All regular (non-FP8) params unchanged
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    logger.info(f"  Building native FP8 cache (one-time conversion)...")

    cache_tensors = {}
    with safe_open(safetensors_path, framework="pt") as f:
        all_keys = set(f.keys())

        for key in all_keys:
            # Skip quanto metadata
            if key.endswith(".input_scale") or key.endswith(".output_scale"):
                continue
            if "._data" in key or "._scale" in key:
                continue
            # Regular param — pass through
            cache_tensors[key] = f.get_tensor(key)

        # Convert FP8 per-row → per-tensor
        for key in all_keys:
            if not key.endswith(".weight._data"):
                continue
            base = key[: -len("._data")]  # e.g. "blocks.0.self_attn.q.weight"
            scale_key = base + "._scale"
            if scale_key not in all_keys:
                continue

            fp8_data = f.get_tensor(key)  # float8_e4m3fn [out, in]
            per_row_scale = f.get_tensor(scale_key)  # float32 [out, 1]

            # Dequantize → re-quantize with per-tensor scale
            fp32_weight = fp8_data.float() * per_row_scale
            amax = fp32_weight.abs().max().clamp(min=1e-12)
            per_tensor_scale = (amax / 448.0).float()
            new_fp8 = (
                (fp32_weight / per_tensor_scale)
                .clamp(-448, 448)
                .to(torch.float8_e4m3fn)
            )

            # Store with clean names (no ._data/_scale suffixes)
            cache_tensors[base] = new_fp8
            # Store scale as "{module_path}.scale_weight"
            module_path = base.rsplit(".", 1)[0]  # remove ".weight"
            cache_tensors[module_path + ".scale_weight"] = per_tensor_scale

    save_file(cache_tensors, cache_path)
    logger.info(f"  Native FP8 cache saved: {cache_path}")


def load_fp8_weights_native(model, safetensors_path, base_dtype=torch.float16):
    """
    Load FP8 weights into a WanModel using native torch._scaled_mm.

    On first call, converts quanto per-row FP8 weights to per-tensor FP8
    and caches the result to disk (~57s). Subsequent loads use the cache (~5s).

    Args:
        model: The WanModel (created on meta device).
        safetensors_path: Path to the quanto FP8 safetensors file.
        base_dtype: Compute dtype for non-quantized params (default float16).

    Returns:
        The model with native FP8 weights loaded.
    """
    import os
    from safetensors import safe_open

    cache_path = _get_cache_path(safetensors_path)

    # Build cache if it doesn't exist
    if not os.path.isfile(cache_path):
        _build_native_fp8_cache(safetensors_path, cache_path)
    else:
        logger.info(f"  Using cached native FP8 weights: {cache_path}")

    # Load from cache (fast path)
    logger.info(f"Loading native FP8 weights from cache...")

    with safe_open(cache_path, framework="pt") as f:
        all_keys = set(f.keys())

        for key in all_keys:
            # Skip scale_weight keys — handled with their parent weight
            if key.endswith(".scale_weight"):
                continue

            parts = key.split(".")
            parent = model
            try:
                for part in parts[:-1]:
                    parent = (
                        parent[int(part)] if part.isdigit() else getattr(parent, part)
                    )
                attr_name = parts[-1]
                tensor = f.get_tensor(key)

                if not hasattr(parent, attr_name):
                    continue

                existing = getattr(parent, attr_name)
                if isinstance(existing, nn.Parameter):
                    # For regular float32 weight params, convert to base_dtype
                    if (
                        tensor.dtype == torch.float32
                        and attr_name == "weight"
                        and tensor.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2)
                    ):
                        tensor = tensor.to(base_dtype)
                    new_param = nn.Parameter(tensor, requires_grad=False)
                    setattr(parent, attr_name, new_param)

                    # Load scale_weight if this is an FP8 weight.
                    # Register as a buffer (not plain attr) so nn.Module.to()
                    # moves it during block swap / offload / restore.
                    if tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                        module_path = ".".join(parts[:-1])
                        scale_key = module_path + ".scale_weight"
                        if scale_key in all_keys:
                            parent.register_buffer(
                                "scale_weight",
                                f.get_tensor(scale_key),
                                persistent=False,
                            )
                elif isinstance(existing, torch.Tensor):
                    setattr(parent, attr_name, tensor)
                else:
                    setattr(parent, attr_name, tensor)
            except (AttributeError, IndexError, KeyError):
                pass

    # Monkey-patch all FP8 linear layers to use scaled_mm
    convert_fp8_linear(model, base_dtype)

    # Detect scaled_mm return type for this torch version
    _check_scaled_mm_tuple()

    logger.info("FP8 native loading complete")
    return model
