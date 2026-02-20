"""
Talking Head Video Generator — GGUF Model Loader

On-demand dequantization for GGUF-quantized models. Loads quantized weights
and dequantizes per-layer during forward passes, keeping only one layer's
worth of float16 data in VRAM at a time.

Based on city96/ComfyUI-GGUF (Apache 2.0 license).
Dequantization functions ported from ComfyUI-GGUF/dequant.py.
"""

import logging
import warnings

import gguf
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GGML quant type helpers
# ---------------------------------------------------------------------------

TORCH_COMPATIBLE_QTYPES = {
    None,
    gguf.GGMLQuantizationType.F32,
    gguf.GGMLQuantizationType.F16,
}


def is_quantized(tensor):
    """Check if a tensor/GGMLTensor is GGML-quantized (not F32/F16)."""
    qtype = getattr(tensor, "tensor_type", None)
    return qtype not in TORCH_COMPATIBLE_QTYPES


# ---------------------------------------------------------------------------
# GGMLTensor — torch.Tensor subclass carrying quantization metadata
# ---------------------------------------------------------------------------


class GGMLTensor(torch.Tensor):
    """
    Tensor subclass that carries GGML quantization metadata.

    The underlying data is the raw packed quantized bytes. The `tensor_shape`
    property returns the original (unquantized) shape so that PyTorch sees
    the correct dimensions for shape checks.
    """

    def __new__(cls, data, tensor_type=None, tensor_shape=None):
        return super().__new__(cls, data)

    def __init__(self, data, tensor_type=None, tensor_shape=None):
        super().__init__()
        self.tensor_type = tensor_type
        self.tensor_shape = tensor_shape

    def to(self, *args, **kwargs):
        new = super().to(*args, **kwargs)
        new.tensor_type = getattr(self, "tensor_type", None)
        new.tensor_shape = getattr(self, "tensor_shape", new.data.shape)
        return new

    def clone(self, *args, **kwargs):
        return self

    def detach(self, *args, **kwargs):
        return self

    @property
    def shape(self):
        if not hasattr(self, "tensor_shape"):
            self.tensor_shape = self.size()
        return self.tensor_shape


# ---------------------------------------------------------------------------
# Dequantization functions (ported from ComfyUI-GGUF dequant.py, Apache 2.0)
# ---------------------------------------------------------------------------

QK_K = 256
K_SCALE_SIZE = 12


def _split_block_dims(blocks, *args):
    n_max = blocks.shape[1]
    dims = list(args) + [n_max - sum(args)]
    return torch.split(blocks, dims, dim=1)


def _to_uint32(x):
    x = x.view(torch.uint8).to(torch.int32)
    return (x[:, 0] | x[:, 1] << 8 | x[:, 2] << 16 | x[:, 3] << 24).unsqueeze(1)


def _get_scale_min(scales):
    n_blocks = scales.shape[0]
    scales = scales.view(torch.uint8)
    scales = scales.reshape((n_blocks, 3, 4))
    d, m, m_d = torch.split(scales, scales.shape[-2] // 3, dim=-2)
    sc = torch.cat([d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)], dim=-1)
    mn = torch.cat([m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)], dim=-1)
    return (sc.reshape((n_blocks, 8)), mn.reshape((n_blocks, 8)))


# --- Full precision ---


def _dequant_BF16(blocks, block_size, type_size, dtype=None):
    return (blocks.view(torch.int16).to(torch.int32) << 16).view(torch.float32)


# --- Legacy quants ---


def _dequant_Q8_0(blocks, block_size, type_size, dtype=None):
    d, x = _split_block_dims(blocks, 2)
    d = d.view(torch.float16).to(dtype)
    x = x.view(torch.int8)
    return d * x


def _dequant_Q4_0(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, qs = _split_block_dims(blocks, 2)
    d = d.view(torch.float16).to(dtype)
    qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8
    ).reshape((1, 1, 2, 1))
    qs = (qs & 0x0F).reshape((n_blocks, -1)).to(torch.int8) - 8
    return d * qs


def _dequant_Q4_1(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, m, qs = _split_block_dims(blocks, 2, 2)
    d = d.view(torch.float16).to(dtype)
    m = m.view(torch.float16).to(dtype)
    qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8
    ).reshape(1, 1, 2, 1)
    qs = (qs & 0x0F).reshape(n_blocks, -1)
    return (d * qs) + m


def _dequant_Q5_0(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, qh, qs = _split_block_dims(blocks, 2, 4)
    d = d.view(torch.float16).to(dtype)
    qh = _to_uint32(qh)
    qh = qh.reshape(n_blocks, 1) >> torch.arange(
        32, device=d.device, dtype=torch.int32
    ).reshape(1, 32)
    ql = qs.reshape(n_blocks, -1, 1, block_size // 2) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8
    ).reshape(1, 1, 2, 1)
    qh = (qh & 1).to(torch.uint8)
    ql = (ql & 0x0F).reshape(n_blocks, -1)
    qs = (ql | (qh << 4)).to(torch.int8) - 16
    return d * qs


def _dequant_Q5_1(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, m, qh, qs = _split_block_dims(blocks, 2, 2, 4)
    d = d.view(torch.float16).to(dtype)
    m = m.view(torch.float16).to(dtype)
    qh = _to_uint32(qh)
    qh = qh.reshape((n_blocks, 1)) >> torch.arange(
        32, device=d.device, dtype=torch.int32
    ).reshape(1, 32)
    ql = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8
    ).reshape(1, 1, 2, 1)
    qh = (qh & 1).to(torch.uint8)
    ql = (ql & 0x0F).reshape((n_blocks, -1))
    qs = ql | (qh << 4)
    return (d * qs) + m


# --- K-quants ---


def _dequant_Q2_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    scales, qs, d, dmin = _split_block_dims(blocks, QK_K // 16, QK_K // 4, 2)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)
    dl = (d * (scales & 0xF)).reshape((n_blocks, QK_K // 16, 1))
    ml = (dmin * (scales >> 4)).reshape((n_blocks, QK_K // 16, 1))
    shift = torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape(
        (1, 1, 4, 1)
    )
    qs = (qs.reshape((n_blocks, -1, 1, 32)) >> shift) & 3
    qs = qs.reshape((n_blocks, QK_K // 16, 16))
    qs = dl * qs - ml
    return qs.reshape((n_blocks, -1))


def _dequant_Q3_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    hmask, qs, scales, d = _split_block_dims(blocks, QK_K // 8, QK_K // 4, 12)
    d = d.view(torch.float16).to(dtype)
    lscales, hscales = scales[:, :8], scales[:, 8:]
    lscales = lscales.reshape((n_blocks, 1, 8)) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8
    ).reshape((1, 2, 1))
    lscales = lscales.reshape((n_blocks, 16))
    hscales = hscales.reshape((n_blocks, 1, 4)) >> torch.tensor(
        [0, 2, 4, 6], device=d.device, dtype=torch.uint8
    ).reshape((1, 4, 1))
    hscales = hscales.reshape((n_blocks, 16))
    scales = (lscales & 0x0F) | ((hscales & 0x03) << 4)
    scales = scales.to(torch.int8) - 32
    dl = (d * scales).reshape((n_blocks, 16, 1))
    ql = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor(
        [0, 2, 4, 6], device=d.device, dtype=torch.uint8
    ).reshape((1, 1, 4, 1))
    qh = hmask.reshape(n_blocks, -1, 1, 32) >> torch.tensor(
        [i for i in range(8)], device=d.device, dtype=torch.uint8
    ).reshape((1, 1, 8, 1))
    ql = ql.reshape((n_blocks, 16, QK_K // 16)) & 3
    qh = (qh.reshape((n_blocks, 16, QK_K // 16)) & 1) ^ 1
    q = ql.to(torch.int8) - (qh << 2).to(torch.int8)
    return (dl * q).reshape((n_blocks, QK_K))


def _dequant_Q4_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, dmin, scales, qs = _split_block_dims(blocks, 2, 2, K_SCALE_SIZE)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)
    sc, m = _get_scale_min(scales)
    d = (d * sc).reshape((n_blocks, -1, 1))
    dm = (dmin * m).reshape((n_blocks, -1, 1))
    qs = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8
    ).reshape((1, 1, 2, 1))
    qs = (qs & 0x0F).reshape((n_blocks, -1, 32))
    return (d * qs - dm).reshape((n_blocks, QK_K))


def _dequant_Q5_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, dmin, scales, qh, qs = _split_block_dims(blocks, 2, 2, K_SCALE_SIZE, QK_K // 8)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)
    sc, m = _get_scale_min(scales)
    d = (d * sc).reshape((n_blocks, -1, 1))
    dm = (dmin * m).reshape((n_blocks, -1, 1))
    ql = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8
    ).reshape((1, 1, 2, 1))
    qh = qh.reshape((n_blocks, -1, 1, 32)) >> torch.tensor(
        [i for i in range(8)], device=d.device, dtype=torch.uint8
    ).reshape((1, 1, 8, 1))
    ql = (ql & 0x0F).reshape((n_blocks, -1, 32))
    qh = (qh & 0x01).reshape((n_blocks, -1, 32))
    q = ql | (qh << 4)
    return (d * q - dm).reshape((n_blocks, QK_K))


def _dequant_Q6_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    ql, qh, scales, d = _split_block_dims(blocks, QK_K // 2, QK_K // 4, QK_K // 16)
    scales = scales.view(torch.int8).to(dtype)
    d = d.view(torch.float16).to(dtype)
    d = (d * scales).reshape((n_blocks, QK_K // 16, 1))
    ql = ql.reshape((n_blocks, -1, 1, 64)) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8
    ).reshape((1, 1, 2, 1))
    ql = (ql & 0x0F).reshape((n_blocks, -1, 32))
    qh = qh.reshape((n_blocks, -1, 1, 32)) >> torch.tensor(
        [0, 2, 4, 6], device=d.device, dtype=torch.uint8
    ).reshape((1, 1, 4, 1))
    qh = (qh & 0x03).reshape((n_blocks, -1, 32))
    q = (ql | (qh << 4)).to(torch.int8) - 32
    q = q.reshape((n_blocks, QK_K // 16, -1))
    return (d * q).reshape((n_blocks, QK_K))


# Dispatch table
_DEQUANT_FUNCTIONS = {
    gguf.GGMLQuantizationType.BF16: _dequant_BF16,
    gguf.GGMLQuantizationType.Q8_0: _dequant_Q8_0,
    gguf.GGMLQuantizationType.Q4_0: _dequant_Q4_0,
    gguf.GGMLQuantizationType.Q4_1: _dequant_Q4_1,
    gguf.GGMLQuantizationType.Q5_0: _dequant_Q5_0,
    gguf.GGMLQuantizationType.Q5_1: _dequant_Q5_1,
    gguf.GGMLQuantizationType.Q2_K: _dequant_Q2_K,
    gguf.GGMLQuantizationType.Q3_K: _dequant_Q3_K,
    gguf.GGMLQuantizationType.Q4_K: _dequant_Q4_K,
    gguf.GGMLQuantizationType.Q5_K: _dequant_Q5_K,
    gguf.GGMLQuantizationType.Q6_K: _dequant_Q6_K,
}


def dequantize_tensor(tensor, dtype=None):
    """Dequantize a GGMLTensor to the target dtype."""
    qtype = getattr(tensor, "tensor_type", None)
    oshape = getattr(tensor, "tensor_shape", tensor.shape)
    return dequantize_weight(tensor, qtype, oshape, dtype)


def dequantize_weight(data, qtype, oshape, dtype=None):
    """Dequantize packed weight data given explicit metadata.

    Unlike dequantize_tensor(), this does NOT rely on tensor attributes.
    Safe to use after .to(device) which strips GGMLTensor metadata.
    """
    if qtype in TORCH_COMPATIBLE_QTYPES:
        return data.view(*oshape).to(dtype)
    elif qtype in _DEQUANT_FUNCTIONS:
        return _dequantize(data, qtype, oshape, dtype).to(dtype)
    else:
        # Fallback to numpy (slow but handles all quant types)
        logger.warning(
            f"Falling back to numpy dequant for qtype: "
            f"{getattr(qtype, 'name', repr(qtype))}"
        )
        new = gguf.quants.dequantize(data.cpu().numpy(), qtype)
        return torch.from_numpy(new).to(data.device, dtype=dtype)


def _dequantize(data, qtype, oshape, dtype=None):
    """Core dequantization: reshape packed bytes → blocks → dequant → reshape."""
    block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
    dequant_fn = _DEQUANT_FUNCTIONS[qtype]
    rows = data.reshape((-1, data.shape[-1])).view(torch.uint8)
    n_blocks = rows.numel() // type_size
    blocks = rows.reshape((n_blocks, type_size))
    blocks = dequant_fn(blocks, block_size, type_size, dtype)
    return blocks.reshape(oshape)


# ---------------------------------------------------------------------------
# GGUF file loading → state_dict of GGMLTensors
# ---------------------------------------------------------------------------


def load_gguf_state_dict(path, prefix="model.diffusion_model."):
    """
    Load a GGUF file and return a state_dict of GGMLTensors.

    Args:
        path: Path to the .gguf file.
        prefix: Key prefix to strip (e.g. "model.diffusion_model.").

    Returns:
        dict: {key: GGMLTensor} state dict with stripped keys.
    """
    reader = gguf.GGUFReader(path)

    # Check if tensors have the prefix
    has_prefix = False
    if prefix is not None:
        prefix_len = len(prefix)
        tensor_names = {t.name for t in reader.tensors}
        has_prefix = any(s.startswith(prefix) for s in tensor_names)

    state_dict = {}
    qtype_counts = {}

    for tensor in reader.tensors:
        sd_key = tensor_name = tensor.name
        if has_prefix:
            if not tensor_name.startswith(prefix):
                continue
            sd_key = tensor_name[prefix_len:]

        # Load tensor data (suppress mmap readonly warning)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="The given NumPy array is not writable"
            )
            torch_tensor = torch.from_numpy(tensor.data)

        # Get original shape from metadata (city96 GGUF stores this)
        shape = _get_orig_shape(reader, tensor_name)
        if shape is None:
            shape = torch.Size(tuple(int(v) for v in reversed(tensor.shape)))

        # F32/F16 tensors can be reshaped directly
        if tensor.tensor_type in {
            gguf.GGMLQuantizationType.F32,
            gguf.GGMLQuantizationType.F16,
        }:
            torch_tensor = torch_tensor.view(*shape)

        state_dict[sd_key] = GGMLTensor(
            torch_tensor, tensor_type=tensor.tensor_type, tensor_shape=shape
        )

        # 1D BF16 tensors (biases, norms) — dequant at load time
        if len(shape) <= 1 and tensor.tensor_type == gguf.GGMLQuantizationType.BF16:
            state_dict[sd_key] = dequantize_tensor(
                state_dict[sd_key], dtype=torch.float32
            )

        # Track quant type distribution
        ttype = getattr(tensor.tensor_type, "name", repr(tensor.tensor_type))
        qtype_counts[ttype] = qtype_counts.get(ttype, 0) + 1

    logger.info(f"GGUF loaded: {len(state_dict)} tensors from {path}")
    logger.info("  qtypes: " + ", ".join(f"{k} ({v})" for k, v in qtype_counts.items()))
    return state_dict


def _get_orig_shape(reader, tensor_name):
    """Get original tensor shape from GGUF metadata (city96 format)."""
    field_key = f"comfy.gguf.orig_shape.{tensor_name}"
    field = reader.get_field(field_key)
    if field is None:
        return None
    if (
        len(field.types) != 2
        or field.types[0] != gguf.GGUFValueType.ARRAY
        or field.types[1] != gguf.GGUFValueType.INT32
    ):
        return None
    return torch.Size(tuple(int(field.parts[part_idx][0]) for part_idx in field.data))


# ---------------------------------------------------------------------------
# Load state_dict into a model
# ---------------------------------------------------------------------------


def _to_plain_tensor(t):
    """Strip GGMLTensor subclass, returning a plain torch.Tensor."""
    if isinstance(t, GGMLTensor):
        return (
            t.data.clone()
            if t.data_ptr() == t.storage().data_ptr()
            else torch.Tensor(t)
        )
    return t


def load_gguf_into_model(model, state_dict, param_dtype=torch.float16):
    """
    Load a GGMLTensor state_dict into a WanModel.

    For quantized Linear weights:
      - The raw packed bytes are stored as a plain nn.Parameter.
      - GGUF metadata (_gguf_tensor_type, _gguf_tensor_shape) is stored on
        the **module** (not the tensor). This survives .to(device) calls
        during block swap.

    All other tensors (biases, norms, Conv3d, etc.) are dequantized at
    load time and stored in param_dtype.

    Args:
        model: The WanModel (created on meta device).
        state_dict: Dict of {key: GGMLTensor} from load_gguf_state_dict().
        param_dtype: Dtype for non-quantized params (default float16).

    Returns:
        int: Number of parameters loaded.
    """
    loaded = 0
    skipped = 0

    for key, tensor in state_dict.items():
        parts = key.split(".")
        parent = model
        try:
            for part in parts[:-1]:
                parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
            attr_name = parts[-1]

            if not hasattr(parent, attr_name):
                skipped += 1
                continue

            existing = getattr(parent, attr_name)

            if isinstance(tensor, GGMLTensor) and is_quantized(tensor):
                # Check if this is a weight in a Linear layer
                if isinstance(parent, nn.Linear) and attr_name == "weight":
                    # Store raw packed bytes as a plain Parameter.
                    # Metadata goes on the MODULE so it survives .to(device).
                    raw = tensor.data  # plain torch.Tensor (packed bytes)
                    parent._parameters[attr_name] = nn.Parameter(
                        raw, requires_grad=False
                    )
                    parent._gguf_tensor_type = tensor.tensor_type
                    parent._gguf_tensor_shape = tensor.tensor_shape
                else:
                    # Non-Linear quantized tensor — dequant now
                    dequant = dequantize_tensor(tensor, dtype=param_dtype)
                    dequant = _to_plain_tensor(dequant)
                    if isinstance(existing, nn.Parameter):
                        parent._parameters[attr_name] = nn.Parameter(
                            dequant, requires_grad=False
                        )
                    else:
                        setattr(parent, attr_name, dequant)
            else:
                # Non-quantized (F16/F32/BF16) — convert to param_dtype
                if isinstance(tensor, GGMLTensor):
                    t = dequantize_tensor(tensor, dtype=param_dtype)
                    t = _to_plain_tensor(t)
                else:
                    t = tensor.to(param_dtype)

                if isinstance(existing, nn.Parameter):
                    parent._parameters[attr_name] = nn.Parameter(t, requires_grad=False)
                elif isinstance(existing, torch.Tensor):
                    setattr(parent, attr_name, t)
                else:
                    setattr(parent, attr_name, t)

            loaded += 1
        except (AttributeError, IndexError, KeyError) as e:
            logger.debug(f"  Skipping GGUF key {key}: {e}")
            skipped += 1

    logger.info(f"  Loaded {loaded} tensors into model ({skipped} skipped)")
    return loaded


# ---------------------------------------------------------------------------
# Monkey-patch Linear layers for on-the-fly dequantization
# ---------------------------------------------------------------------------


def _gguf_linear_forward(self, input):
    """
    Replacement forward for nn.Linear with GGUF-quantized weights.

    Reads GGUF metadata from the MODULE (not the tensor) so it works
    correctly after block swap .to(device) calls. Dequantizes the weight
    to input dtype, applies LoRA patches, then runs F.linear.
    """
    weight = self.weight

    # Dequantize using module-level metadata (survives .to() calls)
    qtype = getattr(self, "_gguf_tensor_type", None)
    if qtype is not None and qtype not in TORCH_COMPATIBLE_QTYPES:
        oshape = self._gguf_tensor_shape
        weight = dequantize_weight(weight.data, qtype, oshape, dtype=input.dtype)

    # Apply LoRA up/down patch if present
    # NOTE: .to(device, dtype) is needed because LoRA buffers may be on CPU
    # when the block has been moved to GPU by block_swap after LoRA was applied.
    if hasattr(self, "_lora_up"):
        up = self._lora_up.to(device=weight.device, dtype=weight.dtype)
        down = self._lora_down.to(device=weight.device, dtype=weight.dtype)
        delta = torch.matmul(up, down) * self._lora_scale
        weight = weight + delta

    # Apply LoRA diff patch if present
    if hasattr(self, "_lora_diff_weight"):
        weight = weight + self._lora_diff_weight.to(
            device=weight.device, dtype=weight.dtype
        )

    bias = self.bias
    if bias is not None and hasattr(self, "_lora_diff_bias"):
        bias = bias + self._lora_diff_bias.to(device=bias.device, dtype=bias.dtype)

    return F.linear(input, weight, bias)


def patch_gguf_linears(model):
    """
    Monkey-patch all nn.Linear layers with GGUF-quantized weights
    to use on-the-fly dequantization during forward.

    Checks for _gguf_tensor_type on the MODULE (set by load_gguf_into_model),
    not on the tensor itself.

    Returns:
        int: Number of layers patched.
    """
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and hasattr(module, "_gguf_tensor_type"):
            module.forward = lambda input, m=module: _gguf_linear_forward(m, input)
            count += 1
    logger.info(f"  GGUF dequant patched {count} linear layers")
    return count


# ---------------------------------------------------------------------------
# LoRA as patches — attach LoRA deltas to Linear modules as buffers
# ---------------------------------------------------------------------------


def apply_lora_as_patches(model, lora_path, lora_scale=1.0):
    """
    Load LoRA weights and attach as patches to model Linear layers.

    For GGUF-quantized weights, LoRA deltas are stored as registered buffers
    on the module and applied during the monkey-patched forward. This avoids
    needing to dequantize → merge → re-quantize at load time.

    For non-quantized weights, deltas are merged directly (same as FP8 path).

    Args:
        model: The WanModel with GGUF weights loaded.
        lora_path: Path to the LoRA .safetensors file.
        lora_scale: LoRA merge strength (default 1.0).

    Returns:
        int: Number of LoRA weights applied/attached.
    """
    from safetensors import safe_open
    from tqdm import tqdm

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
                lora_pairs[base_name] = ("updown", key, up_key)
        elif key.endswith("diff_b") and key.startswith(prefix):
            base_name = key[len(prefix) :].replace("diff_b", "bias")
            lora_pairs[base_name] = ("diff_b", key)
        elif key.endswith("diff") and key.startswith(prefix):
            base_name = key[len(prefix) :].replace("diff", "weight")
            lora_pairs[base_name] = ("diff", key)

    applied = 0
    skipped = 0
    for name in tqdm(lora_pairs, desc="Attaching LoRA patches"):
        parts = name.split(".")
        parent = model
        try:
            for part in parts[:-1]:
                parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
            attr_name = parts[-1]
            param = getattr(parent, attr_name)
        except (AttributeError, IndexError, KeyError):
            skipped += 1
            continue

        entry = lora_pairs[name]
        is_gguf_weight = hasattr(parent, "_gguf_tensor_type") and attr_name == "weight"

        if entry[0] == "updown":
            _, down_key, up_key = entry
            lora_down = lora_weights[down_key].half()
            lora_up = lora_weights[up_key].half()

            if is_gguf_weight:
                # Attach as buffers — applied during monkey-patched forward
                parent.register_buffer("_lora_up", lora_up, persistent=False)
                parent.register_buffer("_lora_down", lora_down, persistent=False)
                parent._lora_scale = lora_scale
            else:
                # Merge directly into non-quantized weight
                delta = torch.matmul(lora_up.float(), lora_down.float()) * lora_scale
                param.data.add_(delta.to(param.device, param.dtype))
            applied += 1

        elif entry[0] == "diff":
            _, diff_key = entry
            delta = lora_weights[diff_key].half() * lora_scale

            if is_gguf_weight:
                parent.register_buffer("_lora_diff_weight", delta, persistent=False)
            else:
                param.data.add_(delta.to(param.device, param.dtype))
            applied += 1

        elif entry[0] == "diff_b":
            _, diff_key = entry
            delta = lora_weights[diff_key].half() * lora_scale

            if isinstance(param, nn.Parameter) or isinstance(param, torch.Tensor):
                if is_gguf_weight:
                    parent.register_buffer("_lora_diff_bias", delta, persistent=False)
                else:
                    param.data.add_(delta.to(param.device, param.dtype))
            applied += 1

    logger.info(f"  Attached {applied} LoRA patches to GGUF model ({skipped} skipped)")
    return applied


# ---------------------------------------------------------------------------
# High-level: load GGUF base + adapter into a WanModel
# ---------------------------------------------------------------------------


def load_gguf_weights(
    model, base_gguf_path, adapter_gguf_path, param_dtype=torch.float16
):
    """
    Load GGUF base model + InfiniteTalk adapter into a WanModel.

    1. Loads base Wan2.1 GGUF (standard DiT blocks)
    2. Loads InfiniteTalk adapter GGUF (audio cross-attention layers)
    3. Merges into a single state_dict and loads into the model
    4. Monkey-patches Linear layers for on-the-fly dequantization

    Args:
        model: WanModel created on meta device.
        base_gguf_path: Path to base Wan2.1 I2V 14B GGUF file.
        adapter_gguf_path: Path to InfiniteTalk adapter GGUF file.
        param_dtype: Dtype for non-quantized params.

    Returns:
        The model with GGUF weights loaded.
    """
    import time

    start = time.time()

    # Load base model state_dict
    logger.info(f"  Loading base GGUF: {base_gguf_path}")
    base_sd = load_gguf_state_dict(base_gguf_path)

    # Load adapter state_dict
    logger.info(f"  Loading adapter GGUF: {adapter_gguf_path}")
    adapter_sd = load_gguf_state_dict(adapter_gguf_path)

    # Merge: adapter overrides base for any overlapping keys
    merged_sd = {**base_sd, **adapter_sd}
    logger.info(
        f"  Merged state_dict: {len(base_sd)} base + {len(adapter_sd)} adapter "
        f"= {len(merged_sd)} total keys"
    )

    # Load into model
    loaded = load_gguf_into_model(model, merged_sd, param_dtype=param_dtype)

    # Set eval mode
    model.eval().requires_grad_(False)

    # Monkey-patch Linear layers for on-the-fly dequant
    patch_gguf_linears(model)

    elapsed = time.time() - start
    logger.info(f"  GGUF weights loaded in {elapsed:.1f}s")

    return model
