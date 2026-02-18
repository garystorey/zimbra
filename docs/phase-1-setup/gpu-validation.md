# GPU Validation

Actual verification output from Phase 1 install (Feb 2026).

---

## Verification Output

```
Python: 3.10.11 (tags/v3.10.11:7d4cc5a, Apr  5 2023, 00:38:17) [MSC v.1929 64 bit (AMD64)]

PyTorch: 2.5.1+cu124
CUDA available: True
CUDA device: NVIDIA GeForce RTX 4090 Laptop GPU
CUDA version: 12.4
VRAM: 16.0 GB

xformers: 0.0.28.post3
diffusers: 0.31.0
numpy: 1.26.4
mediapipe: 0.10.32
onnxruntime: 1.20.1
opencv: 4.10.0
rich: 14.3.2
clip: OK
Pillow: 10.2.0
soundfile: 0.13.1
librosa: 0.11.0
Chatterbox: OK (ChatterboxTTS importable)

========================================
  ALL CHECKS PASSED
========================================
```

**Note:** The Triton warning ("A matching Triton is not available") is expected — Triton has no Windows wheels. It is only an optional xformers optimization, not required for inference.

## Validation Checklist

| Check                       | Expected | Notes                                  |
| --------------------------- | -------- | -------------------------------------- |
| Python version              | 3.10.x   | Required for all package compatibility |
| `torch.cuda.is_available()` | `True`   | NVIDIA driver must be installed        |
| CUDA version                | 12.x     | Bundled with cu124 wheels              |
| VRAM                        | ~16 GB   | RTX 4090 Laptop GPU                    |
| xformers import             | No error | Must match torch version               |
| Chatterbox import           | No error | Installed with --no-deps               |
| mediapipe import            | No error | Windows wheel available                |
| onnxruntime-gpu import      | No error | Requires CUDA 12.x                     |

## Troubleshooting

| Symptom                    | Cause                          | Fix                                                                                                |
| -------------------------- | ------------------------------ | -------------------------------------------------------------------------------------------------- |
| `CUDA available: False`    | Missing/outdated NVIDIA driver | Install latest Game Ready or Studio driver from nvidia.com                                         |
| `xformers` import error    | Version mismatch with torch    | Reinstall: `pip install xformers==0.0.28.post3 --index-url https://download.pytorch.org/whl/cu124` |
| `Chatterbox: IMPORT ERROR` | Missing unique dep             | Run step 5b of install manually                                                                    |
| `onnxruntime` CUDA error   | CUDA version mismatch          | Ensure CUDA 12.x is present (check `nvidia-smi`)                                                   |

## How to Re-run Validation

```bash
# Activate venv first
.\venv\Scripts\activate

# Then run the verification inline:
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```
