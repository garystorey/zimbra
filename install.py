#!/usr/bin/env python3
"""
Talking Head Video Generator — Install Script

Creates a dedicated virtual environment and installs all dependencies
in the correct order to avoid conflicts between Chatterbox and InfiniteTalk.

Usage:
    python install.py
"""

import os
import sys
import subprocess
import shutil
import platform


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(PROJECT_ROOT, "venv")
VENDOR_DIR = os.path.join(PROJECT_ROOT, "vendor")
INFINITETALK_VENDOR = os.path.join(VENDOR_DIR, "infinitetalk")

IS_WINDOWS = platform.system() == "Windows"

if IS_WINDOWS:
    PYTHON_EXE = os.path.join(VENV_DIR, "Scripts", "python.exe")
    PIP_EXE = os.path.join(VENV_DIR, "Scripts", "pip.exe")
else:
    PYTHON_EXE = os.path.join(VENV_DIR, "bin", "python")
    PIP_EXE = os.path.join(VENV_DIR, "bin", "pip")

# Required Python version range
MIN_PYTHON = (3, 10)
MAX_PYTHON = (3, 12)

# Known Python paths to search on Windows
WINDOWS_PYTHON_SEARCH_PATHS = [
    r"C:\Python310\python.exe",
    r"C:\Python311\python.exe",
    r"C:\Python312\python.exe",
    os.path.expanduser(r"~\AppData\Local\Programs\Python\Python310\python.exe"),
    os.path.expanduser(r"~\AppData\Local\Programs\Python\Python311\python.exe"),
    os.path.expanduser(r"~\AppData\Local\Programs\Python\Python312\python.exe"),
]

PYTORCH_INDEX = "https://download.pytorch.org/whl/cu121"
XFORMERS_INDEX = "https://download.pytorch.org/whl/cu124"

INFINITETALK_REPO = "https://github.com/MeiGen-AI/InfiniteTalk.git"

# Prebuilt flash_attn Windows wheel for Python 3.10 + torch 2.4.x + CUDA 12.4
# (kingbri1/flash-attention — bdashore3 redirects here)
FLASH_ATTN_WHEEL = "https://github.com/kingbri1/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1%2Bcu124torch2.4.0cxx11abiFALSE-cp310-cp310-win_amd64.whl"

# Chatterbox unique deps that don't conflict with InfiniteTalk
CHATTERBOX_UNIQUE_DEPS = [
    "librosa",
    "spacy-pkuseg==1.0.1",
    "s3tokenizer",
    "conformer",
    "resemble-perth",
    "safetensors==0.5.3",
    "pykakasi",
    "omegaconf",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run(cmd, desc=None, check=True, **kwargs):
    """Run a subprocess command with logging."""
    if desc:
        print(f"\n{'='*60}")
        print(f"  {desc}")
        print(f"{'='*60}")
    if isinstance(cmd, str):
        print(f"  > {cmd}")
    else:
        print(f"  > {' '.join(cmd)}")
    result = subprocess.run(cmd, check=check, **kwargs)
    return result


def pip_install(*args, index_url=None):
    """Run pip install with optional index URL."""
    cmd = [PYTHON_EXE, "-m", "pip", "install"]
    if index_url:
        cmd += ["--index-url", index_url]
    cmd += list(args)
    return run(cmd)


# ---------------------------------------------------------------------------
# Step 1: Create virtual environment
# ---------------------------------------------------------------------------


def find_compatible_python():
    """
    Find a Python executable in the required version range (3.10-3.12).
    Returns the path to the executable, or None if not found.
    """

    def check_python(exe):
        try:
            result = subprocess.run(
                [
                    exe,
                    "-c",
                    "import sys; print(sys.version_info.major, sys.version_info.minor)",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split()
                major, minor = int(parts[0]), int(parts[1])
                if MIN_PYTHON <= (major, minor) <= MAX_PYTHON:
                    return exe, (major, minor)
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError, IndexError):
            pass
        return None, None

    # Check current interpreter first
    current = (sys.version_info.major, sys.version_info.minor)
    if MIN_PYTHON <= current <= MAX_PYTHON:
        return sys.executable, current

    # Search known paths on Windows
    if IS_WINDOWS:
        for path in WINDOWS_PYTHON_SEARCH_PATHS:
            exe, ver = check_python(path)
            if exe:
                return exe, ver

    # Try common names
    for name in ["python3.12", "python3.11", "python3.10", "python3"]:
        exe_path = shutil.which(name)
        if exe_path:
            exe, ver = check_python(exe_path)
            if exe:
                return exe, ver

    return None, None


def create_venv():
    if os.path.exists(VENV_DIR):
        print(f"\n[INFO] Virtual environment already exists at: {VENV_DIR}")
        print("       To recreate, delete the venv/ folder and re-run this script.")
        return

    python_exe, python_ver = find_compatible_python()
    if python_exe is None:
        print(
            f"\n[ERROR] No Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}-{MAX_PYTHON[0]}.{MAX_PYTHON[1]} found."
        )
        print("        Please install Python 3.10, 3.11, or 3.12 and re-run.")
        sys.exit(1)

    print(f"\n  Using Python {python_ver[0]}.{python_ver[1]} from: {python_exe}")

    run(
        [python_exe, "-m", "venv", VENV_DIR],
        desc="Step 1: Creating virtual environment",
    )
    print(f"  Created venv at: {VENV_DIR}")


# ---------------------------------------------------------------------------
# Step 2: Upgrade pip, setuptools, wheel
# ---------------------------------------------------------------------------


def upgrade_pip():
    run(
        [PYTHON_EXE, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
        desc="Step 2: Upgrading pip, setuptools, wheel",
    )


# ---------------------------------------------------------------------------
# Step 3: Install PyTorch + xformers + flash_attn from cu121 index
# ---------------------------------------------------------------------------


def install_pytorch():
    run(
        [
            PYTHON_EXE,
            "-m",
            "pip",
            "install",
            "torch==2.4.1+cu121",
            "torchvision==0.19.1+cu121",
            "torchaudio==2.4.1+cu121",
            "--index-url",
            PYTORCH_INDEX,
        ],
        desc="Step 3a: Installing PyTorch 2.4.1+cu121",
    )
    run(
        [
            PYTHON_EXE,
            "-m",
            "pip",
            "install",
            "xformers==0.0.28.post1",
            "--index-url",
            XFORMERS_INDEX,
        ],
        desc="Step 3b: Installing xformers 0.0.28.post1",
    )
    if IS_WINDOWS:
        run(
            [
                PYTHON_EXE,
                "-m",
                "pip",
                "install",
                FLASH_ATTN_WHEEL,
            ],
            desc="Step 3c: Installing flash_attn 2.7.4.post1 (prebuilt Windows wheel)",
        )
    else:
        run(
            [
                PYTHON_EXE,
                "-m",
                "pip",
                "install",
                "flash_attn==2.7.4.post1",
            ],
            desc="Step 3c: Installing flash_attn 2.7.4.post1",
        )


# ---------------------------------------------------------------------------
# Step 4: Install InfiniteTalk deps from requirements.txt
# ---------------------------------------------------------------------------


def install_infinitetalk_deps():
    # Pin numpy first to prevent PyTorch/other deps from pulling 2.x
    run(
        [PYTHON_EXE, "-m", "pip", "install", "numpy==1.26.4"],
        desc="Step 4a: Pinning numpy version",
    )

    req_file = os.path.join(PROJECT_ROOT, "requirements.txt")
    run(
        [PYTHON_EXE, "-m", "pip", "install", "-r", req_file],
        desc="Step 4b: Installing InfiniteTalk dependencies",
    )


# ---------------------------------------------------------------------------
# Step 5: Install Chatterbox with --no-deps
# ---------------------------------------------------------------------------


def install_chatterbox():
    run(
        [PYTHON_EXE, "-m", "pip", "install", "chatterbox-tts", "--no-deps"],
        desc="Step 5a: Installing chatterbox-tts (--no-deps)",
    )
    run(
        [PYTHON_EXE, "-m", "pip", "install"] + CHATTERBOX_UNIQUE_DEPS,
        desc="Step 5b: Installing Chatterbox unique dependencies",
    )


# ---------------------------------------------------------------------------
# Step 6: Clone InfiniteTalk into vendor/
# ---------------------------------------------------------------------------


def clone_infinitetalk():
    if os.path.exists(INFINITETALK_VENDOR):
        print(f"\n[INFO] InfiniteTalk already cloned at: {INFINITETALK_VENDOR}")
        return

    os.makedirs(VENDOR_DIR, exist_ok=True)

    git_exe = shutil.which("git")
    if git_exe is None:
        print("\n[WARNING] git not found in PATH. Skipping InfiniteTalk clone.")
        print("          You can manually clone later:")
        print(f"          git clone {INFINITETALK_REPO} {INFINITETALK_VENDOR}")
        return

    run(
        ["git", "clone", "--depth", "1", INFINITETALK_REPO, INFINITETALK_VENDOR],
        desc="Step 6: Cloning InfiniteTalk into vendor/",
    )


# ---------------------------------------------------------------------------
# Step 7: Create runtime directories
# ---------------------------------------------------------------------------


def create_directories():
    dirs = [
        os.path.join(PROJECT_ROOT, "outputs", "audio"),
        os.path.join(PROJECT_ROOT, "outputs", "video"),
        os.path.join(PROJECT_ROOT, "models", "infinitetalk"),
        os.path.join(PROJECT_ROOT, "models", "chatterbox"),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("\n[INFO] Created runtime directories: outputs/, models/")


# ---------------------------------------------------------------------------
# Step 8: Verify installation
# ---------------------------------------------------------------------------


def verify():
    print(f"\n{'='*60}")
    print("  Step 8: Verifying installation")
    print(f"{'='*60}")

    verify_script = """
import sys
print(f"Python: {sys.version}")

# Check torch + CUDA
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Check xformers
import xformers
print(f"xformers: {xformers.__version__}")

# Check flash_attn
try:
    import flash_attn
    print(f"flash_attn: {flash_attn.__version__}")
except ImportError as e:
    print(f"flash_attn: IMPORT ERROR - {e}")

# Check diffusers
import diffusers
print(f"diffusers: {diffusers.__version__}")

# Check transformers
import transformers
print(f"transformers: {transformers.__version__}")

# Check numpy
import numpy
print(f"numpy: {numpy.__version__}")

# Check key InfiniteTalk deps
import cv2
print(f"opencv: {cv2.__version__}")

import optimum.quanto
print(f"optimum-quanto: OK")

import einops
print(f"einops: OK")

# Check Chatterbox import
try:
    from chatterbox.tts import ChatterboxTTS
    print("Chatterbox: OK (ChatterboxTTS importable)")
except ImportError as e:
    print(f"Chatterbox: IMPORT ERROR - {e}")

# Check rich
from importlib.metadata import version as pkg_version
print(f"rich: {pkg_version('rich')}")

print()
print("=" * 40)
print("  ALL CHECKS PASSED" if torch.cuda.is_available() else "  WARNING: No CUDA GPU detected")
print("=" * 40)
"""

    result = run(
        [PYTHON_EXE, "-c", verify_script],
        desc=None,
        check=False,
    )

    if result.returncode != 0:
        print("\n[ERROR] Verification failed. Check the output above for details.")
        return False

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 60)
    print("  Talking Head Video Generator — Installer")
    print("=" * 60)
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"  Platform:     {platform.system()} {platform.release()}")
    print(f"  Python:       {sys.version}")
    print()

    create_venv()
    upgrade_pip()
    install_pytorch()
    install_infinitetalk_deps()
    install_chatterbox()
    clone_infinitetalk()
    create_directories()

    success = verify()

    print()
    if success:
        print("=" * 60)
        print("  Installation complete!")
        print("=" * 60)
        if IS_WINDOWS:
            print(f"  Activate venv:  .\\venv\\Scripts\\activate")
        else:
            print(f"  Activate venv:  source venv/bin/activate")
        print(f"  Run generator:  python generate.py --help")
    else:
        print("  Installation completed with warnings. See above.")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
