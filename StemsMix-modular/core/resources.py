"""System prerequisite checks, resource probing, and package installs."""
import importlib
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import psutil

try:
    import sounddevice as sd
    SOUNDDEVICE_IMPORT_ERROR = None
except OSError as e:
    sd = None
    SOUNDDEVICE_IMPORT_ERROR = str(e)

try:
    import demucs_onnx
    DEMUCS_IMPORT_ERROR = None
except ImportError as e:
    demucs_onnx = None
    DEMUCS_IMPORT_ERROR = str(e)


MIN_CPU_CORES_FAIL = 2
MIN_CPU_CORES_WARN = 4
MIN_RAM_GB_FAIL = 4
MIN_RAM_GB_WARN = 8
MIN_DISK_GB_FAIL = 1
MIN_DISK_GB_WARN = 5
MIN_RAM_6_STEM_GB = 6
MIN_CORES_6_STEM = 4


@dataclass
class CheckResult:
    name: str
    status: str
    message: str
    blocking: bool
    # pip package specs to install when this check fails/warns
    pip_packages: List[str] = field(default_factory=list)
    # optional brew formula (macOS) for system deps like ffmpeg
    brew_packages: List[str] = field(default_factory=list)


def ffmpeg_available():
    try:
        r = subprocess.run(["ffmpeg", "-version"],
                           capture_output=True, timeout=5)
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _module_ok(name):
    try:
        importlib.import_module(name)
        return True, None
    except Exception as e:
        return False, str(e)


def mlx_available():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return False
    ok, _ = _module_ok("demucs_mlx")
    return ok


def mps_available():
    try:
        import torch
        return (hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available())
    except Exception:
        return False


def separation_backend_label(force_cpu=False):
    """Human-readable backend the app will use for Demucs."""
    if force_cpu:
        return "ONNX Runtime (CPU) — slow"
    if mlx_available():
        return "demucs-mlx (Apple Metal) — fast"
    if DEMUCS_IMPORT_ERROR is None:
        return "demucs-onnx (CPU) — slower"
    return "No separation backend installed"


def run_prerequisite_checks():
    results = []
    is_mac = platform.system() == "Darwin"
    is_apple_silicon = is_mac and platform.machine() == "arm64"

    if SOUNDDEVICE_IMPORT_ERROR:
        brew = ["portaudio"] if is_mac else []
        msg = ("Audio playback unavailable: {}. ".format(
            SOUNDDEVICE_IMPORT_ERROR))
        if is_mac:
            msg += "Install with: brew install portaudio"
        else:
            msg += "Install with: sudo apt install portaudio19-dev"
        results.append(CheckResult(
            "PortAudio", "fail", msg, True, brew_packages=brew))
    else:
        results.append(CheckResult("PortAudio", "pass", "Available", False))

    if DEMUCS_IMPORT_ERROR:
        results.append(CheckResult(
            "demucs-onnx", "fail",
            "Separation engine unavailable: {}. "
            "Can be installed from this dialog.".format(DEMUCS_IMPORT_ERROR),
            True, pip_packages=["demucs-onnx"]))
    else:
        results.append(CheckResult(
            "demucs-onnx", "pass", "Available", False))

    if is_apple_silicon:
        ok, err = _module_ok("demucs_mlx")
        if ok:
            results.append(CheckResult(
                "demucs-mlx", "pass",
                "Metal GPU acceleration ready", False))
        else:
            results.append(CheckResult(
                "demucs-mlx", "warn",
                "Not installed — separation will use slow CPU. "
                "Install from this dialog for a large speedup. "
                "({})".format(err or "missing"),
                False, pip_packages=["demucs-mlx"]))

    ok, err = _module_ok("mdxnet_infer")
    if ok:
        device = "MPS" if mps_available() else (
            "CUDA" if _cuda_available() else "CPU")
        results.append(CheckResult(
            "Drum split", "pass",
            "mdxnet-infer ready ({})".format(device), False))
    else:
        results.append(CheckResult(
            "Drum split", "warn",
            "mdxnet-infer missing — Split Drums disabled. "
            "({})".format(err or "missing"),
            False, pip_packages=["mdxnet-infer"]))

    if ffmpeg_available():
        results.append(CheckResult("FFmpeg", "pass",
                                   "Found on PATH", False))
    else:
        brew = ["ffmpeg"] if is_mac else []
        tip = ("brew install ffmpeg" if is_mac
               else "sudo apt install ffmpeg")
        results.append(CheckResult(
            "FFmpeg", "warn",
            "Not found — MP3/AAC/M4A input and export will not work. "
            "Install with: " + tip,
            False, brew_packages=brew))

    from .timestretch import available_backend
    backend = available_backend()
    if backend == "pylibrb":
        results.append(CheckResult(
            "Tempo/Key", "pass", "Rubber Band (pylibrb)", False))
    elif backend == "rubberband":
        results.append(CheckResult(
            "Tempo/Key", "pass", "Rubber Band command-line tool", False))
    elif backend == "librosa":
        results.append(CheckResult(
            "Tempo/Key", "warn",
            "Only the librosa fallback was found (lower quality). "
            "Install Rubber Band for better results.",
            False, pip_packages=["pylibrb"]))
    else:
        brew = ["rubberband"] if is_mac else []
        results.append(CheckResult(
            "Tempo/Key", "warn",
            "Tempo and key change are disabled.",
            False, pip_packages=["pylibrb"], brew_packages=brew))

    try:
        cores = psutil.cpu_count(logical=True) or 1
    except Exception:
        cores = 1

    if cores >= MIN_CPU_CORES_WARN:
        results.append(CheckResult("CPU", "pass",
                                   "{} logical cores".format(cores), False))
    elif cores >= MIN_CPU_CORES_FAIL:
        results.append(CheckResult(
            "CPU", "warn",
            "Only {} cores - separation will be slow".format(cores),
            False))
    else:
        results.append(CheckResult(
            "CPU", "fail",
            "Only {} core - at least {} required".format(
                cores, MIN_CPU_CORES_FAIL),
            True))

    try:
        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024 ** 3)
        avail_gb = mem.available / (1024 ** 3)
    except Exception:
        total_gb, avail_gb = 0.0, 0.0

    ram_msg = "{:.1f} GB total, {:.1f} GB available".format(
        total_gb, avail_gb)
    if total_gb >= MIN_RAM_GB_WARN:
        results.append(CheckResult("RAM", "pass", ram_msg, False))
    elif total_gb >= MIN_RAM_GB_FAIL:
        results.append(CheckResult(
            "RAM", "warn",
            ram_msg + " - 6-stem separation may freeze the system",
            False))
    else:
        results.append(CheckResult(
            "RAM", "fail",
            ram_msg + " - at least {} GB required".format(MIN_RAM_GB_FAIL),
            True))

    try:
        disk = psutil.disk_usage(os.path.expanduser("~"))
        free_gb = disk.free / (1024 ** 3)
    except Exception:
        free_gb = 0.0

    disk_msg = "{:.1f} GB free in home directory".format(free_gb)
    if free_gb >= MIN_DISK_GB_WARN:
        results.append(CheckResult("Disk space", "pass", disk_msg, False))
    elif free_gb >= MIN_DISK_GB_FAIL:
        results.append(CheckResult(
            "Disk space", "warn",
            disk_msg + " - model download may fail", False))
    else:
        results.append(CheckResult(
            "Disk space", "fail",
            disk_msg + " - at least {} GB required".format(
                MIN_DISK_GB_FAIL),
            True))

    # Acceleration summary
    if mlx_available():
        results.append(CheckResult(
            "Acceleration", "pass",
            "Apple Metal via demucs-mlx (leave GPU on for Demucs)", False))
    elif mps_available():
        results.append(CheckResult(
            "Acceleration", "info",
            "PyTorch MPS available for drum split; "
            "install demucs-mlx for fast Demucs",
            False, pip_packages=["demucs-mlx"] if is_apple_silicon else []))
    elif _cuda_available():
        results.append(CheckResult(
            "Acceleration", "info", "CUDA available", False))
    else:
        results.append(CheckResult(
            "Acceleration", "info",
            "CPU only — Demucs and drum split will be slow", False))

    return results


def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def collect_install_plan(results):
    """Deduplicate pip / brew packages suggested by failing checks."""
    pip, brew = [], []
    seen_pip, seen_brew = set(), set()
    for r in results:
        if r.status not in ("fail", "warn"):
            # still allow optional acceleration installs from info? skip
            continue
        for p in r.pip_packages:
            if p not in seen_pip:
                seen_pip.add(p)
                pip.append(p)
        for b in r.brew_packages:
            if b not in seen_brew:
                seen_brew.add(b)
                brew.append(b)
    # Also pick up demucs-mlx from Acceleration info on Apple Silicon
    for r in results:
        if r.name in ("demucs-mlx", "Acceleration") and r.pip_packages:
            for p in r.pip_packages:
                if p not in seen_pip:
                    seen_pip.add(p)
                    pip.append(p)
    return pip, brew


def pip_install(packages, progress_cb=None):
    """Install packages with the current interpreter. Returns (ok, log)."""
    if not packages:
        return True, "Nothing to install."
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade"] + list(packages)
    if progress_cb:
        progress_cb("Running: " + " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        return False, "pip install timed out."
    log = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode == 0, log


def brew_install(packages, progress_cb=None):
    if not packages:
        return True, "Nothing to install."
    if platform.system() != "Darwin":
        return False, "Homebrew installs are only available on macOS."
    cmd = ["brew", "install"] + list(packages)
    if progress_cb:
        progress_cb("Running: " + " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=1800)
    except FileNotFoundError:
        return False, "Homebrew not found. Install from https://brew.sh"
    except subprocess.TimeoutExpired:
        return False, "brew install timed out."
    log = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode == 0, log


def check_system_resources():
    try:
        mem = psutil.virtual_memory()
        avail_gb = mem.available / (1024 ** 3)
        total_gb = mem.total / (1024 ** 3)
        cores = psutil.cpu_count(logical=True) or 1
    except Exception:
        return {
            "ram_gb": 0.0, "cores": 0, "can_run_6_stem": False,
            "reason": "Could not determine system resources.",
            "mlx": False,
        }

    has_mlx = mlx_available()
    # MLX uses unified memory efficiently; gate on total RAM there.
    # ONNX/CPU path is heavier — require more free RAM.
    ram_ok = (
        total_gb >= MIN_RAM_GB_WARN if has_mlx
        else avail_gb >= MIN_RAM_6_STEM_GB)

    can_run = True
    reason = None
    if not ram_ok:
        can_run = False
        if has_mlx:
            reason = ("Only {:.1f} GB total RAM; 6-stem needs about "
                      "{} GB.".format(total_gb, MIN_RAM_GB_WARN))
        else:
            reason = ("Only {:.1f} GB RAM available; 6-stem needs "
                      "{} GB (or install demucs-mlx).".format(
                          avail_gb, MIN_RAM_6_STEM_GB))
    elif cores < MIN_CORES_6_STEM:
        can_run = False
        reason = "Only {} CPU core(s); 6-stem needs at least {}.".format(
            cores, MIN_CORES_6_STEM)

    return {
        "ram_gb": avail_gb, "cores": cores,
        "can_run_6_stem": can_run, "reason": reason,
        "mlx": has_mlx,
    }
