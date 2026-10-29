"""System prerequisite checks and resource probing."""
import os
import subprocess
from dataclasses import dataclass

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


def ffmpeg_available():
    try:
        r = subprocess.run(["ffmpeg", "-version"],
                           capture_output=True, timeout=5)
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def run_prerequisite_checks():
    results = []

    if SOUNDDEVICE_IMPORT_ERROR:
        results.append(CheckResult(
            "PortAudio", "fail",
            "Audio playback unavailable: {}. "
            "Install with: sudo apt install portaudio19-dev".format(
                SOUNDDEVICE_IMPORT_ERROR),
            True))
    else:
        results.append(CheckResult("PortAudio", "pass", "Available", False))

    if DEMUCS_IMPORT_ERROR:
        results.append(CheckResult(
            "demucs-onnx", "fail",
            "Separation engine unavailable: {}. "
            "Install with: pip install demucs-onnx".format(
                DEMUCS_IMPORT_ERROR),
            True))
    else:
        results.append(CheckResult("demucs-onnx", "pass",
                                   "Available", False))

    if ffmpeg_available():
        results.append(CheckResult("FFmpeg", "pass",
                                   "Found on PATH", False))
    else:
        results.append(CheckResult(
            "FFmpeg", "warn",
            "Not found - MP3/AAC/M4A input and export will not work.",
            False))

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
            "Install Rubber Band: pip install pylibrb", False))
    else:
        results.append(CheckResult(
            "Tempo/Key", "warn",
            "Tempo and key change are disabled. Install: "
            "pip install pylibrb   (or: brew install rubberband)", False))

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

    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in providers:
            results.append(CheckResult(
                "GPU", "info", "CUDA available", False))
        elif "CoreMLExecutionProvider" in providers:
            results.append(CheckResult(
                "GPU", "info", "CoreML available (Apple Silicon)", False))
        else:
            results.append(CheckResult(
                "GPU", "info", "CPU only (no GPU acceleration)", False))
    except ImportError:
        results.append(CheckResult(
            "GPU", "info", "onnxruntime not installed", False))

    return results


def check_system_resources():
    try:
        mem = psutil.virtual_memory()
        ram_gb = mem.available / (1024 ** 3)
        cores = psutil.cpu_count(logical=True) or 1
    except Exception:
        return {
            "ram_gb": 0.0, "cores": 0, "can_run_6_stem": False,
            "reason": "Could not determine system resources.",
        }

    can_run = True
    reason = None
    if ram_gb < MIN_RAM_6_STEM_GB:
        can_run = False
        reason = "Only {:.1f} GB RAM available; 6-stem needs {} GB.".format(
            ram_gb, MIN_RAM_6_STEM_GB)
    elif cores < MIN_CORES_6_STEM:
        can_run = False
        reason = "Only {} CPU core(s); 6-stem needs at least {}.".format(
            cores, MIN_CORES_6_STEM)

    return {
        "ram_gb": ram_gb, "cores": cores,
        "can_run_6_stem": can_run, "reason": reason,
    }
