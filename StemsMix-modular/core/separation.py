"""Separation engine. Prefers demucs-mlx on Apple Silicon.

Single pass is the default and the fastest path. Chunked mode exists only
for low-RAM machines; it now uses overlapping chunks with a linear
crossfade so there are no audible seams at chunk boundaries.
"""
import os
import platform
import shutil
import tempfile
import threading
import time

import numpy as np
import soundfile as sf

from .audio_io import convert_to_wav, needs_ffmpeg_conversion
from .resources import DEMUCS_IMPORT_ERROR

_MLX_SEPARATOR = None
_MLX_MODEL_NAME = None
_MLX_LOCK = threading.Lock()
try:
    if (platform.system() == "Darwin"
            and platform.machine() == "arm64"):
        from demucs_mlx import Separator as MLXSeparator
        MLX_AVAILABLE = True
    else:
        MLXSeparator = None
        MLX_AVAILABLE = False
except ImportError:
    MLXSeparator = None
    MLX_AVAILABLE = False

if DEMUCS_IMPORT_ERROR is None:
    import demucs_onnx
else:
    demucs_onnx = None


def separate_file(input_path, output_dir, model_name,
                  providers="auto", precision="fp32",
                  chunked=False, chunk_seconds=60, overlap_seconds=2.0,
                  progress_cb=None):
    def emit(pct, msg):
        if progress_cb:
            progress_cb(int(pct), str(msg))

    force_cpu = providers in ("cpu", ["cpu"])
    use_mlx = MLX_AVAILABLE and not force_cpu

    if not use_mlx and DEMUCS_IMPORT_ERROR:
        raise RuntimeError(
            "No separation backend available. Install demucs-onnx "
            "or demucs-mlx.")

    t0 = time.monotonic()
    converted = None
    input_to_use = input_path
    try:
        if needs_ffmpeg_conversion(input_path):
            emit(1, "Converting to WAV...")
            converted = convert_to_wav(input_path)
            input_to_use = converted
            emit(5, "Converted.")

        os.makedirs(output_dir, exist_ok=True)

        result = None
        if use_mlx:
            try:
                result = _run_mlx(input_to_use, output_dir,
                                  model_name, emit)
            except Exception as e:
                if DEMUCS_IMPORT_ERROR:
                    raise
                emit(10, "MLX failed ({}); falling back to ONNX "
                         "on CPU...".format(e))
        if result is None:
            onnx_providers = _onnx_providers(providers)
            if chunked:
                result = _run_chunked(
                    input_to_use, output_dir, model_name,
                    onnx_providers, precision, chunk_seconds,
                    overlap_seconds, emit)
            else:
                result = _run_single_pass(
                    input_to_use, output_dir, model_name,
                    onnx_providers, precision, emit)

        elapsed = time.monotonic() - t0
        try:
            info = sf.info(input_to_use)
            audio_len = info.frames / float(info.samplerate)
            emit(100, "Done in {:.0f}s ({:.1f}x realtime)".format(
                elapsed, audio_len / max(elapsed, 1e-6)))
        except Exception:
            emit(100, "Done in {:.0f}s".format(elapsed))
        return result
    finally:
        if converted and os.path.exists(converted):
            try:
                os.remove(converted)
            except OSError:
                pass


def separate_task(input_path, output_dir, model_name, task="full",
                  providers="auto", precision="fp32", chunked=False,
                  progress_cb=None):
    """Run a supported separation workflow and return its output stems."""
    if task not in ("full", "vocal_accompaniment", "drum_parts"):
        raise ValueError("Unknown separation task: {}".format(task))
    if task == "full":
        stems = separate_file(
            input_path, output_dir, model_name, providers=providers,
            precision=precision, chunked=chunked,
            progress_cb=progress_cb)
        if not stems:
            raise RuntimeError("The separator returned no usable stems.")
        return stems

    os.makedirs(output_dir, exist_ok=True)
    temp_dir = tempfile.mkdtemp(
        prefix="stemsmix_task_", dir=os.path.dirname(output_dir) or None)
    try:
        stems = separate_file(
            input_path, temp_dir, model_name, providers=providers,
            precision=precision, chunked=chunked,
            progress_cb=progress_cb)
        if not stems:
            raise RuntimeError("The separator returned no usable stems.")
        if task == "vocal_accompaniment":
            return _write_vocal_accompaniment(stems, output_dir)
        return _write_drum_parts(
            stems, output_dir, providers, progress_cb)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _write_vocal_accompaniment(stems, output_dir):
    """Create a vocal stem and a backing made from the other model stems."""
    vocal_path = next(
        (path for name, path in stems.items()
         if name.lower() == "vocals"), None)
    backing_paths = [
        path for name, path in stems.items() if name.lower() != "vocals"]
    if vocal_path is None or not backing_paths:
        raise RuntimeError(
            "The selected model did not return vocals and accompaniment "
            "stems.")

    vocal, vocal_sr = sf.read(
        vocal_path, dtype="float32", always_2d=True)
    backing = None
    for path in backing_paths:
        audio, sr = sf.read(path, dtype="float32", always_2d=True)
        if sr != vocal_sr:
            raise RuntimeError("Model stems have mismatched sample rates.")
        if audio.shape[1] != vocal.shape[1]:
            raise RuntimeError("Model stems have mismatched channel counts.")
        length = min(len(audio), len(vocal))
        if backing is None:
            backing = np.zeros_like(vocal[:length])
        length = min(length, len(backing))
        backing[:length] += audio[:length]
        if len(backing) > length:
            backing = backing[:length]
        if len(vocal) > length:
            vocal = vocal[:length]

    os.makedirs(output_dir, exist_ok=True)
    vocal_out = os.path.join(output_dir, "vocals.wav")
    backing_out = os.path.join(output_dir, "accompaniment.wav")
    sf.write(vocal_out, vocal, vocal_sr, subtype="FLOAT")
    sf.write(backing_out, backing, vocal_sr, subtype="FLOAT")
    return {"vocals": vocal_out, "accompaniment": backing_out}


def _write_drum_parts(stems, output_dir, providers, progress_cb):
    drum_path = next(
        (path for name, path in stems.items()
         if name.lower() == "drums"), None)
    if drum_path is None:
        raise RuntimeError("The selected model did not return a drums stem.")
    try:
        from .drumsep import get_engine, preferred_device, inference_tuning
        from mdxnet_infer import MDX23CInference  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "Drum-part separation requires mdxnet-infer. Install it with "
            "`pip install mdxnet-infer`.") from e

    device = ("cpu" if providers in ("cpu", ["cpu"])
              else preferred_device())
    if progress_cb:
        progress_cb(15, "Loading drum-part model on {}...".format(device))
    audio, sr = sf.read(drum_path, dtype="float32", always_2d=True)
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    elif audio.shape[1] > 2:
        audio = audio[:, :2]
    engine = get_engine(device=device, progress_cb=progress_cb)
    tuning = inference_tuning(device)
    if progress_cb:
        progress_cb(35, "Separating kick, snare, toms and cymbals...")
    parts = engine.separate(
        audio, sample_rate=sr, progress=False, **tuning)
    if not parts:
        raise RuntimeError("Drum-part model returned no stems.")

    os.makedirs(output_dir, exist_ok=True)
    output = {}
    sample_rate = engine.config.audio.sample_rate
    for name, data in parts.items():
        path = os.path.join(output_dir, "{}.wav".format(name))
        sf.write(path, np.asarray(data), sample_rate)
        output[str(name)] = path
    return output


def _onnx_providers(providers):
    """ONNX Runtime must not auto-pick CoreML (it cannot compile the
    Demucs graph), so "auto" means the CPU provider here."""
    return ["cpu"] if providers == "auto" else providers


def _run_mlx(input_path, output_dir, model_name, emit):
    global _MLX_SEPARATOR, _MLX_MODEL_NAME
    with _MLX_LOCK:
        if _MLX_SEPARATOR is None or _MLX_MODEL_NAME != model_name:
            emit(10, "Loading MLX model (one-time compile)...")
            _MLX_SEPARATOR = MLXSeparator(model=model_name)
            _MLX_MODEL_NAME = model_name
        else:
            emit(10, "Reusing cached MLX model...")
        emit(20, "Separating with Metal GPU...")
        origin, stems = _MLX_SEPARATOR.separate_audio_file(input_path)
        sr = getattr(_MLX_SEPARATOR, "samplerate", 44100)
    emit(90, "Writing stems...")
    out_paths = {}
    for name, audio in stems.items():
        out = np.transpose(audio) if audio.ndim == 2 else audio
        out_path = os.path.join(output_dir, name + ".wav")
        sf.write(out_path, out, sr)
        out_paths[name] = out_path
    return out_paths


def _run_single_pass(input_path, output_dir, model_name,
                     providers, precision, emit):
    emit(10, "Loading model and separating...")
    emit(20, "Running separation...")
    result = demucs_onnx.separate(
        input_path, output_dir, model=model_name, providers=providers,
        precision=precision, progress=False)
    emit(95, "Finalizing...")
    return _normalize_result(result)


def _chunk_starts(total_frames, chunk_frames, overlap_frames):
    """Start frame of each chunk. Consecutive chunks overlap by
    overlap_frames and the last chunk always reaches the end."""
    step = chunk_frames - overlap_frames
    starts = [0]
    while starts[-1] + chunk_frames < total_frames:
        starts.append(starts[-1] + step)
    return starts


def _run_chunked(input_path, output_dir, model_name, providers,
                 precision, chunk_seconds, overlap_seconds, emit):
    emit(10, "Reading audio file...")
    audio, sr = sf.read(input_path, dtype="float32", always_2d=True)
    total = audio.shape[0]
    chunk_frames = max(int(chunk_seconds * sr), sr * 10)
    overlap = int(min(overlap_seconds * sr, chunk_frames // 4))
    starts = _chunk_starts(total, chunk_frames, overlap)
    n_chunks = len(starts)

    fade_in = np.linspace(0.0, 1.0, overlap, endpoint=False,
                          dtype="float32")[:, None]
    fade_out = 1.0 - fade_in

    mixed = {}  # stem name -> (total, channels) accumulator
    tmp_root = tempfile.mkdtemp(prefix="stemsmix_chunks_")
    try:
        for i, start in enumerate(starts):
            end = min(start + chunk_frames, total)
            chunk_path = os.path.join(tmp_root, "chunk.wav")
            chunk_out = os.path.join(tmp_root, "out_{:04d}".format(i))
            os.makedirs(chunk_out)
            sf.write(chunk_path, audio[start:end], sr)
            emit(10 + int(i / n_chunks * 80),
                 "Separating chunk {}/{}...".format(i + 1, n_chunks))

            result = demucs_onnx.separate(
                chunk_path, chunk_out, model=model_name,
                providers=providers, precision=precision, progress=False)

            for name, path in _normalize_result(result).items():
                data, _ = sf.read(path, dtype="float32", always_2d=True)
                n = min(data.shape[0], end - start)
                data = data[:n].copy()
                if overlap and i > 0 and n >= overlap:
                    data[:overlap] *= fade_in
                if overlap and i < n_chunks - 1 and n >= overlap:
                    data[n - overlap:n] *= fade_out
                acc = mixed.get(name)
                if acc is None:
                    acc = np.zeros((total, data.shape[1]), dtype="float32")
                    mixed[name] = acc
                acc[start:start + n] += data
            shutil.rmtree(chunk_out, ignore_errors=True)

        emit(92, "Writing final stems...")
        out_paths = {}
        for name, combined in mixed.items():
            out_path = os.path.join(output_dir, name + ".wav")
            sf.write(out_path, combined, sr)
            out_paths[name] = out_path
        return out_paths
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def _normalize_result(result):
    if isinstance(result, dict):
        return {k: v for k, v in result.items()
                if isinstance(v, str) and os.path.exists(v)}
    if isinstance(result, (list, tuple)):
        out = {}
        for item in result:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                name, path = item
                if isinstance(path, str) and os.path.exists(path):
                    out[str(name)] = path
            elif isinstance(item, str) and os.path.exists(item):
                name = os.path.splitext(os.path.basename(item))[0]
                out[name] = item
        return out
    return {}
