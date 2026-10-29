"""Offline tempo (time-stretch) and key (pitch-shift) rendering.

Audio is processed ahead of playback, never inside the audio callback.
Backends, best first:
  1. pylibrb         in-process Rubber Band (pip install pylibrb)
  2. rubberband CLI  (brew install rubberband)
  3. librosa         phase vocoder - works, but audibly lower quality
Rubber Band is GPL-licensed software; if you distribute StemsMix you must
comply with its licence (or buy a commercial one from Breakfast Quay).
"""
import os
import shutil
import subprocess
import tempfile

import numpy as np
import soundfile as sf

NEUTRAL_EPS = 1e-6


def needs_processing(tempo, semitones):
    return abs(tempo - 1.0) > NEUTRAL_EPS or abs(semitones) > NEUTRAL_EPS


# ------------------------------------------------------------------ backends
def _has_pylibrb():
    try:
        import pylibrb  # noqa: F401
        return True
    except Exception:
        return False


def _rubberband_exe():
    return shutil.which("rubberband")


def _has_librosa():
    try:
        import librosa  # noqa: F401
        return True
    except Exception:
        return False


def available_backend():
    """Name of the best usable backend, or None."""
    if _has_pylibrb():
        return "pylibrb"
    if _rubberband_exe():
        return "rubberband"
    if _has_librosa():
        return "librosa"
    return None


INSTALL_HINT = ("No time-stretch backend found. Install one: "
                "pip install pylibrb   (or: brew install rubberband)")


def _fit_length(y, n):
    """Trim or zero-pad to exactly n frames (the stretchers can be a few
    frames off)."""
    if y.shape[0] > n:
        return y[:n]
    if y.shape[0] < n:
        pad = np.zeros((n - y.shape[0], y.shape[1]), dtype=y.dtype)
        return np.vstack([y, pad])
    return y


def _run_pylibrb(data, sr, tempo, semitones, formant, quality, progress_cb):
    from pylibrb import RubberBandStretcher, Option
    opts = Option.PROCESS_OFFLINE | Option.CHANNELS_TOGETHER
    opts |= (Option.ENGINE_FINER if quality == "high"
             else Option.ENGINE_FASTER)
    if formant and abs(semitones) > NEUTRAL_EPS:
        opts |= Option.FORMANT_PRESERVED
    rb = RubberBandStretcher(
        int(sr), data.shape[1], opts,
        initial_time_ratio=1.0 / tempo,
        initial_pitch_scale=2.0 ** (semitones / 12.0))
    x = np.ascontiguousarray(data.T, dtype=np.float32)   # (channels, frames)
    n = x.shape[1]
    block = 8192
    rb.set_max_process_size(block)
    rb.set_expected_input_duration(n)
    for i in range(0, n, block):
        rb.study(np.ascontiguousarray(x[:, i:i + block]),
                 final=(i + block >= n))
        if progress_cb:
            progress_cb(int(40 * min(1.0, (i + block) / n)))
    out = []
    for i in range(0, n, block):
        rb.process(np.ascontiguousarray(x[:, i:i + block]),
                   final=(i + block >= n))
        if rb.available() > 0:
            out.append(rb.retrieve_available())
        if progress_cb:
            progress_cb(40 + int(60 * min(1.0, (i + block) / n)))
    while not rb.is_done():
        chunk = rb.retrieve_available()
        if chunk.shape[1] == 0:
            break
        out.append(chunk)
    if not out:
        raise RuntimeError("Rubber Band produced no output.")
    return np.concatenate(out, axis=1).T


_CLI_FLAGS = {}


def _cli_supports(exe, flag):
    if exe not in _CLI_FLAGS:
        try:
            r = subprocess.run([exe, "--help"], capture_output=True,
                               text=True, timeout=10)
            _CLI_FLAGS[exe] = r.stdout + r.stderr
        except Exception:
            _CLI_FLAGS[exe] = ""
    return flag in _CLI_FLAGS[exe]


def _run_cli(data, sr, tempo, semitones, formant, quality, progress_cb):
    exe = _rubberband_exe()
    tmp = tempfile.mkdtemp(prefix="stemsmix_rb_")
    try:
        src = os.path.join(tmp, "in.wav")
        dst = os.path.join(tmp, "out.wav")
        sf.write(src, data, int(sr), subtype="FLOAT")
        cmd = [exe, "-q", "-T", "{:.8f}".format(tempo),
               "-p", "{:.6f}".format(semitones)]
        if quality == "high" and _cli_supports(exe, "--fine"):
            cmd.append("--fine")
        if formant and abs(semitones) > NEUTRAL_EPS:
            cmd.append("--formant")
        cmd += [src, dst]
        if progress_cb:
            progress_cb(10)
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(dst):
            raise RuntimeError("rubberband failed: " + (r.stderr or "")[-300:])
        out, _ = sf.read(dst, dtype="float32", always_2d=True)
        return out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _run_librosa(data, sr, tempo, semitones, formant, quality, progress_cb):
    import librosa
    chans = []
    for c in range(data.shape[1]):
        y = np.ascontiguousarray(data[:, c])
        if abs(semitones) > NEUTRAL_EPS:
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=semitones)
        if abs(tempo - 1.0) > NEUTRAL_EPS:
            y = librosa.effects.time_stretch(y, rate=tempo)
        chans.append(y)
        if progress_cb:
            progress_cb(int(100 * (c + 1) / data.shape[1]))
    n = min(len(c) for c in chans)
    return np.stack([c[:n] for c in chans], axis=1)


_RUNNERS = {"pylibrb": _run_pylibrb, "rubberband": _run_cli,
            "librosa": _run_librosa}


def process(data, sr, tempo=1.0, semitones=0.0, formant=False,
            quality="high", backend=None, progress_cb=None):
    """Return `data` (frames, channels) stretched by `tempo` (>1 = faster,
    shorter) and shifted by `semitones`, as float32 with exactly
    round(frames / tempo) frames."""
    tempo = float(min(max(tempo, 0.1), 10.0))
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 1:
        data = np.column_stack((data, data))
    if not needs_processing(tempo, semitones):
        return data
    backend = backend or available_backend()
    if backend is None:
        raise RuntimeError(INSTALL_HINT)
    out = _RUNNERS[backend](data, sr, tempo, semitones, formant, quality,
                            progress_cb)
    out = _fit_length(np.ascontiguousarray(out, dtype=np.float32),
                      int(round(data.shape[0] / tempo)))
    if progress_cb:
        progress_cb(100)
    return out


def clip_needs_work(clip, quality="fast"):
    """True if the clip must be (re)rendered for these parameters."""
    if not clip.needs_render():
        return clip.render is not None       # stale render to free
    if clip.render is None or clip.render_key != clip.render_params():
        return True
    return quality == "high" and clip.render_quality != "high"


def ensure_clip_render(clip, quality="fast", progress_cb=None):
    """Make sure clip.render matches the clip's current tempo/key (and is
    at least `quality`). Safe to call from a worker thread."""
    if not clip.needs_render():
        clip.render, clip.render_key, clip.render_quality = None, None, None
        return False
    if not clip_needs_work(clip, quality):
        return False
    key = clip.render_params()
    out = process(clip.data, clip.samplerate, tempo=clip.stretch,
                  semitones=clip.semitones,
                  formant=(clip.stem_name == "vocals"),
                  quality=quality, progress_cb=progress_cb)
    # Order matters: the audio thread treats the render as valid as soon
    # as render_key matches.
    clip.render = out
    clip.render_quality = quality
    clip.render_key = key
    return True


def render_clips(clips, quality="fast", progress_cb=None, max_workers=None):
    """Render all clips that need it, in parallel (pylibrb and the CLI
    both run outside the GIL). progress_cb(percent, message)."""
    todo = [c for c in clips if clip_needs_work(c, quality)]
    if not todo:
        return 0
    workers = max_workers or max(1, min(len(todo), os.cpu_count() or 2))
    pct = {id(c): 0 for c in todo}
    finished = [0]

    def clip_progress(c):
        def cb(p):
            pct[id(c)] = p
            if progress_cb:
                progress_cb(int(sum(pct.values()) / len(pct)),
                            "Rendering tempo/key: {} of {} done".format(
                                finished[0], len(todo)))
        return cb

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(ensure_clip_render, c, quality, clip_progress(c))
                for c in todo]
        for f in as_completed(futs):
            f.result()                      # re-raise worker errors
            finished[0] += 1
    return len(todo)
