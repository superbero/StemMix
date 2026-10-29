"""Chord recognition from audio. numpy + scipy only (no ML packages).

Pipeline: resample -> STFT -> tuning-corrected chroma -> cosine match
against chord templates -> Viterbi smoothing (chords are sticky) ->
merged segments. Works best on harmonic stems (other / guitar / piano /
bass); it is a template matcher, not a neural network, so expect it to
be good on clean material and to confuse close chords (e.g. C vs Cmaj7)
on dense mixes.
"""
import math
from math import gcd

import numpy as np
import soundfile as sf
from scipy import signal as sp_signal

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F",
              "F#", "G", "G#", "A", "A#", "B"]

# (name suffix, {semitone above root: weight}, log-prior)
CHORD_TYPES = [
    ("", {0: 1.0, 4: 0.9, 7: 0.8}, 0.0),
    ("m", {0: 1.0, 3: 0.9, 7: 0.8}, 0.0),
    ("7", {0: 1.0, 4: 0.9, 7: 0.8, 10: 0.65}, -0.10),
    ("maj7", {0: 1.0, 4: 0.9, 7: 0.8, 11: 0.65}, -0.10),
    ("m7", {0: 1.0, 3: 0.9, 7: 0.8, 10: 0.65}, -0.10),
]

TARGET_SR = 11025
N_FFT = 8192
HOP = 1024
FMIN, FMAX = 65.0, 2200.0


def _templates():
    """(n_states, 12) unit-norm templates, names, log-priors.
    State 0 is 'no chord'."""
    names, priors, rows = ["N"], [0.0], [np.zeros(12)]
    for root in range(12):
        for suffix, notes, prior in CHORD_TYPES:
            t = np.zeros(12)
            for iv, w in notes.items():
                t[(root + iv) % 12] = w
            rows.append(t / np.linalg.norm(t))
            names.append(NOTE_NAMES[root] + suffix)
            priors.append(prior)
    return np.array(rows), names, np.array(priors)


def resample_mono(y, sr, target=TARGET_SR):
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if sr != target:
        g = gcd(int(sr), target)
        y = sp_signal.resample_poly(y, target // g, int(sr) // g)
    return y.astype(np.float32)


def load_mix(paths_weights, progress_cb=None):
    """Mix several stems into one mono signal at TARGET_SR."""
    mix = None
    for i, (path, weight) in enumerate(paths_weights):
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        y = resample_mono(data, sr) * float(weight)
        if mix is None:
            mix = y
        else:
            n = min(len(mix), len(y))
            mix = mix[:n] + y[:n]
        if progress_cb:
            progress_cb(5 + int(20 * (i + 1) / len(paths_weights)),
                        "Reading audio...")
    return mix


def _stft_mag(y, progress_cb=None):
    pad = N_FFT // 2
    y = np.pad(y, (pad, pad))
    n_frames = 1 + (len(y) - N_FFT) // HOP
    win = np.hanning(N_FFT).astype(np.float32)
    view = np.lib.stride_tricks.sliding_window_view(y, N_FFT)[::HOP]
    mags = np.empty((n_frames, N_FFT // 2 + 1), dtype=np.float32)
    step = 128
    for i in range(0, n_frames, step):
        blk = view[i:i + step] * win
        mags[i:i + step] = np.abs(np.fft.rfft(blk, axis=1))
        if progress_cb:
            progress_cb(25 + int(35 * min(1.0, (i + step) / n_frames)),
                        "Analysing spectrum...")
    return mags


def compute_chroma(y, progress_cb=None):
    """Returns (chroma (T, 12), frame_energy (T,), seconds_per_frame)."""
    mags = _stft_mag(y, progress_cb)
    freqs = np.arange(mags.shape[1]) * TARGET_SR / N_FFT
    sel = (freqs >= FMIN) & (freqs <= FMAX)
    f = freqs[sel]
    m = mags[:, sel]
    midi = 69.0 + 12.0 * np.log2(f / 440.0)

    # Tuning: circular mean of the fractional semitone, weighted by the
    # long-term spectrum, so a recording that is 30 cents sharp still maps
    # to the right pitch classes.
    mean_spec = (m ** 2).mean(axis=0)
    ang = 2 * np.pi * (midi - np.round(midi))
    z = (mean_spec * np.exp(1j * ang)).sum()
    tuning = (np.angle(z) / (2 * np.pi)) if abs(z) > 0 else 0.0
    midi = midi - tuning

    frac = midi - np.round(midi)
    pc = np.mod(np.round(midi), 12).astype(int)
    w_close = np.clip(1.0 - 2.0 * np.abs(frac), 0.0, 1.0)
    w_freq = np.exp(-0.5 * (np.log2(f / 440.0) / 1.6) ** 2)
    proj = np.zeros((len(f), 12), dtype=np.float32)
    proj[np.arange(len(f)), pc] = (w_close * w_freq).astype(np.float32)

    energy = np.sqrt((m ** 2).sum(axis=1))
    ref = max(float(energy.max()), 1e-9)
    comp = np.log1p(30.0 * m / ref)          # log compression
    chroma = comp @ proj
    # Remove the per-frame floor: broadband noise lifts all 12 classes
    # equally, which otherwise makes 4-note templates (7ths) win.
    chroma = np.maximum(
        chroma - np.median(chroma, axis=1, keepdims=True), 0.0)
    norm = np.linalg.norm(chroma, axis=1, keepdims=True)
    chroma = chroma / np.maximum(norm, 1e-9)
    return chroma, energy / ref, HOP / float(TARGET_SR)


def _viterbi(emis, log_prior, p_stay):
    T, S = emis.shape
    log_stay = math.log(p_stay)
    log_sw = math.log((1.0 - p_stay) / (S - 1))
    delta = emis[0] + log_prior
    back = np.zeros((T, S), dtype=np.int32)
    idx = np.arange(S)
    for t in range(1, T):
        arg = int(delta.argmax())
        stay = delta + log_stay
        sw = delta[arg] + log_sw
        keep = stay >= sw
        back[t] = np.where(keep, idx, arg)
        delta = np.where(keep, stay, sw) + emis[t]
    path = np.empty(T, dtype=np.int32)
    path[-1] = int(delta.argmax())
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def detect_chords_from_signal(y, min_duration=0.5, p_stay=0.94,
                              progress_cb=None):
    """y: mono float signal at TARGET_SR. Returns [(start, end, name)]."""
    if progress_cb:
        progress_cb(25, "Analysing spectrum...")
    chroma, energy, dt = compute_chroma(y, progress_cb)
    if progress_cb:
        progress_cb(65, "Matching chords...")
    temps, names, priors = _templates()
    cos = chroma @ temps.T                    # (T, S), N column = 0
    tau = 0.09
    emis = cos / tau
    silent = energy < 0.03
    emis[:, 0] = np.where(silent, 6.0, -2.0)
    emis[silent, 1:] -= 6.0
    emis = emis - np.logaddexp.reduce(emis, axis=1, keepdims=True)
    if progress_cb:
        progress_cb(80, "Smoothing...")
    path = _viterbi(emis, priors, p_stay)

    # frame labels -> segments
    segs = []
    start = 0
    for t in range(1, len(path) + 1):
        if t == len(path) or path[t] != path[start]:
            segs.append([start * dt, t * dt, int(path[start])])
            start = t
    # absorb very short segments into their longer neighbour
    changed = True
    while changed and len(segs) > 1:
        changed = False
        for i, (a, b, s) in enumerate(segs):
            if b - a < min_duration:
                left = segs[i - 1][1] - segs[i - 1][0] if i > 0 else -1
                right = (segs[i + 1][1] - segs[i + 1][0]
                         if i < len(segs) - 1 else -1)
                j = i - 1 if left >= right else i + 1
                segs[j][0] = min(segs[j][0], a)
                segs[j][1] = max(segs[j][1], b)
                del segs[i]
                changed = True
                break
    merged = []
    for a, b, s in segs:
        if merged and merged[-1][2] == s:
            merged[-1][1] = b
        else:
            merged.append([a, b, s])
    out = [(float(a), float(b), names[s]) for a, b, s in merged if s != 0]
    if progress_cb:
        progress_cb(100, "Detected {} chords.".format(len(out)))
    return out


def detect_chords(paths_weights, progress_cb=None):
    """paths_weights: [(wav_path, weight), ...] harmonic stems to analyse."""
    y = load_mix(paths_weights, progress_cb)
    if y is None or len(y) < N_FFT:
        return []
    return detect_chords_from_signal(y, progress_cb=progress_cb)


def transpose_chord(name, semitones):
    """'Am' + 2 -> 'Bm'; 'G7' + -5 -> 'D7'."""
    if not semitones or name == "N":
        return name
    root = name[:2] if len(name) > 1 and name[1] == "#" else name[:1]
    if root not in NOTE_NAMES:
        return name
    return NOTE_NAMES[(NOTE_NAMES.index(root) + int(round(semitones))) % 12] \
        + name[len(root):]
