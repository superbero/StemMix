"""Tempo and beat detection. numpy + scipy only (no ML packages).

Pipeline: resample -> log-band spectral flux (onset envelope) ->
autocorrelation tempo estimate with a "typical tempo" prior ->
dynamic-programming beat tracker (Ellis 2007) -> if the tempo is steady,
fit a regular grid (a click that jitters by 20 ms sounds sloppy) ->
downbeat guess from low-frequency (kick) strength.

Returns beat times in the *original* audio's time; the app scales them by
the project tempo.
"""
import math

import numpy as np
import soundfile as sf
from scipy import signal as sp_signal

from .chords import resample_mono

SR = 22050
N_FFT = 1024
HOP = 256
FPS = SR / float(HOP)
BPM_MIN, BPM_MAX = 55.0, 200.0
PRIOR_BPM = 120.0

# Seconds added to detected beat times so clicks land on the transients
# (measured with synthetic drum tracks; see tests).
ONSET_LATENCY = 0.011


def load_mix(paths_weights, progress_cb=None):
    mix = None
    for i, (path, weight) in enumerate(paths_weights):
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        y = resample_mono(data, sr, SR) * float(weight)
        if mix is None:
            mix = y
        else:
            n = min(len(mix), len(y))
            mix = mix[:n] + y[:n]
        if progress_cb:
            progress_cb(5 + int(25 * (i + 1) / len(paths_weights)),
                        "Reading audio...")
    return mix


def _band_matrix(n_bins, n_bands=28, fmin=30.0, fmax=9000.0):
    freqs = np.arange(n_bins) * SR / float(N_FFT)
    edges = np.geomspace(fmin, fmax, n_bands + 1)
    m = np.zeros((n_bins, n_bands), dtype=np.float32)
    for b in range(n_bands):
        sel = (freqs >= edges[b]) & (freqs < edges[b + 1])
        if not sel.any():
            sel = np.zeros(n_bins, bool)
            sel[int(np.argmin(np.abs(freqs - edges[b])))] = True
        m[sel, b] = 1.0
    centers = np.sqrt(edges[:-1] * edges[1:])
    return m, centers


def band_log_energy(y, progress_cb=None):
    pad = N_FFT // 2
    y = np.pad(y, (pad, pad))
    n_frames = 1 + (len(y) - N_FFT) // HOP
    win = np.hanning(N_FFT).astype(np.float32)
    view = np.lib.stride_tricks.sliding_window_view(y, N_FFT)[::HOP]
    bm, centers = _band_matrix(N_FFT // 2 + 1)
    out = np.empty((n_frames, bm.shape[1]), dtype=np.float32)
    step = 1024
    for i in range(0, n_frames, step):
        mag = np.abs(np.fft.rfft(view[i:i + step] * win, axis=1))
        out[i:i + step] = (mag ** 2) @ bm
        if progress_cb:
            progress_cb(30 + int(30 * min(1.0, (i + step) / n_frames)),
                        "Finding onsets...")
    db = 10.0 * np.log10(out + 1e-10)
    db = np.maximum(db, db.max() - 70.0)
    return db, centers


def _clean(env):
    """Remove the slow trend and normalise."""
    k = max(3, int(0.4 * FPS))
    trend = sp_signal.convolve(env, np.ones(k) / k, mode="same")
    env = np.maximum(env - trend, 0.0)
    s = env.std()
    return env / s if s > 1e-9 else env


def onset_envelopes(db, centers):
    flux = np.maximum(db[1:] - db[:-1], 0.0)
    flux = np.vstack([np.zeros((1, db.shape[1]), dtype=flux.dtype), flux])
    full = _clean(flux.mean(axis=1))
    low = _clean(flux[:, centers < 200.0].mean(axis=1)
                 if (centers < 200.0).any() else flux.mean(axis=1))
    return full, low


def estimate_tempo(env):
    n = len(env)
    x = env - env.mean()
    spec = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(np.abs(spec) ** 2)[:n]
    if ac[0] <= 0:
        return PRIOR_BPM
    ac = ac / ac[0]
    lags = np.arange(1, n)
    bpm = 60.0 * FPS / lags
    score = ac[1:].copy()
    for mult, wt in ((2, 0.5), (4, 0.25)):
        idx = lags * mult
        ok = idx < n
        score[ok] += wt * ac[idx[ok]]
    prior = np.exp(-0.5 * (np.log2(bpm / PRIOR_BPM) / 1.1) ** 2)
    valid = (bpm >= BPM_MIN) & (bpm <= BPM_MAX)
    score = np.where(valid, score * prior, -np.inf)
    best = int(np.argmax(score)) + 1
    # parabolic refinement of the autocorrelation peak
    if 1 < best < n - 1:
        a, b, c = ac[best - 1], ac[best], ac[best + 1]
        denom = a - 2 * b + c
        if abs(denom) > 1e-12:
            best = best + 0.5 * (a - c) / denom
    return 60.0 * FPS / best


def track_beats(env, bpm, tightness=100.0):
    """Ellis-style dynamic-programming beat tracker. Returns frame idx."""
    n = len(env)
    period = 60.0 * FPS / bpm
    sd = max(1.0, period / 32.0)
    kern = np.exp(-0.5 * (np.arange(-int(3 * sd), int(3 * sd) + 1) / sd) ** 2)
    local = sp_signal.convolve(env, kern, mode="same")
    offs = np.arange(max(1, int(round(period / 2))),
                     int(round(2 * period)) + 1)
    txwt = -tightness * np.log(offs / period) ** 2
    cum = np.zeros(n)
    back = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        j = i - offs
        v = j >= 0
        if v.any():
            cand = cum[j[v]] + txwt[v]
            k = int(cand.argmax())
            cum[i] = local[i] + cand[k]
            back[i] = j[v][k]
        else:
            cum[i] = local[i]
    peaks = np.nonzero((cum[1:-1] > cum[:-2]) & (cum[1:-1] >= cum[2:]))[0] + 1
    if len(peaks) == 0:
        return np.array([], dtype=np.int64)
    thr = 0.5 * np.median(cum[peaks])
    last = peaks[cum[peaks] >= thr][-1]
    beats = [last]
    while back[beats[-1]] >= 0:
        beats.append(back[beats[-1]])
    return np.array(beats[::-1], dtype=np.int64)


def fit_grid(beats, duration):
    """If the tracked beats lie on a straight line (steady tempo), return
    a regular grid covering [0, duration] and the period; else None."""
    if len(beats) < 8:
        return None
    k = np.arange(len(beats), dtype=float)
    keep = np.ones(len(beats), bool)
    for _ in range(3):
        if keep.sum() < 8:
            return None
        p, t0 = np.polyfit(k[keep], beats[keep], 1)
        resid = beats - (t0 + p * k)
        keep = np.abs(resid) < 0.05
    if keep.mean() < 0.8 or np.percentile(np.abs(resid[keep]), 90) > 0.03:
        return None
    k_min = int(math.ceil(-t0 / p))        # first grid beat at t >= 0
    k_max = int(math.ceil((duration - t0) / p))
    ks = np.arange(k_min, k_max + 1)
    times = t0 + p * ks
    return times[(times >= 0) & (times <= duration)], p


def estimate_downbeat_phase(low_env, beat_times, meter=4):
    """Which beat (0..meter-1) is the downbeat: the one whose position in
    the bar carries the strongest low-frequency (kick) onsets."""
    fr = np.round(np.asarray(beat_times) * FPS).astype(int)
    fr = np.clip(fr, 1, len(low_env) - 2)
    strength = np.maximum.reduce([low_env[fr - 1], low_env[fr],
                                  low_env[fr + 1]])
    scores = [strength[p::meter].mean() if len(strength[p::meter]) else 0.0
              for p in range(meter)]
    return int(np.argmax(scores))


def bpm_from_beats(beats):
    beats = np.asarray(beats)
    if len(beats) < 2:
        return None
    return 60.0 / float(np.median(np.diff(beats)))


def halve(beats):
    return np.asarray(beats)[::2]


def double(beats):
    b = np.asarray(beats)
    if len(b) < 2:
        return b
    mid = (b[:-1] + b[1:]) / 2.0
    out = np.empty(len(b) + len(mid))
    out[0::2], out[1::2] = b, mid
    return out


def analyze_signal(y, progress_cb=None):
    """y: mono float signal at SR. Returns a dict or None."""
    duration = len(y) / float(SR)
    if duration < 6.0:
        return None
    db, centers = band_log_energy(y, progress_cb)
    if progress_cb:
        progress_cb(65, "Estimating tempo...")
    env, low = onset_envelopes(db, centers)
    if env.max() <= 0:
        return None
    bpm0 = estimate_tempo(env)
    if progress_cb:
        progress_cb(75, "Tracking beats...")
    frames = track_beats(env, bpm0)
    if len(frames) < 4:
        return None
    # frame t is centred at t*HOP/SR
    beats = frames * HOP / float(SR) + ONSET_LATENCY
    steady = False
    fit = fit_grid(beats, duration)
    if fit is not None:
        beats, _ = fit
        steady = True
    bpm = bpm_from_beats(beats) or bpm0
    phase = estimate_downbeat_phase(low, beats)
    if progress_cb:
        progress_cb(100, "{:.1f} BPM".format(bpm))
    return {"bpm": float(bpm), "beats": np.asarray(beats, dtype=float),
            "steady": steady, "phase": int(phase), "duration": duration}


def analyze_beats(paths_weights, progress_cb=None):
    y = load_mix(paths_weights, progress_cb)
    if y is None:
        return None
    return analyze_signal(y, progress_cb)
