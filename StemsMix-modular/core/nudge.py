"""Nudge and quantize helpers for the selected clip."""
import math


def quantize_time(t, beats, mode="nearest"):
    if not beats:
        return t
    import bisect
    i = bisect.bisect_left(beats, t)
    if i == 0:
        return beats[0]
    if i >= len(beats):
        return beats[-1]
    if mode == "floor":
        return beats[i - 1]
    if mode == "ceil":
        return beats[i]
    return beats[i] if abs(beats[i] - t) < abs(t - beats[i - 1]) else beats[i - 1]


def nudge_seconds(t, seconds, min_time=0.0):
    return max(min_time, t + seconds)


def quantize_by_fraction(t, bpm, fraction=1.0, mode="nearest"):
    """Snap t to a fraction of a beat. fraction=1 -> beats, 0.5 -> 8ths."""
    if bpm is None or bpm <= 0:
        return t
    step = (60.0 / bpm) * fraction
    if step <= 0:
        return t
    if mode == "floor":
        return math.floor(t / step) * step
    if mode == "ceil":
        return math.ceil(t / step) * step
    return round(t / step) * step
