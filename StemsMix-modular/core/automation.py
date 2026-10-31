"""Track automation point validation and evaluation."""
import math

import numpy as np


AUTOMATION_DEFAULTS = {"volume": 1.0, "pan": 0.0}
AUTOMATION_LIMITS = {"volume": (0.0, 1.5), "pan": (-1.0, 1.0)}


def normalize_points(points, parameter):
    """Return finite, sorted [time, value] pairs for a supported lane."""
    if parameter not in AUTOMATION_LIMITS:
        raise ValueError("Unsupported automation parameter: " + str(parameter))
    low, high = AUTOMATION_LIMITS[parameter]
    normalized = {}
    if isinstance(points, (list, tuple)):
        for point in points:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                continue
            try:
                time, value = float(point[0]), float(point[1])
            except (TypeError, ValueError):
                continue
            if not math.isfinite(time) or not math.isfinite(value):
                continue
            normalized[max(0.0, time)] = min(high, max(low, value))
    return [[time, normalized[time]] for time in sorted(normalized)]


def evaluate_points(points, parameter, start_frame, frame_count, samplerate,
                    default=None):
    """Interpolate automation for a contiguous block of audio frames."""
    base = (AUTOMATION_DEFAULTS[parameter] if default is None
            else float(default))
    if frame_count <= 0:
        return np.zeros(0, dtype=np.float32)
    points = normalize_points(points, parameter)
    if not points:
        return np.full(frame_count, base, dtype=np.float32)

    times = np.asarray([point[0] for point in points], dtype=np.float64)
    values = np.asarray([point[1] for point in points], dtype=np.float64)
    if times[0] > 0:
        times = np.insert(times, 0, 0.0)
        values = np.insert(values, 0, base)

    frame_times = (start_frame + np.arange(frame_count)) / float(samplerate)
    result = np.interp(frame_times, times, values)
    return result.astype(np.float32)
