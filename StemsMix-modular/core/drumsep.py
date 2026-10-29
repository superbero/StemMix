"""Cached drum-separation engine (mdxnet-infer / MDX23C).

Loading the ~438 MB drumsep model dominates the first run. Keep one
shared instance per process and run inference with larger batches on
GPU/MPS so Split Drums is not stuck on the library default of batch=1.
"""
import threading

_LOCK = threading.Lock()
_ENGINE = None
_ENGINE_DEVICE = None


def preferred_device():
    """cuda > Apple MPS > cpu."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if (hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()):
            return "mps"
    except Exception:
        pass
    return "cpu"


def inference_tuning(device):
    """batch_size / overlap tuned for speed with acceptable quality.

    Library defaults are batch_size=1 and overlap=4 (very slow). On
    GPU/MPS we raise the batch and cut overlap to 2 (~2x fewer windows).
    """
    if device in ("cuda", "mps"):
        return {"batch_size": 8, "overlap": 2}
    return {"batch_size": 2, "overlap": 2}


def get_engine(device=None, progress_cb=None):
    """Return a process-wide cached MDX23CInference for drumsep-6stem."""
    global _ENGINE, _ENGINE_DEVICE
    from mdxnet_infer import MDX23CInference

    device = device or preferred_device()
    with _LOCK:
        if _ENGINE is not None and _ENGINE_DEVICE == device:
            if progress_cb:
                progress_cb(20, "Reusing cached drum model on {}...".format(
                    device))
            return _ENGINE

        if progress_cb:
            progress_cb(
                15,
                "Loading drum model on {} (first run downloads "
                "~438 MB)...".format(device))
        _ENGINE = MDX23CInference.from_pretrained(
            "drumsep-6stem", device=device, progress=False)
        _ENGINE_DEVICE = device
        return _ENGINE


def preload(device=None):
    """Warm the cache in the background. Safe to call at app startup."""
    try:
        get_engine(device=device)
        return preferred_device() if device is None else device
    except Exception:
        return None
