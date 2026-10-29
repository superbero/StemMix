"""QThread workers."""
import os
import traceback

import numpy as np
import soundfile as sf

from PyQt6.QtCore import QThread, pyqtSignal

from core.separation import separate_file
from core.chords import detect_chords
from core.beats import analyze_beats
from core.timestretch import render_clips
from core.export import render_mix_to_wav, encode_with_ffmpeg


# ---------------------------------------------------------------------------
# Optional dependencies (guarded so module imports even when missing)
# ---------------------------------------------------------------------------

try:
    from mdxnet_infer import MDX23CInference
    MDXNET_AVAILABLE = True
    MDXNET_ERROR = None
except ImportError as e:
    MDX23CInference = None
    MDXNET_AVAILABLE = False
    MDXNET_ERROR = str(e)


# ---------------------------------------------------------------------------
# Separation worker (4-stem / 6-stem)
# ---------------------------------------------------------------------------

class SeparationWorker(QThread):
    finished_ok = pyqtSignal(dict)
    failed = pyqtSignal(str)
    progress = pyqtSignal(int, str)

    def __init__(self, input_file, output_dir, model_name,
                 providers="auto", precision="fp32", chunked=False):
        super().__init__()
        self.input_file = input_file
        self.output_dir = output_dir
        self.model_name = model_name
        # "auto" is passed through untouched. core.separation uses it to
        # pick the MLX backend on Apple Silicon, and maps it to the CPU
        # execution provider for ONNX (CoreML cannot compile the Demucs
        # graph: "generic_general_slice: Invalid values 2048").
        self.providers = providers
        self.precision = precision
        self.chunked = chunked

    def run(self):
        try:
            def cb(pct, msg):
                self.progress.emit(pct, msg)
            result = separate_file(
                self.input_file, self.output_dir, self.model_name,
                providers=self.providers, precision=self.precision,
                chunked=self.chunked, progress_cb=cb)
            self.finished_ok.emit(result)
        except Exception as e:
            traceback.print_exc()
            self.failed.emit(str(e))


# ---------------------------------------------------------------------------
# Export worker
# ---------------------------------------------------------------------------

class ExportWorker(QThread):
    progress = pyqtSignal(int)
    finished_ok = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, tracks, dsp_chains, output_path,
                 fmt_key, bitrate):
        super().__init__()
        self.tracks = tracks
        self.dsp_chains = dsp_chains
        self.output_path = output_path
        self.fmt_key = fmt_key
        self.bitrate = bitrate

    def run(self):
        tmp_wav = self.output_path + ".tmp.wav"
        try:
            def p1(p):
                self.progress.emit(int(p * 0.6))
            ok = render_mix_to_wav(
                self.tracks, self.dsp_chains, tmp_wav,
                progress_cb=p1)
            if not ok:
                self.failed.emit("No active tracks to export.")
                return

            def p2(p):
                self.progress.emit(60 + int(p * 0.4))
            ok = encode_with_ffmpeg(
                tmp_wav, self.output_path, self.fmt_key,
                self.bitrate, progress_cb=p2)
            if not ok:
                self.failed.emit("FFmpeg encoding failed.")
                return
            self.finished_ok.emit(self.output_path)
        except Exception as e:
            traceback.print_exc()
            self.failed.emit(str(e))
        finally:
            if os.path.exists(tmp_wav):
                try:
                    os.remove(tmp_wav)
                except OSError:
                    pass


# ---------------------------------------------------------------------------
# Drum splitting worker
# ---------------------------------------------------------------------------

def _preferred_torch_device():
    """Pick the fastest torch device available (cuda > mps > cpu)."""
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


class DrumSplitWorker(QThread):
    """Splits a drums stem into kick/snare/toms/hh/ride/crash.

    Uses mdxnet-infer's MDX23CInference. device=None auto-picks
    cuda / Apple MPS / cpu (MPS is typically many times faster than CPU).
    """
    finished_ok = pyqtSignal(dict)
    failed = pyqtSignal(str)
    progress = pyqtSignal(int, str)

    def __init__(self, drums_path, output_dir, device=None):
        super().__init__()
        self.drums_path = drums_path
        self.output_dir = output_dir
        self.device = device

    def run(self):
        if not MDXNET_AVAILABLE:
            self.failed.emit(
                "mdxnet-infer is not usable: " + str(MDXNET_ERROR)
                + "  (install with: pip install mdxnet-infer)")
            return

        try:
            os.makedirs(self.output_dir, exist_ok=True)
            device = self.device or _preferred_torch_device()

            self.progress.emit(5, "Loading drums file...")
            audio, sr = sf.read(
                self.drums_path, dtype="float32", always_2d=True)
            # The model wants (frames, 2).
            if audio.shape[1] == 1:
                audio = np.repeat(audio, 2, axis=1)
            elif audio.shape[1] > 2:
                audio = audio[:, :2]

            self.progress.emit(
                15, "Loading model on {} (first run downloads "
                    "~438 MB)...".format(device))
            engine = MDX23CInference.from_pretrained(
                "drumsep-6stem", device=device, progress=False)

            self.progress.emit(
                35, "Splitting drums on {}...".format(device))
            stems = engine.separate(
                audio, sample_rate=sr, progress=False)
            if not stems:
                self.failed.emit("Model returned no stems.")
                return

            # The model resamples to its own rate (44100), so write the
            # parts at that rate, not at the input file's.
            out_sr = engine.config.audio.sample_rate

            self.progress.emit(85, "Writing stem files...")
            paths = {}
            for name, data in stems.items():
                out_path = os.path.join(
                    self.output_dir, name + ".wav")
                sf.write(out_path, np.asarray(data), out_sr)
                paths[name] = out_path

            self.progress.emit(100, "Drum split complete.")
            self.finished_ok.emit(paths)
        except Exception as e:
            traceback.print_exc()
            self.failed.emit(str(e))


# ---------------------------------------------------------------------------
# Chord detection worker
# ---------------------------------------------------------------------------

class ChordAnalysisWorker(QThread):
    """Detects chords from the harmonic stems (core.chords: chroma +
    template matching + Viterbi; numpy/scipy only, no ML packages)."""
    finished_ok = pyqtSignal(list)  # [(start_s, end_s, chord_name)]
    failed = pyqtSignal(str)
    progress = pyqtSignal(int, str)

    def __init__(self, paths_weights):
        super().__init__()
        self.paths_weights = paths_weights

    def run(self):
        try:
            chords = detect_chords(
                self.paths_weights,
                progress_cb=lambda p, m: self.progress.emit(int(p), m))
            self.finished_ok.emit(chords)
        except Exception as e:
            traceback.print_exc()
            self.failed.emit(str(e))


class BeatAnalysisWorker(QThread):
    """Finds the tempo and beat positions (core.beats)."""
    finished_ok = pyqtSignal(object)   # dict, or None if no beat was found
    failed = pyqtSignal(str)
    progress = pyqtSignal(int, str)

    def __init__(self, paths_weights):
        super().__init__()
        self.paths_weights = paths_weights

    def run(self):
        try:
            res = analyze_beats(
                self.paths_weights,
                progress_cb=lambda p, m: self.progress.emit(int(p), m))
            self.finished_ok.emit(res)
        except Exception as e:
            traceback.print_exc()
            self.failed.emit(str(e))


class RenderWorker(QThread):
    """Renders tempo/key changes for a set of clips ahead of playback."""
    progress = pyqtSignal(int, str)
    finished_ok = pyqtSignal(int)
    failed = pyqtSignal(str)

    def __init__(self, clips, quality="fast"):
        super().__init__()
        self.clips = list(clips)
        self.quality = quality

    def run(self):
        try:
            n = render_clips(
                self.clips, self.quality,
                progress_cb=lambda p, m: self.progress.emit(int(p), m))
            self.finished_ok.emit(n)
        except Exception as e:
            traceback.print_exc()
            self.failed.emit(str(e))


class BackgroundExportWorker(QThread):
    """Runs export without blocking playback."""
    finished_ok = pyqtSignal(str)
    failed = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, tracks, dsp_chains, output_path,
                 fmt_key, bitrate):
        super().__init__()
        self.tracks = tracks
        self.dsp_chains = dsp_chains
        self.output_path = output_path
        self.fmt_key = fmt_key
        self.bitrate = bitrate

    def run(self):
        from core.export import (render_mix_to_wav,
                                 encode_with_ffmpeg)
        tmp = self.output_path + ".tmp.wav"
        try:
            def p1(p):
                self.progress.emit(int(p * 0.6))
            ok = render_mix_to_wav(
                self.tracks, self.dsp_chains, tmp,
                progress_cb=p1)
            if not ok:
                self.failed.emit("Render failed.")
                return

            def p2(p):
                self.progress.emit(60 + int(p * 0.4))
            ok = encode_with_ffmpeg(
                tmp, self.output_path, self.fmt_key,
                self.bitrate, progress_cb=p2)
            if not ok:
                self.failed.emit("Encode failed.")
                return
            self.finished_ok.emit(self.output_path)
        except Exception as e:
            self.failed.emit(str(e))
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
