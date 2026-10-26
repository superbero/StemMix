"""QThread workers."""
import os
import traceback

import numpy as np
import soundfile as sf

from PyQt6.QtCore import QThread, pyqtSignal

from core.separation import separate_file
from core.export import render_mix_to_wav, encode_with_ffmpeg


# ---------------------------------------------------------------------------
# Optional dependencies (guarded so module imports even when missing)
# ---------------------------------------------------------------------------

try:
    from mdxnet_infer import MDXNetSession
    MDXNET_AVAILABLE = True
    MDXNET_ERROR = None
except ImportError as e:
    MDXNetSession = None
    MDXNET_AVAILABLE = False
    MDXNET_ERROR = str(e)

try:
    import denote
    import celeritas
    CHORD_AVAILABLE = True
    CHORD_ERROR = None
except ImportError as e:
    denote = None
    celeritas = None
    CHORD_AVAILABLE = False
    CHORD_ERROR = str(e)


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

class DrumSplitWorker(QThread):
    """Splits a drums stem into kick/snare/toms/hh/ride/crash.

    Runs on CPU by design: mdxnet-infer does not support Apple MPS.
    On a CUDA machine, change device to 'cuda' for a big speedup.
    """
    finished_ok = pyqtSignal(dict)
    failed = pyqtSignal(str)
    progress = pyqtSignal(int, str)

    def __init__(self, drums_path, output_dir, device="cpu"):
        super().__init__()
        self.drums_path = drums_path
        self.output_dir = output_dir
        self.device = device

    def run(self):
        if not MDXNET_AVAILABLE:
            self.failed.emit(
                "mdxnet-infer not installed: " + str(MDXNET_ERROR))
            return

        try:
            os.makedirs(self.output_dir, exist_ok=True)

            self.progress.emit(
                5, "Loading drums file...")
            audio, sr = sf.read(
                self.drums_path, dtype="float32", always_2d=True)

            # mdxnet expects channels-first: (channels, frames)
            audio_cf = np.transpose(audio, (1, 0))

            self.progress.emit(
                15, "Loading model (first run downloads ~438 MB)...")
            stems = None
            with MDXNetSession(
                    model_name="drumsep-6stem",
                    device=self.device) as session:

                self.progress.emit(35, "Splitting drums...")
                stems = session.infer(audio_cf, sample_rate=sr)

            if not stems:
                self.failed.emit("Model returned no stems.")
                return

            self.progress.emit(85, "Writing stem files...")
            paths = {}
            for name, data in stems.items():
                # data is channels-first; convert back to frames-first
                data_cf = np.asarray(data)
                if data_cf.ndim == 2:
                    out = np.transpose(data_cf, (1, 0))
                else:
                    out = data_cf

                out_path = os.path.join(
                    self.output_dir, name + ".wav")
                sf.write(out_path, out, sr)
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
    """Transcribes a harmonic stem to notes, then groups them into chords."""
    finished_ok = pyqtSignal(list)  # list of (time_seconds, chord_name)
    failed = pyqtSignal(str)
    progress = pyqtSignal(int, str)

    def __init__(self, harmonic_stem_path):
        super().__init__()
        self.path = harmonic_stem_path

    def run(self):
        if not CHORD_AVAILABLE:
            self.failed.emit(
                "denote or celeritas not installed: "
                + str(CHORD_ERROR))
            return

        try:
            self.progress.emit(15, "Transcribing audio to notes...")
            result = denote.transcribe(self.path)

            notes = getattr(result, "notes", None) or []
            if not notes:
                self.progress.emit(100, "No notes detected.")
                self.finished_ok.emit([])
                return

            self.progress.emit(55, "Grouping notes into chords...")

            # Group note pitches into 0.5-second buckets
            buckets = {}
            for note in notes:
                start = float(getattr(note, "start_time", 0.0))
                pitch = getattr(note, "pitch", None)
                if pitch is None:
                    continue
                bucket = round(start * 2) / 2.0
                buckets.setdefault(bucket, []).append(
                    int(round(float(pitch))))

            self.progress.emit(75, "Identifying chord names...")
            chords = []
            for time_sec in sorted(buckets.keys()):
                pitches = sorted(set(buckets[time_sec]))
                if len(pitches) < 2:
                    continue
                try:
                    chord_name = celeritas.identify_chord(pitches)
                except Exception:
                    continue
                if chord_name:
                    chords.append((time_sec, str(chord_name)))

            self.progress.emit(
                100, "Detected {} chords.".format(len(chords)))
            self.finished_ok.emit(chords)
        except Exception as e:
            traceback.print_exc()
            self.failed.emit(str(e))
