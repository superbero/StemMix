#!/usr/bin/env bash
# optimize-stemsmix.sh
# Applies MLX-first separation, background export, and UI speed hints
# to an existing StemsMix-modular project. Safe to run multiple times.
# Backs up every file it touches.

set -euo pipefail

# Sanity check: are we in the right folder?
if [ ! -d "core" ] || [ ! -d "ui" ] || [ ! -d "apps" ]; then
    echo "ERROR: Run this from the StemsMix-modular/ project root."
    exit 1
fi

STAMP=$(date +%s)

# ---------------------------------------------------------------------------
# Helper: back up a file
# ---------------------------------------------------------------------------
backup() {
    if [ -f "$1" ] && [ ! -f "$1.bak.$STAMP" ]; then
        cp "$1" "$1.bak.$STAMP"
        echo "   backed up: $1 -> $1.bak.$STAMP"
    fi
}

# ---------------------------------------------------------------------------
# 1. Rewrite core/separation.py with MLX-first logic
# ---------------------------------------------------------------------------
echo ""
echo ">> Optimizing core/separation.py ..."
backup "core/separation.py"

cat > core/separation.py <<'FILE_END'
"""Separation engine. Prefers demucs-mlx on Apple Silicon for speed."""
import os
import platform
import shutil

import numpy as np
import soundfile as sf

from .audio_io import convert_to_wav, needs_ffmpeg_conversion
from .resources import DEMUCS_IMPORT_ERROR

# --- demucs-mlx (Apple Silicon only) ---
_MLX_SEPARATOR = None
_MLX_MODEL_NAME = None
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

# --- demucs-onnx fallback ---
if DEMUCS_IMPORT_ERROR is None:
    import demucs_onnx
else:
    demucs_onnx = None


def separate_file(input_path, output_dir, model_name,
                  providers="auto", precision="fp32",
                  chunked=False, chunk_seconds=60, progress_cb=None):
    def emit(pct, msg):
        if progress_cb:
            progress_cb(int(pct), str(msg))

    force_cpu = providers in ("cpu", ["cpu"])
    use_mlx = MLX_AVAILABLE and not force_cpu

    if not use_mlx and DEMUCS_IMPORT_ERROR:
        raise RuntimeError(
            "No separation backend available. Install demucs-onnx "
            "or demucs-mlx.")

    converted = None
    input_to_use = input_path
    try:
        if needs_ffmpeg_conversion(input_path):
            emit(1, "Converting to WAV...")
            try:
                converted = convert_to_wav(input_path)
                input_to_use = converted
                emit(5, "Converted.")
            except RuntimeError as e:
                raise RuntimeError(str(e))

        os.makedirs(output_dir, exist_ok=True)

        if use_mlx:
            return _run_mlx(input_to_use, output_dir,
                            model_name, emit)
        if chunked:
            return _run_chunked(
                input_to_use, output_dir, model_name,
                providers, precision, chunk_seconds, emit)
        return _run_single_pass(
            input_to_use, output_dir, model_name,
            providers, precision, emit)
    finally:
        if converted and os.path.exists(converted):
            try:
                os.remove(converted)
            except OSError:
                pass


def _run_mlx(input_path, output_dir, model_name, emit):
    global _MLX_SEPARATOR, _MLX_MODEL_NAME

    if _MLX_SEPARATOR is None or _MLX_MODEL_NAME != model_name:
        emit(10, "Loading MLX model (one-time compile)...")
        _MLX_SEPARATOR = MLXSeparator(model=model_name)
        _MLX_MODEL_NAME = model_name
    else:
        emit(10, "Reusing cached MLX model...")

    emit(20, "Separating with Metal GPU...")
    origin, stems = _MLX_SEPARATOR.separate_audio_file(input_path)

    emit(90, "Writing stems...")
    out_paths = {}
    sr = getattr(_MLX_SEPARATOR, "samplerate", 44100)
    for name, audio in stems.items():
        out = np.transpose(audio) if audio.ndim == 2 else audio
        out_path = os.path.join(output_dir, name + ".wav")
        sf.write(out_path, out, sr)
        out_paths[name] = out_path

    emit(100, "Done")
    return out_paths


def _run_single_pass(input_path, output_dir, model_name,
                     providers, precision, emit):
    emit(10, "Loading model and separating...")
    kwargs = dict(model=model_name, providers=providers,
                  precision=precision, progress=False)
    emit(20, "Running separation (this may take a while)...")
    result = demucs_onnx.separate(input_path, output_dir, **kwargs)
    emit(95, "Finalizing...")
    normalized = _normalize_result(result)
    emit(100, "Done")
    return normalized


def _run_chunked(input_path, output_dir, model_name, providers,
                 precision, chunk_seconds, emit):
    emit(10, "Reading audio file...")
    audio, sr = sf.read(input_path, dtype="float32", always_2d=True)
    total_frames = audio.shape[0]
    chunk_frames = int(chunk_seconds * sr)
    n_chunks = max(1, (total_frames + chunk_frames - 1) // chunk_frames)

    stem_chunks = {}
    tmp_dir = os.path.join(output_dir, "_tmp_chunks")
    os.makedirs(tmp_dir, exist_ok=True)

    try:
        for i in range(n_chunks):
            start = i * chunk_frames
            end = min(start + chunk_frames, total_frames)
            chunk = audio[start:end]
            chunk_path = os.path.join(tmp_dir,
                                      "chunk_{:04d}.wav".format(i))
            sf.write(chunk_path, chunk, sr)

            pct = 10 + int((i / n_chunks) * 80)
            emit(pct, "Separating chunk {}/{}...".format(
                i + 1, n_chunks))

            kwargs = dict(model=model_name, providers=providers,
                          precision=precision, progress=False)
            result = demucs_onnx.separate(chunk_path, tmp_dir,
                                          **kwargs)

            for stem_name, stem_path in _normalize_result(
                    result).items():
                data, _ = sf.read(stem_path, dtype="float32",
                                  always_2d=True)
                stem_chunks.setdefault(stem_name, []).append(data)

            try:
                os.remove(chunk_path)
            except OSError:
                pass

        emit(92, "Writing final stems...")
        out_paths = {}
        for stem_name, chunks in stem_chunks.items():
            combined = np.concatenate(chunks, axis=0)
            out_path = os.path.join(output_dir, stem_name + ".wav")
            sf.write(out_path, combined, sr)
            out_paths[stem_name] = out_path

        emit(100, "Done")
        return out_paths
    finally:
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass


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
FILE_END
echo "   core/separation.py rewritten."

# ---------------------------------------------------------------------------
# 2. Add BackgroundExportWorker to ui/workers.py
# ---------------------------------------------------------------------------
echo ""
echo ">> Adding BackgroundExportWorker to ui/workers.py ..."
backup "ui/workers.py"

# Check if already present
if grep -q "class BackgroundExportWorker" ui/workers.py; then
    echo "   already present, skipping."
else
    cat >> ui/workers.py <<'FILE_END'


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
FILE_END
    echo "   BackgroundExportWorker appended."
fi

# ---------------------------------------------------------------------------
# 3. Update ui/dialogs.py warning text for speed
# ---------------------------------------------------------------------------
echo ""
echo ">> Updating ui/dialogs.py speed hints ..."
backup "ui/dialogs.py"

python3 - <<'PYEOF'
import re
path = "ui/dialogs.py"
with open(path, "r") as f:
    src = f.read()

old_htdemucs = '''        elif model_name == "htdemucs":
            self.force_cpu_cb.setChecked(False)
            self.warning_label.setText(
                "Fast 4-stem: single file, ~30% faster.")'''
new_htdemucs = '''        elif model_name == "htdemucs":
            self.force_cpu_cb.setChecked(False)
            self.warning_label.setText(
                "Fast 4-stem: MLX-accelerated on Apple Silicon. "
                "Leave 'Force CPU' unchecked for maximum speed.")'''

old_ft = '''        else:
            self.force_cpu_cb.setChecked(False)
            self.warning_label.setText(
                "High Quality 4-stem: best separation quality.")'''
new_ft = '''        else:
            self.force_cpu_cb.setChecked(False)
            self.warning_label.setText(
                "High Quality 4-stem: MLX-accelerated on Apple Silicon. "
                "Leave 'Force CPU' unchecked for maximum speed.")'''

src = src.replace(old_htdemucs, new_htdemucs)
src = src.replace(old_ft, new_ft)

with open(path, "w") as f:
    f.write(src)
print("   dialog hints updated.")
PYEOF

# ---------------------------------------------------------------------------
# 4. Update apps/daw.py to use BackgroundExportWorker
# ---------------------------------------------------------------------------
echo ""
echo ">> Updating apps/daw.py for background export ..."
backup "apps/daw.py"

python3 - <<'PYEOF'
import re
path = "apps/daw.py"
with open(path, "r") as f:
    src = f.read()

# 4a. Add BackgroundExportWorker to imports
old_import = '''from ui.workers import (ExportWorker, DrumSplitWorker,
                        ChordAnalysisWorker)'''
new_import = '''from ui.workers import (ExportWorker, DrumSplitWorker,
                        ChordAnalysisWorker, BackgroundExportWorker)'''
if old_import in src:
    src = src.replace(old_import, new_import)
    print("   import updated.")
else:
    # try alternate import layout
    if "BackgroundExportWorker" not in src:
        # append to whichever workers import exists
        m = re.search(r"from ui\.workers import \(([^)]+)\)", src)
        if m:
            new_list = m.group(1).rstrip() + ",\n                        BackgroundExportWorker"
            src = src.replace(m.group(0),
                              "from ui.workers import (" + new_list + ")")
            print("   import appended via regex.")
        else:
            print("   WARNING: could not find workers import.")

# 4b. Replace export_mix method body to use BackgroundExportWorker
old_export = '''    def export_mix(self):
        if not self.tracks:
            return
        dlg = ExportDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        fmt = dlg.selected_format()
        br = dlg.selected_bitrate()
        ext = EXPORT_FORMATS[fmt]["ext"]
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Mix As", "mix." + ext,
            EXPORT_FORMATS[fmt]["filter"])
        if not path:
            return
        if not path.lower().endswith("." + ext):
            path += "." + ext

        self.export_worker = ExportWorker(
            self.tracks, self.dsp_chains, path, fmt, br)
        self.export_worker.progress.connect(
            lambda p: self.status.setText(
                "Exporting... {}%".format(p)))
        self.export_worker.finished_ok.connect(
            lambda p: self.status.setText("Exported: " + p))
        self.export_worker.failed.connect(
            lambda e: self.status.setText("Export failed: " + e))
        self.export_worker.start()'''

new_export = '''    def export_mix(self):
        if not self.tracks:
            return
        dlg = ExportDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        fmt = dlg.selected_format()
        br = dlg.selected_bitrate()
        ext = EXPORT_FORMATS[fmt]["ext"]
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Mix As", "mix." + ext,
            EXPORT_FORMATS[fmt]["filter"])
        if not path:
            return
        if not path.lower().endswith("." + ext):
            path += "." + ext

        # Background export: playback continues while rendering
        self.status.setText(
            "Export running in background...")
        self.bg_export = BackgroundExportWorker(
            self.tracks, self.dsp_chains, path, fmt, br)
        self.bg_export.progress.connect(
            lambda p: self.status.setText(
                "Exporting... {}%".format(p)))
        self.bg_export.finished_ok.connect(
            lambda p: self.status.setText("Exported: " + p))
        self.bg_export.failed.connect(
            lambda e: self.status.setText("Export failed: " + e))
        self.bg_export.start()'''

if old_export in src:
    src = src.replace(old_export, new_export)
    print("   export_mix replaced.")
else:
    print("   WARNING: export_mix block not found verbatim.")
    # fallback: just check if it still uses ExportWorker
    if "self.export_worker = ExportWorker" in src:
        print("   (still uses synchronous ExportWorker — manual edit needed)")

with open(path, "w") as f:
    f.write(src)
PYEOF

# ---------------------------------------------------------------------------
# 5. Verify everything parses
# ---------------------------------------------------------------------------
echo ""
echo ">> Verifying syntax ..."
FAIL=0
for f in core/separation.py ui/workers.py ui/dialogs.py apps/daw.py; do
    if python3 -c "import ast; ast.parse(open('$f').read())" 2>/dev/null; then
        echo "   $f: OK"
    else
        echo "   $f: SYNTAX ERROR"
        python3 -c "import ast; ast.parse(open('$f').read())" || true
        FAIL=1
    fi
done

if [ "$FAIL" -ne 0 ]; then
    echo ""
    echo "One or more files failed to parse. Backups are at *.bak.$STAMP"
    exit 1
fi

# ---------------------------------------------------------------------------
# 6. Check MLX availability
# ---------------------------------------------------------------------------
echo ""
echo ">> Checking demucs-mlx availability ..."
python3 -c "
try:
    from demucs_mlx import Separator
    print('   demucs-mlx: INSTALLED')
except ImportError:
    print('   demucs-mlx: NOT INSTALLED')
    print('   To enable MLX acceleration: pip install demucs-mlx')
"

# ---------------------------------------------------------------------------
# 7. Summary
# ---------------------------------------------------------------------------
echo ""
echo "=================================================="
echo " Optimization applied."
echo ""
echo " Changes:"
echo "   - core/separation.py: MLX-first with cached Separator"
echo "   - ui/workers.py: BackgroundExportWorker added"
echo "   - ui/dialogs.py: speed hints in model dialog"
echo "   - apps/daw.py: export no longer blocks playback"
echo ""
echo " Backups: *.bak.$STAMP"
echo " Roll back: cp <file>.bak.$STAMP <file>"
echo "=================================================="
