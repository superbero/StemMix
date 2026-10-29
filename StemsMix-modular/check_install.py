"""Run from the project root:  python3 check_install.py
Reports which files are missing or out of date, and which optional
packages are available."""
import importlib, os, platform, sys
sys.path.insert(0, os.getcwd())
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

NEEDED = {
    "ui.style":            ["COLORS", "QSS", "SCROLLBAR"],
    "ui.widgets":          ["Knob", "Fader", "LevelMeter", "Stepper", "make_icon"],
    "ui.timeline":         ["TimelineScene", "TimelineView", "TimeRuler", "ChordStrip",
                            "TrackHeader", "TrackHeaderColumn", "HEADER_WIDTH",
                            "RULER_HEIGHT", "time_to_x"],
    "ui.channel_strip":    ["MixerPanel", "ChannelStrip"],
    "ui.workers":          ["SeparationWorker", "DrumSplitWorker",
                            "ChordAnalysisWorker", "RenderWorker",
                            "BeatAnalysisWorker"],
    "ui.dialogs":          ["ModelSelectionDialog", "ExportDialog"],
    "core.engine":         ["TimelineEngine"],
    "core.dsp":            ["DSPChain"],
    "core.timeline_model": ["Project", "Track", "Clip"],
    "core.separation":     ["separate_file"],
    "core.chords":         ["detect_chords", "transpose_chord"],
    "core.timestretch":    ["process", "render_clips", "available_backend"],
    "core.beats":          ["analyze_beats", "bpm_from_beats", "halve", "double"],
    "core.resources":      ["run_prerequisite_checks"],
}
bad = 0
for mod, names in NEEDED.items():
    try:
        m = importlib.import_module(mod)
    except Exception as e:
        print("ERROR  %-22s cannot import: %s: %s" % (mod, type(e).__name__, e)); bad += 1; continue
    missing = [n for n in names if not hasattr(m, n)]
    if missing:
        print("STALE  %-22s missing %s\n       -> replace %s" % (mod, missing, m.__file__)); bad += 1
    else:
        print("ok     %s" % mod)

print("\nEnvironment: Python %s, %s %s" % (platform.python_version(), platform.system(), platform.machine()))
try:
    from core.timestretch import available_backend
    b = available_backend()
    msg = {"pylibrb": "pylibrb (best: in-process Rubber Band)",
           "rubberband": "rubberband command-line tool",
           "librosa": "librosa fallback only (lower quality)"}.get(b)
    if msg:
        print("Tempo/Key backend: " + msg)
    else:
        hint = ("pip3 install pylibrb" if not (platform.system() == "Darwin" and platform.machine() == "x86_64")
                else "brew install rubberband   (no pylibrb wheel exists for Intel Macs)")
        print("Tempo/Key backend: NONE - tempo and key change will be disabled.\n                   Install:  " + hint)
except Exception as e:
    print("Tempo/Key backend: could not check (%s)" % e)
print("\nAll files are current." if not bad else "\n%d problem(s): copy the listed files from the update." % bad)
