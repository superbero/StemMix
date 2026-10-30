"""Save / load a StemsMix project as a .stemmix JSON file.

The audio is NOT embedded. Each clip stores the path to its source WAV
plus every editable parameter (start, tempo, key, gain, mute, solo).
Loading re-reads the audio from disk and reapplies the parameters.

A project file references WAVs on disk. If you move a project, move the
stems folder and the .stemmix file together, or re-save after moving.

Schema version 1:

{
  "format": "stemmix",
  "version": 1,
  "saved_at": "2026-10-28T14:32:11",
  "samplerate": 44100,
  "project": {
      "tempo": 1.0, "key": 0.0, "master_volume": 1.0,
      "bpm": null, "beats": null, "click_enabled": false,
      "click_volume": 0.6, "beats_per_bar": 4, "downbeat_phase": 0
  },
  "tracks": [
     {
       "name": "vocals", "volume": 1.0, "pan": 0.0,
       "muted": false, "soloed": false,
       "dsp": { "eq_bass": 0.0, ..., "reverb_mix": 0.0 },
       "clips": [
          {
            "stem_name": "vocals",
            "filepath": "/path/to/vocals.wav",
            "start_time": 0.0,
            "stretch": 1.0,
            "semitones": 0.0,
            "gain": 1.0,
            "muted": false,
            "soloed": false,
            "color_hint": null,
            "missing": false
          }
       ]
     }
  ]
}
"""
import datetime
import json
import os

from .audio_io import load_audio
from .timeline_model import Clip, Track, Project


FORMAT_NAME = "stemmix"
FORMAT_VERSION = 1
FILE_EXT = ".stemmix"


class ProjectLoadError(Exception):
    """Raised when a project file is malformed or unreadable."""


class ProjectLoadResult:
    """Returned by load_project.

    project     : the reconstructed Project, ready to install
    dsp_params  : {track_name: {attribute: value}} to feed DSPChain
    missing     : list of (track_name, filepath) for clips whose WAV
                  could not be found or read. These clips are present in
                  the project with missing=True; the engine skips them.
    path        : the file path that was loaded
    """

    def __init__(self, project, dsp_params, missing, path):
        self.project = project
        self.dsp_params = dsp_params
        self.missing = missing
        self.path = path


# --------------------------------------------------------------------------
# Saving
# --------------------------------------------------------------------------

def _dsp_to_dict(dsp):
    return {
        "eq_bass": float(dsp.eq_bass),
        "eq_mid": float(dsp.eq_mid),
        "eq_treble": float(dsp.eq_treble),
        "comp_threshold": float(dsp.comp_threshold),
        "comp_ratio": float(dsp.comp_ratio),
        "comp_attack": float(dsp.comp_attack),
        "comp_release": float(dsp.comp_release),
        "reverb_mix": float(dsp.reverb_mix),
    }


def _clip_to_dict(clip):
    return {
        "stem_name": clip.stem_name,
        "filepath": os.path.abspath(clip.filepath),
        "start_time": float(clip.start_time),
        "stretch": float(clip.stretch),
        "semitones": float(clip.semitones),
        "gain": float(clip.gain),
        "muted": bool(clip.muted),
        "soloed": bool(clip.soloed),
        "color_hint": clip.color_hint,
        "effects": (None if clip.effects is None
                    else _dsp_to_dict(clip.effects)),
    }

def _track_to_dict(track, dsp):
    return {
        "name": track.name,
        "volume": float(track.volume),
        "pan": float(track.pan),
        "muted": bool(track.muted),
        "soloed": bool(track.soloed),
        "dsp": _dsp_to_dict(dsp),
        "clips": [_clip_to_dict(c) for c in track.clips],
    }


def serialize_project(project, dsp_chains):
    """Return the dict that save_project writes. Separated so tests can
    inspect it without touching disk."""
    beats = project.beat_times
    if beats is not None:
        beats = [float(b) for b in beats]

    return {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "samplerate": int(project.samplerate),
        "project": {
            "tempo": float(project.tempo),
            "key": float(project.key),
            "master_volume": float(project.master_volume),
            "bpm": (None if project.bpm is None else float(project.bpm)),
            "beats": beats,
            "click_enabled": bool(project.click_enabled),
            "click_volume": float(project.click_volume),
            "beats_per_bar": int(project.beats_per_bar),
            "downbeat_phase": int(project.downbeat_phase),
        },
        "tracks": [
            _track_to_dict(t, dsp_chains[t.name])
            for t in project.tracks
            if t.name in dsp_chains
        ],
    }


def save_project(path, project, dsp_chains):
    """Write a .stemmix file. Ensures the extension, writes atomically
    (temp file + rename) so a crash mid-save cannot truncate the file."""
    if not path.lower().endswith(FILE_EXT):
        path = path + FILE_EXT

    data = serialize_project(project, dsp_chains)
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    return path


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def _validate_header(data):
    if not isinstance(data, dict):
        raise ProjectLoadError("File is not a JSON object.")
    if data.get("format") != FORMAT_NAME:
        raise ProjectLoadError(
            "This is not a StemsMix project file "
            "(expected format {!r}, got {!r}).".format(
                FORMAT_NAME, data.get("format")))
    version = data.get("version")
    if not isinstance(version, int) or version > FORMAT_VERSION:
        raise ProjectLoadError(
            "Project version {} is newer than this app supports ({}). "
            "Update StemsMix to open it.".format(version, FORMAT_VERSION))


def load_project(path):
    """Read a .stemmix file and reconstruct a Project.

    Clips whose WAV is missing are kept (missing=True) and reported in
    `missing` so the UI can warn. Everything else loads normally.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise ProjectLoadError("File not found: " + path)
    except json.JSONDecodeError as e:
        raise ProjectLoadError("Not valid JSON: " + str(e))

    _validate_header(data)

    project = Project()
    project.samplerate = int(data.get("samplerate", 44100))
    pdata = data.get("project", {}) or {}

    project.tempo = float(pdata.get("tempo", 1.0))
    project.key = float(pdata.get("key", 0.0))
    project.master_volume = float(pdata.get("master_volume", 1.0))

    raw_bpm = pdata.get("bpm")
    project.bpm = None if raw_bpm is None else float(raw_bpm)

    raw_beats = pdata.get("beats")
    if raw_beats:
        project.beat_times = [float(b) for b in raw_beats]
    else:
        project.beat_times = None

    project.click_enabled = bool(pdata.get("click_enabled", False))
    project.click_volume = float(pdata.get("click_volume", 0.6))
    project.beats_per_bar = int(pdata.get("beats_per_bar", 4))
    project.downbeat_phase = int(pdata.get("downbeat_phase", 0))

    dsp_params = {}
    missing = []

    for tdata in data.get("tracks", []):
        name = tdata.get("name")
        if not name:
            continue
        track = Track(name, color_hint=None)
        track.volume = float(tdata.get("volume", 1.0))
        track.pan = float(tdata.get("pan", 0.0))
        track.muted = bool(tdata.get("muted", False))
        track.soloed = bool(tdata.get("soloed", False))

        dsp_params[name] = dict(tdata.get("dsp", {}) or {})

        for cdata in tdata.get("clips", []):
            filepath = cdata.get("filepath")
            stem_name = cdata.get("stem_name") or name

            audio = None
            try:
                if filepath and os.path.exists(filepath):
                    audio, sr = load_audio(filepath)
                    if sr != project.samplerate:
                        # Trust the file; the project's samplerate is a
                        # hint, not a constraint. Mixed rates across
                        # tracks would be unusual, but this keeps what
                        # the file actually contains.
                        project.samplerate = sr
                else:
                    missing.append((name, filepath or "(no path)"))
            except Exception as e:
                missing.append((name, "{}: {}".format(filepath, e)))

            if audio is None:
                # Placeholder: 1 frame of silence. Keeps the clip in the
                # project (with its parameters) so the user can relink
                # the file later without losing edits.
                import numpy as np
                audio = np.zeros((1, 2), dtype="float32")

            clip = Clip(stem_name, filepath or "", audio,
                        project.samplerate,
                        start_time=float(cdata.get("start_time", 0.0)))
            clip.stretch = float(cdata.get("stretch", 1.0))
            clip.semitones = float(cdata.get("semitones", 0.0))
            clip.gain = float(cdata.get("gain", 1.0))
            clip.muted = bool(cdata.get("muted", False))
            clip.soloed = bool(cdata.get("soloed", False))
            clip.color_hint = cdata.get("color_hint")
            if filepath and (name, filepath) in missing or audio.shape == (1, 2):
                clip.missing = True
            else:
                clip.missing = False

            track.add_clip(clip)

        project.add_track(track)

    project._recompute_duration()
    return ProjectLoadResult(project, dsp_params, missing, path)

    effects = cdata.get("effects")
    if effects:
        from .dsp import DSPChain
        chain = DSPChain(project.samplerate)
        apply_dsp_params(chain, effects)
        clip.effects = chain


def apply_dsp_params(dsp_chain, params):
    """Apply the parameters returned by load_project to a DSPChain.
    Missing keys leave the attribute untouched (so a new DSP field
    doesn't break old project files)."""
    for attr, value in params.items():
        if hasattr(dsp_chain, attr):
            try:
                setattr(dsp_chain, attr, float(value))
            except (TypeError, ValueError):
                pass
    dsp_chain._filters_dirty = True
