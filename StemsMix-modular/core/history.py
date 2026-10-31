"""Undo/redo for the timeline project.

A snapshot is a JSON-serialised form of the *editable* project state —
the things a user can change that should be undoable. Audio data is
never copied: snapshot captures clip parameters and track settings only,
and `restore` rebuilds a Project whose clips reference the same
in-memory audio arrays as the pre-snapshot project.

Undo granularity: one entry per user action. The DAW calls
`push(label)` *before* mutating the project. Ctrl+Z calls `undo()` to
restore the previous snapshot; Ctrl+Shift+Z calls `redo()`. Any new push
clears the redo stack, which is the standard linear-history behaviour.
"""
import json

from .timeline_model import Clip, Track, Project


MAX_HISTORY = 100  # bounded so long sessions don't grow without limit


class History:
    def __init__(self):
        self._undo = []   # list of (label, snapshot_dict)
        self._redo = []

    def clear(self):
        self._undo.clear()
        self._redo.clear()

    def can_undo(self):
        return bool(self._undo)

    def can_redo(self):
        return bool(self._redo)

    def undo_label(self):
        return self._undo[-1][0] if self._undo else None

    def redo_label(self):
        return self._redo[-1][0] if self._redo else None

    def push(self, label, project, dsp_chains=None):
        """Snapshot project. Call *before* applying the change the label
        describes. Clears the redo stack (linear history)."""
        self._undo.append((label, _snapshot(project, dsp_chains)))
        if len(self._undo) > MAX_HISTORY:
            self._undo.pop(0)
        self._redo.clear()

    def undo(self, project, dsp_chains=None):
        """Return (label_that_was_undone, restored_project), or None."""
        if not self._undo:
            return None
        label, snap = self._undo.pop()
        # Push the pre-undo state onto redo. The pre-undo state is what
        # the caller's project currently holds.
        self._redo.append((label, _snapshot(project, dsp_chains)))
        return label, _restore(snap, project)

    def redo(self, project, dsp_chains=None):
        """Return (label_that_was_redone, restored_project), or None."""
        if not self._redo:
            return None
        label, snap = self._redo.pop()
        self._undo.append((label, _snapshot(project, dsp_chains)))
        return label, _restore(snap, project)


# --------------------------------------------------------------------------
# Snapshot / restore
# --------------------------------------------------------------------------

def _snapshot(project, dsp_chains=None):
    """Return a JSON-safe dict capturing everything undo should restore.
    Audio arrays are *not* captured; clip identity is by (track_name,
    filepath, index), which is stable within a session."""
    def track_state(track):
        return {
            "name": track.name,
            "volume": float(track.volume),
            "pan": float(track.pan),
            "muted": bool(track.muted),
            "soloed": bool(track.soloed),
            "automation": {
                parameter: [list(point) for point in points]
                for parameter, points in track.automation.items()
            },
            "clips": [
                {
                    "stem_name": clip.stem_name,
                    "filepath": clip.filepath,
                    "start_time": float(clip.start_time),
                    "stretch": float(clip.stretch),
                    "semitones": float(clip.semitones),
                    "gain": float(clip.gain),
                    "muted": bool(clip.muted),
                    "soloed": bool(clip.soloed),
                    "color_hint": clip.color_hint,
                    "fade_in": float(clip.fade_in_seconds),
                    "fade_out": float(clip.fade_out_seconds),
                }
                for clip in track.clips
            ],
        }

    return {
        "samplerate": int(project.samplerate),
        "tempo": float(project.tempo),
        "key": float(project.key),
        "master_volume": float(project.master_volume),
        "bpm": None if project.bpm is None else float(project.bpm),
        "beats": (None if project.beat_times is None
                  else [float(b) for b in project.beat_times]),
        "click_enabled": bool(project.click_enabled),
        "click_volume": float(project.click_volume),
        "beats_per_bar": int(project.beats_per_bar),
        "downbeat_phase": int(project.downbeat_phase),
        "loop": project.loop.to_dict(),
        "sections": [section.to_dict() for section in project.sections],
        "count_in_bars": int(project.count_in_bars),
        "tracks": [track_state(track) for track in project.tracks],
        "removed_tracks": [
            track_state(track) for track in project.removed_tracks],
        "dsp_chains": {
            name: {
                attr: float(getattr(dsp, attr))
                for attr in (
                    "eq_bass", "eq_mid", "eq_treble",
                    "comp_threshold", "comp_ratio", "comp_attack",
                    "comp_release", "reverb_mix")
            }
            for name, dsp in (dsp_chains or {}).items()
        },
    }


def _restore(snap, current_project):
    """Rebuild a Project from snap, reusing audio arrays from
    current_project where possible.

    Reuses audio by matching (stem_name, filepath). A clip whose source
    is not in the current project — which happens after drum split, when
    a track was replaced — falls back to a 1-frame silent placeholder.
    That keeps history usable even across structural changes; the affected
    clips are silent, not crashy."""
    import numpy as np

    # Index the current project's audio by (stem_name, filepath).
    audio_index = {}
    for t in current_project.tracks:
        for c in t.clips:
            audio_index[(c.stem_name, c.filepath)] = c.data
    for t in current_project.removed_tracks:
        for c in t.clips:
            audio_index[(c.stem_name, c.filepath)] = c.data

    p = Project()
    p.history_dsp_chains = snap.get("dsp_chains", {})
    p.samplerate = int(snap["samplerate"])
    p.tempo = float(snap["tempo"])
    p.key = float(snap["key"])
    p.master_volume = float(snap["master_volume"])
    p.bpm = None if snap["bpm"] is None else float(snap["bpm"])
    p.beat_times = (None if snap["beats"] is None
                    else [float(b) for b in snap["beats"]])
    p.click_enabled = bool(snap["click_enabled"])
    p.click_volume = float(snap["click_volume"])
    p.beats_per_bar = int(snap["beats_per_bar"])
    p.downbeat_phase = int(snap["downbeat_phase"])
    from .loop import LoopRegion
    p.loop = LoopRegion.from_dict(snap.get("loop"))
    from .timeline_model import SectionMarker
    p.sections = [
        SectionMarker(str(section["name"]), float(section["start"]),
                      float(section["end"]))
        for section in snap.get("sections", [])
    ]
    p.count_in_bars = int(snap.get("count_in_bars", 0))

    for field, destination in (
            ("tracks", p.tracks),
            ("removed_tracks", p.removed_tracks)):
        for tdata in snap.get(field, []):
            track = Track(tdata["name"])
            track.volume = float(tdata["volume"])
            track.pan = float(tdata["pan"])
            track.muted = bool(tdata["muted"])
            track.soloed = bool(tdata["soloed"])
            track.automation = {
                parameter: [
                    [float(point[0]), float(point[1])]
                    for point in tdata.get("automation", {}).get(
                        parameter, [])
                ]
                for parameter in ("volume", "pan")
            }
            for cdata in tdata["clips"]:
                key = (cdata["stem_name"], cdata["filepath"])
                audio = audio_index.get(key)
                if audio is None:
                    audio = np.zeros((1, 2), dtype="float32")
                clip = Clip(cdata["stem_name"], cdata["filepath"],
                            audio, p.samplerate,
                            start_time=float(cdata["start_time"]))
                clip.stretch = float(cdata["stretch"])
                clip.semitones = float(cdata["semitones"])
                clip.gain = float(cdata["gain"])
                clip.muted = bool(cdata["muted"])
                clip.soloed = bool(cdata["soloed"])
                clip.color_hint = cdata["color_hint"]
                clip.fade_in_seconds = float(
                    cdata.get("fade_in", 0.0))
                clip.fade_out_seconds = float(
                    cdata.get("fade_out", 0.0))
                track.add_clip(clip)
            if field == "tracks":
                p.add_track(track)
            else:
                destination.append(track)

    p._recompute_duration()
    return p
