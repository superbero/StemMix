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

    def push(self, label, project):
        """Snapshot project. Call *before* applying the change the label
        describes. Clears the redo stack (linear history)."""
        self._undo.append((label, _snapshot(project)))
        if len(self._undo) > MAX_HISTORY:
            self._undo.pop(0)
        self._redo.clear()

    def undo(self, project):
        """Return (label_that_was_undone, restored_project), or None."""
        if not self._undo:
            return None
        label, snap = self._undo.pop()
        # Push the pre-undo state onto redo. The pre-undo state is what
        # the caller's project currently holds.
        self._redo.append((label, _snapshot(project)))
        return label, _restore(snap, project)

    def redo(self, project):
        """Return (label_that_was_redone, restored_project), or None."""
        if not self._redo:
            return None
        label, snap = self._redo.pop()
        self._undo.append((label, _snapshot(project)))
        return label, _restore(snap, project)


# --------------------------------------------------------------------------
# Snapshot / restore
# --------------------------------------------------------------------------

def _snapshot(project):
    """Return a JSON-safe dict capturing everything undo should restore.
    Audio arrays are *not* captured; clip identity is by (track_name,
    filepath, index), which is stable within a session."""
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
        "tracks": [
            {
                "name": t.name,
                "volume": float(t.volume),
                "pan": float(t.pan),
                "muted": bool(t.muted),
                "soloed": bool(t.soloed),
                "clips": [
                    {
                        "stem_name": c.stem_name,
                        "filepath": c.filepath,
                        "start_time": float(c.start_time),
                        "stretch": float(c.stretch),
                        "semitones": float(c.semitones),
                        "gain": float(c.gain),
                        "muted": bool(c.muted),
                        "soloed": bool(c.soloed),
                        "color_hint": c.color_hint,
                    }
                    for c in t.clips
                ],
            }
            for t in project.tracks
        ],
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

    p = Project()
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

    for tdata in snap["tracks"]:
        track = Track(tdata["name"])
        track.volume = float(tdata["volume"])
        track.pan = float(tdata["pan"])
        track.muted = bool(tdata["muted"])
        track.soloed = bool(tdata["soloed"])
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
            track.add_clip(clip)
        p.add_track(track)

    p._recompute_duration()
    return p
