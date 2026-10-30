"""Export each track of the project as its own WAV, with effects,
tempo, key, pan and fader applied.

Reuses the TimelineEngine's offline render path so the individual stems
sum exactly to the full mix when no solo/mute state is active.

Two behaviours worth knowing:
  * Every stem is padded to `project.total_duration`. A shorter track's
    file will end in silence. This is correct for summing stems back
    into a mix.
  * During render, the target track is un-soloed and every other track is
    muted AND un-soloed. `is_audible` respects solo, so muting alone is
    not enough if another track is soloed in the project. State is
    restored in a `finally` block.
  * Export is not cancellable: `_mix_block` runs synchronously per track.
    The caller can only close the window, which will finish the current
    track before the thread joins.
"""
import os
import re

import numpy as np
import soundfile as sf


def _safe_name(name):
    """Turn a track name into a filesystem-safe filename component."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "track"


def _render_track(engine, track_name, total_frames):
    """Render one track in isolation, at its current parameters.

    Clears solo state on every track (not just mutes) because
    `Project.is_audible` treats solo as an override of mute. Restores
    both mute and solo state in a `finally` block so a failed render
    cannot leave the project in a modified state.
    """
    project = engine.project
    saved_mute = {t.name: t.muted for t in project.tracks}
    saved_solo = {t.name: t.soloed for t in project.tracks}

    try:
        for t in project.tracks:
            if t.name == track_name:
                t.muted = False
                t.soloed = True     # force audible even if others soloed
            else:
                t.muted = True
                t.soloed = False
        # `_mix_block` calls `is_audible` per clip. With exactly one
        # soloed track, only that track's clips pass the check.
        out, _ = engine._mix_block(0, total_frames, offline=True)
        return out
    finally:
        for t in project.tracks:
            t.muted = saved_mute.get(t.name, False)
            t.soloed = saved_solo.get(t.name, False)


def _track_has_audio(track):
    """True if at least one clip has real audio (not a missing placeholder)."""
    for clip in track.clips:
        if getattr(clip, "missing", False):
            continue
        data = clip.playback_data()
        if data is not None and len(data) > 1:
            return True
    return False


def export_stems(engine, out_dir, progress_cb=None, prefix=""):
    """Write one WAV per track in out_dir.

    Filenames:
        prefix="":       "<trackname>.wav"
        prefix="foo":    "foo_<trackname>.wav"

    Returns a list of (track_name, filepath).
    Skips tracks with no usable audio (missing files, empty projects) and
    reports the skip via progress_cb with a clear message.
    """
    os.makedirs(out_dir, exist_ok=True)
    sr = engine.samplerate
    project = engine.project
    total_seconds = project.total_duration
    total_frames = int(total_seconds * sr)
    if total_frames <= 0:
        raise RuntimeError("Nothing to export.")

    tracks = list(project.tracks)
    n = len(tracks)
    written = []
    skipped = []

    for i, track in enumerate(tracks):
        if progress_cb:
            progress_cb(
                int(100 * i / n),
                "Rendering stem: {}".format(track.name))

        if not _track_has_audio(track):
            skipped.append(track.name)
            continue

        rendered = _render_track(engine, track.name, total_frames)
        if rendered is None:
            skipped.append(track.name)
            continue

        rendered = np.clip(rendered, -1.0, 1.0)
        if not prefix:
            filename = _safe_name(track.name)
        else:
            filename = "{}_{}".format(
                _safe_name(prefix), _safe_name(track.name))

        path = os.path.join(out_dir, filename + ".wav")
        sf.write(path, rendered, sr, subtype="PCM_24")
        written.append((track.name, path))

    if progress_cb:
        msg = "Wrote {} stems".format(len(written))
        if skipped:
            msg += " (skipped: {})".format(", ".join(skipped))
        progress_cb(100, msg)
    return written
