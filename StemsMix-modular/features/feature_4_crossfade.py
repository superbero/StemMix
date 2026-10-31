#!/usr/bin/env python3
"""feature_4_crossfade.py — crossfade overlapping clips on the same
track so the sum doesn't click at the boundary.

The engine already sums clips by frame. When two clips overlap, a linear
crossfade is applied to the overlap region:
  clip A fades out over the overlap, clip B fades in.

This is implemented in core.engine by adding per-clip overlap masks.
The mask is computed on the fly from the clip start/duration, so it
survives every clip edit (drag, stretch, tempo change).
"""
import os
import shutil
import sys
import time


def backup(path):
    if not os.path.exists(path):
        return
    stamp = time.strftime("%Y%m%d_%H%M%S")
    dst = path + ".bak." + stamp
    shutil.copy2(path, dst)
    print("  backup:", dst)


def main():
    path = "core/engine.py"
    if not os.path.isfile(path):
        print("ERROR: run from project root")
        sys.exit(1)
    backup(path)
    src = open(path).read()

    if "_crossfade_gain" in src:
        print("SKIP (already patched)")
        return

    # Add the helper method before _mix_block
    helper = '''    def _crossfade_gain(self, track, clip, a_frame, b_frame):
        """Return a per-frame gain array for clip between global frames
        a_frame..b_frame, applying crossfades where this clip overlaps
        another on the same track.

        Logic:
          - sort clips by start_time; for clip at index i, check overlap
            with i-1 (fade in over the left overlap) and i+1 (fade out
            over the right overlap)
          - overlap length is limited to 0.5 seconds to avoid mangling
            long overlaps into a 30-second fade
        """
        import numpy as np
        n = b_frame - a_frame
        if n <= 0:
            return np.ones(0, dtype="float32")
        gain = np.ones(n, dtype="float32")
        if len(track.clips) < 2:
            return gain
        sr = self.samplerate
        cs = int(round(clip.start_time * sr))
        ce = cs + len(clip.playback_data() or [])
        max_fade_frames = int(0.5 * sr)

        ordered = sorted(track.clips, key=lambda c: c.start_time)
        try:
            idx = ordered.index(clip)
        except ValueError:
            return gain

        # fade in from previous overlap
        if idx > 0:
            prev = ordered[idx - 1]
            prev_end = int(round(prev.start_time * sr)) + len(
                prev.playback_data() or [])
            overlap = min(prev_end - cs, max_fade_frames)
            if overlap > 0:
                fade_start = max(a_frame, cs)
                fade_end = min(b_frame, cs + overlap)
                if fade_end > fade_start:
                    ramp = np.linspace(
                        0.0, 1.0, overlap, dtype="float32")
                    s = fade_start - a_frame
                    e = fade_end - a_frame
                    gain[s:e] = ramp[:e - s]

        # fade out into next overlap
        if idx < len(ordered) - 1:
            nxt = ordered[idx + 1]
            nxt_start = int(round(nxt.start_time * sr))
            overlap = min(ce - nxt_start, max_fade_frames)
            if overlap > 0:
                fade_start = max(a_frame, nxt_start)
                fade_end = min(b_frame, ce)
                if fade_end > fade_start:
                    ramp = np.linspace(
                        1.0, 0.0, overlap, dtype="float32")
                    s = fade_start - a_frame
                    e = fade_end - a_frame
                    gain[s:e] = ramp[:e - s]

        return gain

'''
    src = src.replace("    def _mix_block(self, start_frame, n_frames, offline=False):\n",
                      helper + "    def _mix_block(self, start_frame, n_frames, offline=False):\n",
                      1)

    # Multiply chunk by the crossfade gain in _mix_block
    old_chunk = (
        "                chunk = src[a - cs:b - cs] * (clip.gain * track.volume)\n"
    )
    new_chunk = (
        "                chunk = src[a - cs:b - cs] * (clip.gain * track.volume)\n"
        "                cf = self._crossfade_gain(track, clip, a, b)\n"
        "                if len(cf) == chunk.shape[0]:\n"
        "                    chunk = chunk * cf[:, None]\n"
    )
    if old_chunk not in src:
        print("FAIL: could not find chunk multiply line")
        return
    src = src.replace(old_chunk, new_chunk, 1)

    try:
        compile(src, path, "exec")
    except SyntaxError as e:
        print("SYNTAX ERROR:", e)
        sys.exit(1)
    open(path, "w").write(src)
    print("Feature 4 (crossfade) applied.")
    print("Test: place two clips on the same track so they overlap;")
    print("      play; the boundary should fade instead of clicking.")


if __name__ == "__main__":
    main()
