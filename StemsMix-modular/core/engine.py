"""Timeline audio engine: per-clip offsets, stretch, pitch, pan, meters."""
import numpy as np
import sounddevice as sd

from .automation import evaluate_points


class TimelineEngine:
    """Reads from a Project at a global playhead position."""

    def __init__(self, project, dsp_chain_lookup):
        self.project = project
        self.dsp_lookup = dsp_chain_lookup
        self.samplerate = project.samplerate
        self.stream = None
        self.is_playing = False
        self.playhead_seconds = 0.0
        self._playhead_frames = 0
        # Peak meters, written by the audio callback, read by the UI.
        self.levels = {}                 # track name -> (left, right)
        self.master_level = (0.0, 0.0)
        self._click = self._make_click(1000.0, 0.5)
        self._click_accent = self._make_click(1650.0, 0.8)
        self._beat_cache = None
        self._count_in_frames = 0
        self._count_in_total = 0
        self._count_in_beat_frames = 0

    # ------------------------------------------------------------ transport
    def start(self, from_seconds=None, count_in_bars=0):
        if self.stream is not None:
            self.stop()
        if from_seconds is not None:
            self.seek(from_seconds)
        self._count_in_frames = 0
        self._count_in_total = 0
        self._count_in_beat_frames = 0
        if count_in_bars > 0:
            p = self.project
            base_bpm = p.bpm
            if p.beat_times is not None and len(p.beat_times) > 1:
                from .beats import bpm_from_beats
                base_bpm = bpm_from_beats(p.beat_times) or base_bpm
            bpm = ((base_bpm * p.tempo) if base_bpm else 120.0)
            self._count_in_beat_frames = max(
                1, int(round(60.0 / max(bpm, 1.0) * self.samplerate)))
            beats = max(1, int(p.beats_per_bar)) * int(count_in_bars)
            self._count_in_total = self._count_in_beat_frames * beats
            self._count_in_frames = self._count_in_total
        self.stream = sd.OutputStream(
            samplerate=self.samplerate, channels=2,
            callback=self._callback, blocksize=512)
        self.stream.start()
        self.is_playing = True

    def stop(self):
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            finally:
                self.stream = None
        self.is_playing = False

    def seek(self, seconds):
        self.playhead_seconds = max(0.0, float(seconds))
        self._playhead_frames = int(self.playhead_seconds * self.samplerate)
        self._count_in_frames = 0

    # ------------------------------------------------------------- metronome
    def _make_click(self, freq, gain):
        sr = self.samplerate
        n = int(0.03 * sr)
        t = np.arange(n) / sr
        env = np.exp(-t / 0.006)
        a = max(1, int(0.0008 * sr))
        env[:a] *= np.linspace(0.0, 1.0, a)
        return (gain * np.sin(2 * np.pi * freq * t) * env).astype('float32')

    def _beat_frames(self):
        p = self.project
        key = (id(p.beat_times), p.tempo, self.samplerate)
        if self._beat_cache is None or self._beat_cache[0] != key:
            frames = np.round(np.asarray(p.beat_times)
                              / max(p.tempo, 1e-3)
                              * self.samplerate).astype(np.int64)
            self._beat_cache = (key, frames)
        return self._beat_cache[1]

    def _mix_click(self, out, start_frame, n_frames):
        """Add the metronome. It is synthesised here (no render step), so
        it follows tempo changes instantly. Never part of an export."""
        p = self.project
        if (not p.click_enabled or p.beat_times is None
                or len(p.beat_times) == 0):
            return
        frames = self._beat_frames()
        length = len(self._click)
        i0 = int(np.searchsorted(frames, start_frame - length))
        i1 = int(np.searchsorted(frames, start_frame + n_frames))
        meter = max(1, int(p.beats_per_bar))
        for k in range(i0, i1):
            bf = int(frames[k])
            a = max(start_frame, bf)
            b = min(start_frame + n_frames, bf + length)
            if b <= a:
                continue
            buf = (self._click_accent
                   if (k - p.downbeat_phase) % meter == 0 else self._click)
            out[a - start_frame:b - start_frame, :] += (
                buf[a - bf:b - bf] * p.click_volume)[:, None]

    def _mix_count_in(self, out, offset, n_frames, elapsed_frames):
        beat_frames = self._count_in_beat_frames
        if beat_frames <= 0 or n_frames <= 0:
            return
        first = elapsed_frames // beat_frames
        last = (elapsed_frames + n_frames - 1) // beat_frames
        for beat in range(first, last + 1):
            beat_start = beat * beat_frames
            start = max(elapsed_frames, beat_start)
            end = min(elapsed_frames + n_frames,
                      beat_start + len(self._click))
            if end <= start:
                continue
            click = self._click_accent if beat == 0 else self._click
            amount = getattr(self.project, "click_volume", 0.6)
            out[offset + start - elapsed_frames:
                offset + end - elapsed_frames] += (
                    click[start - beat_start:end - beat_start] * amount
                )[:, None]

    # ------------------------------------------------------------- mixing
    @staticmethod
    def _apply_pan(chunk, pan):
        """Balance law: unity at centre, the far channel is attenuated."""
        if not pan:
            return chunk
        if pan > 0:
            chunk[:, 0] *= (1.0 - pan)
        else:
            chunk[:, 1] *= (1.0 + pan)
        return chunk

    def _crossfade_gain(self, track, clip, a_frame, b_frame):
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
        ce = cs + len(_pd) if (_pd := clip.playback_data()) is not None else 0
        max_fade_frames = int(0.5 * sr)

        ordered = sorted(track.clips, key=lambda c: c.start_time)
        try:
            idx = ordered.index(clip)
        except ValueError:
            return gain

        # fade in from previous overlap
        if idx > 0:
            prev = ordered[idx - 1]
            prev_end = int(round(prev.start_time * sr)) + len(_pd) if (_pd := prev.playback_data()) is not None else 0
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

    def _apply_clip_fades(self, clip, chunk, a_frame, b_frame):
        """Multiply the head/tail of chunk by a linear ramp so
        fade_in_seconds and fade_out_seconds are honoured."""
        import numpy as np
        n = chunk.shape[0]
        if n <= 0:
            return chunk
        sr = self.samplerate
        cs = int(round(clip.start_time * sr))
        total = len(_pd) if (_pd := clip.playback_data()) is not None else 0
        if total <= 0:
            return chunk

        fi = int(round(getattr(clip, 'fade_in_seconds', 0.0) * sr))
        fo = int(round(getattr(clip, 'fade_out_seconds', 0.0) * sr))
        fi = max(0, min(fi, total))
        fo = max(0, min(fo, total))

        gain = np.ones(n, dtype='float32')
        if fi > 0:
            # region of clip inside [a_frame, b_frame)
            start = max(a_frame, cs)
            end = min(b_frame, cs + fi)
            if end > start:
                ramp = np.linspace(0.0, 1.0, fi, dtype='float32')
                s = start - a_frame
                e = end - a_frame
                gain[s:e] = ramp[start - cs:end - cs]
        if fo > 0:
            fo_start = cs + total - fo
            start = max(a_frame, fo_start)
            end = min(b_frame, cs + total)
            if end > start:
                ramp = np.linspace(1.0, 0.0, fo, dtype='float32')
                s = start - a_frame
                e = end - a_frame
                gain[s:e] = ramp[start - fo_start:end - fo_start]

        return chunk * gain[:, None]

    def _mix_block(self, start_frame, n_frames, offline=False):
        """Mix n_frames starting at start_frame into an (n, 2) array.
        Works in integer frames: float seconds used to drop a sample at
        block edges (int(511.9999) == 511), which clicked.

        Tempo/key changes are already baked into clip.playback_data()
        (rendered ahead of time), so this only slices and sums."""
        sr = self.samplerate
        end_frame = start_frame + n_frames
        out = np.zeros((n_frames, 2), dtype='float32')
        levels = {}
        for track in self.project.tracks:
            dsp = self.dsp_lookup(track.name)
            peak = np.zeros(2)
            volume = evaluate_points(
                track.automation.get("volume", []), "volume",
                start_frame, n_frames, sr)
            pan = evaluate_points(
                track.automation.get("pan", []), "pan",
                start_frame, n_frames, sr, default=track.pan)
            for clip in track.clips:
                if not self.project.is_audible(track, clip):
                    continue
                src = clip.playback_data()
                if src is None or src.shape[0] == 0:
                    continue           # render still pending: silence
                if src.ndim == 1:
                    src = np.column_stack((src, src))
                cs = int(round(clip.start_time * sr))
                a = max(start_frame, cs)
                b = min(end_frame, cs + src.shape[0])
                n = b - a
                if n <= 0:
                    continue
                block_start = a - start_frame
                block_end = b - start_frame
                chunk = src[a - cs:b - cs] * (
                    clip.gain * track.volume
                    * volume[block_start:block_end, None])
                chunk = self._apply_clip_fades(clip, chunk, a, b)
                cf = self._crossfade_gain(track, clip, a, b)
                if len(cf) == chunk.shape[0]:
                    chunk = chunk * cf[:, None]
                chunk = (dsp.process_offline(chunk) if offline
                         else dsp.process(chunk))
                if getattr(clip, "effects", None) is not None:
                    chunk = (clip.effects.process_offline(chunk) if offline
                             else clip.effects.process(chunk))
                pan_values = np.clip(
                    pan[block_start:block_end], -1.0, 1.0)
                chunk[:, 0] *= np.where(pan_values > 0, 1.0 - pan_values, 1.0)
                chunk[:, 1] *= np.where(pan_values < 0, 1.0 + pan_values, 1.0)
                peak = np.maximum(peak, np.abs(chunk).max(axis=0))
                out[a - start_frame:b - start_frame] += chunk
            levels[track.name] = (float(peak[0]), float(peak[1]))
        out *= self.project.master_volume
        if not offline:
            self._mix_click(out, start_frame, n_frames)
        return out, levels

    def _callback(self, outdata, frames, time, status):
        # Never leave stale buffer content behind, even on an underflow
        # flag (the old code returned early and played garbage).
        outdata.fill(0)
        sr = self.samplerate
        out = np.zeros((frames, 2), dtype="float32")
        levels = {}
        pre_roll = min(frames, self._count_in_frames)
        if pre_roll:
            elapsed = self._count_in_total - self._count_in_frames
            self._mix_count_in(out, 0, pre_roll, elapsed)
            self._count_in_frames -= pre_roll
        audio_frames = frames - pre_roll
        if audio_frames <= 0:
            outdata[:] = np.clip(out, -1.0, 1.0)
            self.levels = {}
            self.master_level = tuple(
                float(value) for value in np.abs(out).max(axis=0))
            return

        audio_out = out[pre_roll:]
        loop = self.project.loop
        loop_active = loop.is_active()
        loop_start = int(round(loop.start * sr))
        loop_end = int(round(loop.end * sr))
        loop_xfade = (min(int(0.01 * sr),
                          max(0, (loop_end - loop_start) // 2))
                      if loop_active else 0)
        written = 0

        while written < audio_frames:
            position = self._playhead_frames
            if loop_active and position >= loop_end:
                self.seek(loop.start + loop_xfade / sr)
                position = self._playhead_frames

            boundary = None
            if loop_active:
                boundary = (loop_start if position < loop_start
                            else loop_end)
            count = audio_frames - written
            if boundary is not None:
                count = min(count, boundary - position)
            if count <= 0:
                self.seek(loop.start)
                continue

            block, block_levels = self._mix_block(position, count)
            if loop_xfade:
                fade_start = loop_end - loop_xfade
                overlap_start = max(position, fade_start)
                overlap_end = min(position + count, loop_end)
                if overlap_end > overlap_start:
                    cross_frames = overlap_end - overlap_start
                    head_start = (loop_start + overlap_start - fade_start)
                    head, head_levels = self._mix_block(
                        head_start, cross_frames)
                    phase = np.linspace(
                        0.0, np.pi / 2.0, cross_frames,
                        endpoint=True, dtype="float32")
                    tail_gain = np.cos(phase)[:, None]
                    head_gain = np.sin(phase)[:, None]
                    a = overlap_start - position
                    b = overlap_end - position
                    block[a:b] = (block[a:b] * tail_gain
                                  + head * head_gain)
                    for name, peak in head_levels.items():
                        previous = block_levels.get(name, (0.0, 0.0))
                        block_levels[name] = (
                            max(previous[0], peak[0]),
                            max(previous[1], peak[1]))
            audio_out[written:written + count] = block
            for name, peak in block_levels.items():
                previous = levels.get(name, (0.0, 0.0))
                levels[name] = (max(previous[0], peak[0]),
                                max(previous[1], peak[1]))
            written += count
            self._playhead_frames += count
            if loop_active and self._playhead_frames >= loop_end:
                self.seek(loop.start + loop_xfade / sr)

        self.playhead_seconds = self._playhead_frames / sr
        out = np.clip(out, -1.0, 1.0)
        outdata[:] = out
        self.levels = levels
        mp = np.abs(out).max(axis=0)
        self.master_level = (float(mp[0]), float(mp[1]))

        total = self.project.total_duration
        if (self._count_in_frames == 0 and total > 0
                and self.playhead_seconds >= total):
            self.is_playing = False
            raise sd.CallbackStop

    # --------------------------------------------------------------- export
    def render_offline(self, output_path,
                       start_seconds=0.0, end_seconds=None):
        import soundfile as sf
        if end_seconds is None:
            end_seconds = self.project.total_duration
        total_frames = int((end_seconds - start_seconds) * self.samplerate)
        if total_frames <= 0:
            return False
        out, _ = self._mix_block(int(round(start_seconds * self.samplerate)),
                                 total_frames, offline=True)
        out = np.clip(out, -1, 1)
        sf.write(output_path, out, self.samplerate, subtype='PCM_24')
        return True
