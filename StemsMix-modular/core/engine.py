"""Timeline audio engine: per-clip offsets, stretch, pitch, pan, meters."""
import numpy as np
import sounddevice as sd


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

    # ------------------------------------------------------------ transport
    def start(self, from_seconds=None):
        if self.stream is not None:
            self.stop()
        if from_seconds is not None:
            self.seek(from_seconds)
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
                chunk = src[a - cs:b - cs] * (clip.gain * track.volume)
                chunk = (dsp.process_offline(chunk) if offline
                         else dsp.process(chunk))
                chunk = self._apply_pan(chunk, track.pan)
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
        out, levels = self._mix_block(self._playhead_frames, frames)

        self._playhead_frames += frames
        self.playhead_seconds = self._playhead_frames / sr
        out = np.clip(out, -1.0, 1.0)
        outdata[:] = out
        self.levels = levels
        mp = np.abs(out).max(axis=0)
        self.master_level = (float(mp[0]), float(mp[1]))

        total = self.project.total_duration
        if total > 0 and self.playhead_seconds >= total:
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
