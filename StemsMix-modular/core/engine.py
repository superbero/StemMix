"""Real timeline audio engine with per-clip offsets, stretch, pitch."""
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
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.is_playing = False

    def seek(self, seconds):
        self.playhead_seconds = max(0.0, float(seconds))
        self._playhead_frames = int(self.playhead_seconds * self.samplerate)

    def _callback(self, outdata, frames, time, status):
        if status:
            return
        out = np.zeros((frames, 2), dtype='float32')
        sr = self.samplerate
        start_sec = self._playhead_frames / sr
        end_sec = (self._playhead_frames + frames) / sr

        for track in self.project.tracks:
            dsp = self.dsp_lookup(track.name)
            for clip in track.clips:
                clip_start = clip.start_time
                clip_end = clip.end_time()
                if clip_end <= start_sec or clip_start >= end_sec:
                    continue
                if not self.project.is_audible(track, clip):
                    continue
                block_from = max(start_sec, clip_start)
                block_to = min(end_sec, clip_end)
                if block_to <= block_from:
                    continue
                out_a = int((block_from - start_sec) * sr)
                out_b = int((block_to - start_sec) * sr)
                n = out_b - out_a
                if n <= 0:
                    continue
                in_a_sec = (block_from - clip_start) * clip.stretch
                in_b_sec = (block_to - clip_start) * clip.stretch
                in_a = in_a_sec * sr
                in_b = in_b_sec * sr
                chunk = self._read_stretched(clip, in_a, in_b, n)
                if chunk is None:
                    continue
                if clip.pitch != 1.0:
                    chunk = self._apply_pitch(chunk, clip.pitch)
                if chunk.shape[0] > n:
                    chunk = chunk[:n]
                elif chunk.shape[0] < n:
                    pad = np.zeros((n - chunk.shape[0], 2),
                                   dtype='float32')
                    chunk = np.vstack([chunk, pad])
                chunk = chunk * (clip.gain * track.volume)
                chunk = dsp.process(chunk)
                out[out_a:out_b] += chunk

        self._playhead_frames += frames
        self.playhead_seconds = self._playhead_frames / sr

        if (self.project.total_duration > 0
                and self.playhead_seconds >=
                self.project.total_duration):
            outdata[:] = np.clip(out, -1, 1)
            self.is_playing = False
            return

        outdata[:] = np.clip(out, -1.0, 1.0)

    def _read_stretched(self, clip, in_a, in_b, n_out):
        src = clip.data
        if src is None or src.shape[0] == 0:
            return None
        if src.ndim == 1:
            src = np.column_stack((src, src))
        t = np.linspace(in_a, in_b, n_out, endpoint=False)
        i0 = np.floor(t).astype(int)
        i1 = i0 + 1
        frac = (t - i0).astype('float32')
        i0 = np.clip(i0, 0, src.shape[0] - 1)
        i1 = np.clip(i1, 0, src.shape[0] - 1)
        a = src[i0]
        b = src[i1]
        frac2 = frac[:, np.newaxis]
        return a * (1 - frac2) + b * frac2

    def _apply_pitch(self, chunk, pitch_factor):
        if abs(pitch_factor - 1.0) < 1e-4:
            return chunk
        n_out = chunk.shape[0]
        n_in = max(2, int(n_out / pitch_factor))
        src = chunk[:n_in] if chunk.shape[0] >= n_in else np.pad(
            chunk, ((0, n_in - chunk.shape[0]), (0, 0)))
        t = np.linspace(0, src.shape[0] - 1, n_out)
        i0 = np.floor(t).astype(int)
        i1 = np.clip(i0 + 1, 0, src.shape[0] - 1)
        frac = (t - i0).astype('float32')[:, np.newaxis]
        out = src[i0] * (1 - frac) + src[i1] * frac
        return out.astype('float32')

    def render_offline(self, output_path,
                       start_seconds=0.0, end_seconds=None):
        import soundfile as sf
        if end_seconds is None:
            end_seconds = self.project.total_duration
        total_frames = int((end_seconds - start_seconds) * self.samplerate)
        if total_frames <= 0:
            return False
        out = np.zeros((total_frames, 2), dtype='float32')
        for track in self.project.tracks:
            dsp = self.dsp_lookup(track.name)
            for clip in track.clips:
                if not self.project.is_audible(track, clip):
                    continue
                cs = clip.start_time
                ce = clip.end_time()
                if ce <= start_seconds or cs >= end_seconds:
                    continue
                block_from = max(start_seconds, cs)
                block_to = min(end_seconds, ce)
                n = int((block_to - block_from) * self.samplerate)
                if n <= 0:
                    continue
                in_a = (block_from - cs) * clip.stretch * self.samplerate
                in_b = (block_to - cs) * clip.stretch * self.samplerate
                chunk = self._read_stretched(clip, in_a, in_b, n)
                if chunk is None:
                    continue
                if clip.pitch != 1.0:
                    chunk = self._apply_pitch(chunk, clip.pitch)
                chunk = chunk * (clip.gain * track.volume)
                chunk = dsp.process_offline(chunk)
                out_a = int((block_from - start_seconds) * self.samplerate)
                out_b = out_a + n
                out[out_a:out_b] += chunk[:out_b - out_a]
        out = np.clip(out, -1, 1)
        sf.write(output_path, out, self.samplerate, subtype='PCM_24')
        return True
