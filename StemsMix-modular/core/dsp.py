"""Pure-numpy DSP: EQ, compressor, reverb."""
import math

import numpy as np
from scipy import signal as sp_signal


class DSPChain:
    """3-band EQ, compressor, reverb. Neutral by default."""

    def __init__(self, samplerate):
        self.sr = samplerate
        self.eq_bass = 0.0
        self.eq_mid = 0.0
        self.eq_treble = 0.0
        self.comp_threshold = 0.0
        self.comp_ratio = 4.0
        self.comp_attack = 0.01
        self.comp_release = 0.1
        self.reverb_mix = 0.0
        self._filters_dirty = True
        self._bass_sos = None
        self._mid_sos = None
        self._treble_sos = None
        self._bass_zi = None
        self._mid_zi = None
        self._treble_zi = None
        self._comp_env = 0.0
        self._reverb_buffers = None

    def _rebuild_filters(self):
        sr = self.sr

        A = 10 ** (self.eq_bass / 40.0)
        w0 = 2 * math.pi * 250 / sr
        alpha = math.sin(w0) / 2 * math.sqrt(
            (A + 1 / A) * (1 / 0.707 - 1) + 2)
        b0 = A * ((A + 1) - (A - 1) * math.cos(w0)
                  + 2 * math.sqrt(A) * alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * math.cos(w0))
        b2 = A * ((A + 1) - (A - 1) * math.cos(w0)
                  - 2 * math.sqrt(A) * alpha)
        a0 = (A + 1) + (A - 1) * math.cos(w0) + 2 * math.sqrt(A) * alpha
        a1 = -2 * ((A - 1) + (A + 1) * math.cos(w0))
        a2 = (A + 1) + (A - 1) * math.cos(w0) - 2 * math.sqrt(A) * alpha
        self._bass_sos = np.array(
            [[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])

        A = 10 ** (self.eq_mid / 40.0)
        w0 = 2 * math.pi * 1000 / sr
        alpha = math.sin(w0) / 2
        b0 = 1 + alpha * A
        b1 = -2 * math.cos(w0)
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * math.cos(w0)
        a2 = 1 - alpha / A
        self._mid_sos = np.array(
            [[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])

        A = 10 ** (self.eq_treble / 40.0)
        w0 = 2 * math.pi * 4000 / sr
        alpha = math.sin(w0) / 2 * math.sqrt(
            (A + 1 / A) * (1 / 0.707 - 1) + 2)
        b0 = A * ((A + 1) + (A - 1) * math.cos(w0)
                  + 2 * math.sqrt(A) * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * math.cos(w0))
        b2 = A * ((A + 1) + (A - 1) * math.cos(w0)
                  - 2 * math.sqrt(A) * alpha)
        a0 = (A + 1) - (A - 1) * math.cos(w0) + 2 * math.sqrt(A) * alpha
        a1 = 2 * ((A - 1) - (A + 1) * math.cos(w0))
        a2 = (A + 1) - (A - 1) * math.cos(w0) - 2 * math.sqrt(A) * alpha
        self._treble_sos = np.array(
            [[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])

        # One filter state per channel (the old single shared state let
        # left and right corrupt each other). States survive coefficient
        # changes, so turning an EQ knob does not click.
        if self._bass_zi is None:
            self._bass_zi = np.zeros((2, 1, 2))
        if self._mid_zi is None:
            self._mid_zi = np.zeros((2, 1, 2))
        if self._treble_zi is None:
            self._treble_zi = np.zeros((2, 1, 2))
        self._filters_dirty = False

    def process(self, audio):
        if audio.shape[0] == 0:
            return audio
        if (self.eq_bass == 0 and self.eq_mid == 0
                and self.eq_treble == 0 and self.comp_threshold == 0
                and self.reverb_mix == 0):
            return audio

        if self._filters_dirty:
            self._rebuild_filters()

        out = audio.copy()
        if self.eq_bass or self.eq_mid or self.eq_treble:
            for ch in range(out.shape[1]):
                if self.eq_bass:
                    out[:, ch], self._bass_zi[ch] = sp_signal.sosfilt(
                        self._bass_sos, out[:, ch],
                        zi=self._bass_zi[ch])
                if self.eq_mid:
                    out[:, ch], self._mid_zi[ch] = sp_signal.sosfilt(
                        self._mid_sos, out[:, ch],
                        zi=self._mid_zi[ch])
                if self.eq_treble:
                    out[:, ch], self._treble_zi[ch] = sp_signal.sosfilt(
                        self._treble_sos, out[:, ch],
                        zi=self._treble_zi[ch])

        if self.comp_threshold < 0 and self.comp_ratio > 1.0:
            out = self._compress(out)

        if self.reverb_mix > 0:
            if self._reverb_buffers is None:
                self._reverb_buffers = self._make_reverb_buffers()
            wet = np.zeros_like(out)
            for ch in range(out.shape[1]):
                wet[:, ch] = self._reverb_channel(out[:, ch], ch)
            out = (1 - self.reverb_mix) * out + self.reverb_mix * wet

        return out

    def _compress(self, x):
        """Feed-forward compressor. The level detector runs per 32-sample
        block (vectorised peak, short Python loop over blocks) and the
        gain is interpolated between blocks. The old per-sample Python
        loop was too slow for a real-time callback."""
        B = 32
        n = x.shape[0]
        thresh = 10 ** (self.comp_threshold / 20.0)
        att = math.exp(-B / (self.comp_attack * self.sr))
        rel = math.exp(-B / (self.comp_release * self.sr))
        slope = 1.0 - 1.0 / self.comp_ratio
        peak = np.abs(x).max(axis=1)
        pad = (-n) % B
        if pad:
            peak = np.concatenate([peak, np.zeros(pad, peak.dtype)])
        blocks = peak.reshape(-1, B).max(axis=1) + 1e-12
        env = self._comp_env
        gains = np.ones(len(blocks))
        for i, p in enumerate(blocks):
            c = att if p > env else rel
            env = c * env + (1 - c) * p
            if env > thresh:
                gains[i] = 10 ** (-20 * math.log10(env / thresh)
                                  * slope / 20.0)
        self._comp_env = env
        centers = (np.arange(len(blocks)) + 0.5) * B
        g = np.interp(np.arange(n), centers, gains)
        return (x * g[:, None]).astype(x.dtype, copy=False)

    def _make_reverb_buffers(self):
        comb_delays = [1116, 1188, 1277, 1356]
        allpass_delays = [556, 441]
        scale = self.sr / 44100.0
        buffers = []
        for ch in range(2):
            ch_bufs = []
            for d in comb_delays:
                n = max(1, int(d * scale))
                ch_bufs.append({"buf": np.zeros(n), "pos": 0,
                                "zi": np.zeros(1)})
            for d in allpass_delays:
                n = max(1, int(d * scale))
                ch_bufs.append({"buf": np.zeros(n), "pos": 0})
            buffers.append(ch_bufs)
        return buffers

    def _reverb_channel(self, x, ch):
        """Freeverb-style comb/allpass network, vectorised. Every delay
        line is longer than the sub-block, so a whole sub-block can be
        read from the delay buffer at once (no sample depends on one
        written in the same sub-block)."""
        bufs = self._reverb_buffers[ch]
        feedback = 0.84
        damp = 0.4
        n = len(x)
        wet = np.empty(n, dtype=np.float64)
        sub = min(len(b["buf"]) for b in bufs)
        i0 = 0
        while i0 < n:
            m = min(sub, n - i0)
            blk = x[i0:i0 + m].astype(np.float64)
            acc = np.zeros(m)
            for c in bufs[:4]:
                buf = c["buf"]
                idx = (c["pos"] + np.arange(m)) % len(buf)
                delayed = buf[idx]
                filt, c["zi"] = sp_signal.lfilter(
                    [1 - damp], [1, -damp], delayed, zi=c["zi"])
                buf[idx] = blk + filt * feedback
                c["pos"] = (c["pos"] + m) % len(buf)
                acc += delayed
            acc *= 0.25
            for a in bufs[4:]:
                buf = a["buf"]
                idx = (a["pos"] + np.arange(m)) % len(buf)
                buf_out = buf[idx]
                buf[idx] = acc + buf_out * 0.5
                a["pos"] = (a["pos"] + m) % len(buf)
                acc = buf_out - acc
            wet[i0:i0 + m] = acc
            i0 += m
        return wet

    def process_offline(self, audio):
        self._filters_dirty = True
        self._bass_zi = self._mid_zi = self._treble_zi = None
        self._comp_env = 0.0
        self._reverb_buffers = None
        return self.process(audio)
