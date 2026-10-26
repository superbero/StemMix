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

        self._bass_zi = np.zeros((1, 2))
        self._mid_zi = np.zeros((1, 2))
        self._treble_zi = np.zeros((1, 2))
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
                    out[:, ch], self._bass_zi[0] = sp_signal.sosfilt(
                        self._bass_sos, out[:, ch],
                        zi=self._bass_zi[0])
                if self.eq_mid:
                    out[:, ch], self._mid_zi[0] = sp_signal.sosfilt(
                        self._mid_sos, out[:, ch],
                        zi=self._mid_zi[0])
                if self.eq_treble:
                    out[:, ch], self._treble_zi[0] = sp_signal.sosfilt(
                        self._treble_sos, out[:, ch],
                        zi=self._treble_zi[0])

        if self.comp_threshold < 0 and self.comp_ratio > 1.0:
            thresh_lin = 10 ** (self.comp_threshold / 20.0)
            attack_coeff = math.exp(
                -1.0 / (self.comp_attack * self.sr))
            release_coeff = math.exp(
                -1.0 / (self.comp_release * self.sr))
            for i in range(out.shape[0]):
                peak = float(np.max(np.abs(out[i]))) + 1e-12
                if peak > self._comp_env:
                    self._comp_env = (attack_coeff * self._comp_env
                                      + (1 - attack_coeff) * peak)
                else:
                    self._comp_env = (release_coeff * self._comp_env
                                      + (1 - release_coeff) * peak)
                if self._comp_env > thresh_lin:
                    over_db = 20 * math.log10(
                        self._comp_env / thresh_lin)
                    gain_db = -over_db * (1 - 1.0 / self.comp_ratio)
                    out[i] *= 10 ** (gain_db / 20.0)

        if self.reverb_mix > 0:
            if self._reverb_buffers is None:
                self._reverb_buffers = self._make_reverb_buffers()
            wet = np.zeros_like(out)
            for ch in range(out.shape[1]):
                wet[:, ch] = self._reverb_channel(out[:, ch], ch)
            out = (1 - self.reverb_mix) * out + self.reverb_mix * wet

        return out

    def _make_reverb_buffers(self):
        comb_delays = [1116, 1188, 1277, 1356]
        allpass_delays = [556, 441]
        scale = self.sr / 44100.0
        buffers = []
        for ch in range(2):
            ch_bufs = []
            for d in comb_delays:
                n = max(1, int(d * scale))
                ch_bufs.append({"buf": np.zeros(n),
                                "pos": 0, "filter": 0.0})
            for d in allpass_delays:
                n = max(1, int(d * scale))
                ch_bufs.append({"buf": np.zeros(n), "pos": 0})
            buffers.append(ch_bufs)
        return buffers

    def _reverb_channel(self, x, ch):
        bufs = self._reverb_buffers[ch]
        feedback = 0.84
        damp = 0.4
        wet = np.zeros_like(x)
        for i in range(len(x)):
            sample = x[i]
            acc = 0.0
            for c in bufs[:4]:
                buf = c["buf"]
                pos = c["pos"]
                out = buf[pos]
                c["filter"] = out * (1 - damp) + c["filter"] * damp
                buf[pos] = sample + c["filter"] * feedback
                c["pos"] = (pos + 1) % len(buf)
                acc += out
            acc *= 0.25
            for a in bufs[4:]:
                buf = a["buf"]
                pos = a["pos"]
                buf_out = buf[pos]
                buf[pos] = acc + buf_out * 0.5
                a["pos"] = (pos + 1) % len(buf)
                acc = buf_out - acc
            wet[i] = acc
        return wet

    def process_offline(self, audio):
        self._filters_dirty = True
        self._comp_env = 0.0
        self._reverb_buffers = None
        return self.process(audio)
