"""Timeline project state: tracks, clips, playhead. No audio, no Qt."""
import copy


class Clip:
    """A piece of audio placed on a timeline.

    `stretch` is the tempo ratio (>1 = faster/shorter) and `semitones`
    the key shift. Both are applied by rendering ahead of time
    (core.timestretch), never in the audio callback: `render` holds the
    processed audio for the parameters in `render_key`. While a render is
    pending the clip is not ready and the engine plays silence for it.
    """

    def __init__(self, stem_name, filepath, audio_data,
                 samplerate, start_time=0.0):
        self.id = id(self)
        self.stem_name = stem_name
        self.filepath = filepath
        self.data = audio_data
        self.samplerate = samplerate
        self.start_time = start_time
        self.duration = len(audio_data) / samplerate
        self.stretch = 1.0
        self.semitones = 0.0
        self.gain = 1.0
        self.muted = False
        self.soloed = False
        self.color_hint = None
        self.render = None
        self.render_key = None
        self.render_quality = None   # 'fast' (preview) or 'high'

    # -- rendering state
    def render_params(self):
        return (round(float(self.stretch), 6), round(float(self.semitones), 4))

    def needs_render(self):
        return (abs(self.stretch - 1.0) > 1e-6
                or abs(self.semitones) > 1e-6)

    def is_ready(self):
        return (not self.needs_render()
                or (self.render is not None
                    and self.render_key == self.render_params()))

    def playback_data(self):
        """Audio to play at normal speed, or None while a render is
        pending."""
        if not self.needs_render():
            return self.data
        return self.render if self.is_ready() else None

    def eff_duration(self):
        if self.needs_render() and self.is_ready():
            return len(self.render) / self.samplerate
        return self.duration / self.stretch

    def end_time(self):
        return self.start_time + self.eff_duration()

    def contains_time(self, seconds):
        return self.start_time <= seconds < self.end_time()


class Track:
    """A lane that holds clips. Each track has its own DSP chain."""

    def __init__(self, name, color_hint=None):
        self.name = name
        self.color_hint = color_hint
        self.clips = []
        self.muted = False
        self.soloed = False
        self.volume = 1.0
        self.pan = 0.0  # -1 (left) .. +1 (right), balance law

    def add_clip(self, clip):
        self.clips.append(clip)

    def remove_clip(self, clip):
        if clip in self.clips:
            self.clips.remove(clip)

    def clips_at(self, seconds):
        return [c for c in self.clips if c.contains_time(seconds)]


class Project:
    """Top-level container for all tracks and global state."""

    def __init__(self):
        self.tracks = []
        self.samplerate = 44100
        self.total_duration = 0.0
        self.master_volume = 1.0
        self.tempo = 1.0   # cumulative tempo ratio vs. the original
        # Beat grid / click (core.beats). Beat times are in the ORIGINAL
        # audio's time; the engine and UI divide them by `tempo`.
        self.bpm = None
        self.beat_times = None
        self.click_enabled = False
        self.click_volume = 0.6
        self.beats_per_bar = 4
        self.downbeat_phase = 0
        self.key = 0.0     # cumulative key shift in semitones

    def add_track(self, track):
        self.tracks.append(track)
        self._recompute_duration()

    def remove_track(self, track):
        if track in self.tracks:
            self.tracks.remove(track)
        self._recompute_duration()

    def _recompute_duration(self):
        end = 0.0
        for t in self.tracks:
            for c in t.clips:
                end = max(end, c.end_time())
        self.total_duration = end

    def any_soloed(self):
        return any(t.soloed or any(c.soloed for c in t.clips)
                   for t in self.tracks)

    def is_audible(self, track, clip=None):
        any_solo = self.any_soloed()
        if track.muted:
            return False
        track_soloed = track.soloed
        clip_soloed = clip.soloed if clip else False
        if any_solo and not (track_soloed or clip_soloed):
            return False
        if clip is not None and clip.muted:
            return False
        return True

    def snapshot(self):
        return copy.deepcopy(self)
