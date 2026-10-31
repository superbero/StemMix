# StemsMix

Stem separation, mixing, and DAW-style timeline for Linux and macOS.

## Structure

- `core/` — pure Python, no Qt. DSP, audio I/O, separation, export.
- `ui/`   — Qt widgets and dialogs.
- `apps/` — thin entry points: `mixer_app.py` and `daw.py`.

## Setup

    python3.12 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

Linux: sudo apt install portaudio19-dev ffmpeg
macOS: brew install portaudio ffmpeg

## Run

    python apps/mixer_app.py
    python apps/daw.py

## Timeline editing

- Drag across the time ruler to select a section. Use **Loop** to enable or
  disable repeat playback for that range; drag either marker to adjust it.
  The section bar can name ranges, jump to them, or loop a saved section.
  Loop boundaries use a short crossfade to reduce clicks.
- Enable **Count-in** for a one-bar metronome lead-in before playback.
- Select a clip and press `[` / `]` to nudge it by 10 ms, `{` / `}` for 100 ms,
  or `Shift+[` / `Shift+]` for one second. `N`, `Shift+N`, and `Alt+N` snap to
  the nearest, previous, or next detected beat.
- Choose **File → Import Track from Project…** to select a track from another
  `.stemmix` project and add it to the current arrangement.
- Drag a track name vertically to reorder its timeline lane and mixer strip.
  Track order is saved with the project and restored by undo/redo.
- Use **File → Setlist Manager…** to organize projects, then navigate songs
  with **Prev Song** / **Next Song** or `Ctrl+Alt+Left` / `Ctrl+Alt+Right`.

## Separation tasks

When opening audio, choose a task: standard 4/6 instrument stems, a
two-output vocal/accompaniment workflow, or drum-kit parts. The vocal
accompaniment is assembled from the selected Demucs model's outputs; it does
not isolate lead and backing vocals. Drum-kit parts use the optional
`mdxnet-infer` model after the main separator isolates drums.
