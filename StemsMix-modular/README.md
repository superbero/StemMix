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
