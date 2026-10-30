#!/usr/bin/env bash
set -e

echo "=== StemsMix Installer ==="

# System dependencies (macOS)
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo ">> Checking Homebrew packages..."
    brew install portaudio ffmpeg rubberband 2>/dev/null || true
fi

# Python dependencies
echo ">> Installing Python packages..."
pip install --upgrade pip
pip install -r requirements.txt

# Verify critical imports
echo ">> Verifying imports..."
python -c "
import PyQt6, numpy, scipy, soundfile, sounddevice
print('  Core: OK')
try:
    import demucs_mlx
    print('  demucs-mlx (Metal): OK')
except ImportError:
    print('  demucs-mlx: not installed (will use demucs-onnx CPU)')
try:
    import mdxnet_infer
    print('  mdxnet-infer (drum split): OK')
except ImportError:
    print('  mdxnet-infer: MISSING')
try:
    import celeritas
    print('  celeritas (chords): OK')
except ImportError:
    print('  celeritas: MISSING')
"

echo "=== Done. Run: python apps/daw.py ==="
