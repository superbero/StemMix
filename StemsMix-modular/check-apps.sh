#!/usr/bin/env bash
cd "$(dirname "$0")"

echo ">> Syntax check (all modules)"
python3 -m py_compile apps/mixer_app.py apps/daw.py \
    ui/workers.py ui/timeline.py ui/dialogs.py \
    ui/channel_strip.py ui/style.py \
    core/separation.py core/export.py core/dsp.py \
    core/audio_io.py core/resources.py \
    && echo "   All files parse OK" \
    || echo "   SYNTAX ERROR — see above"

echo ""
echo ">> Import check (qt free modules)"
python3 -c "import core.separation, core.export, core.dsp, core.audio_io, core.resources; print('   core/* imports OK')" \
    || echo "   core/* import FAILED"

echo ""
echo ">> Import check (UI modules)"
python3 -c "import ui.style; print('   ui.style OK')" \
    || echo "   ui.style FAILED"
python3 -c "import ui.workers; print('   ui.workers OK')" \
    || echo "   ui.workers FAILED"
python3 -c "import ui.dialogs; print('   ui.dialogs OK')" \
    || echo "   ui.dialogs FAILED"
python3 -c "import ui.timeline; print('   ui.timeline OK')" \
    || echo "   ui.timeline FAILED"
python3 -c "import ui.channel_strip; print('   ui.channel_strip OK')" \
    || echo "   ui.channel_strip FAILED"

echo ""
echo ">> Optional dependency check"
python3 -c "
try:
    import mdxnet_infer
    print('   mdxnet_infer: OK')
except ImportError as e:
    print('   mdxnet_infer: MISSING (' + str(e) + ')')
try:
    import denote, celeritas
    print('   denote + celeritas: OK')
except ImportError as e:
    print('   denote/celeritas: MISSING (' + str(e) + ')')
try:
    import demucs_onnx
    print('   demucs_onnx: OK')
except ImportError as e:
    print('   demucs_onnx: MISSING (' + str(e) + ')')
"
