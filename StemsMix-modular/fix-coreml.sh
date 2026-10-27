#!/usr/bin/env bash
# fix-coreml.sh
# Forces CPU provider in SeparationWorker to work around a CoreML
# compilation error on Apple Silicon ("generic_general_slice: Invalid
# values 2048 in begin_ids"). Safe to run multiple times.

set -euo pipefail

WORKERS="ui/workers.py"
SEPARATION="core/separation.py"

if [ ! -f "$WORKERS" ]; then
    echo "ERROR: $WORKERS not found. Run this from StemsMix-modular/."
    exit 1
fi

# ---------------------------------------------------------------------------
# 1. Back up the files
# ---------------------------------------------------------------------------
echo ">> Backing up files..."
cp "$WORKERS" "$WORKERS.bak.$(date +%s)"
if [ -f "$SEPARATION" ]; then
    cp "$SEPARATION" "$SEPARATION.bak.$(date +%s)"
fi

# ---------------------------------------------------------------------------
# 2. Patch ui/workers.py — force CPU in SeparationWorker
# ---------------------------------------------------------------------------
echo ">> Patching $WORKERS..."

python3 - <<'PYEOF'
import re
import sys

path = "ui/workers.py"
with open(path, "r") as f:
    src = f.read()

# Guard against double-patching
if "CoreML" in src and "providers == \"auto\"" in src:
    print("   Already patched — skipping.")
    sys.exit(0)

# Replace the body of SeparationWorker.__init__ so that "auto" becomes ["cpu"]
old_block = '''    def __init__(self, input_file, output_dir, model_name,
                 providers="auto", precision="fp32", chunked=False):
        super().__init__()
        self.input_file = input_file
        self.output_dir = output_dir
        self.model_name = model_name
        self.providers = providers
        self.precision = precision
        self.chunked = chunked
'''

new_block = '''    def __init__(self, input_file, output_dir, model_name,
                 providers="auto", precision="fp32", chunked=False):
        super().__init__()
        self.input_file = input_file
        self.output_dir = output_dir
        self.model_name = model_name
        # CoreML cannot compile the Demucs graph on Apple Silicon
        # ("generic_general_slice: Invalid values 2048 in begin_ids").
        # Force the CPU execution provider unless the caller asked for
        # a specific provider explicitly.
        if providers == "auto":
            self.providers = ["cpu"]
        else:
            self.providers = providers
        self.precision = precision
        self.chunked = chunked
'''

if old_block in src:
    src = src.replace(old_block, new_block)
    with open(path, "w") as f:
        f.write(src)
    print("   SeparationWorker.__init__ patched.")
else:
    print("   WARNING: could not find exact __init__ block.")
    print("   Falling back to targeted line edit...")
    # Fallback: replace the single line that assigns self.providers
    pat = re.compile(
        r"(def __init__\(self, input_file, output_dir, model_name,.*?"
        r"self\.model_name = model_name\s*\n)"
        r"(\s*)self\.providers = providers",
        re.DOTALL)
    def repl(m):
        return (m.group(1) +
                m.group(2) + 'if providers == "auto":\n' +
                m.group(2) + '    self.providers = ["cpu"]\n' +
                m.group(2) + 'else:\n' +
                m.group(2) + '    self.providers = providers')
    new_src, n = pat.subn(repl, src)
    if n == 0:
        print("   ERROR: manual edit required. See instructions.")
        sys.exit(1)
    with open(path, "w") as f:
        f.write(new_src)
    print("   Patched via fallback ({} replacements).".format(n))
PYEOF

# ---------------------------------------------------------------------------
# 3. Verify the patch
# ---------------------------------------------------------------------------
echo ">> Verifying..."
if grep -q 'self.providers = \["cpu"\]' "$WORKERS"; then
    echo "   OK: SeparationWorker now defaults to CPU provider."
else
    echo "   ERROR: patch did not apply. Please edit manually."
    exit 1
fi

# ---------------------------------------------------------------------------
# 4. Sanity check: import the module
# ---------------------------------------------------------------------------
echo ">> Checking syntax..."
python3 -c "import ast; ast.parse(open('$WORKERS').read())" \
    && echo "   $WORKERS: syntax OK" \
    || { echo "   $WORKERS: SYNTAX ERROR"; exit 1; }

# ---------------------------------------------------------------------------
# 5. Summary
# ---------------------------------------------------------------------------
echo ""
echo "=================================================="
echo " Done. SeparationWorker now forces CPU provider."
echo ""
echo " If separation is too slow on CPU and you want to"
echo " try GPU later on an NVIDIA machine, edit"
echo " ui/workers.py and remove the 'if providers == \"auto\"'"
echo " block so the caller's choice is respected again."
echo "=================================================="
