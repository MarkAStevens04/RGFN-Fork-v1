#!/bin/bash
# setup_fraggfn.sh — OURS. Install the fraggfn baseline generator for validation.
#
# The fraggfn adapter in validation/generators/fraggfn/ is a THIN wrapper; the
# heavy upstream code is installed here (not vendored into the repo). See
# validation/generators/README.md for the entrant table and the boundary rule.
#
# STATUS: placeholder stub — not implemented yet. Fill in the upstream clone /
# pip install / weights download below, mirroring setup_reinvent.sh.

if [ $# -eq 0 ]; then
    echo "Usage: $0 <workspace_directory>"
    exit 1
fi

WORKSPACE=$1
mkdir -p "$WORKSPACE"
cd "$WORKSPACE"

echo "[setup_fraggfn] TODO: clone + install the fraggfn baseline into $WORKSPACE"
echo "[setup_fraggfn] not implemented yet — see validation/generators/README.md"
exit 1
