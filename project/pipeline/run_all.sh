#!/bin/bash
# Runs the whole pipeline in order. =
#   ./run_all.sh            everything
#   ./run_all.sh 6 7        only the calibration steps
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
steps=("$@")
[[ $# -eq 0 ]] && steps=(1 2 3 4 5 6 7)
for s in "${steps[@]}"; do
  f=$(ls 0${s}_*.py)
  echo "=== $f"
  $PY $f
done
