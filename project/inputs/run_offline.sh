#!/bin/bash
# The offline stage, in order.  Draws the training inputs, solves the linear
# programs in Julia, and trains the offline networks.  Everything resolves
# paths against its own config, so this can run from anywhere.
#   ./run_offline.sh inputs      draw the (x, p) and x observations        (numpy)
#   ./run_offline.sh lps         solve the M and drift linear programs     (Julia)
#   ./run_offline.sh train       train the M network and the drift networks (torch)
#   ./run_offline.sh             all three
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
OFF=../offline_approximation
stage=${1:-all}

if [[ $stage == inputs || $stage == all ]]; then
  $PY $OFF/m_term/separate/input_data/m_net_input_data.py --config_path $OFF/m_term/separate/input_data/config.json
  $PY $OFF/drift_term/separate/input_data/d_net_input_data.py --config_path $OFF/drift_term/separate/input_data/config.json
fi
if [[ $stage == lps || $stage == all ]]; then
  julia $OFF/m_term/separate/output_data/m_term_offline.jl $OFF/m_term/separate/output_data/config.json
  for p in fsf cmu cmu_theta myopic myopic_theta value; do
    julia $OFF/drift_term/separate/output_data/d_term_offline.jl $OFF/drift_term/separate/output_data/config_$p.json
  done
fi
if [[ $stage == train || $stage == all ]]; then
  (cd $OFF/m_term/pooled/nn_training && $PY main.py --config_path config.json --run_name m)
  for p in fsf cmu cmu_theta myopic myopic_theta value; do
    (cd $OFF/drift_term/separate/nn_training && $PY main.py --config_path config_$p.json --run_name drift_$p)
  done
fi
