#!/bin/bash
# Simulates the index rules on the test problem.
#   ./run.sh                     all six rules, replications 0..999
#   ./run.sh 0 100 fsf value     replications 0..99 of the listed rules
set -e
cd "$(dirname "$0")"
start=${1:-0}
end=${2:-1000}
shift 2 2>/dev/null || true
policies=("$@")
[[ $# -eq 0 ]] && policies=(cmu cmu_theta fsf myopic myopic_theta value)
[[ -x ./transfer_sim ]] || ./build.sh
mkdir -p results
for p in "${policies[@]}"; do
  echo "=== $p"
  ./transfer_sim config_$p.json $start $end results/$p.csv
done
