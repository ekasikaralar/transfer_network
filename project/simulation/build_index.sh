#!/bin/bash
# Builds ./index_sim, the non preemptive index rule simulator.  Needs only a
# C++17 compiler and OpenMP, the JSON header is in third_party.
set -e
cd "$(dirname "$0")"
if [[ $(uname) == Darwin ]]; then
  OMP=$(brew --prefix libomp)
  ARCH=$(file "$OMP/lib/libomp.dylib" | grep -q x86_64 && echo x86_64 || echo arm64)
  clang++ -std=c++17 -O2 -arch $ARCH -Xpreprocessor -fopenmp -Ithird_party -I"$OMP/include" -L"$OMP/lib" -lomp -o index_sim index_sim.cpp
else
  g++ -std=c++17 -O2 -fopenmp -Ithird_party -o index_sim index_sim.cpp
fi
echo "built ./index_sim"
