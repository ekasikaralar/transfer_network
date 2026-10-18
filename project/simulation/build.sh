#!/bin/bash
# Builds ./transfer_sim against Gurobi.  The JSON header is in third_party, so
# only Gurobi and OpenMP are needed.
#   GUROBI_HOME   the Gurobi install; a cluster module usually sets it, the
#                 default below is the Mac install.  The versioned library,
#                 libgurobi110 or libgurobi120, is picked up from its lib folder.
# On Linux the C++ wrapper libgurobi_c++ is compiled here from the sources
# Gurobi ships in $GUROBI_HOME/src/cpp, with the same compiler as the
# simulator.  The prebuilt one in lib/ is often built with an older compiler
# ABI and then every GRB* symbol comes back undefined at link time.
set -e
cd "$(dirname "$0")"
G=${GUROBI_HOME:-/Library/gurobi1101/macos_universal2}
LIB=$(ls "$G/lib" | grep -o 'libgurobi[0-9]*\.' | head -1 | sed 's/^lib//; s/\.$//')
[[ -n "$LIB" ]] || { echo "no libgurobi<version> in $G/lib; set GUROBI_HOME"; exit 1; }
if [[ $(uname) == Darwin ]]; then
  OMP=$(brew --prefix libomp)
  ARCH=$(file "$OMP/lib/libomp.dylib" | grep -q x86_64 && echo x86_64 || echo arm64)
  clang++ -std=c++17 -O2 -arch $ARCH -Xpreprocessor -fopenmp \
    -Ithird_party -I"$G/include" -I"$OMP/include" transfer_sim.cpp \
    -L"$G/lib" -L"$OMP/lib" -lgurobi_c++ -l$LIB -lomp \
    -o transfer_sim
else
  CPPLIB=build_gurobi_cpp
  if [[ -d "$G/src/cpp" ]]; then
    mkdir -p $CPPLIB
    if [[ ! -f $CPPLIB/libgurobi_c++.a || "$G/src/cpp" -nt $CPPLIB/libgurobi_c++.a ]]; then
      echo "compiling the Gurobi C++ wrapper from $G/src/cpp"
      for f in "$G"/src/cpp/*.cpp; do g++ -std=c++17 -O2 -I"$G/include" -c "$f" -o $CPPLIB/$(basename "${f%.cpp}").o; done
      ar rcs $CPPLIB/libgurobi_c++.a $CPPLIB/*.o
    fi
    CPPFLAGS="-L$CPPLIB"
  else
    CPPFLAGS=""
  fi
  # the source comes before the libraries: GNU ld scans a static archive
  # only for symbols already undefined when it reaches it
  g++ -std=c++17 -O2 -fopenmp -Ithird_party -I"$G/include" transfer_sim.cpp \
    $CPPFLAGS -L"$G/lib" -Wl,-rpath,"$G/lib" -lgurobi_c++ -l$LIB -o transfer_sim
fi
echo "built ./transfer_sim with Gurobi from $G ($LIB)"
