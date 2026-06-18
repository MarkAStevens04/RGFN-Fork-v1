#!/bin/bash
set -o pipefail
source /home/markymoo/miniconda3/etc/profile.d/conda.sh
WS=/home/markymoo/projects/RGFN_Fork/RGFN-Fork/vina_workspace

echo "===== [0/4] re-extract pristine boost source ====="
cd "$WS" || exit 1
rm -rf boost_1_83_0
tar -xzf boost_1_83_0.tar.gz || { echo "EXTRACT FAILED"; exit 1; }
echo "check_cxx11.cpp first line: $(head -1 boost_1_83_0/tools/build/src/engine/check_cxx11.cpp)"

echo "===== [1/4] boost bootstrap (gcc only) ====="
( eval "$($LMOD_CMD bash load gcc/11.4.0)"
  echo "  g++=$(which g++)"
  cd "$WS/boost_1_83_0" || exit 1
  ./bootstrap.sh --prefix="$WS/boost" --with-toolset=gcc \
      --with-libraries=program_options,system,filesystem,thread ) || { echo "BOOTSTRAP FAILED"; exit 1; }

echo "===== [2/4] boost b2 install (long step) ====="
( eval "$($LMOD_CMD bash load gcc/11.4.0)"
  cd "$WS/boost_1_83_0" || exit 1
  ./b2 install -j8 >/dev/null ) || { echo "B2 INSTALL FAILED"; exit 1; }
echo "  boost libs: $(ls "$WS/boost/lib" 2>/dev/null | tr '\n' ' ')"

echo "===== [3/4] compile QuickVina2-GPU-2.1 (gcc + cuda) ====="
( eval "$($LMOD_CMD bash load gcc/11.4.0)"
  eval "$($LMOD_CMD bash load cuda/11.8.0)"
  echo "  gcc=$(which gcc) nvcc=$(which nvcc)"
  cd "$WS/Vina-GPU-2.1/QuickVina2-GPU-2.1" || exit 1
  make clean
  make source ) || { echo "VINA MAKE FAILED"; exit 1; }

echo "===== [4/4] verify ====="
BIN="$WS/Vina-GPU-2.1/QuickVina2-GPU-2.1/QuickVina2-GPU-2-1"
if [ -f "$BIN" ]; then echo "BUILD OK: $BIN"; ls -la "$BIN"; else echo "BUILD INCOMPLETE: binary missing"; exit 1; fi
