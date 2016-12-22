ToT HCC Clang
=============

This repository hosts ToT HCC Clang which is synchronized with upstream Clang.

Branches
========
- master : holds production-ready codes

- upstream : holds commits from upstream clang
  The branch always have the latest vanilla clang.

- clang_tot_upgrade : holds hcc-specific codes
  Developments are always conducted here.

- release_YYWW : release branches for week WW year YY
  Periodically release branches would be created to merge all latest commits
  upstream and develop branch. Once tested, it would be promoted to master.

How to Build It
===============
This is how I build it now. The commands assumes:
- ROCm stack is already installed
- ROCm-Device-Libs is built, and installed at ~/hcc/ROCm-Device-Libs/build/dist
- N is the number of threads available for make

```bash
git clone --recursive -b clang_tot_upgrade git@github.com:RadeonOpenCompute/hcc.git hcc_upstream
mkdir build_upstream
cd build_upstream
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DHSA_AMDGPU_GPU_TARGET=gfx803 \
    -DROCM_DEVICE_LIB_DIR=~/hcc/ROCm-Device-Libs/build/dist/lib \
    ../hcc_upstream
make -jN world
make -jN
```
