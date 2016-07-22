HCC Clang frontend upgrade repository
=====================================

This repository holds the work of upgrading clang frontend used in HCC from
3.5 to tip of tree. The goal is to align the version used in the frontend and 
backend.

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
- LLVM / LLD ToT are built at ~/hcc/llvm_upstream/build .

```
git clone -b clang_tot_upgrade git@github.com:RadeonOpenCompute/hcc.git hcc_upstream
mkdir build_upstream
cd build_upstream
cmake -Wno-dev \
    -DCMAKE_BUILD_TYPE=Release \
    -DHSA_LLVM_BIN_DIR=~/hcc/llvm_upstream/build/bin \
    -DHSA_AMDGPU_GPU_TARGET=fiji \
    -DHSA_USE_AMDGPU_BACKEND=ON \
    ../hcc_upstream
make -j16 world
make -j16
```
