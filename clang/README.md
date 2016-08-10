ToT HCC Clang 
=============

This repository ToT HCC Clang which is synchronized with upstream Clang.

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
git clone --recursive -b clang_tot_upgrade git@github.com:RadeonOpenCompute/hcc.git hcc_upstream
mkdir build_upstream
cd build_upstream
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DHSA_LLVM_BIN_DIR=~/hcc/llvm_upstream/build/bin \
    -DHSA_AMDGPU_GPU_TARGET=AMD:AMDGPU:8:0:3 \
    -DHSA_USE_AMDGPU_BACKEND=ON \
    ../hcc_upstream
make -j40 world
make -j40
```
