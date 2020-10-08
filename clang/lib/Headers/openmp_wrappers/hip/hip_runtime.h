/*===-- complex --- OpenMP complex wrapper for target regions --------- c++ -===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */

#ifndef __CLANG_OPENMP_HIP_RUNTIME_H__
#define __CLANG_OPENMP_HIP_RUNTIME_H__

#ifndef _OPENMP
#error "This file is for OpenMP compilation only."
#endif

#include <time.h>
#warning ====== START OVERLAY hip_runtime.
#define __OPENMP_AMDGCN__
#include_next <hip/hip_runtime.h>
#warning ====== DONE INCLUDE of ACTUAL hip_runtime.

#pragma omp begin declare variant match(                                       \
    device = {arch(amdgcn)}, implementation = {extension(match_any)})

// #pragma omp declare target

//long long int clock();
#pragma omp end declare variant
// #pragma omp end declare target

#warning ====== DONE OVERLAY
// Now get the actual hip headers 

#endif
