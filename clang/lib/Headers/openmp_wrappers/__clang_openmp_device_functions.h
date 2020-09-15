/*===- __clang_openmp_device_functions.h - OpenMP device function declares -===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */

#ifndef __CLANG_OPENMP_DEVICE_FUNCTIONS_H__
#define __CLANG_OPENMP_DEVICE_FUNCTIONS_H__

#ifndef _OPENMP
#error "This file is for OpenMP compilation only."
#endif

// This autoheader is included for device pass only and only for OpenMP.
// Neither the cuda or hip autoheaders are active. An autoheaderis the first
// header seen in a cc1 pass. One objective of this header is to ensure that
// user-included math.h or <cmath> headers works correctly.
// The include search path order ensures the math.h or <cmath> from clang's
// openmp_wrappers directory are found first(overlays). Those overlays
// include_next the system version of those header files.
// Read the comments in those overlay files for more information.

// __NO_INLINE__ prevents some x86 optimized macro definitions in system headers
#define __NO_INLINE__ 1

#pragma omp begin declare variant match(                                       \
    device = {arch(nvptx, nvptx64)}, implementation = {extension(match_any)})

#ifdef __cplusplus
extern "C" {
#endif

#define __CUDA__
#define __OPENMP_NVPTX__

/// Include declarations for libdevice functions.
#include <__clang_cuda_libdevice_declares.h>

/// Provide definitions for some of the above declares
/// that are not linkable when creating the GPU images
#include <__clang_cuda_device_functions.h>

#undef __OPENMP_NVPTX__
#undef __CUDA__

#ifdef __cplusplus
} // extern "C"
#endif

#pragma omp end declare variant

#ifdef __cplusplus
extern "C" {
#endif

#pragma omp begin declare variant match(                                       \
    device = {arch(amdgcn)}, implementation = {extension(match_any)})

#define __HIP__ 1
#define __OPENMP_AMDGCN__

#define __host__ __attribute__((host))
#define __device__ __attribute__((device))
#define __global__ __attribute__((global))
#define __shared__ __attribute__((shared))
#define __constant__ __attribute__((constant))
#define __private __attribute__((address_space(5)))

// Q: Does declare variant work when the device variants for amdgcn
//    are the same name as nvptx?
#include <__clang_hip_libdevice_declares.h>

// FIXME: undef all the above defs
#undef __OPENMP_AMDGCN__
#undef __HIP__

#pragma omp end declare variant

#ifdef __cplusplus
} // extern "C"
#endif

#endif
