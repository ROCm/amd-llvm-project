/*===---- offload_macros.h -------------------------------------------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===

 offload_macros.h: Define Universal Offload Macros _DEVICE_ARCH & _DEVICE_GPU
 
   This header creates macros _DEVICE_ARCH and _DEVICE_GPU with values.
   This header exists because compiler macros are inconsistent in specifying if
   a compilation is a device pass or a host pass. There is also inconsistency
   in how the device architecture and type are specified during a device pass.
   The inconsistencies are between OpenMP, CUDA, HIP, and OpenCL.
   The macro logic in this header is aware of these inconsistencies and
   sets useful values for _DEVICE_ARCH and _DEVICE_GPU during a device
   compilation. The macros will NOT be defined during a host compilation pass.
   So "#ifndef _DEVICE_ARCH" can be used by users to mark code for host-only
   compilation. Updates to this header to cover other architectures or other
   compilation environments are very welcome. This header must remain
   a preprocessing only header because it is intended to be used by
   different languages.
*/

#ifndef __OFFLOAD_MACROS_H__
#define __OFFLOAD_MACROS_H__

#undef _DEVICE_GPU
#undef _DEVICE_ARCH

#if defined(_OPENMP)
  // OpenMP does not set architecture macros on host pass.
  // So if either set, this is an OpenMP  device pass.
  #if defined(__AMDGCN__) || defined(__NVPTX__)
    #if defined(__AMDGCN__)
      #define _DEVICE_ARCH amdgcn
      // _DEVICE_GPU set below
    #else
      #define _DEVICE_ARCH nvptx64
      #define _DEVICE_GPU __CUDA_ARCH__
    #endif
  #endif
#elif defined(__CUDA_ARCH__) 
  // CUDA sets macros __NVPTX__ on host pass. So use __CUDA_ARCH__
  // to determine if this is device pass.
  #define _DEVICE_ARCH nvptx64
  #define _DEVICE_GPU __CUDA_ARCH__
#elif defined(__HIP_DEVICE_COMPILE__)
  // HIP sets macros __AMDGCN__ on host pass. So use __HIP_DEVICE_COMPILE__
  // to determine if this is device pass.
  #define _DEVICE_ARCH amdgcn
  // _DEVICE_GPU set below
#elif defined(__SYCL_DEVICE_ONLY__)
  #if defined(__AMDGCN__)
    #define _DEVICE_ARCH amdgcn
    // _DEVICE_GPU set below
  #else
    #define _DEVICE_ARCH nvptx64
    // FIXME: Check that SYCL sets __CUDA_ARCH__ for nvptx
    #define _DEVICE_GPU __CUDA_ARCH__
  #endif
#elif defined(__OPENCL_C_VERSION__) || defined(__OPENCL_CPP_VERSION__)
  // FIXME: check that these macros are not set on OpenCL host pass
  #if defined(__AMDGCN__) 
    #define _DEVICE_ARCH amdgcn
    // _DEVICE_GPU set below
  #endif
  #if defined(__NVPTX__)
    #define _DEVICE_ARCH nvptx64
    // FIXME: Check that OpenCL sets __CUDA_ARCH__ for nvptx
    #define _DEVICE_GPU __CUDA_ARCH__
  #endif
#endif
 
#if defined(_DEVICE_ARCH) && ( _DEVICE_ARCH == amdgcn )
  // AMD uses binary macros for GPU identification
  // Create a generational value x10 for expansion
  #if defined(__gfx601__)
    #define _DEVICE_GPU 6010
  #elif defined(__gfx700__)
    #define _DEVICE_GPU 7000
  #elif defined(__gfx701__)
    #define _DEVICE_GPU 7010
  #elif defined(__gfx702__)
    #define _DEVICE_GPU 7020
  #elif defined(__gfx703__)
    #define _DEVICE_GPU 7030
  #elif defined(__gfx801__)
    #define _DEVICE_GPU 8010
  #elif defined(__gfx802__)
    #define _DEVICE_GPU 8020
  #elif defined(__gfx803__)
    #define _DEVICE_GPU 8030
  #elif defined(__gfx810__)
    #define _DEVICE_GPU 8100
  #elif defined(__gfx900__)
    #define _DEVICE_GPU 9000
  #elif defined(__gfx902__)
    #define _DEVICE_GPU 9020
  #elif defined(__gfx904__)
    #define _DEVICE_GPU 9040
  #elif defined(__gfx906__)
    #define _DEVICE_GPU 9060
  #elif defined(__gfx908__)
    #define _DEVICE_GPU 9080
  #elif defined(__gfx909__)
    #define _DEVICE_GPU 9090
  #elif defined(__gfx1010__)
    #define _DEVICE_GPU 10100
  #elif defined(__gfx1011__)
    #define _DEVICE_GPU 10110
  #elif defined(__gfx1012__)
    #define _DEVICE_GPU 10120
  #else
    #define _DEVICE_GPU UNKNOWN
  #endif
#endif

#endif // __OFFLOAD_MACROS_H__
