//===--- amdgcn_sync.hip - GPU OpenMP synchronization extensions -- HIP -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements amdgcn extensions to sync
//
//===----------------------------------------------------------------------===//

#include "common/omptarget.h"
#include "target_impl.h"

///////////////////////////////////////////////////////////////////////////////
/// The following five functions are used to coordinate barriers between
/// the master and worker warps in a generic kernel.  The amdgcn architecture
/// does not have a partial barrier so this coordination needs to be
/// recover if a user or implicit barrier is encountered in the sequential
/// part of the master warp or the parallel part of the worker warps.
///
/// __kmpc_amd_worker_start has a barrier to prevent worker warps
/// from starting till the master warp sets the workFn.  This function sets
/// workers_active to true after the barrier to keep the master warp
/// at its barrier in case a worker encounters an explicit or implicit
/// barrier.

volatile DEVICE SHARED unsigned * omptarget_generic_state_ptr;
volatile DEVICE unsigned omptarget_generic_state;
#define set_master_active *omptarget_generic_state_ptr = 0;
#define master_active (!(*omptarget_generic_state_ptr))
#define set_workers_active *omptarget_generic_state_ptr = 1;
#define workers_active (*omptarget_generic_state_ptr)

// Use the following to switch to atomic state management
//#define set_master_active __kmpc_atomic_cas(omptarget_generic_state_ptr, 1u, 0u);
//#define master_active (!__kmpc_atomic_add(omptarget_generic_state_ptr,0u))
//#define set_workers_active __kmpc_atomic_cas(omptarget_generic_state_ptr, 0u, 1u);
//#define workers_active __kmpc_atomic_add(omptarget_generic_state_ptr,0u)

EXTERN void __kmpc_amd_worker_start(kmp_Ident *loc_ref, int32_t tid) {
  omptarget_generic_state_ptr = &omptarget_generic_state;
  set_master_active
  __kmpc_impl_syncthreads();
  while(master_active) __kmpc_impl_syncthreads();
  set_workers_active
}

/// __kmpc_amd_worker_end sets workers_active to false and then
/// issues a barrier to release the master warp to terminate or get
/// the next subregion of work to process.
EXTERN void __kmpc_amd_worker_end(kmp_Ident *loc_ref, int32_t tid) {
  set_master_active
  __kmpc_impl_syncthreads(); // to release 2nd or looped barrier in master_end
}

/// __kmpc_amd_master_start is executed first by the master.  It sets
/// master_active to true to keep the worker warps at its first barrier
/// in case the sequential part of the target region encounters any
/// barrier, explicit or otherwise.
EXTERN void __kmpc_amd_master_start(kmp_Ident *loc_ref, int32_t tid) {
  omptarget_generic_state_ptr = &omptarget_generic_state;
  set_master_active
}

/// __kmpc_amd_master_end sets master_active to false and then enters
/// the double barrier. The first releases the worker warps. The 2nd
/// barrier holds the master warp until the workers are done. If the worker
/// warps encounter a user barrier (implicitly or explicitly), the master
/// warp needs to loop at the barrier until it knows the worker is
/// really done by testing omptarget_generic_state
EXTERN void __kmpc_amd_master_end(kmp_Ident *loc_ref, int32_t tid) {
  set_workers_active
  __kmpc_impl_syncthreads(); // Release workers
  // This is either the barrier at end of worker or spurious barrier
  __kmpc_impl_syncthreads();
  // If it was spurious, master needs another barrier to hold it
  while(workers_active)  __kmpc_impl_syncthreads();
  set_master_active
}

// Release the last set of workers waiting at amd_worker_start
EXTERN void __kmpc_amd_master_terminate(kmp_Ident *loc_ref, int32_t tid) {
  set_workers_active
  __kmpc_impl_syncthreads();
  __kmpc_impl_syncthreads();
}
