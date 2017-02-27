// REQUIRES: nvptx-registered-target

// Make sure we don't allow dynamic initialization for device
// variables, but accept empty constructors allowed by CUDA.

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fcuda-is-device -std=c++11 \
// RUN:     -fno-threadsafe-statics -emit-llvm -o - %s | FileCheck --check-prefixes=CHECK,NVPTX %s

// RUN: %clang_cc1 -triple amdgcn -fcuda-is-device -std=c++11 \
// RUN:     -fno-threadsafe-statics -emit-llvm -o - %s | FileCheck --check-prefixes=CHECK,AMDGCN %s

#ifdef __clang__
#include "Inputs/cuda.h"
#endif

// Use the types we share with Sema tests.
#include "Inputs/cuda-initializers.h"

__device__ int d_v;
// CHECK: @d_v = addrspace(1) externally_initialized global i32 0,
__shared__ int s_v;
// CHECK: @s_v = addrspace(3) global i32 undef,
__constant__ int c_v;
// NVPTX: addrspace(4) externally_initialized global i32 0,
// AMDGCN: addrspace(2) externally_initialized global i32 0,

__device__ int d_v_i = 1;
// CHECK: @d_v_i = addrspace(1) externally_initialized global i32 1,

// trivial constructor -- allowed
__device__ T d_t;
// CHECK: @d_t = addrspace(1) externally_initialized global %struct.T zeroinitializer
__shared__ T s_t;
// CHECK: @s_t = addrspace(3) global %struct.T undef,
__constant__ T c_t;
// NVPTX: @c_t = addrspace(4) externally_initialized global %struct.T zeroinitializer,
// AMDGCN: @c_t = addrspace(2) externally_initialized global %struct.T zeroinitializer,

__device__ T d_t_i = {2};
// CHECK: @d_t_i = addrspace(1) externally_initialized global %struct.T { i32 2 },
__constant__ T c_t_i = {2};
// NVPTX: @c_t_i = addrspace(4) externally_initialized global %struct.T { i32 2 },
// AMDGCN: @c_t_i = addrspace(2) externally_initialized global %struct.T { i32 2 },

// empty constructor
__device__ EC d_ec;
// CHECK: @d_ec = addrspace(1) externally_initialized global %struct.EC zeroinitializer,
__shared__ EC s_ec;
// CHECK: @s_ec = addrspace(3) global %struct.EC undef,
__constant__ EC c_ec;
// NVPTX: @c_ec = addrspace(4) externally_initialized global %struct.EC zeroinitializer,
// AMDGCN: @c_ec = addrspace(2) externally_initialized global %struct.EC zeroinitializer,

// empty destructor
__device__ ED d_ed;
// CHECK: @d_ed = addrspace(1) externally_initialized global %struct.ED zeroinitializer,
__shared__ ED s_ed;
// CHECK: @s_ed = addrspace(3) global %struct.ED undef,
__constant__ ED c_ed;
// NVPTX: @c_ed = addrspace(4) externally_initialized global %struct.ED zeroinitializer,
// AMDGCN: @c_ed = addrspace(2) externally_initialized global %struct.ED zeroinitializer,

__device__ ECD d_ecd;
// CHECK: @d_ecd = addrspace(1) externally_initialized global %struct.ECD zeroinitializer,
__shared__ ECD s_ecd;
// CHECK: @s_ecd = addrspace(3) global %struct.ECD undef,
__constant__ ECD c_ecd;
// NVPTX: @c_ecd = addrspace(4) externally_initialized global %struct.ECD zeroinitializer,
// AMDGCN: @c_ecd = addrspace(2) externally_initialized global %struct.ECD zeroinitializer,

// empty templated constructor -- allowed with no arguments
__device__ ETC d_etc;
// CHECK: @d_etc = addrspace(1) externally_initialized global %struct.ETC zeroinitializer,
__shared__ ETC s_etc;
// CHECK: @s_etc = addrspace(3) global %struct.ETC undef,
__constant__ ETC c_etc;
// NVPTX: @c_etc = addrspace(4) externally_initialized global %struct.ETC zeroinitializer,
// AMDGCN: @c_etc = addrspace(2) externally_initialized global %struct.ETC zeroinitializer,

__device__ NCFS d_ncfs;
// CHECK: @d_ncfs = addrspace(1) externally_initialized global %struct.NCFS { i32 3 }
__constant__ NCFS c_ncfs;
// NVPTX: @c_ncfs = addrspace(4) externally_initialized global %struct.NCFS { i32 3 }
// AMDGCN: @c_ncfs = addrspace(2) externally_initialized global %struct.NCFS { i32 3 }

// Regular base class -- allowed
__device__ T_B_T d_t_b_t;
// CHECK: @d_t_b_t = addrspace(1) externally_initialized global %struct.T_B_T zeroinitializer,
__shared__ T_B_T s_t_b_t;
// CHECK: @s_t_b_t = addrspace(3) global %struct.T_B_T undef,
__constant__ T_B_T c_t_b_t;
// NVPTX: @c_t_b_t = addrspace(4) externally_initialized global %struct.T_B_T zeroinitializer,
// AMDGCN: @c_t_b_t = addrspace(2) externally_initialized global %struct.T_B_T zeroinitializer,

// Incapsulated object of allowed class -- allowed
__device__ T_F_T d_t_f_t;
// CHECK: @d_t_f_t = addrspace(1) externally_initialized global %struct.T_F_T zeroinitializer,
__shared__ T_F_T s_t_f_t;
// CHECK: @s_t_f_t = addrspace(3) global %struct.T_F_T undef,
__constant__ T_F_T c_t_f_t;
// NVPTX: @c_t_f_t = addrspace(4) externally_initialized global %struct.T_F_T zeroinitializer,
// AMDGCN: @c_t_f_t = addrspace(2) externally_initialized global %struct.T_F_T zeroinitializer,

// array of allowed objects -- allowed
__device__ T_FA_T d_t_fa_t;
// CHECK: @d_t_fa_t = addrspace(1) externally_initialized global %struct.T_FA_T zeroinitializer,
__shared__ T_FA_T s_t_fa_t;
// CHECK: @s_t_fa_t = addrspace(3) global %struct.T_FA_T undef,
__constant__ T_FA_T c_t_fa_t;
// NVPTX: @c_t_fa_t = addrspace(4) externally_initialized global %struct.T_FA_T zeroinitializer,
// AMDGCN: @c_t_fa_t = addrspace(2) externally_initialized global %struct.T_FA_T zeroinitializer,


// Calling empty base class initializer is OK
__device__ EC_I_EC d_ec_i_ec;
// CHECK: @d_ec_i_ec = addrspace(1) externally_initialized global %struct.EC_I_EC zeroinitializer,
__shared__ EC_I_EC s_ec_i_ec;
// CHECK: @s_ec_i_ec = addrspace(3) global %struct.EC_I_EC undef,
__constant__ EC_I_EC c_ec_i_ec;
// NVPTX: @c_ec_i_ec = addrspace(4) externally_initialized global %struct.EC_I_EC zeroinitializer,
// AMDGCN: @c_ec_i_ec = addrspace(2) externally_initialized global %struct.EC_I_EC zeroinitializer,

// We should not emit global initializers for device-side variables.
// CHECK-NOT: @__cxx_global_var_init

// Make sure that initialization restrictions do not apply to local
// variables.
__device__ void df() {
  T t;
  // CHECK-NOT: call
  EC ec;
  // NVPTX:   call void @_ZN2ECC1Ev(%struct.EC* %ec)
  // AMDGCN:   call void @_ZN2ECC1Ev(%struct.EC addrspace(4)* %ec)
  ED ed;
  // CHECK-NOT: call
  ECD ecd;
  // NVPTX:   call void @_ZN3ECDC1Ev(%struct.ECD* %ecd)
  // AMDGCN:   call void @_ZN3ECDC1Ev(%struct.ECD addrspace(4)* %ecd)
  ETC etc;
  // NVPTX:   call void @_ZN3ETCC1IJEEEDpT_(%struct.ETC* %etc)
  // AMDGCN:   call void @_ZN3ETCC1IJEEEDpT_(%struct.ETC addrspace(4)* %etc)
  UC uc;
  // undefined constructor -- not allowed
  // NVPTX:   call void @_ZN2UCC1Ev(%struct.UC* %uc)
  // AMDGCN:   call void @_ZN2UCC1Ev(%struct.UC addrspace(4)* %uc)
  UD ud;
  // undefined destructor -- not allowed
  // CHECK-NOT: call
  ECI eci;
  // empty constructor w/ initializer list -- not allowed
  // NVPTX:   call void @_ZN3ECIC1Ev(%struct.ECI* %eci)
  // AMDGCN:   call void @_ZN3ECIC1Ev(%struct.ECI addrspace(4)* %eci)
  NEC nec;
  // non-empty constructor -- not allowed
  // NVPTX:   call void @_ZN3NECC1Ev(%struct.NEC* %nec)
  // AMDGCN:   call void @_ZN3NECC1Ev(%struct.NEC addrspace(4)* %nec)
  // non-empty destructor -- not allowed
  NED ned;
  // no-constructor,  virtual method -- not allowed
  // NVPTX:   call void @_ZN3NCVC1Ev(%struct.NCV* %ncv)
  // AMDGCN:   call void @_ZN3NCVC1Ev(%struct.NCV addrspace(4)* %ncv)
  NCV ncv;
  // CHECK-NOT: call
  VD vd;
  // NVPTX:   call void @_ZN2VDC1Ev(%struct.VD* %vd)
  // AMDGCN:   call void @_ZN2VDC1Ev(%struct.VD addrspace(4)* %vd)
  NCF ncf;
  // NVPTX:   call void @_ZN3NCFC1Ev(%struct.NCF* %ncf)
  // AMDGCN:   call void @_ZN3NCFC1Ev(%struct.NCF addrspace(4)* %ncf)
  NCFS ncfs;
  // NVPTX:   call void @_ZN4NCFSC1Ev(%struct.NCFS* %ncfs)
  // AMDGCN:   call void @_ZN4NCFSC1Ev(%struct.NCFS addrspace(4)* %ncfs)
  UTC utc;
  // NVPTX:   call void @_ZN3UTCC1IJEEEDpT_(%struct.UTC* %utc)
  // AMDGCN:   call void @_ZN3UTCC1IJEEEDpT_(%struct.UTC addrspace(4)* %utc)
  NETC netc;
  // NVPTX:   call void @_ZN4NETCC1IJEEEDpT_(%struct.NETC* %netc)
  // AMDGCN:   call void @_ZN4NETCC1IJEEEDpT_(%struct.NETC addrspace(4)* %netc)
  T_B_T t_b_t;
  // CHECK-NOT: call
  T_F_T t_f_t;
  // CHECK-NOT: call
  T_FA_T t_fa_t;
  // CHECK-NOT: call
  EC_I_EC ec_i_ec;
  // NVPTX:   call void @_ZN7EC_I_ECC1Ev(%struct.EC_I_EC* %ec_i_ec)
  // AMDGCN:   call void @_ZN7EC_I_ECC1Ev(%struct.EC_I_EC addrspace(4)* %ec_i_ec)
  EC_I_EC1 ec_i_ec1;
  // NVPTX:   call void @_ZN8EC_I_EC1C1Ev(%struct.EC_I_EC1* %ec_i_ec1)
  // AMDGCN:   call void @_ZN8EC_I_EC1C1Ev(%struct.EC_I_EC1 addrspace(4)* %ec_i_ec1)
  T_V_T t_v_t;
  // NVPTX:   call void @_ZN5T_V_TC1Ev(%struct.T_V_T* %t_v_t)
  // AMDGCN:   call void @_ZN5T_V_TC1Ev(%struct.T_V_T addrspace(4)* %t_v_t)
  T_B_NEC t_b_nec;
  // NVPTX:   call void @_ZN7T_B_NECC1Ev(%struct.T_B_NEC* %t_b_nec)
  // AMDGCN:   call void @_ZN7T_B_NECC1Ev(%struct.T_B_NEC addrspace(4)* %t_b_nec)
  T_F_NEC t_f_nec;
  // NVPTX:   call void @_ZN7T_F_NECC1Ev(%struct.T_F_NEC* %t_f_nec)
  // AMDGCN:   call void @_ZN7T_F_NECC1Ev(%struct.T_F_NEC addrspace(4)* %t_f_nec)
  T_FA_NEC t_fa_nec;
  // NVPTX:   call void @_ZN8T_FA_NECC1Ev(%struct.T_FA_NEC* %t_fa_nec)
  // AMDGCN:   call void @_ZN8T_FA_NECC1Ev(%struct.T_FA_NEC addrspace(4)* %t_fa_nec)
  T_B_NED t_b_ned;
  // CHECK-NOT: call
  T_F_NED t_f_ned;
  // CHECK-NOT: call
  T_FA_NED t_fa_ned;
  // CHECK-NOT: call
  static __shared__ EC s_ec;
  // NVPTX-NOT: call void @_ZN2ECC1Ev(%struct.EC* addrspacecast (%struct.EC addrspace(3)* @_ZZ2dfvE4s_ec to %struct.EC*))
  // AMDGCN-NOT: call void @_ZN2ECC1Ev(%struct.EC addrspace(4)* addrspacecast (%struct.EC addrspace(3)* @_ZZ2dfvE4s_ec to %struct.EC addrspace(4)*))
  static __shared__ ETC s_etc;
  // NVPTX-NOT: call void @_ZN3ETCC1IJEEEDpT_(%struct.ETC* addrspacecast (%struct.ETC addrspace(3)* @_ZZ2dfvE5s_etc to %struct.ETC*))
  // AMDGCN-NOT: call void @_ZN3ETCC1IJEEEDpT_(%struct.ETC addrspace(4)* addrspacecast (%struct.ETC addrspace(3)* @_ZZ2dfvE5s_etc to %struct.ETC addrspace(4)*))

  // anchor point separating constructors and destructors
  df(); // CHECK: call void @_Z2dfv()

  // Verify that we only call non-empty destructors
  // NVPTX-NEXT: call void @_ZN8T_FA_NEDD1Ev(%struct.T_FA_NED* %t_fa_ned)
  // NVPTX-NEXT: call void @_ZN7T_F_NEDD1Ev(%struct.T_F_NED* %t_f_ned)
  // NVPTX-NEXT: call void @_ZN7T_B_NEDD1Ev(%struct.T_B_NED* %t_b_ned)
  // NVPTX-NEXT: call void @_ZN2VDD1Ev(%struct.VD* %vd)
  // NVPTX-NEXT: call void @_ZN3NEDD1Ev(%struct.NED* %ned)
  // NVPTX-NEXT: call void @_ZN2UDD1Ev(%struct.UD* %ud)
  // NVPTX-NEXT: call void @_ZN3ECDD1Ev(%struct.ECD* %ecd)
  // NVPTX-NEXT: call void @_ZN2EDD1Ev(%struct.ED* %ed)

  // AMDGCN-NEXT: call void @_ZN8T_FA_NEDD1Ev(%struct.T_FA_NED addrspace(4)* %t_fa_ned)
  // AMDGCN-NEXT: call void @_ZN7T_F_NEDD1Ev(%struct.T_F_NED addrspace(4)* %t_f_ned)
  // AMDGCN-NEXT: call void @_ZN7T_B_NEDD1Ev(%struct.T_B_NED addrspace(4)* %t_b_ned)
  // AMDGCN-NEXT: call void @_ZN2VDD1Ev(%struct.VD addrspace(4)* %vd)
  // AMDGCN-NEXT: call void @_ZN3NEDD1Ev(%struct.NED addrspace(4)* %ned)
  // AMDGCN-NEXT: call void @_ZN2UDD1Ev(%struct.UD addrspace(4)* %ud)
  // AMDGCN-NEXT: call void @_ZN3ECDD1Ev(%struct.ECD addrspace(4)* %ecd)
  // AMDGCN-NEXT: call void @_ZN2EDD1Ev(%struct.ED addrspace(4)* %ed)

  // CHECK-NEXT: ret void
}

// We should not emit global init function.
// CHECK-NOT: @_GLOBAL__sub_I
