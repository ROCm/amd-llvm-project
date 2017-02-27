// RUN: %clang_cc1 -emit-llvm %s -o - -fcuda-is-device -triple nvptx-unknown-unknown | FileCheck --check-prefixes=NVPTX,CHECK %s
// RUN: %clang_cc1 -emit-llvm %s -o - -fcuda-is-device -triple amdgcn | FileCheck --check-prefixes=AMDGCN,CHECK %s

// Verifies Clang emits correct address spaces and addrspacecast instructions
// for CUDA code.

#include "Inputs/cuda.h"

// CHECK: @i = addrspace(1) externally_initialized global
__device__ int i;

// AMDGCN: @j = addrspace(2) externally_initialized global
// NVPTX: @j = addrspace(4) externally_initialized global
__constant__ int j;

// CHECK: @k = addrspace(3) global
__shared__ int k;

struct MyStruct {
  int data1;
  int data2;
};

// CHECK: @_ZZ5func0vE1a = internal addrspace(3) global %struct.MyStruct zeroinitializer
// CHECK: @_ZZ5func1vE1a = internal addrspace(3) global float 0.000000e+00
// CHECK: @_ZZ5func2vE1a = internal addrspace(3) global [256 x float] zeroinitializer
// CHECK: @_ZZ5func3vE1a = internal addrspace(3) global float 0.000000e+00
// CHECK: @_ZZ5func4vE1a = internal addrspace(3) global float 0.000000e+00
// CHECK: @b = addrspace(3) global float undef

__device__ void foo() {
  // NVPTX: load i32, i32* addrspacecast (i32 addrspace(1)* @i to i32*)
  // AMDGCN: load i32, i32 addrspace(4)* addrspacecast (i32 addrspace(1)* @i to i32 addrspace(4)*)
  i++;

  // NVPTX: load i32, i32* addrspacecast (i32 addrspace(4)* @j to i32*)
  // AMDGCN: load i32, i32 addrspace(4)* addrspacecast (i32 addrspace(2)* @j to i32 addrspace(4)*)
  j++;

  // NVPTX: load i32, i32* addrspacecast (i32 addrspace(3)* @k to i32*)
  // AMDGCN: load i32, i32 addrspace(4)* addrspacecast (i32 addrspace(3)* @k to i32 addrspace(4)*)
  k++;

  __shared__ int lk;
  // NVPTX: load i32, i32* addrspacecast (i32 addrspace(3)* @_ZZ3foovE2lk to i32*)
  // AMDGCN: load i32, i32 addrspace(4)* addrspacecast (i32 addrspace(3)* @_ZZ3foovE2lk to i32 addrspace(4)*)
  lk++;
}

__device__ void func0() {
  __shared__ MyStruct a;
  MyStruct *ap = &a; // composite type
  ap->data1 = 1;
  ap->data2 = 2;
}
// CHECK-LABEL: define void @_Z5func0v()
// NVPTX: store %struct.MyStruct* addrspacecast (%struct.MyStruct addrspace(3)* @_ZZ5func0vE1a to %struct.MyStruct*), %struct.MyStruct** %ap
// AMDGCN: store %struct.MyStruct addrspace(4)* addrspacecast (%struct.MyStruct addrspace(3)* @_ZZ5func0vE1a to %struct.MyStruct addrspace(4)*), %struct.MyStruct addrspace(4)* addrspace(4)* %ap

__device__ void callee(float *ap) {
  *ap = 1.0f;
}

__device__ void func1() {
  __shared__ float a;
  callee(&a); // implicit cast from parameters
}
// CHECK-LABEL: define void @_Z5func1v()
// NVPTX: call void @_Z6calleePf(float* addrspacecast (float addrspace(3)* @_ZZ5func1vE1a to float*))
// AMDGCN: call void @_Z6calleePf(float addrspace(4)* addrspacecast (float addrspace(3)* @_ZZ5func1vE1a to float addrspace(4)*))

__device__ void func2() {
  __shared__ float a[256];
  float *ap = &a[128]; // implicit cast from a decayed array
  *ap = 1.0f;
}
// CHECK-LABEL: define void @_Z5func2v()
// NVPTX: store float* getelementptr inbounds ([256 x float], [256 x float]* addrspacecast ([256 x float] addrspace(3)* @_ZZ5func2vE1a to [256 x float]*), i32 0, i32 128), float** %ap
// AMDGCN: store float addrspace(4)* getelementptr inbounds ([256 x float], [256 x float] addrspace(4)* addrspacecast ([256 x float] addrspace(3)* @_ZZ5func2vE1a to [256 x float] addrspace(4)*), i64 0, i64 128), float addrspace(4)* addrspace(4)* %ap
__device__ void func3() {
  __shared__ float a;
  float *ap = reinterpret_cast<float *>(&a); // explicit cast
  *ap = 1.0f;
}
// CHECK-LABEL: define void @_Z5func3v()
// NVPTX: store float* addrspacecast (float addrspace(3)* @_ZZ5func3vE1a to float*), float** %ap
// AMDGCN: store float addrspace(4)* addrspacecast (float addrspace(3)* @_ZZ5func3vE1a to float addrspace(4)*), float addrspace(4)* addrspace(4)* %ap

__device__ void func4() {
  __shared__ float a;
  float *ap = (float *)&a; // explicit c-style cast
  *ap = 1.0f;
}
// CHECK-LABEL: define void @_Z5func4v()
// NVPTX: store float* addrspacecast (float addrspace(3)* @_ZZ5func4vE1a to float*), float** %ap
// AMDGCN: store float addrspace(4)* addrspacecast (float addrspace(3)* @_ZZ5func4vE1a to float addrspace(4)*), float addrspace(4)* addrspace(4)* %ap

__shared__ float b;

__device__ float *func5() {
  return &b; // implicit cast from a return value
}
// NVPTX-LABEL: define float* @_Z5func5v()
// AMDGCN-LABEL: define float addrspace(4)* @_Z5func5v()
// NVPTX: ret float* addrspacecast (float addrspace(3)* @b to float*)
// AMDGCN: ret float addrspace(4)* addrspacecast (float addrspace(3)* @b to float addrspace(4)*)
