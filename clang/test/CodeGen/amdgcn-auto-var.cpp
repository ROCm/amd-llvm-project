// RUN: %clang_cc1 -O0 -std=c++11 -emit-llvm -o - -triple amdgcn %s | FileCheck %s

// CHECK: %struct.ATy = type { i32* }
struct ATy {
  int *p;
};

// CHECK-LABEL: @_Z1fPi(i32* %a)
void f(int* a) {
  // CHECK: %[[a_addr:.*]] = alloca i32*
  // CHECK: %[[b:.*]] = alloca i32
  // CHECK: %[[A:.*]] = alloca %struct.ATy, align 8

  // CHECK:  store i32* %a, i32** %[[a_addr]]

  // CHECK:  store i32 1, i32* %[[b]]
  int b = 1;

  // CHECK: %[[p:.*]] = getelementptr inbounds %struct.ATy, %struct.ATy* %[[A]], i32 0, i32 0
  // CHECK: store i32* %[[b]], i32** %[[p]], align 8
  ATy A{&b};

  // CHECK: %[[r0:.*]] = load i32, i32* %b
  // CHECK: %[[r1:.*]] = load i32*, i32** %[[a_addr]]
  // CHECK: store i32 %[[r0]], i32* %[[r1]]
  *a = b;

  // CHECK: store i32* %[[b]], i32** %[[a_addr]], align 8
  a = &b;
}
