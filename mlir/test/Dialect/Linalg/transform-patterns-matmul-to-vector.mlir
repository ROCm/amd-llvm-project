// RUN: mlir-opt %s -test-linalg-transform-patterns=test-matmul-to-vector-patterns-tile-1d | FileCheck %s
// RUN: mlir-opt %s -test-linalg-transform-patterns=test-matmul-to-vector-patterns-tile-2d | FileCheck %s

func @matmul(%A: memref<1584x1584xf32, offset: 0, strides: [1584, 1]>,
                  %B: memref<1584x1584xf32, offset: 0, strides: [1584, 1]>,
                  %C: memref<1584x1584xf32, offset: 0, strides: [1584, 1]>) {
  linalg.matmul %A, %B, %C {__internal_linalg_transform__ = "START"} :
    (memref<1584x1584xf32, offset: 0, strides: [1584, 1]>,
     memref<1584x1584xf32, offset: 0, strides: [1584, 1]>,
     memref<1584x1584xf32, offset: 0, strides: [1584, 1]>)
  return
}

// CHECK-LABEL:func @matmul
//      CHECK: vector.broadcast {{.*}} : f32 to vector<8x16xf32>
//      CHECK: store {{.*}}[] : memref<vector<8x16xf32>>
//
//      CHECK: vector.broadcast {{.*}} : f32 to vector<16x12xf32>
//      CHECK: store {{.*}}[] : memref<vector<16x12xf32>>
//
//      CHECK: vector.broadcast {{.*}} : f32 to vector<8x12xf32>
//      CHECK: store {{.*}}[] : memref<vector<8x12xf32>>
//
//      CHECK: linalg.copy
//      CHECK: linalg.copy
//      CHECK: linalg.copy
//
//      CHECK: vector.contract
// CHECK-SAME:   iterator_types = ["parallel", "parallel", "reduction"]
// CHECK-SAME:   : vector<8x16xf32>, vector<16x12xf32> into vector<8x12xf32>
//
//      CHECK: linalg.copy
