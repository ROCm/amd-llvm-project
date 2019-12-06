; RUN: llc -mtriple amdgcn-amd-amdhsa -mcpu=gfx900 < %s | FileCheck --check-prefix=GCN %s
; GCN: mul
define weak_odr amdgpu_kernel void @test_mul24_knownbits_kernel(float addrspace(1)* %p) {
entry:
  %0 = tail call i32 @llvm.amdgcn.workitem.id.x() #28
  %tid = and i32 %0, 3
  %1 = mul nsw i32 %tid, -5
  %v1 = and i32 %1, -32
  %v2 = sext i32 %v1 to i64
  %v3 = getelementptr inbounds float, float addrspace(1)* %p, i64 %v2
  store float 0.000, float addrspace(1)* %v3, align 4
  ret void
}

; Function Attrs: nounwind readnone speculatable
declare i32 @llvm.amdgcn.workitem.id.x() #20
