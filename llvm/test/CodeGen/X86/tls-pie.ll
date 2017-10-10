; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc < %s -mcpu=generic -mtriple=i386-linux-gnu -relocation-model=pic | FileCheck %s --check-prefix=X86
; RUN: llc < %s -mcpu=generic -mtriple=x86_64-linux-gnux32 -relocation-model=pic | FileCheck %s --check-prefix=X32
; RUN: llc < %s -mcpu=generic -mtriple=x86_64-linux-gnu -relocation-model=pic | FileCheck %s --check-prefix=X64

@i = thread_local global i32 15
@i2 = external thread_local global i32

define i32 @f1() {
; X86-LABEL: f1:
; X86:       # BB#0: # %entry
; X86-NEXT:    movl %gs:i@NTPOFF, %eax
; X86-NEXT:    retl
;
; X32-LABEL: f1:
; X32:       # BB#0: # %entry
; X32-NEXT:    movl %fs:i@TPOFF, %eax
; X32-NEXT:    retq
;
; X64-LABEL: f1:
; X64:       # BB#0: # %entry
; X64-NEXT:    movl %fs:i@TPOFF, %eax
; X64-NEXT:    retq
entry:
	%tmp1 = load i32, i32* @i
	ret i32 %tmp1
}

define i32* @f2() {
; X86-LABEL: f2:
; X86:       # BB#0: # %entry
; X86-NEXT:    movl %gs:0, %eax
; X86-NEXT:    leal i@NTPOFF(%eax), %eax
; X86-NEXT:    retl
;
; X32-LABEL: f2:
; X32:       # BB#0: # %entry
; X32-NEXT:    movl %fs:0, %eax
; X32-NEXT:    leal i@TPOFF(%rax), %eax
; X32-NEXT:    retq
;
; X64-LABEL: f2:
; X64:       # BB#0: # %entry
; X64-NEXT:    movq %fs:0, %rax
; X64-NEXT:    leaq i@TPOFF(%rax), %rax
; X64-NEXT:    retq
entry:
	ret i32* @i
}

define i32 @f3() {
; X86-LABEL: f3:
; X86:       # BB#0: # %entry
; X86-NEXT:    calll .L2$pb
; X86-NEXT:    .cfi_adjust_cfa_offset 4
; X86-NEXT:  .L2$pb:
; X86-NEXT:    popl %eax
; X86-NEXT:    .cfi_adjust_cfa_offset -4
; X86-NEXT:  .Ltmp0:
; X86-NEXT:    addl $_GLOBAL_OFFSET_TABLE_+(.Ltmp0-.L2$pb), %eax
; X86-NEXT:    movl i2@GOTNTPOFF(%eax), %eax
; X86-NEXT:    movl %gs:(%eax), %eax
; X86-NEXT:    retl
;
; X32-LABEL: f3:
; X32:       # BB#0: # %entry
; X32-NEXT:    movl i2@{{.*}}(%rip), %eax
; X32-NEXT:    movl %fs:(%eax), %eax
; X32-NEXT:    retq
;
; X64-LABEL: f3:
; X64:       # BB#0: # %entry
; X64-NEXT:    movq i2@{{.*}}(%rip), %rax
; X64-NEXT:    movl %fs:(%rax), %eax
; X64-NEXT:    retq
entry:
	%tmp1 = load i32, i32* @i2
	ret i32 %tmp1
}

define i32* @f4() {
; X86-LABEL: f4:
; X86:       # BB#0: # %entry
; X86-NEXT:    calll .L3$pb
; X86-NEXT:    .cfi_adjust_cfa_offset 4
; X86-NEXT:  .L3$pb:
; X86-NEXT:    popl %ecx
; X86-NEXT:    .cfi_adjust_cfa_offset -4
; X86-NEXT:  .Ltmp1:
; X86-NEXT:    addl $_GLOBAL_OFFSET_TABLE_+(.Ltmp1-.L3$pb), %ecx
; X86-NEXT:    movl %gs:0, %eax
; X86-NEXT:    addl i2@GOTNTPOFF(%ecx), %eax
; X86-NEXT:    retl
;
; X32-LABEL: f4:
; X32:       # BB#0: # %entry
; X32-NEXT:    movl %fs:0, %eax
; X32-NEXT:    addl i2@{{.*}}(%rip), %eax
; X32-NEXT:    retq
;
; X64-LABEL: f4:
; X64:       # BB#0: # %entry
; X64-NEXT:    movq %fs:0, %rax
; X64-NEXT:    addq i2@{{.*}}(%rip), %rax
; X64-NEXT:    retq
entry:
	ret i32* @i2
}

!llvm.module.flags = !{!0, !1}
!0 = !{i32 1, !"PIC Level", i32 1}
!1 = !{i32 1, !"PIE Level", i32 1}
