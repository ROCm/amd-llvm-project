# RUN: llvm-mc %s -filetype=obj -triple=i686-pc-linux -o %t
# RUN: llvm-dwarfdump -v %t | FileCheck %s

# CHECK:      .eh_frame contents:
# CHECK:          FDE
# CHECK-NEXT:     Format:       DWARF32
# CHECK-NEXT:     DW_CFA_LLVM_def_aspace_cfa: reg48 +2 as0
# CHECK-NEXT:     DW_CFA_advance_loc1: 0
# CHECK-NEXT:     DW_CFA_restore_extended: reg0
# CHECK-NEXT:     DW_CFA_nop:

.text
.globl foo
.type  foo,@function
foo:
 .cfi_startproc
.cfi_llvm_def_aspace_cfa %edx, 0, 6
 .cfi_endproc
