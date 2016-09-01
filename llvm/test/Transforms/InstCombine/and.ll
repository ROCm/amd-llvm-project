; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt < %s -instcombine -S | FileCheck %s

; There should be no 'and' instructions left in any test.

define i32 @test1(i32 %A) {
; CHECK-LABEL: @test1(
; CHECK-NEXT:    ret i32 0
;
  %B = and i32 %A, 0
  ret i32 %B
}

define i32 @test2(i32 %A) {
; CHECK-LABEL: @test2(
; CHECK-NEXT:    ret i32 %A
;
  %B = and i32 %A, -1
  ret i32 %B
}

define i1 @test3(i1 %A) {
; CHECK-LABEL: @test3(
; CHECK-NEXT:    ret i1 false
;
  %B = and i1 %A, false
  ret i1 %B
}

define i1 @test4(i1 %A) {
; CHECK-LABEL: @test4(
; CHECK-NEXT:    ret i1 %A
;
  %B = and i1 %A, true
  ret i1 %B
}

define i32 @test5(i32 %A) {
; CHECK-LABEL: @test5(
; CHECK-NEXT:    ret i32 %A
;
  %B = and i32 %A, %A
  ret i32 %B
}

define i1 @test6(i1 %A) {
; CHECK-LABEL: @test6(
; CHECK-NEXT:    ret i1 %A
;
  %B = and i1 %A, %A
  ret i1 %B
}

; A & ~A == 0
define i32 @test7(i32 %A) {
; CHECK-LABEL: @test7(
; CHECK-NEXT:    ret i32 0
;
  %NotA = xor i32 %A, -1
  %B = and i32 %A, %NotA
  ret i32 %B
}

; AND associates
define i8 @test8(i8 %A) {
; CHECK-LABEL: @test8(
; CHECK-NEXT:    ret i8 0
;
  %B = and i8 %A, 3
  %C = and i8 %B, 4
  ret i8 %C
}

; Test of sign bit, convert to setle %A, 0
define i1 @test9(i32 %A) {
; CHECK-LABEL: @test9(
; CHECK-NEXT:    [[C:%.*]] = icmp slt i32 %A, 0
; CHECK-NEXT:    ret i1 [[C]]
;
  %B = and i32 %A, -2147483648
  %C = icmp ne i32 %B, 0
  ret i1 %C
}

; Test of sign bit, convert to setle %A, 0
define i1 @test9a(i32 %A) {
; CHECK-LABEL: @test9a(
; CHECK-NEXT:    [[C:%.*]] = icmp slt i32 %A, 0
; CHECK-NEXT:    ret i1 [[C]]
;
  %B = and i32 %A, -2147483648
  %C = icmp ne i32 %B, 0
  ret i1 %C
}

define i32 @test10(i32 %A) {
; CHECK-LABEL: @test10(
; CHECK-NEXT:    ret i32 1
;
  %B = and i32 %A, 12
  %C = xor i32 %B, 15
  ; (X ^ C1) & C2 --> (X & C2) ^ (C1&C2)
  %D = and i32 %C, 1
  ret i32 %D
}

define i32 @test11(i32 %A, i32* %P) {
; CHECK-LABEL: @test11(
; CHECK-NEXT:    [[B:%.*]] = or i32 %A, 3
; CHECK-NEXT:    [[C:%.*]] = xor i32 [[B]], 12
; CHECK-NEXT:    store i32 [[C]], i32* %P, align 4
; CHECK-NEXT:    ret i32 3
;
  %B = or i32 %A, 3
  %C = xor i32 %B, 12
  ; additional use of C
  store i32 %C, i32* %P
  ; %C = and uint %B, 3 --> 3
  %D = and i32 %C, 3
  ret i32 %D
}

define i1 @test12(i32 %A, i32 %B) {
; CHECK-LABEL: @test12(
; CHECK-NEXT:    [[TMP1:%.*]] = icmp ult i32 %A, %B
; CHECK-NEXT:    ret i1 [[TMP1]]
;
  %C1 = icmp ult i32 %A, %B
  %C2 = icmp ule i32 %A, %B
  ; (A < B) & (A <= B) === (A < B)
  %D = and i1 %C1, %C2
  ret i1 %D
}

define i1 @test13(i32 %A, i32 %B) {
; CHECK-LABEL: @test13(
; CHECK-NEXT:    ret i1 false
;
  %C1 = icmp ult i32 %A, %B
  %C2 = icmp ugt i32 %A, %B
  ; (A < B) & (A > B) === false
  %D = and i1 %C1, %C2
  ret i1 %D
}

define i1 @test14(i8 %A) {
; CHECK-LABEL: @test14(
; CHECK-NEXT:    [[C:%.*]] = icmp slt i8 %A, 0
; CHECK-NEXT:    ret i1 [[C]]
;
  %B = and i8 %A, -128
  %C = icmp ne i8 %B, 0
  ret i1 %C
}

define i8 @test15(i8 %A) {
; CHECK-LABEL: @test15(
; CHECK-NEXT:    ret i8 0
;
  %B = lshr i8 %A, 7
  ; Always equals zero
  %C = and i8 %B, 2
  ret i8 %C
}

define i8 @test16(i8 %A) {
; CHECK-LABEL: @test16(
; CHECK-NEXT:    ret i8 0
;
  %B = shl i8 %A, 2
  %C = and i8 %B, 3
  ret i8 %C
}

;; ~(~X & Y) --> (X | ~Y)
define i8 @test17(i8 %X, i8 %Y) {
; CHECK-LABEL: @test17(
; CHECK-NEXT:    [[Y_NOT:%.*]] = xor i8 %Y, -1
; CHECK-NEXT:    [[D:%.*]] = or i8 %X, [[Y_NOT]]
; CHECK-NEXT:    ret i8 [[D]]
;
  %B = xor i8 %X, -1
  %C = and i8 %B, %Y
  %D = xor i8 %C, -1
  ret i8 %D
}

define i1 @test18(i32 %A) {
; CHECK-LABEL: @test18(
; CHECK-NEXT:    [[C:%.*]] = icmp ugt i32 %A, 127
; CHECK-NEXT:    ret i1 [[C]]
;
  %B = and i32 %A, -128
  ;; C >= 128
  %C = icmp ne i32 %B, 0
  ret i1 %C
}

define <2 x i1> @test18_vec(<2 x i32> %A) {
; CHECK-LABEL: @test18_vec(
; CHECK-NEXT:    [[C:%.*]] = icmp ugt <2 x i32> %A, <i32 127, i32 127>
; CHECK-NEXT:    ret <2 x i1> [[C]]
;
  %B = and <2 x i32> %A, <i32 -128, i32 -128>
  %C = icmp ne <2 x i32> %B, zeroinitializer
  ret <2 x i1> %C
}

define i1 @test18a(i8 %A) {
; CHECK-LABEL: @test18a(
; CHECK-NEXT:    [[C:%.*]] = icmp ult i8 %A, 2
; CHECK-NEXT:    ret i1 [[C]]
;
  %B = and i8 %A, -2
  %C = icmp eq i8 %B, 0
  ret i1 %C
}

define <2 x i1> @test18a_vec(<2 x i8> %A) {
; CHECK-LABEL: @test18a_vec(
; CHECK-NEXT:    [[C:%.*]] = icmp ult <2 x i8> %A, <i8 2, i8 2>
; CHECK-NEXT:    ret <2 x i1> [[C]]
;
  %B = and <2 x i8> %A, <i8 -2, i8 -2>
  %C = icmp eq <2 x i8> %B, zeroinitializer
  ret <2 x i1> %C
}

define i32 @test19(i32 %A) {
; CHECK-LABEL: @test19(
; CHECK-NEXT:    [[B:%.*]] = shl i32 %A, 3
; CHECK-NEXT:    ret i32 [[B]]
;
  %B = shl i32 %A, 3
  ;; Clearing a zero bit
  %C = and i32 %B, -2
  ret i32 %C
}

define i8 @test20(i8 %A) {
; CHECK-LABEL: @test20(
; CHECK-NEXT:    [[C:%.*]] = lshr i8 %A, 7
; CHECK-NEXT:    ret i8 [[C]]
;
  %C = lshr i8 %A, 7
  ;; Unneeded
  %D = and i8 %C, 1
  ret i8 %D
}

define i1 @test23(i32 %A) {
; CHECK-LABEL: @test23(
; CHECK-NEXT:    [[TMP1:%.*]] = icmp eq i32 %A, 2
; CHECK-NEXT:    ret i1 [[TMP1]]
;
  %B = icmp sgt i32 %A, 1
  %C = icmp sle i32 %A, 2
  %D = and i1 %B, %C
  ret i1 %D
}

; FIXME: Vectors should fold too.
define <2 x i1> @test23vec(<2 x i32> %A) {
; CHECK-LABEL: @test23vec(
; CHECK-NEXT:    [[B:%.*]] = icmp sgt <2 x i32> %A, <i32 1, i32 1>
; CHECK-NEXT:    [[C:%.*]] = icmp slt <2 x i32> %A, <i32 3, i32 3>
; CHECK-NEXT:    [[D:%.*]] = and <2 x i1> [[B]], [[C]]
; CHECK-NEXT:    ret <2 x i1> [[D]]
;
  %B = icmp sgt <2 x i32> %A, <i32 1, i32 1>
  %C = icmp sle <2 x i32> %A, <i32 2, i32 2>
  %D = and <2 x i1> %B, %C
  ret <2 x i1> %D
}

define i1 @test24(i32 %A) {
; CHECK-LABEL: @test24(
; CHECK-NEXT:    [[TMP1:%.*]] = icmp sgt i32 %A, 2
; CHECK-NEXT:    ret i1 [[TMP1]]
;
  %B = icmp sgt i32 %A, 1
  %C = icmp ne i32 %A, 2
  ;; A > 2
  %D = and i1 %B, %C
  ret i1 %D
}

define i1 @test25(i32 %A) {
; CHECK-LABEL: @test25(
; CHECK-NEXT:    [[A_OFF:%.*]] = add i32 %A, -50
; CHECK-NEXT:    [[TMP1:%.*]] = icmp ult i32 [[A_OFF]], 50
; CHECK-NEXT:    ret i1 [[TMP1]]
;
  %B = icmp sge i32 %A, 50
  %C = icmp slt i32 %A, 100
  %D = and i1 %B, %C
  ret i1 %D
}

; FIXME: Vectors should fold too.
define <2 x i1> @test25vec(<2 x i32> %A) {
; CHECK-LABEL: @test25vec(
; CHECK-NEXT:    [[B:%.*]] = icmp sgt <2 x i32> %A, <i32 49, i32 49>
; CHECK-NEXT:    [[C:%.*]] = icmp slt <2 x i32> %A, <i32 100, i32 100>
; CHECK-NEXT:    [[D:%.*]] = and <2 x i1> [[B]], [[C]]
; CHECK-NEXT:    ret <2 x i1> [[D]]
;
  %B = icmp sge <2 x i32> %A, <i32 50, i32 50>
  %C = icmp slt <2 x i32> %A, <i32 100, i32 100>
  %D = and <2 x i1> %B, %C
  ret <2 x i1> %D
}

define i1 @test26(i32 %A) {
; CHECK-LABEL: @test26(
; CHECK-NEXT:    [[A_OFF:%.*]] = add i32 %A, -49
; CHECK-NEXT:    [[A_CMP:%.*]] = icmp ugt i32 [[A_OFF]], 1
; CHECK-NEXT:    ret i1 [[A_CMP]]
;
  %B = icmp ne i32 %A, 49
  %C = icmp ne i32 %A, 50
  ;; (A-49) > 1
  %D = and i1 %B, %C
  ret i1 %D
}

define i8 @test27(i8 %A) {
; CHECK-LABEL: @test27(
; CHECK-NEXT:    ret i8 0
;
  %B = and i8 %A, 4
  %C = sub i8 %B, 16
  ;; 0xF0
  %D = and i8 %C, -16
  %E = add i8 %D, 16
  ret i8 %E
}

;; This is juse a zero extending shr.
define i32 @test28(i32 %X) {
; CHECK-LABEL: @test28(
; CHECK-NEXT:    [[Y1:%.*]] = lshr i32 %X, 24
; CHECK-NEXT:    ret i32 [[Y1]]
;
  ;; Sign extend
  %Y = ashr i32 %X, 24
  ;; Mask out sign bits
  %Z = and i32 %Y, 255
  ret i32 %Z
}

define i32 @test29(i8 %X) {
; CHECK-LABEL: @test29(
; CHECK-NEXT:    [[Y:%.*]] = zext i8 %X to i32
; CHECK-NEXT:    ret i32 [[Y]]
;
  %Y = zext i8 %X to i32
  ;; Zero extend makes this unneeded.
  %Z = and i32 %Y, 255
  ret i32 %Z
}

define i32 @test30(i1 %X) {
; CHECK-LABEL: @test30(
; CHECK-NEXT:    [[Y:%.*]] = zext i1 %X to i32
; CHECK-NEXT:    ret i32 [[Y]]
;
  %Y = zext i1 %X to i32
  %Z = and i32 %Y, 1
  ret i32 %Z
}

define i32 @test31(i1 %X) {
; CHECK-LABEL: @test31(
; CHECK-NEXT:    [[Y:%.*]] = zext i1 %X to i32
; CHECK-NEXT:    [[Z:%.*]] = shl nuw nsw i32 [[Y]], 4
; CHECK-NEXT:    ret i32 [[Z]]
;
  %Y = zext i1 %X to i32
  %Z = shl i32 %Y, 4
  %A = and i32 %Z, 16
  ret i32 %A
}

define i32 @test32(i32 %In) {
; CHECK-LABEL: @test32(
; CHECK-NEXT:    ret i32 0
;
  %Y = and i32 %In, 16
  %Z = lshr i32 %Y, 2
  %A = and i32 %Z, 1
  ret i32 %A
}

;; Code corresponding to one-bit bitfield ^1.
define i32 @test33(i32 %b) {
; CHECK-LABEL: @test33(
; CHECK-NEXT:    [[TMP_13:%.*]] = xor i32 %b, 1
; CHECK-NEXT:    ret i32 [[TMP_13]]
;
  %tmp.4.mask = and i32 %b, 1
  %tmp.10 = xor i32 %tmp.4.mask, 1
  %tmp.12 = and i32 %b, -2
  %tmp.13 = or i32 %tmp.12, %tmp.10
  ret i32 %tmp.13
}

define i32 @test34(i32 %A, i32 %B) {
; CHECK-LABEL: @test34(
; CHECK-NEXT:    ret i32 %B
;
  %tmp.2 = or i32 %B, %A
  %tmp.4 = and i32 %tmp.2, %B
  ret i32 %tmp.4
}

