// RUN: %check_clang_tidy %s google-explicit-constructor,clang-diagnostic-unused-variable %t -- -extra-arg=-Wunused-variable --

class A { A(int i); };
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: single-argument constructors must be marked explicit

class B { B(int i); }; // NOLINT

class C { C(int i); }; // NOLINT(we-dont-care-about-categories-yet)

void f() {
  int i;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: unused variable 'i' [clang-diagnostic-unused-variable]
  int j; // NOLINT
}

#define MACRO(X) class X { X(int i); };
MACRO(D)
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: single-argument constructors must be marked explicit
MACRO(E) // NOLINT

#define MACRO_NOARG class F { F(int i); };
MACRO_NOARG // NOLINT

#define MACRO_NOLINT class G { G(int i); }; // NOLINT
MACRO_NOLINT

#define DOUBLE_MACRO MACRO(H) // NOLINT
DOUBLE_MACRO

// CHECK-MESSAGES: Suppressed 7 warnings (7 NOLINT)
