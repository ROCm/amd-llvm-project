// RUN: %clang_cc1 %s -cl-std=CL2.0 -include opencl-c.h -triple amdgcn -fno-common -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s -O0 -cl-std=CL2.0 -include opencl-c.h -triple amdgcn -fno-common -emit-llvm -o - | FileCheck --check-prefix=NOOPT %s

typedef struct {
  private char *p1;
  local char *p2;
  constant char *p3;
  global char *p4;
  generic char *p5;
} StructTy1;

typedef struct {
  constant char *p3;
  global char *p4;
  generic char *p5;
} StructTy2;

// LLVM requests global variable with common linkage to be initialized with zeroinitializer, therefore use -fno-common
// to suppress common linkage for tentative definition.

// Test 0 as initializer.

// CHECK: @private_p = local_unnamed_addr addrspace(1) global i8* addrspacecast (i8 addrspace(4)* null to i8*), align 4
private char *private_p = 0;

// CHECK: @local_p = local_unnamed_addr addrspace(1) global i8 addrspace(3)* addrspacecast (i8 addrspace(4)* null to i8 addrspace(3)*), align 4
local char *local_p = 0;

// CHECK: @global_p = local_unnamed_addr addrspace(1) global i8 addrspace(1)* null, align 4
global char *global_p = 0;

// CHECK: @constant_p = local_unnamed_addr addrspace(1) global i8 addrspace(2)* null, align 4
constant char *constant_p = 0;

// CHECK: @generic_p = local_unnamed_addr addrspace(1) global i8 addrspace(4)* null, align 4
generic char *generic_p = 0;

// Test NULL as initializer.

// CHECK: @private_p_NULL = local_unnamed_addr addrspace(1) global i8* addrspacecast (i8 addrspace(4)* null to i8*), align 4
private char *private_p_NULL = NULL;

// CHECK: @local_p_NULL = local_unnamed_addr addrspace(1) global i8 addrspace(3)* addrspacecast (i8 addrspace(4)* null to i8 addrspace(3)*), align 4
local char *local_p_NULL = NULL;

// CHECK: @global_p_NULL = local_unnamed_addr addrspace(1) global i8 addrspace(1)* null, align 4
global char *global_p_NULL = NULL;

// CHECK: @constant_p_NULL = local_unnamed_addr addrspace(1) global i8 addrspace(2)* null, align 4
constant char *constant_p_NULL = NULL;

// CHECK: @generic_p_NULL = local_unnamed_addr addrspace(1) global i8 addrspace(4)* null, align 4
generic char *generic_p_NULL = NULL;

// Test constant folding of null pointer.
// A null pointer should be folded to a null pointer in the target address space.

// CHECK: @fold_generic = local_unnamed_addr addrspace(1) global i32 addrspace(4)* null, align 4
generic int *fold_generic = (global int*)(generic float*)(private char*)0;

// CHECK: @fold_priv = local_unnamed_addr addrspace(1) global i16* addrspacecast (i16 addrspace(4)* null to i16*), align 4
private short *fold_priv = (private short*)(generic int*)(global void*)0;

// CHECK: @fold_priv_arith = local_unnamed_addr addrspace(1) global i8* inttoptr (i32 9 to i8*), align 4
private char *fold_priv_arith = (private char*)0 + 10;

// CHECK: @fold_int = local_unnamed_addr addrspace(1) global i32 13, align 4
int fold_int = (int)(private void*)(generic char*)(global int*)0 + 14;

// CHECK: @fold_int2 = local_unnamed_addr addrspace(1) global i32 12, align 4
int fold_int2 = (int) ((private void*)0 + 13);

// CHECK: @fold_int3 = local_unnamed_addr addrspace(1) global i32 -1, align 4
int fold_int3 = (int) ((private int*)0);

// CHECK: @fold_int4 = local_unnamed_addr addrspace(1) global i32 7, align 4
int fold_int4 = (int) &((private int*)0)[2];

// CHECK: @fold_int5 = local_unnamed_addr addrspace(1) global i32 3, align 4
int fold_int5 = (int) &((private StructTy1*)0)->p2;

// Test static variable initialization.

// NOOPT: @test_static_var.sp1 = internal addrspace(1) global i8* addrspacecast (i8 addrspace(4)* null to i8*), align 4
// NOOPT: @test_static_var.sp2 = internal addrspace(1) global i8* addrspacecast (i8 addrspace(4)* null to i8*), align 4
// NOOPT: @test_static_var.sp3 = internal addrspace(1) global i8* addrspacecast (i8 addrspace(4)* null to i8*), align 4
// NOOPT: @test_static_var.sp4 = internal addrspace(1) global i8* null, align 4
// NOOPT: @test_static_var.sp5 = internal addrspace(1) global i8* null, align 4
// NOOPT: @test_static_var.SS1 = internal addrspace(1) global %struct.StructTy1 { i8* addrspacecast (i8 addrspace(4)* null to i8*), i8 addrspace(3)* addrspacecast (i8 addrspace(4)* null to i8 addrspace(3)*), i8 addrspace(2)* null, i8 addrspace(1)* null, i8 addrspace(4)* null }, align 4
// NOOPT: @test_static_var.SS2 = internal addrspace(1) global %struct.StructTy2 zeroinitializer, align 4

void test_static_var(void) {
  static private char *sp1 = 0;
  static private char *sp2 = NULL;
  static private char *sp3;
  static private char *sp4 = (private char*)((void)0, 0);
  const int x = 0;
  static private char *sp5 = (private char*)x;
  static StructTy1 SS1;
  static StructTy2 SS2;
}

// Test function-scope variable initialization.
// NOOPT-LABEL: test_func_scope_var
// NOOPT: store i8* addrspacecast (i8 addrspace(4)* null to i8*), i8** %sp1, align 4
// NOOPT: store i8* addrspacecast (i8 addrspace(4)* null to i8*), i8** %sp2, align 4
// NOOPT: store i8* null, i8** %sp3, align 4
// NOOPT: store i8* null, i8** %sp4, align 4
// NOOPT: %[[SS1:.*]] = bitcast %struct.StructTy1* %SS1 to i8*
// NOOPT: call void @llvm.memcpy.p0i8.p2i8.i64(i8* %[[SS1]], i8 addrspace(2)* bitcast (%struct.StructTy1 addrspace(2)* @test_func_scope_var.SS1 to i8 addrspace(2)*), i64 32, i32 4, i1 false)
// NOOPT: %[[SS2:.*]] = bitcast %struct.StructTy2* %SS2 to i8*
// NOOPT: call void @llvm.memset.p0i8.i64(i8* %[[SS2]], i8 0, i64 24, i32 4, i1 false)

void test_func_scope_var(void) {
  private char *sp1 = 0;
  private char *sp2 = NULL;
  private char *sp3 = (private char*)((void)0, 0);
  const int x = 0;
  private char *sp4 = (private char*)x;
  StructTy1 SS1 = {0, 0, 0, 0, 0};
  StructTy2 SS2 = {0, 0, 0};
}

// Test default initialization of pointers.

// CHECK: @p1 = local_unnamed_addr addrspace(1) global i8* addrspacecast (i8 addrspace(4)* null to i8*), align 4
private char *p1;

// CHECK: @p2 = local_unnamed_addr addrspace(1) global i8 addrspace(3)* addrspacecast (i8 addrspace(4)* null to i8 addrspace(3)*), align 4
local char *p2;

// CHECK: @p3 = local_unnamed_addr addrspace(1) global i8 addrspace(2)* null, align 4
constant char *p3;

// CHECK: @p4 = local_unnamed_addr addrspace(1) global i8 addrspace(1)* null, align 4
global char *p4;

// CHECK: @p5 = local_unnamed_addr addrspace(1) global i8 addrspace(4)* null, align 4
generic char *p5;

// Test default initialization of sturcture.

// CHECK: @S1 = local_unnamed_addr addrspace(1) global %struct.StructTy1 { i8* addrspacecast (i8 addrspace(4)* null to i8*), i8 addrspace(3)* addrspacecast (i8 addrspace(4)* null to i8 addrspace(3)*), i8 addrspace(2)* null, i8 addrspace(1)* null, i8 addrspace(4)* null }, align 4
StructTy1 S1;

// CHECK: @S2 = local_unnamed_addr addrspace(1) global %struct.StructTy2 zeroinitializer, align 4
StructTy2 S2;

// Test default initialization of array.
// CHECK: @A1 = local_unnamed_addr addrspace(1) global [2 x %struct.StructTy1] [%struct.StructTy1 { i8* addrspacecast (i8 addrspace(4)* null to i8*), i8 addrspace(3)* addrspacecast (i8 addrspace(4)* null to i8 addrspace(3)*), i8 addrspace(2)* null, i8 addrspace(1)* null, i8 addrspace(4)* null }, %struct.StructTy1 { i8* addrspacecast (i8 addrspace(4)* null to i8*), i8 addrspace(3)* addrspacecast (i8 addrspace(4)* null to i8 addrspace(3)*), i8 addrspace(2)* null, i8 addrspace(1)* null, i8 addrspace(4)* null }], align 4
StructTy1 A1[2];

// CHECK: @A2 = local_unnamed_addr addrspace(1) global [2 x %struct.StructTy2] zeroinitializer, align 4
StructTy2 A2[2];

// Test comparison with 0.

// CHECK-LABEL: cmp_private
// CHECK: icmp eq i8* %p, addrspacecast (i8 addrspace(4)* null to i8*)
void cmp_private(private char* p) {
  if (p != 0)
    *p = 0;
}

// CHECK-LABEL: cmp_local
// CHECK: icmp eq i8 addrspace(3)* %p, addrspacecast (i8 addrspace(4)* null to i8 addrspace(3)*)
void cmp_local(local char* p) {
  if (p != 0)
    *p = 0;
}

// CHECK-LABEL: cmp_global
// CHECK: icmp eq i8 addrspace(1)* %p, null
void cmp_global(global char* p) {
  if (p != 0)
    *p = 0;
}

// CHECK-LABEL: cmp_constant
// CHECK: icmp eq i8 addrspace(2)* %p, null
char cmp_constant(constant char* p) {
  if (p != 0)
    return *p;
  else
    return 0;
}

// CHECK-LABEL: cmp_generic
// CHECK: icmp eq i8 addrspace(4)* %p, null
void cmp_generic(generic char* p) {
  if (p != 0)
    *p = 0;
}

// Test comparison with NULL.

// CHECK-LABEL: cmp_NULL_private
// CHECK: icmp eq i8* %p, addrspacecast (i8 addrspace(4)* null to i8*)
void cmp_NULL_private(private char* p) {
  if (p != NULL)
    *p = 0;
}

// CHECK-LABEL: cmp_NULL_local
// CHECK: icmp eq i8 addrspace(3)* %p, addrspacecast (i8 addrspace(4)* null to i8 addrspace(3)*)
void cmp_NULL_local(local char* p) {
  if (p != NULL)
    *p = 0;
}

// CHECK-LABEL: cmp_NULL_global
// CHECK: icmp eq i8 addrspace(1)* %p, null
void cmp_NULL_global(global char* p) {
  if (p != NULL)
    *p = 0;
}

// CHECK-LABEL: cmp_NULL_constant
// CHECK: icmp eq i8 addrspace(2)* %p, null
char cmp_NULL_constant(constant char* p) {
  if (p != NULL)
    return *p;
  else
    return 0;
}

// CHECK-LABEL: cmp_NULL_generic
// CHECK: icmp eq i8 addrspace(4)* %p, null
void cmp_NULL_generic(generic char* p) {
  if (p != NULL)
    *p = 0;
}

// Test storage 0 as null pointer.
// CHECK-LABEL: test_storage_null_pointer
// CHECK: store i8* addrspacecast (i8 addrspace(4)* null to i8*), i8* addrspace(4)* %arg_private
// CHECK: store i8 addrspace(3)* addrspacecast (i8 addrspace(4)* null to i8 addrspace(3)*), i8 addrspace(3)* addrspace(4)* %arg_local
// CHECK: store i8 addrspace(1)* null, i8 addrspace(1)* addrspace(4)* %arg_global
// CHECK: store i8 addrspace(2)* null, i8 addrspace(2)* addrspace(4)* %arg_constant
// CHECK: store i8 addrspace(4)* null, i8 addrspace(4)* addrspace(4)* %arg_generic
void test_storage_null_pointer(private char** arg_private,
                               local char** arg_local,
                               global char** arg_global,
                               constant char** arg_constant,
                               generic char** arg_generic) {
   *arg_private = 0;
   *arg_local = 0;
   *arg_global = 0;
   *arg_constant = 0;
   *arg_generic = 0;
}

// Test storage NULL as null pointer.
// CHECK-LABEL: test_storage_null_pointer_NULL
// CHECK: store i8* addrspacecast (i8 addrspace(4)* null to i8*), i8* addrspace(4)* %arg_private
// CHECK: store i8 addrspace(3)* addrspacecast (i8 addrspace(4)* null to i8 addrspace(3)*), i8 addrspace(3)* addrspace(4)* %arg_local
// CHECK: store i8 addrspace(1)* null, i8 addrspace(1)* addrspace(4)* %arg_global
// CHECK: store i8 addrspace(2)* null, i8 addrspace(2)* addrspace(4)* %arg_constant
// CHECK: store i8 addrspace(4)* null, i8 addrspace(4)* addrspace(4)* %arg_generic
void test_storage_null_pointer_NULL(private char** arg_private,
                                    local char** arg_local,
                                    global char** arg_global,
                                    constant char** arg_constant,
                                    generic char** arg_generic) {
   *arg_private = NULL;
   *arg_local = NULL;
   *arg_global = NULL;
   *arg_constant = NULL;
   *arg_generic = NULL;
}

// Test pass null pointer to function as argument.
void test_pass_null_pointer_arg_calee(private char* arg_private,
                                      local char* arg_local,
                                      global char* arg_global,
                                      constant char* arg_constant,
                                      generic char* arg_generic);

// CHECK-LABEL: test_pass_null_pointer_arg
// CHECK: call void @test_pass_null_pointer_arg_calee(i8* addrspacecast (i8 addrspace(4)* null to i8*), i8 addrspace(3)* addrspacecast (i8 addrspace(4)* null to i8 addrspace(3)*), i8 addrspace(1)* null, i8 addrspace(2)* null, i8 addrspace(4)* null)
// CHECK: call void @test_pass_null_pointer_arg_calee(i8* addrspacecast (i8 addrspace(4)* null to i8*), i8 addrspace(3)* addrspacecast (i8 addrspace(4)* null to i8 addrspace(3)*), i8 addrspace(1)* null, i8 addrspace(2)* null, i8 addrspace(4)* null)
void test_pass_null_pointer_arg(void) {
  test_pass_null_pointer_arg_calee(0, 0, 0, 0, 0);
  test_pass_null_pointer_arg_calee(NULL, NULL, NULL, NULL, NULL);
}

// Test cast null pointer to size_t.
void test_cast_null_pointer_to_sizet_calee(size_t arg_private,
                                           size_t arg_local,
                                           size_t arg_global,
                                           size_t arg_constant,
                                           size_t arg_generic);

// CHECK-LABEL: test_cast_null_pointer_to_sizet
// CHECK: call void @test_cast_null_pointer_to_sizet_calee(i64 -1, i64 -1, i64 0, i64 0, i64 0)
// CHeCK: call void @test_cast_null_pointer_to_sizet_calee(i64 -1, i64 -1, i64 0, i64 0, i64 0)
void test_cast_null_pointer_to_sizet(void) {
  test_cast_null_pointer_to_sizet_calee((size_t)((private char*)0),
                                        (size_t)((local char*)0),
                                        (size_t)((global char*)0),
                                        (size_t)((constant char*)0),
                                        (size_t)((generic char*)0));
  test_cast_null_pointer_to_sizet_calee((size_t)((private char*)NULL),
                                        (size_t)((local char*)NULL),
                                        (size_t)((global char*)NULL),
                                        (size_t)((constant char*)0), // NULL cannot be casted to constant pointer since it is defined as a generic pointer
                                        (size_t)((generic char*)NULL));
}

// Test comparision between null pointers.
#define TEST_EQ00(addr1, addr2) int test_eq00_##addr1##_##addr2(void) { return (addr1 char*)0 == (addr2 char*)0; }
#define TEST_EQ0N(addr1, addr2) int test_eq0N_##addr1##_##addr2(void) { return (addr1 char*)0 == (addr2 char*)NULL; }
#define TEST_EQN0(addr1, addr2) int test_eqN0_##addr1##_##addr2(void) { return (addr1 char*)NULL == (addr2 char*)0; }
#define TEST_EQNN(addr1, addr2) int test_eqNN_##addr1##_##addr2(void) { return (addr1 char*)0 == (addr2 char*)NULL; }
#define TEST_NE00(addr1, addr2) int test_ne00_##addr1##_##addr2(void) { return (addr1 char*)0 != (addr2 char*)0; }
#define TEST_NE0N(addr1, addr2) int test_ne0N_##addr1##_##addr2(void) { return (addr1 char*)0 != (addr2 char*)NULL; }
#define TEST_NEN0(addr1, addr2) int test_neN0_##addr1##_##addr2(void) { return (addr1 char*)NULL != (addr2 char*)0; }
#define TEST_NENN(addr1, addr2) int test_neNN_##addr1##_##addr2(void) { return (addr1 char*)0 != (addr2 char*)NULL; }
#define TEST(addr1, addr2) \
        TEST_EQ00(addr1, addr2) \
        TEST_EQ0N(addr1, addr2) \
        TEST_EQN0(addr1, addr2) \
        TEST_EQNN(addr1, addr2) \
        TEST_NE00(addr1, addr2) \
        TEST_NE0N(addr1, addr2) \
        TEST_NEN0(addr1, addr2) \
        TEST_NENN(addr1, addr2)

// CHECK-LABEL: test_eq00_generic_private
// CHECK: ret i32 1
// CHECK-LABEL: test_eq0N_generic_private
// CHECK: ret i32 1
// CHECK-LABEL: test_eqN0_generic_private
// CHECK: ret i32 1
// CHECK-LABEL: test_eqNN_generic_private
// CHECK: ret i32 1
// CHECK-LABEL: test_ne00_generic_private
// CHECK: ret i32 0
// CHECK-LABEL: test_ne0N_generic_private
// CHECK: ret i32 0
// CHECK-LABEL: test_neN0_generic_private
// CHECK: ret i32 0
// CHECK-LABEL: test_neNN_generic_private
// CHECK: ret i32 0
TEST(generic, private)

// CHECK-LABEL: test_eq00_generic_local
// CHECK: ret i32 1
// CHECK-LABEL: test_eq0N_generic_local
// CHECK: ret i32 1
// CHECK-LABEL: test_eqN0_generic_local
// CHECK: ret i32 1
// CHECK-LABEL: test_eqNN_generic_local
// CHECK: ret i32 1
// CHECK-LABEL: test_ne00_generic_local
// CHECK: ret i32 0
// CHECK-LABEL: test_ne0N_generic_local
// CHECK: ret i32 0
// CHECK-LABEL: test_neN0_generic_local
// CHECK: ret i32 0
// CHECK-LABEL: test_neNN_generic_local
// CHECK: ret i32 0
TEST(generic, local)

// CHECK-LABEL: test_eq00_generic_global
// CHECK: ret i32 1
// CHECK-LABEL: test_eq0N_generic_global
// CHECK: ret i32 1
// CHECK-LABEL: test_eqN0_generic_global
// CHECK: ret i32 1
// CHECK-LABEL: test_eqNN_generic_global
// CHECK: ret i32 1
// CHECK-LABEL: test_ne00_generic_global
// CHECK: ret i32 0
// CHECK-LABEL: test_ne0N_generic_global
// CHECK: ret i32 0
// CHECK-LABEL: test_neN0_generic_global
// CHECK: ret i32 0
// CHECK-LABEL: test_neNN_generic_global
// CHECK: ret i32 0
TEST(generic, global)

// CHECK-LABEL: test_eq00_generic_generic
// CHECK: ret i32 1
// CHECK-LABEL: test_eq0N_generic_generic
// CHECK: ret i32 1
// CHECK-LABEL: test_eqN0_generic_generic
// CHECK: ret i32 1
// CHECK-LABEL: test_eqNN_generic_generic
// CHECK: ret i32 1
// CHECK-LABEL: test_ne00_generic_generic
// CHECK: ret i32 0
// CHECK-LABEL: test_ne0N_generic_generic
// CHECK: ret i32 0
// CHECK-LABEL: test_neN0_generic_generic
// CHECK: ret i32 0
// CHECK-LABEL: test_neNN_generic_generic
// CHECK: ret i32 0
TEST(generic, generic)

// CHECK-LABEL: test_eq00_constant_constant
// CHECK: ret i32 1
TEST_EQ00(constant, constant)

// Test cast to bool.

// CHECK-LABEL: cast_bool_private
// CHECK: icmp eq i8* %p, addrspacecast (i8 addrspace(4)* null to i8*)
void cast_bool_private(private char* p) {
  if (p)
    *p = 0;
}

// CHECK-LABEL: cast_bool_local
// CHECK: icmp eq i8 addrspace(3)* %p, addrspacecast (i8 addrspace(4)* null to i8 addrspace(3)*)
void cast_bool_local(local char* p) {
  if (p)
    *p = 0;
}

// CHECK-LABEL: cast_bool_global
// CHECK: icmp eq i8 addrspace(1)* %p, null
void cast_bool_global(global char* p) {
  if (p)
    *p = 0;
}

// CHECK-LABEL: cast_bool_constant
// CHECK: icmp eq i8 addrspace(2)* %p, null
char cast_bool_constant(constant char* p) {
  if (p)
    return *p;
  else
    return 0;
}

// CHECK-LABEL: cast_bool_generic
// CHECK: icmp eq i8 addrspace(4)* %p, null
void cast_bool_generic(generic char* p) {
  if (p)
    *p = 0;
}

// Test initialize a struct using memset.
// For large structures which is mostly zero, clang generats llvm.memset for
// the zero part and store for non-zero members.
typedef struct {
  long a, b, c, d;
  private char *p;
} StructTy3;

// CHECK-LABEL: test_memset
// CHECK: call void @llvm.memset.p0i8.i64(i8* {{.*}}, i8 0, i64 32, i32 8, i1 false)
// CHECK: store i8* addrspacecast (i8 addrspace(4)* null to i8*), i8** {{.*}}
StructTy3 test_memset(void) {
  StructTy3 S3 = {0, 0, 0, 0, 0};
  return S3;
}

// Test casting literal 0 to pointer.
// A 0 literal casted to pointer should become a null pointer.

// CHECK-LABEL: test_cast_0_to_ptr
// CHECK: ret i32* addrspacecast (i32 addrspace(4)* null to i32*)
private int* test_cast_0_to_ptr(void) {
  return (private int*)0;
}

// Test casting non-literal integer with 0 value to pointer.
// A non-literal integer expression with 0 value is casted to a pointer with
// zero value. 

// CHECK-LABEL: test_cast_int_to_ptr1
// CHECK: ret i32* null
private int* test_cast_int_to_ptr1(void) {
  return (private int*)((void)0, 0);
}

// CHECK-LABEL: test_cast_int_to_ptr2
// CHECK: ret i32* null
private int* test_cast_int_to_ptr2(void) {
  int x = 0;
  return (private int*)x;
}

// Test logical operations.
// CHECK-LABEL: test_not_nullptr
// CHECK: ret i32 1
int test_not_nullptr(void) {
  return !(private char*)NULL;
}

// CHECK-LABEL: test_and_nullptr
// CHECK: ret i32 0
int test_and_nullptr(int a) {
  return a && ((private char*)NULL);
}

// CHECK-LABEL: test_not_ptr
// CHECK: %[[lnot:.*]] = icmp eq i8* %p, addrspacecast (i8 addrspace(4)* null to i8*)
// CHECK: %[[lnot_ext:.*]] = zext i1 %[[lnot]] to i32
// CHECK: ret i32 %[[lnot_ext]]
int test_not_ptr(private char* p) {
  return !p;
}
// CHECK-LABEL: test_and_ptr
// CHECK: %[[tobool:.*]] = icmp ne i8* %p1, addrspacecast (i8 addrspace(4)* null to i8*)
// CHECK: %[[tobool1:.*]] = icmp ne i8 addrspace(3)* %p2, addrspacecast (i8 addrspace(4)* null to i8 addrspace(3)*)
// CHECK: %[[res:.*]] = and i1 %[[tobool]], %[[tobool1]]
// CHECK: %[[land_ext:.*]] = zext i1 %[[res]] to i32
// CHECK: ret i32 %[[land_ext]]
int test_and_ptr(private char* p1, local char* p2) {
  return p1 && p2;
}

// Test folding of null pointer in function scope.
// NOOPT-LABEL: test_fold
// NOOPT: store i32 addrspace(1)* null, i32 addrspace(1)** %glob, align 4
// NOOPT: %{{.*}} = sub i64 %{{.*}}, 0
// NOOPT: %{{.*}} = add nsw i64 %{{.*}}, -1
// NOOPT: %{{.*}} = sub nsw i64 %{{.*}}, 1
void test_fold(void) {
  global int* glob = (global int*)(generic char*)0;
  long x = glob - (global int*)(generic char*)0;
  x = x + (int)(private int*)(generic char*)(global short*)0;
  x = x - (int)((private int*)0 == (private int*)(generic char*)0);
}
