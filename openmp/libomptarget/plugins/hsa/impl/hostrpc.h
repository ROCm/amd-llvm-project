/*  hostrpc_varfn_uint_.c

MIT License

Copyright © 2020 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished
to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

// Need to get llvm typeID enum from Type.h but cant include LLVM headers in
// runtime #include "../../../../../llvm/include/llvm/IR/Type.h" For now, we a
// have a manual copy of llvm TypeID enum
enum TypeID {
  // PrimitiveTypes - make sure LastPrimitiveTyID stays up to date.
  VoidTyID = 0,  ///<  0: type with no size
  HalfTyID,      ///<  1: 16-bit floating point type
  FloatTyID,     ///<  2: 32-bit floating point type
  DoubleTyID,    ///<  3: 64-bit floating point type
  X86_FP80TyID,  ///<  4: 80-bit floating point type (X87)
  FP128TyID,     ///<  5: 128-bit floating point type (112-bit mantissa)
  PPC_FP128TyID, ///<  6: 128-bit floating point type (two 64-bits, PowerPC)
  LabelTyID,     ///<  7: Labels
  MetadataTyID,  ///<  8: Metadata
  X86_MMXTyID,   ///<  9: MMX vectors (64 bits, X86 specific)
  TokenTyID,     ///< 10: Tokens

  // Derived types... see DerivedTypes.h file.
  // Make sure FirstDerivedTyID stays up to date!
  IntegerTyID,       ///< 11: Arbitrary bit width integers
  FunctionTyID,      ///< 12: Functions
  StructTyID,        ///< 13: Structures
  ArrayTyID,         ///< 14: Arrays
  PointerTyID,       ///< 15: Pointers
  FixedVectorTyID,   ///< 16: Fixed width SIMD vector type
  ScalableVectorTyID ///< 17: Scalable SIMD vector type
};

typedef enum hostrpc_status_t {
  HOSTRPC_SUCCESS = 0,
  HOSTRPC_STATUS_UNKNOWN = 1,
  HOSTRPC_STATUS_ERROR = 2,
  HOSTRPC_STATUS_TERMINATE = 3,
  HOSTRPC_DATA_USED_ERROR = 4,
  HOSTRPC_ADDINT_ERROR = 5,
  HOSTRPC_ADDFLOAT_ERROR = 6,
  HOSTRPC_ADDSTRING_ERROR = 7,
  HOSTRPC_UNSUPPORTED_ID_ERROR = 8,
  HOSTRPC_INVALID_ID_ERROR = 9,
  HOSTRPC_ERROR_INVALID_REQUEST = 10,
  HOSTRPC_EXCEED_MAX_ARGS_ERROR = 11,
} hostrpc_status_t;

#define NUMFPREGS 8
#define FPREGSZ 16

typedef int uint128_t __attribute__((mode(TI)));
struct hostrpc_pfIntRegs {
  uint64_t rdi, rsi, rdx, rcx, r8, r9;
};
typedef struct hostrpc_pfIntRegs hostrpc_pfIntRegs_t; // size = 48 bytes

struct hostrpc_pfRegSaveArea {
  hostrpc_pfIntRegs_t iregs;
  uint128_t freg[NUMFPREGS];
};
typedef struct hostrpc_pfRegSaveArea
    hostrpc_pfRegSaveArea_t; // size = 304 bytes

struct hostrpc_ValistExt {
  uint32_t gp_offset;      /* offset to next available gpr in reg_save_area */
  uint32_t fp_offset;      /* offset to next available fpr in reg_save_area */
  void *overflow_arg_area; /* args that are passed on the stack */
  hostrpc_pfRegSaveArea_t *reg_save_area; /* int and fp registers */
  size_t overflow_size;
} __attribute__((packed));
typedef struct hostrpc_ValistExt hostrpc_ValistExt_t;

/// Prototype for host fallback function also stored in hostcall_stubs.h
typedef uint32_t hostrpc_varfn_uint_t(void *, ...);
typedef uint64_t hostrpc_varfn_uint64_t(void *, ...);
typedef double hostrpc_varfn_double_t(void *, ...);

hostrpc_status_t hostrpc_printf(char *buf, size_t bufsz, uint *rc);
hostrpc_status_t hostrpc_varfn_uint_(char *buf, size_t bufsz, uint *rc);
hostrpc_status_t hostrpc_varfn_uint64_(char *buf, size_t bufsz, uint64_t *rc);
hostrpc_status_t hostrpc_varfn_double_(char *buf, size_t bufsz, double *rc);
