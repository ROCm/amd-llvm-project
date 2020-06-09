/*

hostrpc_varfn.c : Service functions for hostrpc functions
                  hostrpc_printf
                  hostrpc_varfn_uint

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

#include "hostrpc.h"
#include <ctype.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//------------------------------------------------------------------------
//
// hostrpc_printf:

// These are the static helper functions to support hostrpc_printf

// Handle overflow when building va_list for vprintf
static hostrpc_status_t hostrpc_pfGetOverflow(hostrpc_ValistExt_t *valist,
                                              size_t needsize) {
  if (needsize < valist->overflow_size)
    return HOSTRPC_SUCCESS;

  // Make the overflow area bigger
  size_t stacksize;
  void *newstack;
  if (valist->overflow_size == 0) {
    // Make initial save area big to reduce mallocs
    stacksize = (FPREGSZ * NUMFPREGS) * 2;
    if (needsize > stacksize)
      stacksize = needsize; // maybe a big string
  } else {
    // Initial save area not big enough, double it
    stacksize = valist->overflow_size * 2;
  }
  if (!(newstack = malloc(stacksize))) {
    return HOSTRPC_STATUS_ERROR;
  }
  memset(newstack, 0, stacksize);
  if (valist->overflow_size) {
    memcpy(newstack, valist->overflow_arg_area, valist->overflow_size);
    free(valist->overflow_arg_area);
  }
  valist->overflow_arg_area = newstack;
  valist->overflow_size = stacksize;
  return HOSTRPC_SUCCESS;
}

// Add an integer to the va_list for vprintf
static hostrpc_status_t hostrpc_pfAddInteger(hostrpc_ValistExt_t *valist,
                                             char *val, size_t valsize,
                                             size_t *stacksize) {
  uint64_t ival;
  switch (valsize) {
  case 1:
    ival = *(uint8_t *)val;
    break;
  case 2:
    ival = *(uint32_t *)val;
    break;
  case 4:
    ival = (*(uint32_t *)val);
    break;
  case 8:
    ival = *(uint64_t *)val;
    break;
  default: {
    return HOSTRPC_STATUS_ERROR;
  }
  }
  //  Always copy 8 bytes, sizeof(ival)
  if ((valist->gp_offset + sizeof(ival)) <= sizeof(hostrpc_pfIntRegs_t)) {
    memcpy(((char *)valist->reg_save_area + valist->gp_offset), &ival,
           sizeof(ival));
    valist->gp_offset += sizeof(ival);
    return HOSTRPC_SUCCESS;
  }
  // Ensure valist overflow area is big enough
  size_t needsize = (size_t)*stacksize + sizeof(ival);
  if (hostrpc_pfGetOverflow(valist, needsize) != HOSTRPC_SUCCESS)
    return HOSTRPC_STATUS_ERROR;
  // Copy to overflow
  memcpy((char *)(valist->overflow_arg_area) + (size_t)*stacksize, &ival,
         sizeof(ival));

  *stacksize += sizeof(ival);
  return HOSTRPC_SUCCESS;
}

// Add a String argument when building va_list for vprintf
static hostrpc_status_t hostrpc_pfAddString(hostrpc_ValistExt_t *valist,
                                            char *val, size_t strsz,
                                            size_t *stacksize) {
  size_t valsize =
      sizeof(char *); // ABI captures pointer to string,  not string
  if ((valist->gp_offset + valsize) <= sizeof(hostrpc_pfIntRegs_t)) {
    memcpy(((char *)valist->reg_save_area + valist->gp_offset), val, valsize);
    valist->gp_offset += valsize;
    return HOSTRPC_SUCCESS;
  }
  size_t needsize = (size_t)*stacksize + valsize;
  if (hostrpc_pfGetOverflow(valist, needsize) != HOSTRPC_SUCCESS)
    return HOSTRPC_STATUS_ERROR;
  memcpy((char *)(valist->overflow_arg_area) + (size_t)*stacksize, val,
         valsize);
  *stacksize += valsize;
  return HOSTRPC_SUCCESS;
}

// Add a floating point value when building va_list for vprintf
static hostrpc_status_t hostrpc_pfAddFloat(hostrpc_ValistExt_t *valist,
                                           char *numdata, size_t valsize,
                                           size_t *stacksize) {
  // FIXME, we can used load because doubles are now aligned
  double dval;
  if (valsize == 4) {
    float fval;
    memcpy(&fval, numdata, 4);
    dval = (double)fval; // Extend single to double per abi
  } else if (valsize == 8) {
    memcpy(&dval, numdata, 8);
  } else {
    return HOSTRPC_STATUS_ERROR;
  }
  if ((valist->fp_offset + FPREGSZ) <= sizeof(hostrpc_pfRegSaveArea_t)) {
    memcpy(((char *)valist->reg_save_area + (size_t)(valist->fp_offset)), &dval,
           sizeof(double));
    valist->fp_offset += FPREGSZ;
    return HOSTRPC_SUCCESS;
  }
  size_t needsize = (size_t)*stacksize + sizeof(double);
  if (hostrpc_pfGetOverflow(valist, needsize) != HOSTRPC_SUCCESS)
    return HOSTRPC_STATUS_ERROR;
  memcpy((char *)(valist->overflow_arg_area) + (size_t)*stacksize, &dval,
         sizeof(double));
  // move only by the size of the double (8 bytes)
  *stacksize += sizeof(double);
  return HOSTRPC_SUCCESS;
}

// Build an extended va_list for vprintf by unpacking the buffer
static hostrpc_status_t hostrpc_pfBuildValist(hostrpc_ValistExt_t *valist,
                                              int NumArgs, char *keyptr,
                                              char *dataptr, char *strptr,
                                              size_t *data_not_used) {
  hostrpc_pfRegSaveArea_t *regs;
  size_t regs_size = sizeof(*regs);
  regs = (hostrpc_pfRegSaveArea_t *)malloc(regs_size);
  if (!regs)
    return HOSTRPC_STATUS_ERROR;
  memset(regs, 0, regs_size);
  *valist = (hostrpc_ValistExt_t){
      .gp_offset = 0,
      .fp_offset = 0,
      .overflow_arg_area = NULL,
      .reg_save_area = regs,
      .overflow_size = 0,
  };

  size_t num_bytes;
  size_t bytes_consumed;
  size_t strsz;
  size_t fillerNeeded;

  size_t stacksize = 0;

  for (int argnum = 0; argnum < NumArgs; argnum++) {
    num_bytes = 0;
    strsz = 0;
    unsigned int key = *(unsigned int *)keyptr;
    unsigned int llvmID = key >> 16;
    unsigned int numbits = (key << 16) >> 16;

    switch (llvmID) {
    case FloatTyID:  ///<  2: 32-bit floating point type
    case DoubleTyID: ///<  3: 64-bit floating point type
    case FP128TyID:  ///<  5: 128-bit floating point type (112-bit mantissa)
      num_bytes = numbits / 8;
      bytes_consumed = num_bytes;
      fillerNeeded = ((size_t)dataptr) % num_bytes;
      if (fillerNeeded) {
        dataptr += fillerNeeded;
        bytes_consumed += fillerNeeded;
      }
      if ((*data_not_used) < bytes_consumed)
        return HOSTRPC_DATA_USED_ERROR;
      if (valist->fp_offset == 0)
        valist->fp_offset = sizeof(hostrpc_pfIntRegs_t);
      if (hostrpc_pfAddFloat(valist, dataptr, num_bytes, &stacksize))
        return HOSTRPC_ADDFLOAT_ERROR;
      break;

    case IntegerTyID: ///< 11: Arbitrary bit width integers
      num_bytes = numbits / 8;
      bytes_consumed = num_bytes;
      fillerNeeded = ((size_t)dataptr) % num_bytes;
      if (fillerNeeded) {
        dataptr += fillerNeeded;
        bytes_consumed += fillerNeeded;
      }
      if ((*data_not_used) < bytes_consumed)
        return HOSTRPC_DATA_USED_ERROR;
      if (hostrpc_pfAddInteger(valist, dataptr, num_bytes, &stacksize))
        return HOSTRPC_ADDINT_ERROR;
      break;

    case PointerTyID:     ///< 15: Pointers
      if (numbits == 1) { // This is a pointer to string
        num_bytes = 4;
        bytes_consumed = num_bytes;
        strsz = (size_t) * (unsigned int *)dataptr;
        if ((*data_not_used) < bytes_consumed)
          return HOSTRPC_DATA_USED_ERROR;
        if (hostrpc_pfAddString(valist, (char *)&strptr, strsz, &stacksize))
          return HOSTRPC_ADDSTRING_ERROR;
      } else {
        num_bytes = 8;
        bytes_consumed = num_bytes;
        fillerNeeded = ((size_t)dataptr) % num_bytes;
        if (fillerNeeded) {
          dataptr += fillerNeeded; // dataptr is now aligned
          bytes_consumed += fillerNeeded;
        }
        if ((*data_not_used) < bytes_consumed)
          return HOSTRPC_DATA_USED_ERROR;
        if (hostrpc_pfAddInteger(valist, dataptr, num_bytes, &stacksize))
          return HOSTRPC_ADDINT_ERROR;
      }
      break;

    case HalfTyID:           ///<  1: 16-bit floating point type
    case ArrayTyID:          ///< 14: Arrays
    case StructTyID:         ///< 13: Structures
    case FunctionTyID:       ///< 12: Functions
    case TokenTyID:          ///< 10: Tokens
    case X86_MMXTyID:        ///<  9: MMX vectors (64 bits, X86 specific)
    case MetadataTyID:       ///<  8: Metadata
    case LabelTyID:          ///<  7: Labels
    case PPC_FP128TyID:      ///<  6: 128-bit floating point type (two 64-bits,
                             ///<  PowerPC)
    case X86_FP80TyID:       ///<  4: 80-bit floating point type (X87)
    case FixedVectorTyID:    ///< 16: Fixed width SIMD vector type
    case ScalableVectorTyID: ///< 17: Scalable SIMD vector type
    case VoidTyID:
      return HOSTRPC_UNSUPPORTED_ID_ERROR;
      break;
    default:
      return HOSTRPC_INVALID_ID_ERROR;
    }

    dataptr += num_bytes;
    strptr += strsz;
    *data_not_used -= bytes_consumed;
    keyptr += 4;
  }
  return HOSTRPC_SUCCESS;
} // end hostrpc_pfBuildValist

// -----
//  hostrpc_printf:  This the main service routine for printf
// -----
hostrpc_status_t hostrpc_printf(char *buf, size_t bufsz, uint *rc) {
  if (bufsz == 0)
    return HOSTRPC_SUCCESS;

  // Get 6 values needed to unpack the buffer
  int *datalen = (int *)buf;
  int NumArgs = *((int *)(buf + sizeof(int)));
  size_t data_not_used =
      (size_t)(*datalen) - ((size_t)(2 + NumArgs) * sizeof(int));
  char *keyptr = buf + (2 * sizeof(int));
  char *dataptr = keyptr + (NumArgs * sizeof(int));
  char *strptr = buf + (size_t)*datalen;

  if (NumArgs <= 0)
    return HOSTRPC_ERROR_INVALID_REQUEST;

  // Skip past the format string argument
  char *fmtstr = strptr;
  NumArgs--;
  keyptr += 4;
  size_t strsz = (size_t) * (unsigned int *)dataptr;
  dataptr += 4; // for strings the data value is the size, not a real pointer
  strptr += strsz;
  data_not_used -= 4;

  hostrpc_ValistExt_t valist;
  va_list *real_va_list;
  real_va_list = (va_list *)&valist;

  if (hostrpc_pfBuildValist(&valist, NumArgs, keyptr, dataptr, strptr,
                            &data_not_used) != HOSTRPC_SUCCESS)
    return HOSTRPC_ERROR_INVALID_REQUEST;

  // Roll back offsets and save stack pointer for hostrpc_varfn_uint to consume
  valist.gp_offset = 0;
  valist.fp_offset = sizeof(hostrpc_pfIntRegs_t);
  void *save_stack = valist.overflow_arg_area;

  *rc = vprintf(fmtstr, *real_va_list);

  if (valist.reg_save_area)
    free(valist.reg_save_area);
  if (save_stack)
    free(save_stack);

  return HOSTRPC_SUCCESS;
}

//---------------- Support for hostrpc_varfn_* service ---------------------
//
#define MAXVARGS 32

// These are the helper functions for hostrpc_varfn_uint_
uint64_t getuint32(char *val) {
  uint32_t i32 = *(uint32_t *)val;
  return (uint64_t)i32;
}
uint64_t getuint64(char *val) { return *(uint64_t *)val; }

void *getfnptr(char *val) {
  uint64_t ival = *(uint64_t *)val;
  return (void *)ival;
}

// build argument array
static hostrpc_status_t hostrpc_build_vargs_array(int NumArgs, char *keyptr,
                                                  char *dataptr, char *strptr,
                                                  size_t *data_not_used,
                                                  uint64_t *a[MAXVARGS]) {
  size_t num_bytes;
  size_t bytes_consumed;
  size_t strsz;
  size_t fillerNeeded;

  uint argcount = 0;

  for (int argnum = 0; argnum < NumArgs; argnum++) {
    num_bytes = 0;
    strsz = 0;
    unsigned int key = *(unsigned int *)keyptr;
    unsigned int llvmID = key >> 16;
    unsigned int numbits = (key << 16) >> 16;

    switch (llvmID) {
    case FloatTyID:  ///<  2: 32-bit floating point type
    case DoubleTyID: ///<  3: 64-bit floating point type
    case FP128TyID:  ///<  5: 128-bit floating point type (112-bit mantissa)
      num_bytes = numbits / 8;
      bytes_consumed = num_bytes;
      fillerNeeded = ((size_t)dataptr) % num_bytes;
      if (fillerNeeded) {
        dataptr += fillerNeeded;
        bytes_consumed += fillerNeeded;
      }
      if ((*data_not_used) < bytes_consumed)
        return HOSTRPC_DATA_USED_ERROR;

      if (num_bytes == 4)
        a[argcount] = (uint64_t *)getuint32(dataptr);
      else
        a[argcount] = (uint64_t *)getuint64(dataptr);

      break;

    case IntegerTyID: ///< 11: Arbitrary bit width integers
      num_bytes = numbits / 8;
      bytes_consumed = num_bytes;
      fillerNeeded = ((size_t)dataptr) % num_bytes;
      if (fillerNeeded) {
        dataptr += fillerNeeded;
        bytes_consumed += fillerNeeded;
      }
      if ((*data_not_used) < bytes_consumed)
        return HOSTRPC_DATA_USED_ERROR;

      if (num_bytes == 4)
        a[argcount] = (uint64_t *)getuint32(dataptr);
      else
        a[argcount] = (uint64_t *)getuint64(dataptr);

      break;

    case PointerTyID:     ///< 15: Pointers
      if (numbits == 1) { // This is a pointer to string
        num_bytes = 4;
        bytes_consumed = num_bytes;
        strsz = (size_t) * (unsigned int *)dataptr;
        if ((*data_not_used) < bytes_consumed)
          return HOSTRPC_DATA_USED_ERROR;
        a[argcount] = (uint64_t *)((char *)strptr);

      } else {
        num_bytes = 8;
        bytes_consumed = num_bytes;
        fillerNeeded = ((size_t)dataptr) % num_bytes;
        if (fillerNeeded) {
          dataptr += fillerNeeded; // dataptr is now aligned
          bytes_consumed += fillerNeeded;
        }
        if ((*data_not_used) < bytes_consumed)
          return HOSTRPC_DATA_USED_ERROR;

        a[argcount] = (uint64_t *)getuint64(dataptr);
      }
      break;

    case HalfTyID:           ///<  1: 16-bit floating point type
    case ArrayTyID:          ///< 14: Arrays
    case StructTyID:         ///< 13: Structures
    case FunctionTyID:       ///< 12: Functions
    case TokenTyID:          ///< 10: Tokens
    case X86_MMXTyID:        ///<  9: MMX vectors (64 bits, X86 specific)
    case MetadataTyID:       ///<  8: Metadata
    case LabelTyID:          ///<  7: Labels
    case PPC_FP128TyID:      ///<  6: 128-bit floating point type (two 64-bits,
                             ///<  PowerPC)
    case X86_FP80TyID:       ///<  4: 80-bit floating point type (X87)
    case FixedVectorTyID:    ///< 16: Fixed width SIMD vector type
    case ScalableVectorTyID: ///< 17: Scalable SIMD vector type
    case VoidTyID:
      return HOSTRPC_UNSUPPORTED_ID_ERROR;
      break;
    default:
      return HOSTRPC_INVALID_ID_ERROR;
    }

    // Move to next argument
    dataptr += num_bytes;
    strptr += strsz;
    *data_not_used -= bytes_consumed;
    keyptr += 4;
    argcount++;
  }
  return HOSTRPC_SUCCESS;
}

// Make the vargs function call to the function pointer fnptr
// by casting fnptr to vfnptr. Return uint32_t
static hostrpc_status_t hostrpc_call_fnptr_uint(uint32_t NumArgs, void *fnptr,
                                                uint64_t *a[MAXVARGS],
                                                uint32_t *rc) {
  //
  // Users are instructed that their first arg must be a dummy
  // so that device interface is same as host interface. To match device
  // interface we make the first arg be the function pointer.
  //
  hostrpc_varfn_uint_t *vfnptr = (hostrpc_varfn_uint_t *)fnptr;

  switch (NumArgs) {
  case 1:
    *rc = vfnptr(fnptr, a[0]);
    break;
  case 2:
    *rc = vfnptr(fnptr, a[0], a[1]);
    break;
  case 3:
    *rc = vfnptr(fnptr, a[0], a[1], a[2]);
    break;
  case 4:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3]);
    break;
  case 5:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4]);
    break;
  case 6:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5]);
    break;
  case 7:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6]);
    break;
  case 8:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);
    break;
  case 9:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]);
    break;
  case 10:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9]);
    break;
  case 11:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10]);
    break;
  case 12:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11]);
    break;
  case 13:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12]);
    break;
  case 14:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13]);
    break;
  case 15:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14]);
    break;
  case 16:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15]);
    break;
  case 17:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16]);
    break;
  case 18:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17]);
    break;
  case 19:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18]);
    break;
  case 20:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19]);
    break;
  case 21:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20]);
    break;
  case 22:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21]);
    break;
  case 23:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22]);
    break;
  case 24:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22], a[23]);
    break;
  case 25:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22], a[23], a[24]);
    break;
  case 26:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25]);
    break;
  case 27:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26]);
    break;
  case 28:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26],
                 a[27]);
    break;
  case 29:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26],
                 a[27], a[28]);
    break;
  case 30:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26],
                 a[27], a[28], a[29]);
    break;
  case 31:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26],
                 a[27], a[28], a[29], a[30]);
    break;
  case 32:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26],
                 a[27], a[28], a[29], a[30], a[31]);
    break;
  default:
    return HOSTRPC_EXCEED_MAX_ARGS_ERROR;
  }
  return HOSTRPC_SUCCESS;
}

// Make the vargs function call to the function pointer fnptr
// by casting fnptr to vfnptr. Return uint64
static hostrpc_status_t hostrpc_call_fnptr_uint64(uint32_t NumArgs, void *fnptr,
                                                  uint64_t *a[MAXVARGS],
                                                  uint64_t *rc) {
  //
  // Users are instructed that their first arg must be a dummy
  // so that device interface is same as host interface. To match device
  // interface we make the first arg be the function pointer.
  //
  hostrpc_varfn_uint64_t *vfnptr = (hostrpc_varfn_uint64_t *)fnptr;

  switch (NumArgs) {
  case 1:
    *rc = vfnptr(fnptr, a[0]);
    break;
  case 2:
    *rc = vfnptr(fnptr, a[0], a[1]);
    break;
  case 3:
    *rc = vfnptr(fnptr, a[0], a[1], a[2]);
    break;
  case 4:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3]);
    break;
  case 5:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4]);
    break;
  case 6:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5]);
    break;
  case 7:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6]);
    break;
  case 8:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);
    break;
  case 9:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]);
    break;
  case 10:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9]);
    break;
  case 11:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10]);
    break;
  case 12:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11]);
    break;
  case 13:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12]);
    break;
  case 14:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13]);
    break;
  case 15:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14]);
    break;
  case 16:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15]);
    break;
  case 17:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16]);
    break;
  case 18:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17]);
    break;
  case 19:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18]);
    break;
  case 20:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19]);
    break;
  case 21:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20]);
    break;
  case 22:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21]);
    break;
  case 23:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22]);
    break;
  case 24:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22], a[23]);
    break;
  case 25:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22], a[23], a[24]);
    break;
  case 26:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25]);
    break;
  case 27:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26]);
    break;
  case 28:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26],
                 a[27]);
    break;
  case 29:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26],
                 a[27], a[28]);
    break;
  case 30:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26],
                 a[27], a[28], a[29]);
    break;
  case 31:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26],
                 a[27], a[28], a[29], a[30]);
    break;
  case 32:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26],
                 a[27], a[28], a[29], a[30], a[31]);
    break;
  default:
    return HOSTRPC_EXCEED_MAX_ARGS_ERROR;
  }
  return HOSTRPC_SUCCESS;
}

// Make the vargs function call to the function pointer fnptr
// by casting fnptr to vfnptr. Return double
static hostrpc_status_t hostrpc_call_fnptr_double(uint32_t NumArgs, void *fnptr,
                                                  uint64_t *a[MAXVARGS],
                                                  double *rc) {
  //
  // Users are instructed that their first arg must be a dummy
  // so that device interface is same as host interface. To match device
  // interface we make the first arg be the function pointer.
  //
  hostrpc_varfn_double_t *vfnptr = (hostrpc_varfn_double_t *)fnptr;

  switch (NumArgs) {
  case 1:
    *rc = vfnptr(fnptr, a[0]);
    break;
  case 2:
    *rc = vfnptr(fnptr, a[0], a[1]);
    break;
  case 3:
    *rc = vfnptr(fnptr, a[0], a[1], a[2]);
    break;
  case 4:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3]);
    break;
  case 5:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4]);
    break;
  case 6:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5]);
    break;
  case 7:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6]);
    break;
  case 8:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);
    break;
  case 9:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]);
    break;
  case 10:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9]);
    break;
  case 11:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10]);
    break;
  case 12:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11]);
    break;
  case 13:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12]);
    break;
  case 14:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13]);
    break;
  case 15:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14]);
    break;
  case 16:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15]);
    break;
  case 17:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16]);
    break;
  case 18:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17]);
    break;
  case 19:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18]);
    break;
  case 20:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19]);
    break;
  case 21:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20]);
    break;
  case 22:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21]);
    break;
  case 23:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22]);
    break;
  case 24:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22], a[23]);
    break;
  case 25:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22], a[23], a[24]);
    break;
  case 26:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25]);
    break;
  case 27:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26]);
    break;
  case 28:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26],
                 a[27]);
    break;
  case 29:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26],
                 a[27], a[28]);
    break;
  case 30:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26],
                 a[27], a[28], a[29]);
    break;
  case 31:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26],
                 a[27], a[28], a[29], a[30]);
    break;
  case 32:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26],
                 a[27], a[28], a[29], a[30], a[31]);
    break;
  default:
    return HOSTRPC_EXCEED_MAX_ARGS_ERROR;
  }
  return HOSTRPC_SUCCESS;
}
// -----
//  hostrpc_varfn_uint_:  This the main service routine for hostrpc_varfn_uint
//                        Static helper functions are defined above
// -----
hostrpc_status_t hostrpc_varfn_uint_(char *buf, size_t bufsz, uint *rc) {
  if (bufsz == 0)
    return HOSTRPC_SUCCESS;

  // Get 6 values needed to unpack the buffer
  int *datalen = (int *)buf;
  int NumArgs = *((int *)(buf + sizeof(int)));
  size_t data_not_used =
      (size_t)(*datalen) - ((size_t)(2 + NumArgs) * sizeof(int));
  char *keyptr = buf + (2 * sizeof(int));
  char *dataptr = keyptr + (NumArgs * sizeof(int));
  char *strptr = buf + (size_t)*datalen;

  // skip the function pointer arg including any align buffer
  if (((size_t)dataptr) % (size_t)8) {
    dataptr += 4;
    data_not_used -= 4;
  }
  void *fnptr = getfnptr(dataptr);
  NumArgs--;
  keyptr += 4;
  dataptr += 8;
  data_not_used -= 4;

  if (NumArgs <= 0)
    return HOSTRPC_ERROR_INVALID_REQUEST;

  uint64_t *a[MAXVARGS];
  if (hostrpc_build_vargs_array(NumArgs, keyptr, dataptr, strptr,
                                &data_not_used, a) != HOSTRPC_SUCCESS)
    return HOSTRPC_ERROR_INVALID_REQUEST;

  if (hostrpc_call_fnptr_uint(NumArgs, fnptr, a, rc) != HOSTRPC_SUCCESS)
    return HOSTRPC_ERROR_INVALID_REQUEST;

  return HOSTRPC_SUCCESS;
}

hostrpc_status_t hostrpc_varfn_uint64_(char *buf, size_t bufsz, uint64_t *rc) {
  if (bufsz == 0)
    return HOSTRPC_SUCCESS;

  // Get 6 tracking values needed to unpack the buffer
  int *datalen = (int *)buf;
  int NumArgs = *((int *)(buf + sizeof(int)));
  size_t data_not_used =
      (size_t)(*datalen) - ((size_t)(2 + NumArgs) * sizeof(int));
  char *keyptr = buf + (2 * sizeof(int));
  char *dataptr = keyptr + (NumArgs * sizeof(int));
  char *strptr = buf + (size_t)*datalen;

  if (NumArgs <= 0)
    return HOSTRPC_ERROR_INVALID_REQUEST;

  // skip the function pointer arg including any align buffer
  if (((size_t)dataptr) % (size_t)8) {
    dataptr += 4;
    data_not_used -= 4;
  }
  void *fnptr = getfnptr(dataptr);
  NumArgs--;
  keyptr += 4;
  dataptr += 8;
  data_not_used -= 4;

  uint64_t *a[MAXVARGS];
  if (hostrpc_build_vargs_array(NumArgs, keyptr, dataptr, strptr,
                                &data_not_used, a) != HOSTRPC_SUCCESS)
    return HOSTRPC_ERROR_INVALID_REQUEST;

  if (hostrpc_call_fnptr_uint64(NumArgs, fnptr, a, rc) != HOSTRPC_SUCCESS)
    return HOSTRPC_ERROR_INVALID_REQUEST;

  return HOSTRPC_SUCCESS;
}

hostrpc_status_t hostrpc_varfn_double_(char *buf, size_t bufsz, double *rc) {
  if (bufsz == 0)
    return HOSTRPC_SUCCESS;

  // Get 6 tracking values needed to unpack the buffer
  int *datalen = (int *)buf;
  int NumArgs = *((int *)(buf + sizeof(int)));
  size_t data_not_used =
      (size_t)(*datalen) - ((size_t)(2 + NumArgs) * sizeof(int));
  char *keyptr = buf + (2 * sizeof(int));
  char *dataptr = keyptr + (NumArgs * sizeof(int));
  char *strptr = buf + (size_t)*datalen;

  if (NumArgs <= 0)
    return HOSTRPC_ERROR_INVALID_REQUEST;

  // skip the function pointer arg including any align buffer
  if (((size_t)dataptr) % (size_t)8) {
    dataptr += 4;
    data_not_used -= 4;
  }
  void *fnptr = getfnptr(dataptr);
  NumArgs--;
  keyptr += 4;
  dataptr += 8;
  data_not_used -= 4;

  uint64_t *a[MAXVARGS];
  if (hostrpc_build_vargs_array(NumArgs, keyptr, dataptr, strptr,
                                &data_not_used, a) != HOSTRPC_SUCCESS)
    return HOSTRPC_ERROR_INVALID_REQUEST;

  if (hostrpc_call_fnptr_double(NumArgs, fnptr, a, rc) != HOSTRPC_SUCCESS)
    return HOSTRPC_ERROR_INVALID_REQUEST;

  return HOSTRPC_SUCCESS;
}
