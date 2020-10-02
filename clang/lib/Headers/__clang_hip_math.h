/*===---- __clang_hip_math.h - Device-side CUDA math support --------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */
#ifndef __CLANG_HIP_MATH_H__
#define __CLANG_HIP_MATH_H__
#ifndef __HIP__
#error "This file is for HIP and OpenMP amdgcn device compilation only."
#endif

#include <limits.h>
#include <stdint.h>

// __DEVICE__ is a helper macro with common set of attributes for the wrappers
// we implement in this file. We need static in order to avoid emitting unused
// functions and __forceinline__ helps inlining these wrappers at -O1.
#pragma push_macro("__DEVICE__")
#ifdef __OPENMP_AMDGCN__
#if defined(__cplusplus)
#define __DEVICE__ static constexpr __attribute__((always_inline, nothrow))
#else
#ifdef __BUILD_MATH_BUILTINS_LIB__
#define __DEVICE__ extern __attribute__((always_inline, nothrow, cold, weak))
#else
#define __DEVICE__ static __attribute__((always_inline, nothrow))
#endif
#endif
#else
//  not openmp, so this is for clang-hip
#define __DEVICE__                                                             \
  static __inline__ __attribute__((always_inline)) __attribute__((device))
#endif

// Specialized version of __DEVICE__ for functions with void return type. Needed
// because the OpenMP overlay requires constexpr functions here but prior to
// c++14 void return functions could not be constexpr.
#pragma push_macro("__DEVICE_VOID__")
#if defined(__OPENMP_AMDGCN__) && defined(__cplusplus) && __cplusplus < 201402L
#define __DEVICE_VOID__ static __attribute__((always_inline, nothrow))
#else
#define __DEVICE_VOID__ __DEVICE__
#endif

// For c++, all functions are defined in __clang_hip_cmath.h, skip if c+_
#ifndef __cplusplus
__DEVICE__ uint64_t __make_mantissa_base8(const char *__tagp) {
  uint64_t r = 0;
  while (__tagp) {
    char tmp = *__tagp;

    if (tmp >= '0' && tmp <= '7')
      r = (r * 8u) + tmp - '0';
    else
      return 0;

    ++__tagp;
  }

  return r;
}

__DEVICE__ uint64_t __make_mantissa_base10(const char *__tagp) {
  uint64_t r = 0;
  while (__tagp) {
    char tmp = *__tagp;

    if (tmp >= '0' && tmp <= '9')
      r = (r * 10u) + tmp - '0';
    else
      return 0;

    ++__tagp;
  }

  return r;
}

__DEVICE__ uint64_t __make_mantissa_base16(const char *__tagp) {
  uint64_t r = 0;
  while (__tagp) {
    char tmp = *__tagp;

    if (tmp >= '0' && tmp <= '9')
      r = (r * 16u) + tmp - '0';
    else if (tmp >= 'a' && tmp <= 'f')
      r = (r * 16u) + tmp - 'a' + 10;
    else if (tmp >= 'A' && tmp <= 'F')
      r = (r * 16u) + tmp - 'A' + 10;
    else
      return 0;

    ++__tagp;
  }

  return r;
}

__DEVICE__ uint64_t __make_mantissa(const char *__tagp) {
  if (!__tagp)
    return 0u;

  if (*__tagp == '0') {
    ++__tagp;

    if (*__tagp == 'x' || *__tagp == 'X')
      return __make_mantissa_base16(__tagp);
    else
      return __make_mantissa_base8(__tagp);
  }

  return __make_mantissa_base10(__tagp);
}

__DEVICE__ float acosf(float x) { return __ocml_acos_f32(x); }
__DEVICE__ float acoshf(float x) { return __ocml_acosh_f32(x); }
__DEVICE__ float asinf(float x) { return __ocml_asin_f32(x); }
__DEVICE__ float asinhf(float x) { return __ocml_asinh_f32(x); }
__DEVICE__ float atan2f(float x, float y) { return __ocml_atan2_f32(x, y); }
__DEVICE__ float atanf(float x) { return __ocml_atan_f32(x); }
__DEVICE__ float atanhf(float x) { return __ocml_atanh_f32(x); }
__DEVICE__ float cbrtf(float x) { return __ocml_cbrt_f32(x); }
__DEVICE__ float ceilf(float x) { return __ocml_ceil_f32(x); }
__DEVICE__ float copysignf(float x, float y) {
  return __ocml_copysign_f32(x, y);
}
__DEVICE__ double acos(double __x) { return __ocml_acos_f64(__x); }
__DEVICE__ double acosh(double __x) { return __ocml_acosh_f64(__x); }
__DEVICE__ double asin(double __x) { return __ocml_asin_f64(__x); }
__DEVICE__ double asinh(double __x) { return __ocml_asinh_f64(__x); }
__DEVICE__ double atan(double __x) { return __ocml_atan_f64(__x); }
__DEVICE__ double atan2(double __x, double __y) {
  return __ocml_atan2_f64(__x, __y);
}
__DEVICE__ double atanh(double __x) { return __ocml_atanh_f64(__x); }
__DEVICE__ double cbrt(double __x) { return __ocml_cbrt_f64(__x); }
__DEVICE__ double ceil(double __x) { return __ocml_ceil_f64(__x); }
__DEVICE__ double copysign(double __x, double __y) {
  return __ocml_copysign_f64(__x, __y);
}
__DEVICE__ float cosf(float x) { return __ocml_cos_f32(x); }
__DEVICE__ float coshf(float x) { return __ocml_cosh_f32(x); }
__DEVICE__ float cospif(float x) { return __ocml_cospi_f32(x); }
__DEVICE__ double cos(double __x) { return __ocml_cos_f64(__x); }
__DEVICE__ double cosh(double __x) { return __ocml_cosh_f64(__x); }
__DEVICE__ double cospi(double __x) { return __ocml_cospi_f64(__x); }
__DEVICE__ float cyl_bessel_i0f(float x) { return __ocml_i0_f32(x); }
__DEVICE__ float cyl_bessel_i1f(float x) { return __ocml_i1_f32(x); }
__DEVICE__ double cyl_bessel_i0(double __x) { return __ocml_i0_f64(__x); }
__DEVICE__ double cyl_bessel_i1(double __x) { return __ocml_i1_f64(__x); }
__DEVICE__ double erf(double __x) { return __ocml_erf_f64(__x); }
__DEVICE__ double erfc(double __x) { return __ocml_erfc_f64(__x); }
__DEVICE__ double erfcinv(double __x) { return __ocml_erfcinv_f64(__x); }
__DEVICE__ double erfcx(double __x) { return __ocml_erfcx_f64(__x); }
__DEVICE__ double erfinv(double __x) { return __ocml_erfinv_f64(__x); }
__DEVICE__ double exp(double __x) { return __ocml_exp_f64(__x); }
__DEVICE__ double exp10(double __x) { return __ocml_exp10_f64(__x); }
__DEVICE__ double exp2(double __x) { return __ocml_exp2_f64(__x); }
__DEVICE__ double expm1(double __x) { return __ocml_expm1_f64(__x); }
__DEVICE__ double fabs(double __x) { return __ocml_fabs_f64(__x); }
__DEVICE__ float fabsf(float x) { return __ocml_fabs_f32(x); }
__DEVICE__ int abs(int __x) {
  int __sgn = __x >> (sizeof(int) * CHAR_BIT - 1);
  return (__x ^ __sgn) - __sgn;
}
__DEVICE__ long labs(long __x) {
  long __sgn = __x >> (sizeof(long) * CHAR_BIT - 1);
  return (__x ^ __sgn) - __sgn;
}
__DEVICE__ long long llabs(long long __x) {
  long long __sgn = __x >> (sizeof(long long) * CHAR_BIT - 1);
  return (__x ^ __sgn) - __sgn;
}

__DEVICE__ double fdim(double __x, double __y) {
  return __ocml_fdim_f64(__x, __y);
}
__DEVICE__ double floor(double __x) { return __ocml_floor_f64(__x); }
__DEVICE__ double fma(double __x, double __y, double __z) {
  return __ocml_fma_f64(__x, __y, __z);
}
__DEVICE__ double fmax(double __x, double __y) {
  return __ocml_fmax_f64(__x, __y);
}
__DEVICE__ double fmin(double __x, double __y) {
  return __ocml_fmin_f64(__x, __y);
}
__DEVICE__ double fmod(double __x, double __y) {
  return __ocml_fmod_f64(__x, __y);
}
__DEVICE__ float erfcf(float x) { return __ocml_erfc_f32(x); }
__DEVICE__ float erfcinvf(float x) { return __ocml_erfcinv_f32(x); }
__DEVICE__ float erfcxf(float x) { return __ocml_erfcx_f32(x); }
__DEVICE__ float erff(float x) { return __ocml_erf_f32(x); }
__DEVICE__ float erfinvf(float x) { return __ocml_erfinv_f32(x); }
__DEVICE__ float exp10f(float x) { return __ocml_exp10_f32(x); }
__DEVICE__ float exp2f(float x) { return __ocml_exp2_f32(x); }
__DEVICE__ float expf(float x) { return __ocml_exp_f32(x); }
__DEVICE__ float expm1f(float x) { return __ocml_expm1_f32(x); }
__DEVICE__ float fdimf(float x, float y) { return __ocml_fdim_f32(x, y); }
__DEVICE__ float fdividef(float x, float y) { return x / y; }
__DEVICE__ float floorf(float x) { return __ocml_floor_f32(x); }
__DEVICE__ float fmaf(float x, float y, float z) {
  return __ocml_fma_f32(x, y, z);
}
__DEVICE__ float fmaxf(float x, float y) { return __ocml_fmax_f32(x, y); }
__DEVICE__ float fminf(float x, float y) { return __ocml_fmin_f32(x, y); }
__DEVICE__ float fmodf(float x, float y) { return __ocml_fmod_f32(x, y); }

__DEVICE__ float hypotf(float x, float y) { return __ocml_hypot_f32(x, y); }
__DEVICE__ double hypot(double __x, double __y) {
  return __ocml_hypot_f64(__x, __y);
}
__DEVICE__ double fdivide(double __a, double __b) { return __a / __b; }
__DEVICE__ int ilogbf(float x) { return __ocml_ilogb_f32(x); }
__DEVICE__ int ilogb(double __x) { return __ocml_ilogb_f64(__x); }
// int isfinite(float x) { return __ocml_isfinite_f32(x); }
__DEVICE__ int __finitef(float x) { return __ocml_isfinite_f32(x); }
__DEVICE__ int __isinff(float x) { return __ocml_isinf_f32(x); }
__DEVICE__ int __isnanf(float x) { return __ocml_isnan_f32(x); }
__DEVICE__ float j0f(float x) { return __ocml_j0_f32(x); }
__DEVICE__ float j1f(float x) { return __ocml_j1_f32(x); }

__DEVICE__ float jnf(int n,
                     float x) { // TODO: we could use Ahmes multiplication and
                                // the Miller & Brown algorithm
  //       for linear recurrences to get O(log n) steps, but it's unclear if
  //       it'd be beneficial in this case.
  if (n == 0)
    return j0f(x);
  if (n == 1)
    return j1f(x);

  float x0 = j0f(x);
  float x1 = j1f(x);
  for (int i = 1; i < n; ++i) {
    float x2 = (2 * i) / x * x1 - x0;
    x0 = x1;
    x1 = x2;
  }
  return x1;
}
__DEVICE__ double j0(double __x) { return __ocml_j0_f64(__x); }
__DEVICE__ double j1(double __x) { return __ocml_j1_f64(__x); }
__DEVICE__ double jn(int __n,
                     double __x) { // TODO: we could use Ahmes multiplication
                                   // and the Miller & Brown algorithm
  //       for linear recurrences to get O(log n) steps, but it's unclear if
  //       it'd be beneficial in this case. Placeholder until OCML adds
  //       support.
  if (__n == 0)
    return j0f(__x);
  if (__n == 1)
    return j1f(__x);

  double __x0 = j0f(__x);
  double __x1 = j1f(__x);
  for (int __i = 1; __i < __n; ++__i) {
    double __x2 = (2 * __i) / __x * __x1 - __x0;
    __x0 = __x1;
    __x1 = __x2;
  }
  return __x1;
}

__DEVICE__ float ldexpf(float x, int e) { return __ocml_ldexp_f32(x, e); }
__DEVICE__ double ldexp(double __x, int __e) {
  return __ocml_ldexp_f64(__x, __e);
}
__DEVICE__ float lgammaf(float x) { return __ocml_lgamma_f32(x); }
__DEVICE__ double lgamma(double __x) { return __ocml_lgamma_f64(__x); }
__DEVICE__ long long int llrintf(float x) { return __ocml_rint_f32(x); }
__DEVICE__ long long int llroundf(float x) { return __ocml_round_f32(x); }
__DEVICE__ float log10f(float x) { return __ocml_log10_f32(x); }
__DEVICE__ float log1pf(float x) { return __ocml_log1p_f32(x); }
__DEVICE__ float log2f(float x) { return __ocml_log2_f32(x); }
__DEVICE__ float logbf(float x) { return __ocml_logb_f32(x); }
__DEVICE__ float logf(float x) { return __ocml_log_f32(x); }
__DEVICE__ double log(double x) { return __ocml_log_f64(x); }
__DEVICE__ double log10(double __x) { return __ocml_log10_f64(__x); }
__DEVICE__ double log1p(double __x) { return __ocml_log1p_f64(__x); }
__DEVICE__ double log2(double __x) { return __ocml_log2_f64(__x); }
__DEVICE__ double logb(double __x) { return __ocml_logb_f64(__x); }
__DEVICE__ long int lrintf(float x) { return __ocml_rint_f32(x); }
__DEVICE__ long int lroundf(float x) { return __ocml_round_f32(x); }
__DEVICE__ long int lround(double __x) { return __ocml_round_f64(__x); }

__DEVICE__ float nanf(const char *__tagp) {
  union {
    float val;
    struct ieee_float {
      unsigned mantissa : 22;
      unsigned quiet : 1;
      unsigned exponent : 8;
      unsigned sign : 1;
    } bits;

    //        static_assert(sizeof(float) == sizeof(ieee_float), "");
  } tmp;

  tmp.bits.sign = 0u;
  tmp.bits.exponent = ~0u;
  tmp.bits.quiet = 1u;
  tmp.bits.mantissa = __make_mantissa(__tagp);

  return tmp.val;
}
__DEVICE__ float nearbyintf(float x) { return __ocml_nearbyint_f32(x); }
__DEVICE__ float nextafterf(float x, float y) {
  return __ocml_nextafter_f32(x, y);
}
__DEVICE__ double nextafter(double __x, double __y) {
  return __ocml_nextafter_f64(__x, __y);
}
__DEVICE__ float norm3df(float x, float y, float z) {
  return __ocml_len3_f32(x, y, z);
}
__DEVICE__ double norm3d(double __x, double __y, double __z) {
  return __ocml_len3_f64(__x, __y, __z);
}
__DEVICE__ float norm4df(float x, float y, float z, float w) {
  return __ocml_len4_f32(x, y, z, w);
}
__DEVICE__ double norm4d(double __x, double __y, double __z, double __w) {
  return __ocml_len4_f64(__x, __y, __z, __w);
}
__DEVICE__ double rnorm3d(double __x, double __y, double __z) {
  return __ocml_rlen3_f64(__x, __y, __z);
}
__DEVICE__ double rnorm4d(double __x, double __y, double __z, double __w) {
  return __ocml_rlen4_f64(__x, __y, __z, __w);
}
__DEVICE__ float normcdff(float x) { return __ocml_ncdf_f32(x); }

__DEVICE__ float normcdfinvf(float x) { return __ocml_ncdfinv_f32(x); }

__DEVICE__ float
normf(int dim,
      const float *a) { // TODO: placeholder until OCML adds support.
  float r = 0;
  while (dim--) {
    r += a[0] * a[0];
    ++a;
  }
  return __ocml_sqrt_f32(r);
}
__DEVICE__ double
norm(int __dim,
     const double *__a) { // TODO: placeholder until OCML adds support.
  double __r = 0;
  while (__dim--) {
    __r += __a[0] * __a[0];
    ++__a;
  }
  return __ocml_sqrt_f64(__r);
}
__DEVICE__ double normcdf(double __x) { return __ocml_ncdf_f64(__x); }
__DEVICE__ double normcdfinv(double __x) { return __ocml_ncdfinv_f64(__x); }
__DEVICE__ double pow(double __x, double __y) {
  return __ocml_pow_f64(__x, __y);
}
__DEVICE__ double rcbrt(double __x) { return __ocml_rcbrt_f64(__x); }
__DEVICE__ double remainder(double __x, double __y) {
  return __ocml_remainder_f64(__x, __y);
}

__DEVICE__ float powf(float x, float y) { return __ocml_pow_f32(x, y); }

__DEVICE__ float rcbrtf(float x) { return __ocml_rcbrt_f32(x); }

__DEVICE__ float remainderf(float x, float y) {
  return __ocml_remainder_f32(x, y);
}

__DEVICE__ float rhypotf(float x, float y) { return __ocml_rhypot_f32(x, y); }
__DEVICE__ double rhypot(double __x, double __y) {
  return __ocml_rhypot_f64(__x, __y);
}

__DEVICE__ float rintf(float x) { return __ocml_rint_f32(x); }

__DEVICE__ float rnorm3df(float x, float y, float z) {
  return __ocml_rlen3_f32(x, y, z);
}

__DEVICE__ float rnorm4df(float x, float y, float z, float w) {
  return __ocml_rlen4_f32(x, y, z, w);
}

__DEVICE__ float
rnormf(int dim, const float *a) { // TODO: placeholder until OCML adds support.
  float r = 0;
  while (dim--) {
    r += a[0] * a[0];
    ++a;
  }
  return __ocml_rsqrt_f32(r);
}
__DEVICE__ double
rnorm(int __dim,
      const double *__a) { // TODO: placeholder until OCML adds support.
  double __r = 0;
  while (__dim--) {
    __r += __a[0] * __a[0];
    ++__a;
  }
  return __ocml_rsqrt_f64(__r);
}

__DEVICE__ float roundf(float x) { return __ocml_round_f32(x); }
__DEVICE__ float rsqrtf(float x) { return __ocml_rsqrt_f32(x); }
__DEVICE__ float scalblnf(float x, long int n) {
  return (n < INT_MAX) ? __ocml_scalbn_f32(x, n) : __ocml_scalb_f32(x, n);
}
__DEVICE__ float scalbnf(float x, int n) { return __ocml_scalbn_f32(x, n); }
__DEVICE__ double round(double __x) { return __ocml_round_f64(__x); }
__DEVICE__ double rsqrt(double __x) { return __ocml_rsqrt_f64(__x); }
__DEVICE__ double scalbln(double __x, long int __n) {
  return (__n < INT_MAX) ? __ocml_scalbn_f64(__x, __n)
                         : __ocml_scalb_f64(__x, __n);
}
__DEVICE__ double scalbn(double __x, int __n) {
  return __ocml_scalbn_f64(__x, __n);
}
__DEVICE__ int __signbitf(float x) { return __ocml_signbit_f32(x); }

//  FIXME: build omp_ versions of these with private pointer to avoid cast and
//  copy ?
__DEVICE_VOID__ void sincos(double __x, double *__sinptr, double *__cosptr) {
  double __tmp;
  *__sinptr = __ocml_sincos_f64(
      __x, (__attribute__((address_space(5))) double *)&__tmp);
  *__cosptr = __tmp;
}
__DEVICE_VOID__ void sincospi(double __x, double *__sinptr, double *__cosptr) {
  double __tmp;
  *__sinptr = __ocml_sincospi_f64(
      __x, (__attribute__((address_space(5))) double *)&__tmp);
  *__cosptr = __tmp;
}
__DEVICE_VOID__ void sincosf(float __x, float *__sinptr, float *__cosptr) {
  float __tmp;
  *__sinptr =
      __ocml_sincos_f32(__x, (__attribute__((address_space(5))) float *)&__tmp);
  *__cosptr = __tmp;
}
__DEVICE_VOID__ void sincospif(float __x, float *__sinptr, float *__cosptr) {
  float __tmp;

  *__sinptr = __ocml_sincospi_f32(
      __x, (__attribute__((address_space(5))) float *)&__tmp);
  *__cosptr = __tmp;
}

__DEVICE__ float sinf(float x) { return __ocml_sin_f32(x); }
__DEVICE__ float sinhf(float x) { return __ocml_sinh_f32(x); }
__DEVICE__ float sinpif(float x) { return __ocml_sinpi_f32(x); }
__DEVICE__ float sqrtf(float x) { return __ocml_sqrt_f32(x); }
__DEVICE__ float tanf(float x) { return __ocml_tan_f32(x); }
__DEVICE__ float tanhf(float x) { return __ocml_tanh_f32(x); }
__DEVICE__ float tgammaf(float x) { return __ocml_tgamma_f32(x); }
__DEVICE__ float truncf(float x) { return __ocml_trunc_f32(x); }
__DEVICE__ float y0f(float x) { return __ocml_y0_f32(x); }
__DEVICE__ float y1f(float x) { return __ocml_y1_f32(x); }

__DEVICE__ float ynf(int n,
                     float x) { // TODO: we could use Ahmes multiplication and
                                // the Miller & Brown algorithm
  //       for linear recurrences to get O(log n) steps, but it's unclear if
  //       it'd be beneficial in this case. Placeholder until OCML adds
  //       support.
  if (n == 0)
    return y0f(x);
  if (n == 1)
    return y1f(x);

  float x0 = y0f(x);
  float x1 = y1f(x);
  for (int i = 1; i < n; ++i) {
    float x2 = (2 * i) / x * x1 - x0;
    x0 = x1;
    x1 = x2;
  }

  return x1;
}

// BEGIN INTRINSICS
__DEVICE__ float __cosf(float x) { return __llvm_amdgcn_cos_f32(x); }
__DEVICE__ float __exp10f(float x) { return __ocml_exp10_f32(x); }
__DEVICE__ float __expf(float x) { return __ocml_exp_f32(x); }
__DEVICE__ float __fadd_rd(float x, float y) {
  return __ocml_add_rtp_f32(x, y);
}
__DEVICE__ float __fadd_rn(float x, float y) {
  return __ocml_add_rte_f32(x, y);
}
__DEVICE__ float __fadd_ru(float x, float y) {
  return __ocml_add_rtn_f32(x, y);
}
__DEVICE__ float __fadd_rz(float x, float y) {
  return __ocml_add_rtz_f32(x, y);
}
__DEVICE__ float __fdiv_rd(float x, float y) { return x / y; }
__DEVICE__ float __fdiv_rn(float x, float y) { return x / y; }
__DEVICE__ float __fdiv_ru(float x, float y) { return x / y; }
__DEVICE__ float __fdiv_rz(float x, float y) { return x / y; }
__DEVICE__ float __fdividef(float x, float y) { return x / y; }
__DEVICE__ float __fmaf_rd(float x, float y, float z) {
  return __ocml_fma_rtp_f32(x, y, z);
}
__DEVICE__ float __fmaf_rn(float x, float y, float z) {
  return __ocml_fma_rte_f32(x, y, z);
}
__DEVICE__ float __fmaf_ru(float x, float y, float z) {
  return __ocml_fma_rtn_f32(x, y, z);
}
__DEVICE__ float __fmaf_rz(float x, float y, float z) {
  return __ocml_fma_rtz_f32(x, y, z);
}
__DEVICE__ float __fmul_rd(float x, float y) {
  return __ocml_mul_rtp_f32(x, y);
}
__DEVICE__ float __fmul_rn(float x, float y) {
  return __ocml_mul_rte_f32(x, y);
}
__DEVICE__ float __fmul_ru(float x, float y) {
  return __ocml_mul_rtn_f32(x, y);
}
__DEVICE__ float __fmul_rz(float x, float y) {
  return __ocml_mul_rtz_f32(x, y);
}
__DEVICE__ float __frcp_rd(float x) { return __llvm_amdgcn_rcp_f32(x); }
__DEVICE__ float __frcp_rn(float x) { return __llvm_amdgcn_rcp_f32(x); }
__DEVICE__ float __frcp_ru(float x) { return __llvm_amdgcn_rcp_f32(x); }
__DEVICE__ float __frcp_rz(float x) { return __llvm_amdgcn_rcp_f32(x); }
__DEVICE__ float __frsqrt_rn(float x) { return __llvm_amdgcn_rsq_f32(x); }
__DEVICE__ float __fsqrt_rd(float x) { return __ocml_sqrt_rtp_f32(x); }
__DEVICE__ float __fsqrt_rn(float x) { return __ocml_sqrt_rte_f32(x); }
__DEVICE__ float __fsqrt_ru(float x) { return __ocml_sqrt_rtn_f32(x); }
__DEVICE__ float __fsqrt_rz(float x) { return __ocml_sqrt_rtz_f32(x); }
__DEVICE__ float __fsub_rd(float x, float y) {
  return __ocml_sub_rtp_f32(x, y);
}
__DEVICE__ float __fsub_rn(float x, float y) {
  return __ocml_sub_rte_f32(x, y);
}
__DEVICE__ float __fsub_ru(float x, float y) {
  return __ocml_sub_rtn_f32(x, y);
}
__DEVICE__ float __fsub_rz(float x, float y) {
  return __ocml_sub_rtz_f32(x, y);
}
__DEVICE__ float __log10f(float x) { return __ocml_log10_f32(x); }
__DEVICE__ float __log2f(float x) { return __ocml_log2_f32(x); }
__DEVICE__ float __logf(float x) { return __ocml_log_f32(x); }
__DEVICE__ float __powf(float x, float y) { return __ocml_pow_f32(x, y); }
__DEVICE__ float __saturatef(float x) {
  return (x < 0) ? 0 : ((x > 1) ? 1 : x);
}
__DEVICE__ double sin(double x) { return __ocml_sinh_f64(x); }
__DEVICE__ double sinh(double x) { return __ocml_sinh_f64(x); }
__DEVICE__ double sinpi(double x) { return __ocml_sinpi_f64(x); }
__DEVICE__ double sqrt(double x) { return __ocml_sqrt_f64(x); }
__DEVICE__ double tan(double x) { return __ocml_tan_f64(x); }
__DEVICE__ double tanh(double x) { return __ocml_tanh_f64(x); }
__DEVICE__ double tgamma(double x) { return __ocml_tgamma_f64(x); }
__DEVICE__ double trunc(double x) { return __ocml_trunc_f64(x); }
__DEVICE__ double y0(double x) { return __ocml_y0_f64(x); }
__DEVICE__ double y1(double x) { return __ocml_y1_f64(x); }

__DEVICE__ double yn(int n,
                     double x) { // TODO: we could use Ahmes multiplication and
                                 // the Miller & Brown algorithm
  //       for linear recurrences to get O(log n) steps, but it's unclear if
  //       it'd be beneficial in this case. Placeholder until OCML adds
  //       support.
  if (n == 0)
    return j0f(x);
  if (n == 1)
    return j1f(x);

  double x0 = j0f(x);
  double x1 = j1f(x);
  for (int i = 1; i < n; ++i) {
    double x2 = (2 * i) / x * x1 - x0;
    x0 = x1;
    x1 = x2;
  }

  return x1;
}

__DEVICE__ double __dadd_rd(double x, double y) {
  return __ocml_add_rtp_f64(x, y);
}
__DEVICE__ double __dadd_rn(double x, double y) {
  return __ocml_add_rte_f64(x, y);
}
__DEVICE__ double __dadd_ru(double x, double y) {
  return __ocml_add_rtn_f64(x, y);
}
__DEVICE__ double __dadd_rz(double x, double y) {
  return __ocml_add_rtz_f64(x, y);
}
__DEVICE__ double __ddiv_rd(double x, double y) { return x / y; }
__DEVICE__ double __ddiv_rn(double x, double y) { return x / y; }
__DEVICE__ double __ddiv_ru(double x, double y) { return x / y; }
__DEVICE__ double __ddiv_rz(double x, double y) { return x / y; }
__DEVICE__ double __dmul_rd(double x, double y) {
  return __ocml_mul_rtp_f64(x, y);
}
__DEVICE__ double __dmul_rn(double x, double y) {
  return __ocml_mul_rte_f64(x, y);
}
__DEVICE__ double __dmul_ru(double x, double y) {
  return __ocml_mul_rtn_f64(x, y);
}
__DEVICE__ double __dmul_rz(double x, double y) {
  return __ocml_mul_rtz_f64(x, y);
}
__DEVICE__ double __drcp_rd(double x) { return __llvm_amdgcn_rcp_f64(x); }
__DEVICE__ double __drcp_rn(double x) { return __llvm_amdgcn_rcp_f64(x); }
__DEVICE__ double __drcp_ru(double x) { return __llvm_amdgcn_rcp_f64(x); }
__DEVICE__ double __drcp_rz(double x) { return __llvm_amdgcn_rcp_f64(x); }
__DEVICE__ double __dsqrt_rd(double x) { return __ocml_sqrt_rtp_f64(x); }
__DEVICE__ double __dsqrt_rn(double x) { return __ocml_sqrt_rte_f64(x); }
__DEVICE__ double __dsqrt_ru(double x) { return __ocml_sqrt_rtn_f64(x); }
__DEVICE__ double __dsqrt_rz(double x) { return __ocml_sqrt_rtz_f64(x); }
__DEVICE__ double __dsub_rd(double x, double y) {
  return __ocml_sub_rtp_f64(x, y);
}
__DEVICE__ double __dsub_rn(double x, double y) {
  return __ocml_sub_rte_f64(x, y);
}
__DEVICE__ double __dsub_ru(double x, double y) {
  return __ocml_sub_rtn_f64(x, y);
}
__DEVICE__ double __dsub_rz(double x, double y) {
  return __ocml_sub_rtz_f64(x, y);
}
__DEVICE__ double __fma_rd(double x, double y, double z) {
  return __ocml_fma_rtp_f64(x, y, z);
}
__DEVICE__ double __fma_rn(double x, double y, double z) {
  return __ocml_fma_rte_f64(x, y, z);
}
__DEVICE__ double __fma_ru(double x, double y, double z) {
  return __ocml_fma_rtn_f64(x, y, z);
}
__DEVICE__ double __fma_rz(double x, double y, double z) {
  return __ocml_fma_rtz_f64(x, y, z);
}
__DEVICE__ int max(int __a, int __b) { return (__a > __b) ? __a : __b; }
__DEVICE__ int min(int __a, int __b) { return (__a < __b) ? __a : __b; }

//  These omp_ functions have private pointer arg and can use the ocml function
//  directly. The non omp_ functions are called without private but cast to
//  private on call to the omp_ function OpenMP target offload users should
//  declare a private pointer and use omp_ version.
__DEVICE__ double omp_frexp(double x, __private int *nptr) {
  return __ocml_frexp_f64(x, nptr);
}
__DEVICE__ double frexp(double __x, int *__nptr) {
  int __tmp;
  double __r = omp_frexp(__x, (__attribute__((address_space(5))) int *)&__tmp);
  *__nptr = __tmp;

  return __r;
}
__DEVICE__ float omp_frexpf(float x, __private int *nptr) {
  return __ocml_frexp_f32(x, nptr);
}
__DEVICE__ float frexpf(float __x, int *__nptr) {
  int __tmp;
  float __r = omp_frexpf(__x, (__attribute__((address_space(5))) int *)&__tmp);
  *__nptr = __tmp;
  return __r;
}
__DEVICE__ double omp_modf(double x, __private double *iptr) {
  return __ocml_modf_f64(x, iptr);
}
__DEVICE__ double modf(double __x, double *__iptr) {
  double __tmp;
  double __r =
      omp_modf(__x, (__attribute__((address_space(5))) double *)&__tmp);
  *__iptr = __tmp;
  return __r;
}
__DEVICE__ float omp_modff(float x, __private float *iptr) {
  return __ocml_modf_f32(x, iptr);
}
__DEVICE__ float modff(float __x, float *__iptr) {
  float __tmp;
  float __r = omp_modff(__x, (__attribute__((address_space(5))) float *)&__tmp);
  *__iptr = __tmp;
  return __r;
}
__DEVICE__ double omp_remquo(double x, double y, __private int *quo) {
  return __ocml_remquo_f64(x, y, quo);
}
__DEVICE__ double remquo(double __x, double __y, int *__quo) {
  int __tmp;
  double __r =
      omp_remquo(__x, __y, (__attribute__((address_space(5))) int *)&__tmp);
  *__quo = __tmp;

  return __r;
}
__DEVICE__ float omp_remquof(float x, float y, __private int *quo) {
  return __ocml_remquo_f32(x, y, quo);
}
__DEVICE__ float remquof(float __x, float __y, int *__quo) {
  int __tmp;
  float __r =
      omp_remquof(__x, __y, (__attribute__((address_space(5))) int *)&__tmp);
  *__quo = __tmp;
  return __r;
}

#endif // end if ! c++

#include <__clang_cuda_complex_builtins.h>

#pragma pop_macro("__DEVICE__")
#pragma pop_macro("__DEVICE_VOID__")
#pragma pop_macro("__FAST_OR_SLOW")

#endif // __CLANG_HIP_MATH_H__
