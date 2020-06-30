#ifndef PLATFORM_HPP_INCLUDED
#define PLATFORM_HPP_INCLUDED

#include <stdint.h>

#include "../base_types.hpp"  // page_t

namespace platform
{
void sleep_briefly(void);
}

#if defined(__x86_64__)
#include <chrono>
#include <unistd.h>

#include <cassert>
#include <cstdio>

namespace platform
{
// local toolchain thinks usleep might throw. That induces a bunch of exception
// control flow where there otherwise wouldn't be any. Will fix by calling into
// std::chrono, bodge for now
static __attribute__((noinline)) void sleep_noexcept(unsigned int t) noexcept
{
  usleep(t);
}

inline void sleep_briefly(void)
{
  // <thread> conflicts with <stdatomic.h>
  // std::this_thread::sleep_for(std::chrono::milliseconds(10));
  sleep_noexcept(10);
}
inline void sleep(void) { sleep_noexcept(1000); }

inline bool is_master_lane(void) { return true; }
inline uint32_t get_lane_id(void) { return 0; }
inline uint32_t broadcast_master(uint32_t x) { return x; }
inline uint64_t broadcast_master(uint64_t x) { return x; }
inline void init_inactive_lanes(hostrpc::page_t *, uint64_t) {}
inline uint32_t client_start_slot() { return 0; }
}  // namespace platform
#endif

#if defined(__AMDGCN__) || defined(__CUDACC__)
// Enough of assert.h, derived from musl
#ifdef NDEBUG
#define assert(x) (void)0
#else
#define assert_str(x) assert_str_1(x)
#define assert_str_1(x) #x
#define assert(x)                                                           \
  ((void)((x) || (__assert_fail("L:" assert_str(__LINE__) " " #x, __FILE__, \
                                __LINE__, __func__),                        \
                  0)))
#endif

#undef static_assert
#define static_assert _Static_assert

__attribute__((always_inline)) inline void __assert_fail(const char *str,
                                                         const char *,
                                                         unsigned int line,
                                                         const char *)
{
  asm("// Assert fail " ::"r"(line), "r"(str));
  __builtin_trap();
}

// stub printf for now
// aomp clang currently rewrites any variadic function to a pair of
// allocate/execute functions, which don't necessarily exist.
// Clobber it with the preprocessor as a workaround.
#define printf(...) __inline_printf()
__attribute__((always_inline)) inline int __inline_printf()
{
  // printf is implement with hostcall, so going to have to do without
  return 0;
}

#endif

#if defined(__AMDGCN__)

namespace platform
{
inline void sleep_briefly(void) { __builtin_amdgcn_s_sleep(0); }
inline void sleep(void) { __builtin_amdgcn_s_sleep(100); }

__attribute__((always_inline)) inline uint32_t get_lane_id(void)
{
  return __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u));
}
__attribute__((always_inline)) inline bool is_master_lane(void)
{
  // TODO: 32 wide wavefront, consider not using raw intrinsics here
  uint64_t activemask = __builtin_amdgcn_read_exec();

  // TODO: check codegen for trunc lowest_active vs expanding lane_id
  // TODO: ffs is lifted from openmp runtime, looks like it should be ctz
  uint32_t lowest_active = __builtin_ffsl(activemask) - 1;
  uint32_t lane_id = get_lane_id();

  // TODO: readfirstlane(lane_id) == lowest_active?
  return lane_id == lowest_active;
}

__attribute__((always_inline)) inline uint32_t broadcast_master(uint32_t x)
{
  return __builtin_amdgcn_readfirstlane(x);
}

__attribute__((always_inline)) inline uint64_t broadcast_master(uint64_t x)
{
  uint32_t lo = x;
  uint32_t hi = x >> 32u;
  lo = broadcast_master(lo);
  hi = broadcast_master(hi);
  return ((uint64_t)hi << 32u) | lo;
}

inline void init_inactive_lanes(hostrpc::page_t *page, uint64_t v)
{
  // may want to do this control flow within the asm
  uint64_t activemask = __builtin_amdgcn_read_exec();
  if (activemask == UINT64_MAX)
    {
      return;
    }

  // 64 bit addition is by 2x32bit, need to convert the pointer to an integer
  uint64_t addr;
  __builtin_memcpy(&addr, &page, 8);

  // need the address and initializer to be lane independent
  uint32_t loclo = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(addr));
  uint32_t lochi =
      __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(addr >> 32u));

  uint32_t scalar_lo = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(v));
  uint32_t scalar_hi =
      __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(v >> 32u));

  // Hard codes 2:5 for the quad store and 6:7 for the computed address
  // Can be cheaper if only the first dwordx2 is written
  uint32_t laneid_scratch;
  asm("// wait, may not be necessary\n\t"
      "s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)\n\t"
      "// Invert active lanes\n\t"
      "s_not_b64 exec, exec\n\t"
      "// broadcast address to v[6:7]\n\t"
      "v_mov_b32_e32 v6, %[loclo]\n\t"
      "v_mov_b32_e32 v7, %[lochi]\n\t"

      "// Calculate offset into page\n\t"
      "v_mbcnt_lo_u32_b32 %[laneid], -1, 0\n\t"
      "v_mbcnt_hi_u32_b32 %[laneid], -1, %[laneid]\n\t"
      "v_lshlrev_b32_e32 %[laneid], 6, %[laneid]\n\t"
      "v_add_co_u32_e32 v6, vcc, v6, %[laneid]\n\t"
      "v_addc_co_u32_e32 v7, vcc, 0, v7, vcc\n\t"

      "// Write payload to v[2:5]\n\t"
      "v_mov_b32_e32 v2, %[vallo]\n\t"
      "v_mov_b32_e32 v3, %[valhi]\n\t"
      "v_mov_b32_e32 v4, %[vallo]\n\t"
      "v_mov_b32_e32 v5, %[valhi]\n\t"

      "// v[0:1] from reg constraints on loclo/hi\n\t"
      // flat_store_dwordx2 v[6:7], v[2:3] if only e[0]
      "flat_store_dwordx4 v[6:7], v[2:5]\n\t"
      "flat_store_dwordx4 v[6:7], v[2:5] offset:16\n\t"
      "flat_store_dwordx4 v[6:7], v[2:5] offset:32\n\t"
      "flat_store_dwordx4 v[6:7], v[2:5] offset:48\n\t"

      "// Restore active lanes\n\t"
      "s_not_b64 exec, exec\n\t"

      "// wait\n\t"
      "s_waitcnt vmcnt(0) lgkmcnt(0)\n\t"

      : [ laneid ] "=&v"(laneid_scratch)
      : [ loclo ] "s"(loclo), [ lochi ] "s"(lochi), [ vallo ] "s"(scalar_lo),
        [ valhi ] "s"(scalar_hi)
      : "v2", "v3", "v4", "v5", "v6", "v7", "vcc", "memory");
}

inline uint32_t client_start_slot()
{
  // Ideally would return something < size
  // Attempt to distibute clients roughly across the array
  // compute unit currently executing the wave is a version of that
  enum
  {
    HW_ID = 4,  // specify that the hardware register to read is HW_ID

    HW_ID_CU_ID_SIZE = 4,    // size of CU_ID field in bits
    HW_ID_CU_ID_OFFSET = 8,  // offset of CU_ID from start of register

    HW_ID_SE_ID_SIZE = 2,     // sizeof SE_ID field in bits
    HW_ID_SE_ID_OFFSET = 13,  // offset of SE_ID from start of register
  };
#define ENCODE_HWREG(WIDTH, OFF, REG) (REG | (OFF << 6) | ((WIDTH - 1) << 11))
  uint32_t cu_id = __builtin_amdgcn_s_getreg(
      ENCODE_HWREG(HW_ID_CU_ID_SIZE, HW_ID_CU_ID_OFFSET, HW_ID));
  uint32_t se_id = __builtin_amdgcn_s_getreg(
      ENCODE_HWREG(HW_ID_SE_ID_SIZE, HW_ID_SE_ID_OFFSET, HW_ID));
  return (se_id << HW_ID_CU_ID_SIZE) + cu_id;
#undef ENCODE_HWREG
}

}  // namespace platform
#endif

#if defined(__CUDACC__)

namespace platform
{
inline void sleep_briefly(void) {}
inline void sleep(void) {}

namespace detail
{
__attribute__((always_inline)) inline uint32_t ballot()
{
#if CUDA_VERSION >= 9000
  return __activemask();
#else
  return 0;  // __ballot undeclared, definitely can't include cuda.h though
             // return __ballot(1);
#endif
}

__attribute__((always_inline)) inline uint32_t get_master_lane_id(void)
{
  return 0;  // todo
}

}  // namespace detail
__attribute__((always_inline)) inline uint32_t get_lane_id(void)
{
  // uses threadIdx.x & 0x1f from cuda, need to find the corresponding intrinsic
  return 0;
}

__attribute__((always_inline)) inline bool is_master_lane(void)
{
  return get_lane_id() == detail::get_master_lane_id();
}

__attribute__((always_inline)) inline uint32_t broadcast_master(uint32_t x)
{
  // involves __shfl_sync
  uint32_t lane_id = get_lane_id();
  uint32_t master_id = detail::get_master_lane_id();
  uint32_t v;
  if (lane_id == master_id)
    {
      v = x;
    }
  // shfl_sync isn't declared either
  // v = __shfl_sync(UINT32_MAX, v, master_id);
  return v;
}

__attribute__((always_inline)) inline uint64_t broadcast_master(uint64_t x)
{
  // probably don't want to model 64 wide warps on nvptx
  return x;
}
}  // namespace platform
#endif

namespace platform
{
template <typename U, typename F>
U critical(F f)
{
  U res = {};
  if (is_master_lane())
    {
      res = f();
    }
  res = broadcast_master(res);
  return res;
}

}  // namespace platform

#endif
