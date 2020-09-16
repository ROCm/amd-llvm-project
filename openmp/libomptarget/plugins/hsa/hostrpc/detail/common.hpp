#ifndef HOSTRPC_COMMON_H_INCLUDED
#define HOSTRPC_COMMON_H_INCLUDED

#include <stdint.h>

#include "../base_types.hpp"
#include "platform.hpp"

namespace hostrpc
{
// The bitmap in question is usually a runtime parameter, but as it's
// invariant during program execution I think it's worth tolerating this
// anyway. Going to lead to a switch somewhere.

// Need a bitmap which sets an element by CAS
// Also need to support find-some-bit-set

// Uses:
// client needs one to track which slots in the buffer are currently owned
// client uses another to publish which slots are available to read
// server reads from one a word at a time, but probably wants another as a cache
// server uses one to track which slots in the buffer are currently owned
// server uses another to publish which slots are available to read
// client probably wants a cache as well
// six instances
// different scoping rules needed for the various updates, e.g. the bitmap
// which is only used from one device should have device scope on the cas
// not sure if we can get away with relaxed atomic read/writes

namespace
{
inline uint64_t index_to_element(uint64_t x) { return x / 64u; }

inline uint64_t index_to_subindex(uint64_t x) { return x % 64u; }

namespace detail
{
inline bool multiple_of_64(uint64_t x) { return (x % 64) == 0; }

inline uint64_t round_up_to_multiple_of_64(uint64_t x)
{
  return 64u * ((x + 63u) / 64u);
}

inline bool nthbitset64(uint64_t x, uint64_t n)
{
  // assert(n < 64);
  return x & (1ull << n);
}

inline uint64_t setnthbit64(uint64_t x, uint64_t n)
{
  assert(n < 64);
  return x | (1ull << n);
}

inline uint64_t clearnthbit64(uint64_t x, uint64_t n)
{
  assert(n < 64);
  return x & ~(1ull << n);
}

inline uint64_t setbitsrange64(uint64_t l, uint64_t h)
{
  uint64_t base = UINT64_MAX;
  uint64_t width = (h - l) + 1;
  // The &63 is eliminated by the backend for x86-64 as that's the
  // behaviour of the shift instruction.
  base >>= (UINT64_C(63) & (UINT64_C(64) - width));
  base <<= (UINT64_C(63) & l);
  return base;
}

inline uint64_t ctz64(uint64_t value)
{
  if (value == 0)
    {
      return 64;
    }
#if defined(__has_builtin) && __has_builtin(__builtin_ctzl)
  static_assert(
      sizeof(unsigned long) == sizeof(uint64_t),
      "Calling __builtin_ctzl on a uint64_t requires 64 bit unsigned long");
  return (uint64_t)__builtin_ctzl(value);
#else
  uint64_t pos = 0;
  while (!(value & 1))
    {
      value >>= 1;
      ++pos;
    }
  return pos;
#endif
}

inline uint64_t clz64(uint64_t value)
{
  if (value == 0)
    {
      return 64;
    }
#if defined(__has_builtin) && __has_builtin(__builtin_clzl)
  static_assert(
      sizeof(unsigned long) == sizeof(uint64_t),
      "Calling __builtin_clzl on a uint64_t requires 64 bit unsigned long");
  return (uint64_t)__builtin_clzl(value);
#else
#error "Unimplemented clz64"
#endif
}

}  // namespace detail
}  // namespace

struct cache
{
  cache() = default;

  void dump()
  {
#ifndef NDEBUG
    printf("[%lu] %lu/%lu/%lu\n", slot, i_, o_, a_);
#else
    printf("cache dump requires assertions enabled\n");
#endif
  }

  bool is(uint8_t s)
  {
#ifndef NDEBUG
    assert(s < 8);
    bool r = s == concat();
    if (!r) dump();
    return r;
#else
    (void)s;
    return true;
#endif
  }

  void init(uint64_t s)
  {
#ifndef NDEBUG
    slot = s;
    word = index_to_element(s);
    subindex = index_to_subindex(s);
#else
    (void)s;
#endif
  }

#ifndef NDEBUG
  void i(uint64_t x) { i_ = x; }
  void o(uint64_t x) { o_ = x; }
  void a(uint64_t x) { a_ = x; }
#else
  void i(uint64_t) {}
  void o(uint64_t) {}
  void a(uint64_t) {}
#endif

#ifndef NDEBUG
 private:
  uint64_t i_ = 0;
  uint64_t o_ = 0;
  uint64_t a_ = 0;
  uint64_t slot = UINT64_MAX;
  uint64_t word = UINT64_MAX;
  uint64_t subindex = UINT64_MAX;

  uint8_t concat()
  {
    unsigned r = detail::nthbitset64(i_, subindex) << 2 |
                 detail::nthbitset64(o_, subindex) << 1 |
                 detail::nthbitset64(a_, subindex) << 0;
    return static_cast<uint8_t>(r);
  }
#endif
};

namespace properties
{
// atomic operations on fine grained memory are limited to those that the
// pci-e bus supports. There is no cache involved to mask this - fetch_and on
// the gpu will silently do the wrong thing if the pci-e bus doesn't support
// it. That means using cas (or swap, or faa) to communicate or buffering. The
// fetch_and works fine on coarse grained memory, but multiple waves will
// clobber each other, leaving the flag flickering from the other device
// perspective. Can downgrade to swap fairly easily, which will be roughly as
// expensive as a load & store.

template <bool HasFetchOpArg>
struct base
{
  static constexpr bool hasFetchOp() { return HasFetchOpArg; }
};

struct fine_grain : public base<false>
{
  using Ty = _Atomic uint64_t *;
};

struct coarse_grain : public base<true>
{
#if defined(__AMDGCN__)
  using Ty = __attribute__((address_space(1))) _Atomic uint64_t *;
#else
  using Ty = _Atomic uint64_t *;
#endif
};

}  // namespace properties

template <size_t scope, typename Prop>
struct slot_bitmap;

using slot_bitmap_all_svm =
    slot_bitmap<__OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES, properties::fine_grain>;

using slot_bitmap_device =
    slot_bitmap<__OPENCL_MEMORY_SCOPE_DEVICE, properties::fine_grain>;

// coarse grain here raises a HSAIL hardware exception
using slot_bitmap_coarse =
    slot_bitmap<__OPENCL_MEMORY_SCOPE_DEVICE, properties::coarse_grain>;

template <size_t scope, typename Prop>
struct slot_bitmap
{
  using Ty = typename Prop::Ty;
  static_assert(sizeof(uint64_t) == sizeof(_Atomic uint64_t), "");
  static_assert(sizeof(_Atomic uint64_t *) == 8, "");

  bool valid(uint64_t N)
  {
    // Notably, default constructed instance isn't valid
    static_assert(sizeof(slot_bitmap<scope, Prop>) == 8, "");
    return (a != nullptr) && (N != 0) && (N != SIZE_MAX) && (N % 64 == 0);
  }
  Ty a;

  slot_bitmap() : a(nullptr) {}

  slot_bitmap(size_t size, Ty d) : a(d)
  {
    assert(valid(size));
    for (size_t i = 0; i < size / 64; i++)
      {
        a[i] = 0;
      }
  }

  Ty data() { return a; }

  ~slot_bitmap() {}

  bool operator()(size_t size, size_t i, uint64_t *loaded) const
  {
    size_t w = index_to_element(i);
    uint64_t d = load_word(size, w);
    *loaded = d;
    return detail::nthbitset64(d, index_to_subindex(i));
  }

  void dump(size_t size) const
  {
    uint64_t w = size / 64;
    printf("Size %lu / words %lu\n", size, w);
    for (uint64_t i = 0; i < w; i++)
      {
        printf("[%2lu]:", i);
        for (uint64_t j = 0; j < 64; j++)
          {
            if (j % 8 == 0)
              {
                printf(" ");
              }
            printf("%c", this->operator()(size, 64 * i + j) ? '1' : '0');
          }
        printf("\n");
      }
  }

  // cas, true on success
  bool try_claim_empty_slot(size_t size, size_t i, uint64_t *,
                            uint64_t *cas_fail_count);

  // assumes slot available
  uint64_t claim_slot_returning_updated_word(size_t size, size_t i)
  {
    (void)size;
    assert(i < size);
    size_t w = index_to_element(i);
    uint64_t subindex = index_to_subindex(i);
    assert(!detail::nthbitset64(load_word(size, w), subindex));

    // or with only the slot set
    uint64_t mask = detail::setnthbit64(0, subindex);

    uint64_t before = fetch_or(w, mask);
    assert(!detail::nthbitset64(before, subindex));
    return before | mask;
  }

  // assumes slot taken
  uint64_t release_slot_returning_updated_word(size_t size, size_t i)
  {
    (void)size;
    assert(i < size);
    size_t w = index_to_element(i);
    uint64_t subindex = index_to_subindex(i);
    assert(detail::nthbitset64(load_word(size, w), subindex));

    // and with everything other than the slot set
    uint64_t mask = ~detail::setnthbit64(0, subindex);

    uint64_t before = fetch_and(w, mask);
    // assert(detail::nthbitset64(before, subindex)); // this is firing on the
    // gpu
    return before & mask;
  }

  uint64_t load_word(size_t size, size_t i) const
  {
    (void)size;
    assert(i < (size / 64));

#if 0
    // Passing a compile time scope to aomp's clang hits a bug:
    // clang++: /home/amd/aomp/amd-llvm-project/clang/lib/AST/ExprConstant.cpp:14617: bool clang::Expr::isIntegerConstantExpr(llvm::APSInt&, const clang::ASTContext&, clang::SourceLocation*, bool) const: Assertion `!isValueDependent() && "Expression evaluator can't be called on a dependent expression."' failed.
        return __opencl_atomic_load(&a[i], __ATOMIC_RELAXED, scope);
#else

    if (scope == __OPENCL_MEMORY_SCOPE_DEVICE)
      {
        return __opencl_atomic_load(&a[i], __ATOMIC_RELAXED,
                                    __OPENCL_MEMORY_SCOPE_DEVICE);
      }
    else
      {
        return __opencl_atomic_load(&a[i], __ATOMIC_RELAXED,
                                    __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES);
      }
#endif
  }

  bool cas(uint64_t element, uint64_t expect, uint64_t replace,
           uint64_t *loaded)
  {
    Ty addr = &a[element];

    // this cas function is not used across devices by this library
    bool r = __opencl_atomic_compare_exchange_weak(
        addr, &expect, replace, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED,
        __OPENCL_MEMORY_SCOPE_DEVICE);

    // on success, bits in memory have been set to replace
    // on failure, value found is now in expect
    // if cas succeeded, the bits in memory matched what was expected and now
    // match replace if it failed, the above call wrote the bits found in memory
    // into expect
    *loaded = expect;
    return r;
  }

  // returns value from before the and/or
  // these are used on memory visible fromi all svm devices

  __attribute__((used)) uint64_t fetch_and(uint64_t element, uint64_t mask)
  {
    Ty addr = &a[element];

    if (Prop::hasFetchOp())
      {
        // This seems to work on amdgcn, but only with acquire. acq/rel fails
        if (scope == __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES)
          {
            return __opencl_atomic_fetch_and(
                addr, mask, __ATOMIC_ACQ_REL,
                __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES);
          }
        else
          {
            return __opencl_atomic_fetch_and(addr, mask, __ATOMIC_ACQ_REL,
                                             __OPENCL_MEMORY_SCOPE_DEVICE);
          }
      }
    else
      {
        // load and atomic cas have similar cost across pcie, may be faster to
        // use a (usually wrong) initial guess instead of a load
        uint64_t current = __opencl_atomic_load(
            addr, __ATOMIC_RELAXED, __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES);
        while (1)
          {
            uint64_t replace = current & mask;

            bool r = __opencl_atomic_compare_exchange_weak(
                addr, &current, replace, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED,
                __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES);

            if (r)
              {
                return current;
              }
          }
      }
  }

  __attribute__((used)) uint64_t fetch_or(uint64_t element, uint64_t mask)
  {
    Ty addr = &a[element];

    if (Prop::hasFetchOp())
      {
        if (scope == __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES)
          {
            return __opencl_atomic_fetch_or(
                addr, mask, __ATOMIC_ACQ_REL,
                __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES);
          }
        else
          {
            return __opencl_atomic_fetch_or(addr, mask, __ATOMIC_ACQ_REL,
                                            __OPENCL_MEMORY_SCOPE_DEVICE);
          }
      }
    else
      {
        uint64_t current = __opencl_atomic_load(
            addr, __ATOMIC_RELAXED, __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES);
        while (1)
          {
            uint64_t replace = current | mask;

            bool r = __opencl_atomic_compare_exchange_weak(
                addr, &current, replace, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED,
                __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES);
            if (r)
              {
                return current;
              }
          }
      }
  }
};

template <size_t Sscope, typename SProp, size_t Vscope, typename VProp>
uint64_t staged_claim_slot_returning_updated_word(
    size_t size, size_t i, slot_bitmap<Sscope, SProp> *staging,
    slot_bitmap<Vscope, VProp> *visible, uint64_t *cas_fail_count,
    uint64_t *cas_help_count)
{
  // claim slot in staging (efficiently) then propagage change to visible
  assert((void *)visible != (void *)staging);
  assert(i < size);
  const size_t w = index_to_element(i);
  const uint64_t subindex = index_to_subindex(i);

  // slot is known clear as lock is held
  assert(!detail::nthbitset64(staging->load_word(size, w), subindex));
  assert(!detail::nthbitset64(visible->load_word(size, w), subindex));

  // fetch_or to update staging
  uint64_t staged_result = staging->claim_slot_returning_updated_word(size, i);
  assert(detail::nthbitset64(staged_result, subindex));

  // propose a value that could plausibly be in visible
  // this was returned by fetch_or, can refactor to drop the arithmetic
  uint64_t guess = detail::clearnthbit64(staged_result, subindex);

  // initialise the value with the latest view of staging that is already
  // available
  uint64_t proposed = staged_result;

  uint64_t local_fail_count = 0;
  uint64_t local_help_count = 0;
  while (!visible->cas(w, guess, proposed, &guess))
    {
      local_fail_count++;
      if (detail::nthbitset64(guess, subindex))
        {
          // Cas failed, but another thread has done our work
          local_help_count++;
          proposed = guess;
          break;
        }

      // Update our view of proposed and try again
      proposed = staging->load_word(size, w);
      assert(detail::nthbitset64(proposed, subindex));
    }
  *cas_fail_count = *cas_fail_count + local_fail_count;
  *cas_help_count = *cas_help_count + local_help_count;
  assert(detail::nthbitset64(visible->load_word(size, w), subindex));
  return proposed;
}

template <size_t Sscope, typename SProp, size_t Vscope, typename VProp>
uint64_t staged_release_slot_returning_updated_word(
    size_t size, size_t i, slot_bitmap<Sscope, SProp> *staging,
    slot_bitmap<Vscope, VProp> *visible, uint64_t *cas_fail_count,
    uint64_t *cas_help_count)
{
  // claim slot in staging (efficiently) then propagage change to visible
  assert((void *)visible != (void *)staging);
  assert(i < size);
  const size_t w = index_to_element(i);
  const uint64_t subindex = index_to_subindex(i);

  // slot is known set as lock is held
  assert(detail::nthbitset64(staging->load_word(size, w), subindex));
  assert(detail::nthbitset64(visible->load_word(size, w), subindex));

  // fetch_and to update staging
  uint64_t staged_result =
      staging->release_slot_returning_updated_word(size, i);
  assert(!detail::nthbitset64(staged_result, subindex));

  // propose a value that could plausibly be in visible
  // this was returned by fetch_or, can refactor to drop the arithmetic
  uint64_t guess = detail::setnthbit64(staged_result, subindex);

  // initialise the value with the latest view of staging that is already
  // available
  uint64_t proposed = staged_result;

  uint64_t local_fail_count = 0;
  uint64_t local_help_count = 0;
  while (!visible->cas(w, guess, proposed, &guess))
    {
      local_fail_count++;
      if (!detail::nthbitset64(guess, subindex))
        {
          // Cas failed, but another thread has done our work
          local_help_count++;
          proposed = guess;
          break;
        }

      // Update our view of proposed and try again
      proposed = staging->load_word(size, w);
      assert(!detail::nthbitset64(proposed, subindex));
    }
  *cas_fail_count = *cas_fail_count + local_fail_count;
  *cas_help_count = *cas_help_count + local_help_count;

  assert(!detail::nthbitset64(visible->load_word(size, w), subindex));
  return proposed;
}

// on return true, loaded contains active[w]
template <size_t scope, typename Prop>
bool slot_bitmap<scope, Prop>::try_claim_empty_slot(size_t size, size_t i,
                                                    uint64_t *loaded,
                                                    uint64_t *cas_fail_count)
{
  assert(i < size);
  size_t w = index_to_element(i);
  uint64_t subindex = index_to_subindex(i);

  uint64_t d = load_word(size, w);

  // printf("Slot %lu, w %lu, subindex %lu, d %lu\n", i, w, subindex, d);
  uint64_t local_fail_count = 0;
  for (;;)
    {
      // if the bit was already set then we've lost the race

      // can either check the bit is zero, or unconditionally set it and check
      // if this changed the value
      uint64_t proposed = detail::setnthbit64(d, subindex);
      if (proposed == d)
        {
          *cas_fail_count = *cas_fail_count + local_fail_count;
          return false;
        }

      // If the bit is known zero, can use fetch_or to set it

      uint64_t unexpected_contents;

      uint32_t r = platform::critical<uint32_t>(
          [&]() { return cas(w, d, proposed, &unexpected_contents); });

      unexpected_contents = platform::broadcast_master(unexpected_contents);

      if (r)
        {
          // success, got the lock, and active word was set to proposed
          *loaded = proposed;
          *cas_fail_count = *cas_fail_count + local_fail_count;
          return true;
        }

      local_fail_count++;
      // cas failed. reasons:
      // we lost the slot
      // another slot in the same word changed
      // spurious

      // try again if the slot is still empty
      // may want a give up count / sleep or similar
      d = unexpected_contents;
    }
}

template <bool enable>
struct slot_owner_t;

#ifdef __CUDACC__
// TODO: amdgcn doesn't have thread_local either, just doesn't error on it
extern unsigned my_id;
#else
extern thread_local unsigned my_id;
#endif

using slot_owner = slot_owner_t<false>;

template <>
struct slot_owner_t<false>
{
  void dump() {}
  void claim(uint64_t) {}
  void release(uint64_t) {}
};

template <>
struct slot_owner_t<true>
{
  void dump()
  {
    for (unsigned i = 0; i < sizeof(slots) / sizeof(slots[0]); i++)
      {
        uint32_t v = __c11_atomic_load(&slots[i], __ATOMIC_SEQ_CST);
        printf("slot[%u] owned by %u\n", i, v);
        (void)v;
      }
  }
  static const bool verbose = false;
  slot_owner_t()
  {
    for (unsigned i = 0; i < sizeof(slots) / sizeof(slots[0]); i++)
      {
        slots[i] = UINT32_MAX;
      }
  }
  _Atomic uint32_t slots[128];

  void claim(uint64_t slot) { claim(my_id, slot); }

  void release(uint64_t slot) { release(my_id, slot); }

  void claim(uint32_t id, uint64_t slot)
  {
    assert(slot < 128);
    uint32_t v = __c11_atomic_load(&slots[slot], __ATOMIC_SEQ_CST);
    if (v != UINT32_MAX)
      {
        printf("slot[%lu] <- %u failed, owned by %u\n", slot, id, slots[slot]);
      }
    assert(v == UINT32_MAX);
    if (verbose)
      {
        printf("slot[%lu] <- %u\n", slot, id);
      }
    __c11_atomic_store(&slots[slot], id, __ATOMIC_SEQ_CST);
  }

  void release(uint32_t id, uint64_t slot)
  {
    assert(slot < 128);
    uint32_t v = __c11_atomic_load(&slots[slot], __ATOMIC_SEQ_CST);
    if (v != id)
      {
        printf("slot[%lu] owned by %u, can't be freed by %u\n", slot,
               slots[slot], id);
      }
    assert(v == id);
    __c11_atomic_store(&slots[slot], UINT32_MAX, __ATOMIC_SEQ_CST);
    if (verbose)
      {
        printf("slot[%lu] -> free\n", slot);
      }
  }
};

inline slot_owner tracker()
{
  static slot_owner t;
  return t;
}

inline void step(_Atomic(uint64_t) * steps_left)
{
  if (__c11_atomic_load(steps_left, __ATOMIC_SEQ_CST) == UINT64_MAX)
    {
      // Disable stepping
      return;
    }
  while (__c11_atomic_load(steps_left, __ATOMIC_SEQ_CST) == 0)
    {
      // Don't burn all the cpu waiting for a step
      platform::sleep_briefly();
    }

  steps_left--;
}

struct nop_stepper
{
  static void call(int, void *) {}
};

struct default_stepper_state
{
  default_stepper_state(_Atomic(uint64_t) * val, bool show_step = false,
                        const char *name = "unknown")
      : val(val), show_step(show_step), name(name)
  {
  }

  _Atomic(uint64_t) * val;
  bool show_step;
  const char *name;
};

struct default_stepper
{
  static void call(int line, void *v)
  {
    default_stepper_state *state = static_cast<default_stepper_state *>(v);
    if (state->show_step)
      {
        printf("%s:%d: step\n", state->name, line);
      }
    (void)line;
    step(state->val);
  }
};

// Depending on the host / client device and how they're connected together,
// copying data can be a no-op (shared memory, single buffer in use),
// pull and push from one of the two, routed through a third buffer

template <typename T>
struct copy_functor_interface
{
  // dst then src, memcpy style. Copies a single page
  static void push_from_client_to_server(page_t *dst, const page_t *src)
  {
    T::push_from_client_to_server_impl(dst, src);
  }
  static void pull_to_client_from_server(page_t *dst, const page_t *src)
  {
    T::pull_to_client_from_server_impl(dst, src);
  }

  static void push_from_server_to_client(page_t *dst, const page_t *src)
  {
    T::push_from_server_to_client_impl(dst, src);
  }
  static void pull_to_server_from_client(page_t *dst, const page_t *src)
  {
    T::pull_to_server_from_client_impl(dst, src);
  }

 private:
  friend T;
  copy_functor_interface() = default;

  // Default implementations are no-ops
  static void push_from_client_to_server_impl(page_t *, const page_t *) {}
  static void pull_to_client_from_server_impl(page_t *, const page_t *) {}
  static void push_from_server_to_client_impl(page_t *, const page_t *) {}
  static void pull_to_server_from_client_impl(page_t *, const page_t *) {}
};

struct copy_functor_memcpy_pull
    : public copy_functor_interface<copy_functor_memcpy_pull>
{
  friend struct copy_functor_interface<copy_functor_memcpy_pull>;

 private:
  static void pull_to_client_from_server_impl(page_t *dst, const page_t *src)
  {
    size_t N = sizeof(page_t);
    __builtin_memcpy(dst, src, N);
  }
  static void pull_to_server_from_client_impl(page_t *dst, const page_t *src)
  {
    size_t N = sizeof(page_t);
    __builtin_memcpy(dst, src, N);
  }
};

struct copy_functor_given_alias
    : public copy_functor_interface<copy_functor_given_alias>
{
  friend struct copy_functor_interface<copy_functor_given_alias>;

  static void push_from_client_to_server_impl(page_t *dst, const page_t *src)
  {
    assert(src == dst);
    (void)src;
    (void)dst;
  }
  static void pull_to_client_from_server_impl(page_t *dst, const page_t *src)
  {
    assert(src == dst);
    (void)src;
    (void)dst;
  }
  static void push_from_server_to_client_impl(page_t *dst, const page_t *src)
  {
    assert(src == dst);
    (void)src;
    (void)dst;
  }
  static void pull_to_server_from_client_impl(page_t *dst, const page_t *src)
  {
    assert(src == dst);
    (void)src;
    (void)dst;
  }
};

}  // namespace hostrpc

#endif
