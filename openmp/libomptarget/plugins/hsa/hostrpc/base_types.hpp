#ifndef BASE_TYPES_HPP_INCLUDED
#define BASE_TYPES_HPP_INCLUDED

#include <stddef.h>
#include <stdint.h>

#if defined(__x86_64__)
#include <stdio.h>
#endif

namespace hostrpc
{
struct cacheline_t
{
  alignas(64) uint64_t element[8];
};
static_assert(sizeof(cacheline_t) == 64, "");

struct page_t
{
  alignas(4096) cacheline_t cacheline[64];
};
static_assert(sizeof(page_t) == 4096, "");

struct size_runtime
{
  size_runtime(size_t N) : SZ(N) {}
  size_t N() const { return SZ; }

 private:
  size_t SZ;
};

template <size_t SZ>
struct size_compiletime
{
  size_compiletime() {}
  size_compiletime(size_t) {}
  constexpr size_t N() const { return SZ; }
};

using closure_func_t = void (*)(page_t*, void*);
struct closure_pair
{
  closure_func_t func;
  void* state;
};

template <size_t Size, size_t Align>
struct storage
{
  static constexpr size_t size() { return Size; }
  static constexpr size_t align() { return Align; }

  template <typename T>
  T* open()
  {
    return __builtin_launder(reinterpret_cast<T*>(data));
  }

  // TODO: Allow move construct into storage
  template <typename T>
  T* construct(T t)
  {
    return new (reinterpret_cast<T*>(data)) T(t);
  }

  template <typename T>
  void destroy()
  {
    open<T>()->~T();
  }

  alignas(Align) unsigned char data[Size];
};

struct client_counters
{
  enum
  {
    cc_no_candidate_slot = 0,
    cc_missed_lock_on_candidate_slot = 1,
    cc_got_lock_after_work_done = 2,
    cc_waiting_for_result = 3,
    cc_cas_lock_fail = 4,
    cc_garbage_cas_fail = 5,
    cc_publish_cas_fail = 6,
    cc_finished_cas_fail = 7,

    cc_garbage_cas_help = 8,
    cc_publish_cas_help = 9,
    cc_finished_cas_help = 10,

    cc_total_count,

  };

  uint64_t state[cc_total_count];
  client_counters()
  {
    for (unsigned i = 0; i < cc_total_count; i++)
      {
        state[i] = 0;
      }
  }

#if defined(__x86_64__)
  void dump() const
  {
    printf("no_candidate_slot: %lu\n", state[cc_no_candidate_slot]);
    printf("missed_lock_on_candidate_slot: %lu\n",
           state[cc_missed_lock_on_candidate_slot]);
    printf("got_lock_after_work_done: %lu\n",
           state[cc_got_lock_after_work_done]);
    printf("waiting_for_result: %lu\n", state[cc_waiting_for_result]);
    printf("cas_lock_fail: %lu\n", state[cc_cas_lock_fail]);
    printf("garbage_cas_fail: %lu\n", state[cc_garbage_cas_fail]);
    printf("garbage_cas_help: %lu\n", state[cc_garbage_cas_help]);
    printf("publish_fail: %lu\n", state[cc_publish_cas_fail]);
    printf("publish_help: %lu\n", state[cc_publish_cas_help]);
    printf("finished_fail: %lu\n", state[cc_finished_cas_fail]);
    printf("finished_help: %lu\n", state[cc_finished_cas_help]);
  }
#endif
};

}  // namespace hostrpc

#endif
