#ifndef BASE_TYPES_HPP_INCLUDED
#define BASE_TYPES_HPP_INCLUDED

#include <stddef.h>
#include <stdint.h>

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

}  // namespace hostrpc

#endif
