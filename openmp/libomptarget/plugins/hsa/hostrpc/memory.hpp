#ifndef HOSTRPC_MEMORY_HPP
#define HOSTRPC_MEMORY_HPP

#include <stddef.h>
#include <stdint.h>

#if defined(__x86_64__)
// strictly, should test for presence of <new>.
// currently x64_64 is a hosted implementation and amdgcn is freestanding
// without headers.
#include <new>
#endif
namespace hostrpc
{
namespace x64_native
{
void* allocate(size_t align, size_t bytes);
void deallocate(void*);
}  // namespace x64_native

namespace hsa
{
void* allocate(uint64_t hsa_region_t_handle, size_t align, size_t bytes);
void deallocate(void*);
}  // namespace hsa

#if defined(__x86_64__)
template <typename T>
T* careful_array_cast(void* data, size_t N)
{
  // allocation functions return void*, but casting that to a T* is not strictly
  // sufficient to satisfy the C++ object model. One should placement new
  // instead. Placement new on arrays isn't especially useful as it needs extra
  // space to store the size of the array, in order for delete[] to work.
  // Instead, walk the memory constructing each element individually.

  // Strictly one should probably do this with destructors as well. That seems
  // less necessary to avoid consequences from the aliasing rules.

  // Handles the invalid combination of nullptr data and N != 0 by returning the
  // cast nullptr, for convenience a the call site.
  T* typed = static_cast<T*>(data);
  if (data != nullptr)
    {
      for (size_t i = 0; i < N; i++)
        {
          new (typed + i) T;
        }
    }
  return typed;
}

#endif
}  // namespace hostrpc

#endif
