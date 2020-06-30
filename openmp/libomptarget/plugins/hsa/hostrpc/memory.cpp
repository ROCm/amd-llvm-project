#include "memory.hpp"

#if defined(__x86_64__)
#include "hsa.h"
#include <stdlib.h>
#endif

namespace hostrpc
{
#if defined(__x86_64__)
namespace x64_native
{
void* allocate(size_t align, size_t bytes)
{
  return ::aligned_alloc(align, bytes);
}
void deallocate(void* d) { free(d); }
}  // namespace x64_native

namespace hsa
{
void* allocate(uint64_t hsa_region_t_handle, size_t align, size_t bytes)
{
  (void)align;  // todo
  hsa_region_t region{.handle = hsa_region_t_handle};
  void* memory;
  if (HSA_STATUS_SUCCESS == hsa_memory_allocate(region, bytes, &memory))
    {
      return memory;
    }
  else
    {
      return nullptr;
    }
}
void deallocate(void* d) { hsa_memory_free(d); }
}  // namespace hsa
#endif

#if defined(__AMDGCN__)

namespace x64_native
{
void* allocate(size_t, size_t) { return nullptr; }
void deallocate(void*) {}
}  // namespace x64_native

namespace hsa
{
void* allocate(hsa_region_t, size_t, size_t) { return nullptr; }
void deallocate(void*) {}
}  // namespace hsa
#endif

}  // namespace hostrpc
