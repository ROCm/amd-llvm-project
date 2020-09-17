#include "memory.hpp"

#if defined(__x86_64__)
#include "hsa.h"
#include "hsa_ext_amd.h"
#include <stdlib.h>
#include <string.h>
#endif

namespace hostrpc
{
#if defined(__x86_64__)
namespace x64_native
{
void* allocate(size_t align, size_t bytes)
{
  void* memory = ::aligned_alloc(align, bytes);
  if (memory)
    {
      memset(memory, 0, bytes);
    }
  return memory;
}
void deallocate(void* d) { free(d); }
}  // namespace x64_native

namespace hsa
{
void* allocate(uint64_t hsa_region_t_handle, size_t align, size_t bytes)
{
  (void)align;  // todo
  hsa_region_t region{.handle = hsa_region_t_handle};

  bytes = 4 * ((bytes + 3)/4); // fill uses a multiple of four
  
  void* memory;
  if (HSA_STATUS_SUCCESS == hsa_memory_allocate(region, bytes, &memory))
    {
      // probably want memset for fine grain, may want it for gfx9
      // memset(memory, 0, bytes);
      // warning: This is likely to be relied on by bitmap
      hsa_status_t  r = hsa_amd_memory_fill(memory, 0, bytes/4);
      if (HSA_STATUS_SUCCESS == r) {
        return memory;      
      }      
    }
  
  return nullptr;
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
