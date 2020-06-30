#ifndef HOSTCALL_HPP_INCLUDED
#define HOSTCALL_HPP_INCLUDED

#include "base_types.hpp"
#include <stddef.h>

// needs to scale with CUs
namespace hostrpc
{
static const constexpr size_t x64_host_amdgcn_array_size = 2048;
}

#if defined(__AMDGCN__)
// amdgcn client API

#include <stdint.h>
void hostcall_client(uint64_t data[8]);
void hostcall_client_async(uint64_t data[8]);
#endif

#if defined(__x86_64__)
// x64 host API

#include "hsa.h"
const char *hostcall_client_symbol();

class hostcall
{
 public:
  hostcall(hsa_executable_t executable, hsa_agent_t kernel_agent);
  hostcall(void *client_symbol_address, hsa_agent_t kernel_agent);
  bool valid();
  int enable_queue(hsa_queue_t *queue);
  int spawn_worker(hsa_queue_t *queue);

 private:
  using state_t = hostrpc::storage<128, 8>;
  state_t state;
};
#endif

// Functions implemented by the user of the library
namespace hostcall_ops
{
#if defined(__x86_64__)
void operate(hostrpc::page_t *page);
void clear(hostrpc::page_t *page);
#endif
#if defined __AMDGCN__
void pass_arguments(hostrpc::page_t *page, uint64_t data[8]);
void use_result(hostrpc::page_t *page, uint64_t data[8]);
#endif
}  // namespace hostcall_ops

#endif
