#include "../../../hostrpc/src/hostrpc.h"
#include "detail/common.hpp"
#include "detail/platform.hpp"
#include "hostcall.hpp"

#if defined(__x86_64__)

extern "C" void handlePayload(uint32_t service, uint64_t *payload);

#endif

namespace hostcall_ops
{
#if defined(__x86_64__)
void operate(hostrpc::page_t *page)
{
  for (unsigned c = 0; c < 64; c++)
    {
      hostrpc::cacheline_t &line = page->cacheline[c];
      uint64_t service_id = line.element[0];
      uint64_t *payload = &line.element[1];

      service_id = ((uint32_t)service_id << 16u) >> 16u;
      assert(service_id <= UINT32_MAX);

      // A bit dubious in that the existing code expects payload to have
      // length 8 and we're passing one of length 7, but nothing yet
      // implemented goes beyond [3]
      handlePayload(static_cast<uint32_t>(service_id), payload);
    }
}
void clear(hostrpc::page_t *page)
{
  for (unsigned c = 0; c < 64; c++)
    {
      hostrpc::cacheline_t &line = page->cacheline[c];
      for (unsigned i = 0; i < 8; i++)
        {
          line.element[i] = HOSTRPC_SERVICE_NO_OPERATION;
        }
    }
}
#endif

#if defined __AMDGCN__
void pass_arguments(hostrpc::page_t *page, uint64_t d[8])
{
  hostrpc::cacheline_t *line = &page->cacheline[platform::get_lane_id()];
  for (unsigned i = 0; i < 8; i++)
    {
      line->element[i] = d[i];
    }
}
void use_result(hostrpc::page_t *page, uint64_t d[8])
{
  hostrpc::cacheline_t *line = &page->cacheline[platform::get_lane_id()];
  for (unsigned i = 0; i < 8; i++)
    {
      d[i] = line->element[i];
    }
}

extern "C" hostrpc_result_t __attribute__((used))
hostrpc_invoke(uint32_t service_id, uint64_t arg0, uint64_t arg1, uint64_t arg2,
               uint64_t arg3, uint64_t arg4, uint64_t arg5, uint64_t arg6,
               uint64_t arg7)
{
  // changes the control flow in hsa/impl
  __asm__(
      "; hostcall_invoke: record need for hostcall support\n\t"
      ".type needs_hostcall_buffer,@object\n\t"
      ".global needs_hostcall_buffer\n\t"
      ".comm needs_hostcall_buffer,4" ::
          :);

  uint64_t buf[8] = {service_id, arg0, arg1, arg2, arg3, arg4, arg5, arg6};

  hostcall_client(buf);

  return {buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7], UINT64_MAX};
}

#endif

}  // namespace hostcall_ops
