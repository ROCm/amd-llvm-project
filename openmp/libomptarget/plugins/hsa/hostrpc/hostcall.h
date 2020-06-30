#ifndef HOSTCALL_H_INCLUDED
#define HOSTCALL_H_INCLUDED

// Limited API for using hostcall.hpp from C
// Spawns one thread for the queue

#include "hsa.h"
#include <stdint.h>

#if __cplusplus
extern "C"
{
#endif


  // pointer is to address of the symbol with name returned by
  // hostcall_client_symbol on this device
  void spawn_hostcall_for_queue(uint32_t device_id,
                               hsa_agent_t agent,
                               hsa_queue_t *queue,
                               void *client_symbol_address);

  void free_hostcall_state();

#if __cplusplus
}
#endif

#endif
