 
/*   
 *   atmi_hostcall.c: atmi integration of hostcall.
 *                    This source implements a linked list queue in c.
 *                    hostcall buffers and pointer to their consumer
 *                    are placed on the linked list queue (hcb).
 *
 *   Written by Greg Rodgers

MIT License

Copyright © 2019 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished
to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

#include <stdio.h>
#include <stdlib.h>
#include "hsa/hsa_ext_amd.h"
#include "amd_hostcall.h"
#include "hostcall_impl.h"
#include "hostcall_service_id.h"
#include "hostcall_internal.h"
#include "atmi_interop_hsa.h"
#include "atmi_runtime.h"

static int atl_hcq_size() { return atl_hcq_count ;}

atl_hcq_element_t * atl_hcq_push(buffer_t * hcb, amd_hostcall_consumer_t * consumer, 
		                 hsa_queue_t * hsa_q, uint32_t devid) {
  // FIXME , check rc of these mallocs
  if (atl_hcq_rear == NULL) {
    atl_hcq_rear = (atl_hcq_element_t *) malloc(sizeof(atl_hcq_element_t));
    atl_hcq_front = atl_hcq_rear;
  } else {
    atl_hcq_element_t * new_rear = (atl_hcq_element_t *) malloc(sizeof(atl_hcq_element_t));
    atl_hcq_rear->next_ptr = new_rear;
    atl_hcq_rear = new_rear;
  }
  atl_hcq_rear->next_ptr  = NULL;
  atl_hcq_rear->hcb       = hcb; 
  atl_hcq_rear->hsa_q     = hsa_q;
  atl_hcq_rear->consumer  = consumer;
  atl_hcq_rear->device_id = devid;
  atl_hcq_count++;
  return atl_hcq_rear;
}

#if 0
// we are not using hcq as a true queue so we do not need the pop operation
static void atl_hcq_pop() {
  if (atl_hcq_front  == NULL) {
    printf("\n Error: Trying to pop an element from empty queue");
    return;
  } else {
    if (atl_hcq_front->next_ptr != NULL) {
      atl_hcq_element_t * new_front = atl_hcq_front->next_ptr;
      free(atl_hcq_front);
      atl_hcq_front = new_front;
    } else {
      free(atl_hcq_front);
      atl_hcq_front = NULL;
      atl_hcq_rear = NULL;
    }
    atl_hcq_count--;
  }
}
#endif
 
static atl_hcq_element_t  * atl_hcq_find_by_hsa_q(hsa_queue_t * hsa_q) {
  atl_hcq_element_t * this_front = atl_hcq_front;
  int reverse_counter = atl_hcq_size();
  while (reverse_counter) {
    if (this_front->hsa_q == hsa_q)
       return this_front;
    this_front = this_front->next_ptr;
    reverse_counter--;
  }
  return NULL;
}

static buffer_t * atl_hcq_create_buffer(unsigned int num_packets) {
    if (num_packets == 0) {
	printf("num_packets cannot be zero \n");
	abort();
    }
    size_t size  = amd_hostcall_get_buffer_size(num_packets);
    uint32_t align = amd_hostcall_get_buffer_alignment();
    void *newbuffer = NULL;
    atmi_mem_place_t place = ATMI_MEM_PLACE_CPU_MEM(0,0,0);
    atmi_status_t err = atmi_malloc(&newbuffer, size+align, place);
    if (!newbuffer || (err != ATMI_STATUS_SUCCESS) ) {
	    printf("call to atmi_malloc failed \n");
	    abort();
    }
    if (amd_hostcall_initialize_buffer(newbuffer, num_packets) != AMD_HOSTCALL_SUCCESS) {
	    printf("call to  amd_hostcall_initialize_buffer failed \n");
	    abort();
    }
    // printf("created hostcall buffer %p with %d packets \n", newbuffer, num_packets);
    return (buffer_t *) newbuffer;
}

// FIXME: Clean up this diagnostic and die properly
hsa_status_t atmi_hostcall_version_check(unsigned int device_vrm) {
    uint device_version_release = device_vrm >> 6;
    if (device_version_release != HOSTCALL_VERSION_RELEASE ) {
      printf("ERROR Incompatible device and host release\n      Device release(%d)\n      Host release(%d)\n",device_version_release, HOSTCALL_VERSION_RELEASE);
      return HSA_STATUS_ERROR;
    }
    if (device_vrm > HOSTCALL_VRM) {
      printf("ERROR Incompatible device and host version \n       Device version(%d)\n      Host version(%d)\n",device_vrm, HOSTCALL_VERSION_RELEASE);
      return HSA_STATUS_ERROR;
    }
    if (device_vrm < HOSTCALL_VRM) {
      unsigned int host_ver = ((unsigned int) HOSTCALL_VRM) >> 12;
      unsigned int host_rel = (((unsigned int) HOSTCALL_VRM) << 20) >>26  ;
      unsigned int host_mod = (((unsigned int) HOSTCALL_VRM) << 26) >>26 ;
      unsigned int dev_ver = ((unsigned int) device_vrm) >> 12;
      unsigned int dev_rel = (((unsigned int) device_vrm) << 20) >>26  ;
      unsigned int dev_mod = (((unsigned int) device_vrm) << 26) >>26 ;
      printf("WARNING:  Device mod version < host mod version \n          Device version: %d.%d.%d\n          Host version:   %d.%d.%d\n",
         dev_ver,dev_rel,dev_mod, host_ver,host_rel,host_mod);
      printf("          Please consider upgrading hostcall on your host\n");
    }
    return HSA_STATUS_SUCCESS;
}

void hostcall_register_all_handlers(amd_hostcall_consumer_t * c, void * cbdata);

// These three external functions are called by atmi.
// ATMI uses the header atmi_hostcall.h to reference these. 
//
unsigned long atmi_hostcall_assign_buffer(
		hsa_queue_t * this_Q,
		uint32_t device_id) {
    atl_hcq_element_t * llq_elem ;
    llq_elem  = atl_hcq_find_by_hsa_q(this_Q);
    if (!llq_elem) {
       hsa_agent_t agent;
       atmi_place_t place = ATMI_PLACE_GPU(0, device_id);
       // FIXME: error check for this function
       // atmi_status_t atmi_err =
       atmi_interop_hsa_get_agent(place, &agent);
       // ATMIErrorCheck(Could not get agent from place, atmi_err);
       uint32_t numCu;
       // hsa_status_t err =
       hsa_agent_get_info(agent, (hsa_agent_info_t)
           HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT, &numCu);
       // ErrorCheck(Could not get number of cus, err);
       uint32_t waverPerCu;
       // err =
       hsa_agent_get_info(agent, (hsa_agent_info_t)
           HSA_AMD_AGENT_INFO_MAX_WAVES_PER_CU, &waverPerCu);
       // ErrorCheck(Could not get number of waves per cu, err);
       unsigned int minpackets = numCu * waverPerCu;
       //  For now, we create one bufer and one consumer per ATMI hsa queue
       buffer_t * hcb  = atl_hcq_create_buffer(minpackets);
       amd_hostcall_consumer_t * c ;
       atl_hcq_element_t * front = atl_hcq_front;
       if (front) 
          c = front->consumer;
       else 
          c = amd_hostcall_create_consumer();
       amd_hostcall_register_buffer(c,hcb);
       // create element of linked list hcq. This will also be the callback data
       llq_elem = atl_hcq_push( hcb , c, this_Q, device_id);
       hostcall_register_all_handlers(c, (void*) llq_elem);
       amd_hostcall_launch_consumer(c);
    }
    return (unsigned long) llq_elem->hcb;
}

hsa_status_t atmi_hostcall_init() {
   atl_hcq_count = 0;
   atl_hcq_front = atl_hcq_rear = NULL;
   // Register atmi_hostcall_assign_buffer with ATMI so that it is
   // called by ATMI during every task launch.
   atmi_status_t status = atmi_register_task_hostcall_handler(
      (atmi_task_hostcall_handler_t)&atmi_hostcall_assign_buffer);
   if(status == ATMI_STATUS_SUCCESS)
    return HSA_STATUS_SUCCESS;
   else
    return HSA_STATUS_ERROR;
}

hsa_status_t atmi_hostcall_terminate() {
   amd_hostcall_consumer_t * c;
   atl_hcq_element_t * this_front = atl_hcq_front;
   atl_hcq_element_t * last_front;
   int reverse_counter = atl_hcq_size();
   while (reverse_counter) {
      if (this_front == atl_hcq_front) {
         c = this_front->consumer;
         amd_hostcall_destroy_consumer(c);
      }
      atmi_free(this_front->hcb);
      last_front = this_front;
      this_front = this_front->next_ptr;
      free(last_front);
      reverse_counter--;
   }
   atl_hcq_count = 0;
   atl_hcq_front = atl_hcq_rear = NULL;
   return HSA_STATUS_SUCCESS;
}

