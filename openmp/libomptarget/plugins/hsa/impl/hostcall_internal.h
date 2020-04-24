 
/*   
 *   hostcall_handlers.c: 
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


typedef struct atl_hcq_element_s atl_hcq_element_t;
struct atl_hcq_element_s {
  buffer_t *            hcb;
  amd_hostcall_consumer_t *  consumer;
  hsa_queue_t *       	hsa_q;
  atl_hcq_element_t *   next_ptr;
  uint32_t              device_id;
};

//  Persistent static values for the hcq linked list 
atl_hcq_element_t * atl_hcq_front;
atl_hcq_element_t * atl_hcq_rear;
int atl_hcq_count;

/// This is called by the registered printf callback handler.  
//  The source code for hostcall_printf is in hostcall_printf.c
amd_hostcall_error_t hostcall_printf(char *buf, size_t bufsz);

