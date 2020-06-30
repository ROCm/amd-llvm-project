#include "hostcall_interface.hpp"
#include "detail/client_impl.hpp"
#include "detail/platform.hpp"
#include "detail/server_impl.hpp"
#include "hostcall.hpp"  // hostcall_ops prototypes
#include "memory.hpp"

// Glue the opaque hostcall_interface class onto the freestanding implementation

// hsa uses freestanding C headers, unlike hsa.hpp
#if defined(__x86_64__)
#include "hsa.h"
#include <new>
#include <string.h>
#endif

namespace hostrpc
{
namespace x64_host_amdgcn_client
{
struct fill
{
  static void call(hostrpc::page_t *page, void *dv)
  {
#if defined(__AMDGCN__)
    uint64_t *d = static_cast<uint64_t *>(dv);
    hostcall_ops::pass_arguments(page, d);
#else
    (void)page;
    (void)dv;
#endif
  };
};

struct use
{
  static void call(hostrpc::page_t *page, void *dv)
  {
#if defined(__AMDGCN__)
    uint64_t *d = static_cast<uint64_t *>(dv);
    hostcall_ops::use_result(page, d);
#else
    (void)page;
    (void)dv;
#endif
  };
};

struct operate
{
  static void call(hostrpc::page_t *page, void *)
  {
#if defined(__x86_64__)
    hostcall_ops::operate(page);
#else
    (void)page;
#endif
  }
};

struct clear
{
  static void call(hostrpc::page_t *page, void *)
  {
#if defined(__x86_64__)
    hostcall_ops::clear(page);
#else
    (void)page;
#endif
  }
};
}  // namespace x64_host_amdgcn_client

template <typename SZ>
using x64_amdgcn_client =
    hostrpc::client_impl<SZ, hostrpc::copy_functor_given_alias,
                         x64_host_amdgcn_client::fill,
                         x64_host_amdgcn_client::use, hostrpc::nop_stepper>;

template <typename SZ>
using x64_amdgcn_server =
    hostrpc::server_impl<SZ, hostrpc::copy_functor_given_alias,
                         x64_host_amdgcn_client::operate,
                         x64_host_amdgcn_client::clear, hostrpc::nop_stepper>;

#if defined(__x86_64__)
namespace
{
inline _Atomic uint64_t *hsa_allocate_slot_bitmap_data_alloc(
    hsa_region_t region, size_t size)
{
  const size_t align = 64;
  void *memory = hostrpc::hsa::allocate(region.handle, align, size);
  return hostrpc::careful_array_cast<_Atomic uint64_t>(memory, size);
}

inline void hsa_allocate_slot_bitmap_data_free(_Atomic uint64_t *d)
{
  hostrpc::hsa::deallocate(static_cast<void *>(d));
}

}  // namespace

#endif

template <typename SZ>
struct x64_amdgcn_pair
{
  using client_type = hostrpc::x64_amdgcn_client<SZ>;
  using server_type = hostrpc::x64_amdgcn_server<SZ>;
  client_type client;
  server_type server;
  SZ sz;

  x64_amdgcn_pair(SZ sz, uint64_t fine_handle, uint64_t coarse_handle) : sz(sz)
  {
#if defined(__x86_64__)
    size_t N = sz.N();
    hsa_region_t fine = {.handle = fine_handle};
    hsa_region_t coarse = {.handle = coarse_handle};

    hostrpc::page_t *client_buffer = hostrpc::careful_array_cast<page_t>(
        hostrpc::hsa::allocate(fine_handle, alignof(page_t),
                               N * sizeof(page_t)),
        N);

    hostrpc::page_t *server_buffer = client_buffer;

    // Put the buffer in a known-good state to begin with
    for (size_t i = 0; i < N; i++)
      {
        x64_host_amdgcn_client::clear::call(&client_buffer[i], nullptr);
      }

    auto *send_data = hsa_allocate_slot_bitmap_data_alloc(fine, N);
    auto *recv_data = hsa_allocate_slot_bitmap_data_alloc(fine, N);
    auto *client_active_data = hsa_allocate_slot_bitmap_data_alloc(coarse, N);
    auto *server_active_data = hsa_allocate_slot_bitmap_data_alloc(fine, N);

    slot_bitmap_all_svm send = {N, send_data};
    slot_bitmap_all_svm recv = {N, recv_data};
    slot_bitmap_device client_active = {N, client_active_data};
    slot_bitmap_device server_active = {N, server_active_data};

    client = {sz, recv, send, client_active, server_buffer, client_buffer};

    server = {sz, send, recv, server_active, client_buffer, server_buffer};
#else
    (void)fine_handle;
    (void)coarse_handle;
#endif
  }

  ~x64_amdgcn_pair()
  {
#if defined(__x86_64__)
    assert(client.inbox.data() == server.outbox.data());
    assert(client.outbox.data() == server.inbox.data());

    hsa_allocate_slot_bitmap_data_free(client.inbox.data());
    hsa_allocate_slot_bitmap_data_free(client.outbox.data());
    hsa_allocate_slot_bitmap_data_free(client.active.data());
    hsa_allocate_slot_bitmap_data_free(server.active.data());

    assert(client.local_buffer == server.remote_buffer);
    assert(client.remote_buffer == server.local_buffer);

    if (client.local_buffer == client.remote_buffer)
      {
        hsa_memory_free(client.local_buffer);
      }
    else
      {
        hsa_memory_free(client.local_buffer);
        hsa_memory_free(server.local_buffer);
      }
#endif
  }
};

using SZ = hostrpc::size_compiletime<hostrpc::x64_host_amdgcn_array_size>;
using ty = x64_amdgcn_pair<SZ>;

hostcall_interface_t::hostcall_interface_t(uint64_t hsa_region_t_fine_handle,
                                           uint64_t hsa_region_t_coarse_handle)
{
  state = nullptr;
#if defined(__x86_64__)
  SZ sz;
  ty *s = new (std::nothrow)
      ty(sz, hsa_region_t_fine_handle, hsa_region_t_coarse_handle);
  state = static_cast<void *>(s);
#else
  (void)hsa_region_t_fine_handle;
  (void)hsa_region_t_coarse_handle;
#endif
}

hostcall_interface_t::~hostcall_interface_t()
{
#if defined(__x86_64__)
  ty *s = static_cast<ty *>(state);
  if (s)
    {
      // Should probably call the destructors on client/server state here
      delete s;
    }
#endif
}

bool hostcall_interface_t::valid() { return state != nullptr; }

hostcall_interface_t::client_t hostcall_interface_t::client()
{
  using res_t = hostcall_interface_t::client_t;
  static_assert(res_t::state_t::size() == sizeof(ty::client_type), "");
  static_assert(res_t::state_t::align() == alignof(ty::client_type), "");

  ty *s = static_cast<ty *>(state);
  assert(s);
  res_t res;
  auto *cl = res.state.construct<ty::client_type>(s->client);
  (void)cl;
  assert(cl == res.state.open<ty::client_type>());
  return res;
}

hostcall_interface_t::server_t hostcall_interface_t::server()
{
  // Construct an opaque server_t into the aligned state field
  using res_t = hostcall_interface_t::server_t;
  static_assert(res_t::state_t::size() == sizeof(ty::server_type), "");
  static_assert(res_t::state_t::align() == alignof(ty::server_type), "");

  ty *s = static_cast<ty *>(state);
  assert(s);
  res_t res;
  auto *sv = res.state.construct<ty::server_type>(s->server);
  (void)sv;
  assert(sv == res.state.open<ty::server_type>());
  return res;
}

bool hostcall_interface_t::client_t::invoke_impl(void *f, void *u)
{
#if defined(__AMDGCN__)
  auto *cl = state.open<ty::client_type>();
  return cl->rpc_invoke<true>(f, u);
#else
  (void)f;
  (void)u;
  return false;
#endif
}

bool hostcall_interface_t::client_t::invoke_async_impl(void *f, void *u)
{
#if defined(__AMDGCN__)
  auto *cl = state.open<ty::client_type>();
  return cl->rpc_invoke<true>(f, u);
#else
  (void)f;
  (void)u;
  return false;
#endif
}

bool hostcall_interface_t::server_t::handle_impl(void *application_state,
                                                 uint64_t *l)
{
#if defined(__x86_64__)
  auto *se = state.open<ty::server_type>();
  return se->rpc_handle(application_state, l);
#else
  (void)application_state;
  (void)l;
  return false;
#endif
}

}  // namespace hostrpc
