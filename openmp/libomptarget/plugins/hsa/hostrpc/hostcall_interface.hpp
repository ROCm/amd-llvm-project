#ifndef HOSTCALL_INTERFACE_HPP_INCLUDED
#define HOSTCALL_INTERFACE_HPP_INCLUDED

#include "base_types.hpp"
#include "interface.hpp"
#include <stddef.h>
#include <stdint.h>
namespace hostrpc
{
namespace client
{
template <typename T>
struct interface
{
  bool invoke(void *fill, void *use) noexcept
  {
    return derived().invoke_impl(fill, use);
  }
  bool invoke_async(void *fill, void *use) noexcept
  {
    return derived().invoke_async_impl(fill, use);
  }

 protected:
  interface() {}

 private:
  friend T;
  T &derived() { return *static_cast<T *>(this); }
  bool invoke_impl(void *, void *) { return false; }
  bool invoke_async_impl(void *, void *) { return false; }
};
}  // namespace client
namespace server
{
template <typename T>
struct interface
{
  bool handle(void *x, uint64_t *location_arg) noexcept
  {
    return derived().handle_impl(x, location_arg);
  }
  bool handle(void *x) noexcept
  {
    uint64_t loc;
    return handle(x, &loc);
  }

 protected:
  interface() {}

 private:
  friend T;
  T &derived() { return *static_cast<T *>(this); }
  bool handle_impl(void *, uint64_t *) { return false; }
};
}  // namespace server

struct hostcall_interface_t
{
  hostcall_interface_t(uint64_t hsa_region_t_fine_handle,
                       uint64_t hsa_region_t_coarse_handle);
  ~hostcall_interface_t();
  hostcall_interface_t(const hostcall_interface_t &) = delete;
  bool valid();

  struct client_t : public client::interface<client_t>
  {
    friend struct client::interface<client_t>;
    friend struct hostcall_interface_t;
    client_t() {}  // would like this to be private
    using state_t = hostrpc::storage<40, 8>;

   private:
    bool invoke_impl(void *, void *);
    bool invoke_async_impl(void *, void *);
    state_t state;
  };

  struct server_t : public server::interface<server_t>
  {
    friend struct server::interface<server_t>;
    friend struct hostcall_interface_t;
    server_t() {}
    using state_t = hostrpc::storage<40, 8>;

   private:
    bool handle_impl(void *, uint64_t *);
    state_t state;
  };

  client_t client();
  server_t server();

 private:
  void *state;
};

}  // namespace hostrpc

#endif
