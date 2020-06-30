#ifndef INTERFACE_HPP_INCLUDED
#define INTERFACE_HPP_INCLUDED

#include "base_types.hpp"
#include <stddef.h>
#include <stdint.h>

namespace hostrpc
{
// Lifecycle management is tricky for objects which are allocated on one system
// and copied to another, where they contain pointers into each other.
// One owning object is created. If successful, that can construct instances of
// a client or server class. These can be copied by memcpy, which is necessary
// to set up the instance across pcie. The client/server objects don't own
// memory so can be copied at will. They can be used until the owning instance
// destructs.

// Notes on the legality of the char state[] handling and aliasing.
// Constructing an instance into state[] is done with placement new, which needs
// the header <new> that is unavailable for amdgcn at present. Following
// libunwind's solution discussed at D57455, operator new is added as a member
// function to client_impl, server_impl. Combined with a reinterpret cast to
// select the right operator new, that creates the object. Access is via
// std::launder'ed reinterpret cast, but as one can't assume C++17 and doesn't
// have <new> for amdgcn, this uses __builtin_launder.

template <typename T>
struct client_invoke_overloads
{
  template <typename Fill, typename Use>
  bool invoke(Fill f, Use u)
  {
    auto cbf = [](hostrpc::page_t *page, void *vf) {
      Fill *f = static_cast<Fill *>(vf);
      (*f)(page);
    };
    auto cbu = [](hostrpc::page_t *page, void *vf) {
      Use *f = static_cast<Use *>(vf);
      (*f)(page);
    };
    return derived().invoke(cbf, static_cast<void *>(&f), cbu,
                            static_cast<void *>(&u));
  }

  template <typename Fill, typename Use>
  bool invoke_async(Fill f, Use u)
  {
    auto cbf = [](hostrpc::page_t *page, void *vf) {
      Fill *f = static_cast<Fill *>(vf);
      (*f)(page);
    };
    auto cbu = [](hostrpc::page_t *page, void *vf) {
      Use *f = static_cast<Use *>(vf);
      (*f)(page);
    };
    return derived().invoke_async(cbf, static_cast<void *>(&f), cbu,
                                  static_cast<void *>(&u));
  }

 private:
  friend T;
  T &derived() { return *static_cast<T *>(this); }

  template <typename ClientType>
  bool invoke(closure_func_t fill, void *fill_state, closure_func_t use,
              void *use_state)
  {
    hostrpc::closure_pair fill_arg = {.func = fill, .state = fill_state};
    hostrpc::closure_pair use_arg = {.func = use, .state = use_state};
    auto *cl = derived().state.template open<ClientType>();
    return cl->template rpc_invoke<true>(static_cast<void *>(&fill_arg),
                                         static_cast<void *>(&use_arg));
  }

  template <typename ClientType>
  bool invoke_async(closure_func_t fill, void *fill_state, closure_func_t use,
                    void *use_state)
  {
    hostrpc::closure_pair fill_arg = {.func = fill, .state = fill_state};
    hostrpc::closure_pair use_arg = {.func = use, .state = use_state};
    auto *cl = derived().state.template open<ClientType>();
    return cl->template rpc_invoke<false>(static_cast<void *>(&fill_arg),
                                          static_cast<void *>(&use_arg));
  }
};

template <typename T>
struct server_handle_overloads
{
  template <typename Func>
  bool handle(Func f, uint64_t *loc)
  {
    auto cb = [](hostrpc::page_t *page, void *vf) {
      Func *f = static_cast<Func *>(vf);
      (*f)(page);
    };
    return derived().handle(cb, static_cast<void *>(&f), loc);
  }

 private:
  friend T;
  T &derived() { return *static_cast<T *>(this); }

  template <typename ServerType>
  bool handle(closure_func_t func, void *application_state, uint64_t *l)
  {
    hostrpc::closure_pair arg = {.func = func, .state = application_state};
    auto *se = derived().state.template open<ServerType>();
    return se->rpc_handle(static_cast<void *>(&arg), l);
  }
};

struct x64_x64_t
{
  // This probably can't be copied, but could be movable
  x64_x64_t(size_t minimum_number_slots);  // != 0
  ~x64_x64_t();
  x64_x64_t(const x64_x64_t &) = delete;
  bool valid();  // true if construction succeeded

  struct client_t : public client_invoke_overloads<client_t>
  {
    friend struct x64_x64_t;
    friend client_invoke_overloads<client_t>;
    client_t() {}  // would like this to be private

    using state_t = hostrpc::storage<48, 8>;
    using client_invoke_overloads::invoke;
    using client_invoke_overloads::invoke_async;

   private:
    template <typename ClientType>
    client_t(ClientType ct)
    {
      static_assert(state_t::size() == sizeof(ClientType), "");
      static_assert(state_t::align() == alignof(ClientType), "");
      auto *cv = state.construct<ClientType>(ct);
      assert(cv == state.open<ClientType>());
    }

    bool invoke(closure_func_t fill, void *fill_state, closure_func_t use,
                void *use_state);
    bool invoke_async(closure_func_t fill, void *fill_state, closure_func_t use,
                      void *use_state);
    state_t state;
  };

  struct server_t : public server_handle_overloads<server_t>
  {
    friend struct x64_x64_t;
    friend server_handle_overloads<server_t>;
    server_t() {}

    using state_t = hostrpc::storage<48, 8>;
    using server_handle_overloads::handle;

   private:
    template <typename ServerType>
    server_t(ServerType st)
    {
      static_assert(state_t::size() == sizeof(ServerType), "");
      static_assert(state_t::align() == alignof(ServerType), "");
      auto *sv = state.construct<ServerType>(st);
      assert(sv == state.open<ServerType>());
    }
    state_t state;
    bool handle(closure_func_t operate, void *state, uint64_t *loc);
  };

  client_t client();
  server_t server();

 private:
  void *state;
};

struct x64_gcn_t
{
  x64_gcn_t(size_t minimum_number_slots, uint64_t hsa_region_t_fine_handle,
            uint64_t hsa_region_t_coarse_handle);

  ~x64_gcn_t();
  x64_gcn_t(const x64_gcn_t &) = delete;
  bool valid();  // true if construction succeeded

  struct client_t
  {
    friend struct x64_gcn_t;
    client_t() {}  // would like this to be private

    using state_t = hostrpc::storage<48, 8>;

    // Lost the friendly interface in favour of hard coding memcpy
    // as part of debugging nullptr deref, hope to reinstate.
    void invoke(hostrpc::page_t *);
    void invoke_async(hostrpc::page_t *);

   private:
    template <typename ClientType>
    client_t(ClientType ct)
    {
      static_assert(state_t::size() == sizeof(ClientType), "");
      static_assert(state_t::align() == alignof(ClientType), "");
      auto *cv = state.construct<ClientType>(ct);
      assert(cv == state.open<ClientType>());
    }

   public:
    state_t state;
  };

  struct server_t : public server_handle_overloads<server_t>
  {
    friend struct x64_gcn_t;
    friend server_handle_overloads<server_t>;
    server_t() {}

    using state_t = hostrpc::storage<48, 8>;
    using server_handle_overloads::handle;

   private:
    template <typename ServerType>
    server_t(ServerType st)
    {
      static_assert(state_t::size() == sizeof(ServerType), "");
      static_assert(state_t::align() == alignof(ServerType), "");
      auto *sv = state.construct<ServerType>(st);
      assert(sv == state.open<ServerType>());
    }
    state_t state;
    bool handle(closure_func_t operate, void *state, uint64_t *loc);
  };

  client_t client();
  server_t server();

 private:
  void *state;
};

}  // namespace hostrpc

#endif
