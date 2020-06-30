#ifndef HOSTRPC_SERVER_HPP_INCLUDED
#define HOSTRPC_SERVER_HPP_INCLUDED

#include "common.hpp"

namespace hostrpc
{
struct operate_nop
{
  static void call(page_t*, void*) {}
};
struct clear_nop
{
  static void call(page_t*, void*) {}
};

enum class server_state : uint8_t
{
  // inbox outbox active
  idle_server = 0b000,
  idle_thread = 0b001,
  garbage_available = 0b010,
  garbage_with_thread = 0b011,
  work_available = 0b100,
  work_with_thread = 0b101,
  result_available = 0b110,
  result_with_thread = 0b111,
};

template <typename SZ, typename Copy, typename Op, typename Clear,
          typename Step>
struct server_impl : public SZ
{
  using inbox_t = slot_bitmap_all_svm;
  using outbox_t = slot_bitmap_all_svm;
  using locks_t = slot_bitmap_device;

  server_impl(SZ sz, inbox_t inbox, outbox_t outbox, locks_t active,
              page_t* remote_buffer, page_t* local_buffer)
      : SZ{sz},

        remote_buffer(remote_buffer),
        local_buffer(local_buffer),

        inbox(inbox),
        outbox(outbox),
        active(active)
  {
  }

  server_impl()
      : SZ{0},
        remote_buffer(nullptr),
        local_buffer(nullptr),
        inbox{},
        outbox{},
        active{}
  {
  }

  static void* operator new(size_t, server_impl* p) { return p; }

  void step(int x, void* y) { Step::call(x, y); }

  void dump_word(size_t size, uint64_t word)
  {
    uint64_t i = inbox.load_word(size, word);
    uint64_t o = outbox.load_word(size, word);
    uint64_t a = active.load_word(size, word);
    (void)(i + o + a);
    printf("%lu %lu %lu\n", i, o, a);
  }

  size_t find_candidate_server_available_bitmap(uint64_t w, uint64_t mask)
  {
    const size_t size = this->size();
    uint64_t i = inbox.load_word(size, w);
    uint64_t o = outbox.load_word(size, w);
    uint64_t a = active.load_word(size, w);
    __c11_atomic_thread_fence(__ATOMIC_ACQUIRE);

    uint64_t work = i & ~o;
    uint64_t garbage = ~i & o;
    uint64_t todo = work | garbage;
    uint64_t available = todo & ~a & mask;
    return available;
  }

  size_t find_candidate_server_slot(uint64_t w, uint64_t mask)
  {
    uint64_t available = find_candidate_server_available_bitmap(w, mask);
    if (available != 0)
      {
        return 64 * w + detail::ctz64(available);
      }
    return SIZE_MAX;
  }

  // return true if no garbage (briefly) during call
  void try_garbage_collect_word_server(size_t size, uint64_t w)
  {
    auto c = [](uint64_t i, uint64_t o) -> uint64_t { return ~i & o; };
    try_garbage_collect_word<decltype(c)>(size, c, inbox, outbox, active, w);
  }

  size_t words() { return size() / 64; }

  // may want to rename this, number-slots?
  size_t size() { return SZ::N(); }
  __attribute__((always_inline)) bool rpc_handle_given_slot(
      void* application_state, size_t slot)
  {
    assert(slot != SIZE_MAX);

    const uint64_t element = index_to_element(slot);
    const uint64_t subindex = index_to_subindex(slot);

    auto lock_held = [&]() -> bool {
      return detail::nthbitset64(active.load_word(size(), element), subindex);
    };
    (void)lock_held;

    const size_t size = this->size();
    cache c;
    c.init(slot);

    uint64_t i = inbox.load_word(size, element);
    uint64_t o = outbox.load_word(size, element);
    uint64_t a = active.load_word(size, element);
    __c11_atomic_thread_fence(__ATOMIC_ACQUIRE);
    c.i = i;
    c.o = o;
    c.a = a;

    // Called with a lock. The corresponding slot can be:
    //  inbox outbox    state  action
    //      0      0     idle    none
    //      0      1  garbage collect
    //      1      0     work    work
    //      1      1  waiting    none

    uint64_t this_slot = detail::setnthbit64(0, subindex);
    uint64_t work_todo = (i & ~o) & this_slot;
    uint64_t garbage_todo = (~i & o) & this_slot;

    assert((work_todo & garbage_todo) == 0);  // disjoint
    assert(lock_held());

    if (garbage_todo)
      {
        assert((o & this_slot) != 0);

        // Move data and clear. TODO: Elide the copy for nop clear
        Copy::pull_to_server_from_client((void*)&local_buffer[slot],
                                         (void*)&remote_buffer[slot],
                                         sizeof(page_t));
        step(__LINE__, application_state);
        Clear::call(&local_buffer[slot], application_state);
        step(__LINE__, application_state);
        Copy::push_from_server_to_client((void*)&remote_buffer[slot],
                                         (void*)&local_buffer[slot],
                                         sizeof(page_t));

        __c11_atomic_thread_fence(__ATOMIC_RELEASE);
        uint64_t updated_out = platform::critical<uint64_t>([&]() {
          return outbox.release_slot_returning_updated_word(size, slot);
        });

        assert((updated_out & this_slot) == 0);
        (void)updated_out;

        assert(lock_held());
        return false;
      }

    if (!work_todo)
      {
        step(__LINE__, application_state);
        assert(lock_held());
        return false;
      }

    assert(c.is(0b101));
    step(__LINE__, application_state);
    tracker().claim(slot);

    // make the calls
    Copy::pull_to_server_from_client((void*)&local_buffer[slot],
                                     (void*)&remote_buffer[slot],
                                     sizeof(page_t));
    step(__LINE__, application_state);
    Op::call(&local_buffer[slot], application_state);
    step(__LINE__, application_state);
    Copy::push_from_server_to_client((void*)&remote_buffer[slot],
                                     (void*)&local_buffer[slot],
                                     sizeof(page_t));
    step(__LINE__, application_state);

    assert(c.is(0b101));

    tracker().release(slot);

    // publish result
    {
      __c11_atomic_thread_fence(__ATOMIC_RELEASE);
      uint64_t o = platform::critical<uint64_t>([&]() {
        return outbox.claim_slot_returning_updated_word(size, slot);
      });
      c.o = o;
    }
    assert(c.is(0b111));
    // leaves outbox live
    assert(lock_held());
    return true;
  }

  // Returns true if it handled one task. Does not attempt multiple tasks

  __attribute__((always_inline)) bool rpc_handle(
      void* application_state) noexcept
  {
    uint64_t location = 0;
    return rpc_handle(application_state, &location);
  }

  // location != NULL, used to round robin across slots
  __attribute__((always_inline)) bool rpc_handle(
      void* application_state, uint64_t* location_arg) noexcept
  {
    step(__LINE__, application_state);
    const size_t size = this->size();
    const size_t words = size / 64;
    // garbage collection should be fairly cheap when there is none,
    // and the presence of any occupied slots can starve the client
    for (uint64_t w = 0; w < words; w++)
      {
        // try_garbage_collect_word_server(size, w);
      }

    step(__LINE__, application_state);

    const uint64_t location = *location_arg % size;
    const uint64_t element = index_to_element(location);

    // skip bits in the first word <= subindex
    uint64_t mask = detail::setbitsrange64(index_to_subindex(location), 63);

    // Tries a few bits in element, then all bits in all the other words, then
    // all bits in element. This overshoots somewhat but ensures that all slots
    // are checked. Could truncate the last word to check each slot exactly once
    for (uint64_t wc = 0; wc < words + 1; wc++)
      {
        uint64_t w = (element + wc) % words;
        uint64_t available = find_candidate_server_available_bitmap(w, mask);
        while (available != 0)
          {
            uint64_t idx = detail::ctz64(available);
            assert(detail::nthbitset64(available, idx));
            uint64_t slot = 64 * w + idx;
            uint64_t active_word;
            if (active.try_claim_empty_slot(size, slot, &active_word))
              {
                // Success, got the lock. Aim location_arg at next slot
                assert(active_word != 0);
                *location_arg = slot + 1;

                bool r = rpc_handle_given_slot(application_state, slot);

                step(__LINE__, application_state);

                uint64_t a = platform::critical<uint64_t>([&]() {
                  return active.release_slot_returning_updated_word(size, slot);
                });
                assert(!detail::nthbitset64(a, index_to_subindex(slot)));
                (void)a;

                return r;
              }

            // don't try the same slot repeatedly
            available = detail::clearnthbit64(available, idx);
          }

        mask = UINT64_MAX;
      }

    // Nothing hit, may as well go from the same location on the next call
    step(__LINE__, application_state);
    return false;
  }

  page_t* remote_buffer;
  page_t* local_buffer;
  inbox_t inbox;
  outbox_t outbox;
  locks_t active;
};

namespace indirect
{
struct operate
{
  static void call(hostrpc::page_t* page, void* pv)
  {
    hostrpc::closure_pair* p = static_cast<hostrpc::closure_pair*>(pv);
    p->func(page, p->state);
  }
};
struct clear
{
  static void call(hostrpc::page_t* page, void* pv)
  {
    hostrpc::closure_pair* p = static_cast<hostrpc::closure_pair*>(pv);
    p->func(page, p->state);
  }
};
}  // namespace indirect

template <typename SZ, typename Copy, typename Step>
using server_indirect_impl =
    server_impl<SZ, Copy, indirect::operate, indirect::clear, Step>;

}  // namespace hostrpc
#endif
