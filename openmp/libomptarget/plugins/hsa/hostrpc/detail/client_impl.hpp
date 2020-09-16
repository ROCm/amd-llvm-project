#ifndef HOSTRPC_CLIENT_HPP_INCLUDED
#define HOSTRPC_CLIENT_HPP_INCLUDED

#include "common.hpp"
// Intend to have call and service working across gcn and x86
// The normal terminology is:
// Client makes a call to the server, which does some work and sends back a
// reply

namespace hostrpc
{
struct fill_nop
{
  static void call(page_t*, void*) {}
};

struct use_nop
{
  static void call(page_t*, void*) {}
};

enum class client_state : uint8_t
{
  // inbox outbox active
  idle_client = 0b000,
  active_thread = 0b001,
  work_available = 0b011,
  async_work_available = 0b010,
  done_pending_server_gc =
      0b100,  // waiting for server to garbage collect, no local thread
  garbage_with_thread = 0b101,  // transient state, 0b100 with local thread
  done_pending_client_gc =
      0b110,                 // created work, result available, no continuation
  result_available = 0b111,  // thread waiting
};

// if inbox is set and outbox not, we are waiting for the server to collect
// garbage that is, can't claim the slot for a new thread is that a sufficient
// criteria for the slot to be awaiting gc?

namespace counters
{
// client_nop compiles to no code
// both are default-constructed

struct client
{
  // Probably want this in the interface, partly to keep size
  // lined up (this will be multiple words)
  client() = default;
  client(const client& o) = default;
  client& operator=(const client& o) = default;

  void no_candidate_slot()
  {
    inc(&state[client_counters::cc_no_candidate_slot]);
  }
  void missed_lock_on_candidate_slot()
  {
    inc(&state[client_counters::cc_missed_lock_on_candidate_slot]);
  }
  void got_lock_after_work_done()
  {
    inc(&state[client_counters::cc_got_lock_after_work_done]);
  }
  void waiting_for_result()
  {
    inc(&state[client_counters::cc_waiting_for_result]);
  }
  void cas_lock_fail(uint64_t c)
  {
    add(&state[client_counters::cc_cas_lock_fail], c);
  }
  void garbage_cas_fail(uint64_t c)
  {
    add(&state[client_counters::cc_garbage_cas_fail], c);
  }
  void publish_cas_fail(uint64_t c)
  {
    add(&state[client_counters::cc_publish_cas_fail], c);
  }
  void finished_cas_fail(uint64_t c)
  {
    // triggers an infinite loop on amdgcn trunk but not amd-stg-open
    add(&state[client_counters::cc_finished_cas_fail], c);
  }

  void garbage_cas_help(uint64_t c)
  {
    add(&state[client_counters::cc_garbage_cas_help], c);
  }
  void publish_cas_help(uint64_t c)
  {
    add(&state[client_counters::cc_publish_cas_help], c);
  }
  void finished_cas_help(uint64_t c)
  {
    add(&state[client_counters::cc_finished_cas_help], c);
  }

  // client_counters contains non-atomic, const version of this state
  // defined in base_types
  client_counters get()
  {
    __c11_atomic_thread_fence(__ATOMIC_RELEASE);
    client_counters res;
    for (unsigned i = 0; i < client_counters::cc_total_count; i++)
      {
        res.state[i] = state[i];
      }
    return res;
  }

 private:
  _Atomic uint64_t state[client_counters::cc_total_count] = {0u};

  static void add(_Atomic uint64_t* addr, uint64_t v)
  {
    if (platform::is_master_lane())
      {
        __opencl_atomic_fetch_add(addr, v, __ATOMIC_RELAXED,
                                  __OPENCL_MEMORY_SCOPE_DEVICE);
      }
  }

  static void inc(_Atomic uint64_t* addr)
  {
    uint64_t v = 1;
    add(addr, v);
  }
};

struct client_nop
{
  client_nop() {}
  client_counters get() { return {}; }

  void no_candidate_slot() {}
  void missed_lock_on_candidate_slot() {}
  void got_lock_after_work_done() {}
  void waiting_for_result() {}
  void cas_lock_fail(uint64_t) {}

  void garbage_cas_fail(uint64_t) {}
  void publish_cas_fail(uint64_t) {}
  void finished_cas_fail(uint64_t) {}
  void garbage_cas_help(uint64_t) {}
  void publish_cas_help(uint64_t) {}
  void finished_cas_help(uint64_t) {}
};

}  // namespace counters

// enabling counters breaks codegen for amdgcn,
template <typename SZ, typename Copy, typename Fill, typename Use,
          typename Step, typename Counter = counters::client>
struct client_impl : public SZ, public Counter
{
  using inbox_t = slot_bitmap_all_svm;
  using outbox_t = slot_bitmap_all_svm;
  using locks_t = slot_bitmap_device;
  using outbox_staging_t = slot_bitmap_coarse;

  client_impl(SZ sz, inbox_t inbox, outbox_t outbox, locks_t active,
              outbox_staging_t outbox_staging, page_t* remote_buffer,
              page_t* local_buffer)

      : SZ{sz},
        Counter{},
        remote_buffer(remote_buffer),
        local_buffer(local_buffer),
        inbox(inbox),
        outbox(outbox),
        active(active),
        outbox_staging(outbox_staging)
  {
    constexpr size_t client_size = 48;

    // SZ is expected to be zero bytes or a uint64_t
    struct SZ_local : public SZ
    {
      float x;
    };
    // Counter is zero bytes for nop or potentially many
    struct Counter_local : public Counter
    {
      float x;
    };
    constexpr bool SZ_empty = sizeof(SZ_local) == sizeof(float);
    constexpr bool Counter_empty = sizeof(Counter_local) == sizeof(float);

    constexpr size_t SZ_size = SZ_empty ? 0 : sizeof(SZ);
    constexpr size_t Counter_size = Counter_empty ? 0 : sizeof(Counter);

    constexpr size_t total_size = client_size + SZ_size + Counter_size;

    static_assert(sizeof(client_impl) == total_size, "");
    static_assert(alignof(client_impl) == 8, "");
  }

  client_impl()
      : SZ{0},
        Counter{},
        remote_buffer(nullptr),
        local_buffer(nullptr),
        inbox{},
        outbox{},
        active{},
        outbox_staging{}
  {
  }

  static void* operator new(size_t, client_impl* p) { return p; }

  void step(int x, void* y, void* z)
  {
    Step::call(x, y);
    Step::call(x, z);
  }

  client_counters get_counters() { return Counter::get(); }

  size_t size() { return SZ::N(); }
  size_t words() { return size() / 64; }

  size_t find_candidate_client_slot(uint64_t w)
  {
    uint64_t i = inbox.load_word(size(), w);
    uint64_t o = outbox_staging.load_word(size(), w);
    uint64_t a = active.load_word(size(), w);
    __c11_atomic_thread_fence(__ATOMIC_ACQUIRE);

    // inbox == outbox == 0 => available for use
    uint64_t available = ~i & ~o & ~a;

    // 1 0 => garbage waiting on server
    // 1 1 => garbage that client can act on
    // Take those that client can act on and are not locked
    uint64_t garbage_todo = i & o & ~a;

    // could also let through inbox == 1 on the basis that
    // the client may have

    uint64_t candidate = available | garbage_todo;
    if (candidate != 0)
      {
        return 64 * w + detail::ctz64(candidate);
      }

    return SIZE_MAX;
  }

  void dump_word(size_t size, uint64_t word)
  {
    uint64_t i = inbox.load_word(size, word);
    uint64_t o = outbox_staging.load_word(size, word);
    uint64_t a = active.load_word(size, word);
    (void)(i + o + a);
    printf("%lu %lu %lu\n", i, o, a);
  }

  // true if it successfully made a call, false if no work to do or only gc
  // If there's no continuation, shouldn't require a use_application_state
  template <bool have_continuation>
  bool rpc_invoke_given_slot(void* fill_application_state,
                             void* use_application_state, size_t slot) noexcept
  {
    assert(slot != SIZE_MAX);
    const uint64_t element = index_to_element(slot);
    const uint64_t subindex = index_to_subindex(slot);

    cache c;
    c.init(slot);
    const size_t size = this->size();
    uint64_t i = inbox.load_word(size, element);
    uint64_t o = outbox_staging.load_word(size, element);
    uint64_t a = active.load_word(size, element);
    __c11_atomic_thread_fence(__ATOMIC_ACQUIRE);
    c.i(i);
    c.o(o);
    c.a(a);

    // Called with a lock. The corresponding slot can be:
    //  inbox outbox    state  action
    //      0      0     work    work
    //      0      1     done    none
    //      1      0  garbage    none (waiting on server)
    //      1      1  garbage   clean
    // Inbox true means the result has come back
    // That this lock has been taken means no other thread is
    // waiting for that result
    uint64_t this_slot = detail::setnthbit64(0, subindex);
    uint64_t garbage_todo = i & o & this_slot;
    uint64_t available = ~i & ~o & this_slot;

    assert((garbage_todo & available) == 0);  // disjoint

    if (garbage_todo)
      {
        __c11_atomic_thread_fence(__ATOMIC_RELEASE);
        uint64_t cas_fail_count = 0;
        uint64_t cas_help_count = 0;
        platform::critical<uint64_t>([&]() {
          return staged_release_slot_returning_updated_word(
              size, slot, &outbox_staging, &outbox, &cas_fail_count,
              &cas_help_count);
          // outbox.release_slot_returning_updated_word(size, slot);
        });
        cas_fail_count = platform::broadcast_master(cas_fail_count);
        cas_help_count = platform::broadcast_master(cas_help_count);
        Counter::garbage_cas_fail(cas_fail_count);
        Counter::garbage_cas_help(cas_help_count);
        return false;
      }

    if (!available)
      {
        Counter::got_lock_after_work_done();
        step(__LINE__, fill_application_state, use_application_state);
        return false;
      }

    assert(c.is(0b001));
    step(__LINE__, fill_application_state, use_application_state);
    tracker().claim(slot);

    // wave_populate

    // Fill may have no precondition, in which case this doesn't need to run
    Copy::pull_to_client_from_server(&local_buffer[slot], &remote_buffer[slot]);
    step(__LINE__, fill_application_state, use_application_state);
    Fill::call(&local_buffer[slot], fill_application_state);
    step(__LINE__, fill_application_state, use_application_state);
    Copy::push_from_client_to_server(&remote_buffer[slot], &local_buffer[slot]);
    step(__LINE__, fill_application_state, use_application_state);

    tracker().release(slot);

    // wave_publish work
    {
      __c11_atomic_thread_fence(__ATOMIC_RELEASE);
      uint64_t cas_fail_count = 0;
      uint64_t cas_help_count = 0;
      uint64_t o = platform::critical<uint64_t>([&]() {
        return staged_claim_slot_returning_updated_word(
            size, slot, &outbox_staging, &outbox, &cas_fail_count,
            &cas_help_count);
        // return outbox.claim_slot_returning_updated_word(size, slot);
      });
      cas_fail_count = platform::broadcast_master(cas_fail_count);
      cas_help_count = platform::broadcast_master(cas_help_count);
      Counter::publish_cas_fail(cas_fail_count);
      Counter::publish_cas_help(cas_help_count);
      c.o(o);
      assert(detail::nthbitset64(o, subindex));
      assert(c.is(0b011));
    }

    step(__LINE__, fill_application_state, use_application_state);

    // current strategy is drop interest in the slot, then wait for the
    // server to confirm, then drop local thread

    // with a continuation, outbox is cleared before this thread returns
    // otherwise, garbage collection needed to clear that outbox

    if (have_continuation)
      {
        // wait for H1, result available
        uint64_t loaded = 0;

        while (true)
          {
            uint32_t got = platform::critical<uint32_t>(
                [&]() { return inbox(size, slot, &loaded); });

            loaded = platform::broadcast_master(loaded);

            c.i(loaded);
            assert(got == 1 ? c.is(0b111) : c.is(0b011));
            if (got == 1)
              {
                break;
              }

            Counter::waiting_for_result();

            // make this spin slightly cheaper
            // todo: can the client do useful work while it waits? e.g. gc?
            // need to avoid taking too many locks at a time given forward
            // progress which makes gc tricky
            // could attempt to propagate the current word from staging to
            // outbox - that's safe because a lock is held, maintaining linear
            // time - but may conflict with other clients trying to do the same
            platform::sleep();
          }

        __c11_atomic_thread_fence(__ATOMIC_ACQUIRE);

        assert(c.is(0b111));
        tracker().claim(slot);

        step(__LINE__, fill_application_state, use_application_state);
        Copy::pull_to_client_from_server(&local_buffer[slot],
                                         &remote_buffer[slot]);
        step(__LINE__, fill_application_state, use_application_state);
        // call the continuation
        Use::call(&local_buffer[slot], use_application_state);

        step(__LINE__, fill_application_state, use_application_state);

        // Copying the state back to the server is a nop for aliased case,
        // and is only necessary if the server has a non-nop garbage clear
        // callback
        Copy::push_from_client_to_server(&remote_buffer[slot],
                                         &local_buffer[slot]);

        step(__LINE__, fill_application_state, use_application_state);

        tracker().release(slot);

        // mark the work as no longer in use
        // todo: is it better to leave this for the GC?
        // can free slots more lazily by updating the staging outbox and
        // leaving the visible one. In that case the update may be transfered
        // for free, or it may never become visible in which case the server
        // won't realise the slot is no longer in use
        __c11_atomic_thread_fence(__ATOMIC_RELEASE);
        uint64_t cas_fail_count = 0;
        uint64_t cas_help_count = 0;
        uint64_t o = platform::critical<uint64_t>([&]() {
          return staged_release_slot_returning_updated_word(
              size, slot, &outbox_staging, &outbox, &cas_fail_count,
              &cas_help_count);
          // return outbox.release_slot_returning_updated_word(size, slot);
        });
        cas_fail_count = platform::broadcast_master(cas_fail_count);
        cas_help_count = platform::broadcast_master(cas_help_count);
        Counter::finished_cas_fail(cas_fail_count);
        Counter::finished_cas_help(cas_help_count);
        c.o(o);
        assert(c.is(0b101));

        step(__LINE__, fill_application_state, use_application_state);
      }

    // if we don't have a continuation, would return on 0b010
    // this wouldn't be considered garbage by client as inbox is clear
    // the server gets 0b100, does the work, sets the result to 0b110
    // that is then picked up by the client as 0b110

    // wait for H0, result has been garbage collected by the host
    // todo: want to get rid of this busy spin in favour of deferred collection
    // I think that will need an extra client side bitmap

    // We could wait for inbox[slot] != 0 which indicates the result
    // has been garbage collected, but that stalls the wave waiting for the hose
    // Instead, drop the warp and let the allocator skip occupied inbox slots
    return true;
  }

  // Returns true if it successfully launched the task
  template <bool have_continuation>
  bool rpc_invoke(void* fill_application_state,
                  void* use_application_state) noexcept
  {
    step(__LINE__, fill_application_state, use_application_state);

    const size_t size = this->size();
    const size_t words = size / 64;
    // 0b111 is posted request, waited for it, got it
    // 0b110 is posted request, nothing waited, got one
    // 0b101 is got a result, don't need it, only spun up a thread for cleanup
    // 0b100 is got a result, don't need it

    step(__LINE__, fill_application_state, use_application_state);

    size_t slot = SIZE_MAX;
    // tries each word in sequnce. A cas failing suggests contention, in which
    // case try the next word instead of the next slot
    // may be worth supporting non-zero starting word for cache locality effects

    // the array is somewhat contended - attempt to spread out the load by
    // starting clients off at different points in the array. Doesn't make an
    // observable difference in the current benchmark.

    // if the invoke call performed garbage collection, the word is not
    // known to be contended so it may be worth trying a different slot
    // before trying a different word
#define CLIENT_OFFSET 0

#if CLIENT_OFFSET
    const uint32_t wstart = platform::client_start_slot();
    const uint32_t wend = wstart + words;
    for (uint64_t wi = wstart; wi != wend; wi++)
      {
        uint64_t w =
            wi % words;  // modulo may hurt here, and probably want 32 bit iv
#else
    for (uint64_t w = 0; w < words; w++)
      {
#endif
        uint64_t active_word;
        slot = find_candidate_client_slot(w);
        if (slot == SIZE_MAX)
          {
            // no slot
            Counter::no_candidate_slot();
          }
        else
          {
            uint64_t cas_fail_count = 0;
            if (active.try_claim_empty_slot(size, slot, &active_word,
                                            &cas_fail_count))
              {
                // Success, got the lock.
                assert(active_word != 0);
                Counter::cas_lock_fail(cas_fail_count);
                bool r = rpc_invoke_given_slot<have_continuation>(
                    fill_application_state, use_application_state, slot);

                // wave release slot
                step(__LINE__, fill_application_state, use_application_state);
                platform::critical<uint64_t>([&]() {
                  return active.release_slot_returning_updated_word(size, slot);
                });
                // returning if the invoke garbage collected is inefficient
                // as the caller will need to try again, better to keep the
                // position in the loop. This raises a memory access error
                // however HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION: The
                // agent attempted to access memory beyond the largest legal
                // address.
#if 0
                if (r)
                  {
                    return true;
                  }
#else
                return r;
#endif
              }
            else
              {
                Counter::missed_lock_on_candidate_slot();
              }
          }
      }

    // couldn't get a slot, won't launch
    step(__LINE__, fill_application_state, use_application_state);
    return false;
  }

  page_t* remote_buffer;
  page_t* local_buffer;
  inbox_t inbox;
  outbox_t outbox;
  locks_t active;
  outbox_staging_t outbox_staging;
};

namespace indirect
{
struct fill
{
  static void call(hostrpc::page_t* page, void* pv)
  {
    hostrpc::closure_pair* p = static_cast<hostrpc::closure_pair*>(pv);
    p->func(page, p->state);
  };
};

struct use
{
  static void call(hostrpc::page_t* page, void* pv)
  {
    hostrpc::closure_pair* p = static_cast<hostrpc::closure_pair*>(pv);
    p->func(page, p->state);
  };
};

}  // namespace indirect

template <typename SZ, typename Copy, typename Step>
using client_indirect_impl =
    client_impl<SZ, Copy, indirect::fill, indirect::use, Step>;

}  // namespace hostrpc

#endif
