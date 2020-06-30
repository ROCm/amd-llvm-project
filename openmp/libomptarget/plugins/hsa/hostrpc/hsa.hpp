#ifndef HSA_HPP_INCLUDED
#define HSA_HPP_INCLUDED

// A C++ wrapper around a subset of the hsa api
#include "hsa.h"
#include <array>
#include <cstdio>

#include <cassert>
#include <cstring>
#include <memory>
#include <string>
#include <type_traits>

namespace hsa
{
inline const char* status_string(hsa_status_t status)
{
  const char* res;
  if (hsa_status_string(status, &res) != HSA_STATUS_SUCCESS)
    {
      res = "unknown";
    }
  return res;
}

#define hsa_success_or_exit(status) \
  hsa::success_or_exit_impl(__LINE__, __FILE__, status)
inline void success_or_exit_impl(int line, const char* file,
                                 hsa_status_t status)
{
  if (status == HSA_STATUS_SUCCESS)
    {
      return;
    }
  fprintf(stderr, "HSA Failure at %s:%d (%u,%s)\n", file, line,
          (unsigned)status, status_string(status));
  exit(1);
}

struct init
{
  init() : status(hsa_init()) { hsa_success_or_exit(status); }
  ~init() { hsa_shut_down(); }
  const hsa_status_t status;
};

#if __cplusplus >= 201703L
#define requires_invocable_r(...) \
  static_assert(std::is_invocable_r<__VA_ARGS__>::value, "")
#else
#define requires_invocable_r(...) (void)0
#endif

template <typename C>
hsa_status_t iterate_agents(C cb)
{
  requires_invocable_r(hsa_status_t, C, hsa_agent_t);

  auto L = [](hsa_agent_t agent, void* data) -> hsa_status_t {
    C* unwrapped = static_cast<C*>(data);
    return (*unwrapped)(agent);
  };
  return hsa_iterate_agents(L, static_cast<void*>(&cb));
}

template <typename T, hsa_agent_info_t req>
struct agent_get_info
{
  static T call(hsa_agent_t agent)
  {
    T res;
    hsa_status_t r = hsa_agent_get_info(agent, req, static_cast<void*>(&res));
    (void)r;
    return res;
  }
};

template <hsa_agent_info_t req, typename e, size_t w>
struct agent_get_info<std::array<e, w>, req>
{
  using T = std::array<e, w>;
  static T call(hsa_agent_t agent)
  {
    T res;
    hsa_status_t rc =
        hsa_agent_get_info(agent, req, static_cast<void*>(res.data()));
    (void)rc;
    return res;
  }
};

inline std::array<char, 64> agent_get_info_name(hsa_agent_t agent)
{
  return agent_get_info<std::array<char, 64>, HSA_AGENT_INFO_NAME>::call(agent);
}

inline std::array<char, 64> agent_get_info_vendor_name(hsa_agent_t agent)
{
  return agent_get_info<std::array<char, 64>, HSA_AGENT_INFO_VENDOR_NAME>::call(
      agent);
}

inline hsa_agent_feature_t agent_get_info_feature(hsa_agent_t agent)
{
  return agent_get_info<hsa_agent_feature_t, HSA_AGENT_INFO_FEATURE>::call(
      agent);
}

inline uint32_t agent_get_info_queues_max(hsa_agent_t agent)
{
  return agent_get_info<uint32_t, HSA_AGENT_INFO_QUEUES_MAX>::call(agent);
}

inline uint32_t agent_get_info_queue_min_size(hsa_agent_t agent)
{
  return agent_get_info<uint32_t, HSA_AGENT_INFO_QUEUE_MIN_SIZE>::call(agent);
}

inline uint32_t agent_get_info_queue_max_size(hsa_agent_t agent)
{
  return agent_get_info<uint32_t, HSA_AGENT_INFO_QUEUE_MAX_SIZE>::call(agent);
}

inline hsa_queue_type32_t agent_get_info_queue_type(hsa_agent_t agent)
{
  return agent_get_info<hsa_queue_type32_t, HSA_AGENT_INFO_QUEUE_TYPE>::call(
      agent);
}

inline hsa_device_type_t agent_get_info_device(hsa_agent_t agent)
{
  return agent_get_info<hsa_device_type_t, HSA_AGENT_INFO_DEVICE>::call(agent);
}

inline std::array<uint8_t, 128> agent_get_info_extensions(hsa_agent_t agent)
{
  return agent_get_info<std::array<uint8_t, 128>,
                        HSA_AGENT_INFO_EXTENSIONS>::call(agent);
}

inline uint16_t agent_get_info_version_major(hsa_agent_t agent)
{
  return agent_get_info<uint16_t, HSA_AGENT_INFO_VERSION_MAJOR>::call(agent);
}

inline uint16_t agent_get_info_version_minor(hsa_agent_t agent)
{
  return agent_get_info<uint16_t, HSA_AGENT_INFO_VERSION_MINOR>::call(agent);
}

template <typename T, hsa_executable_symbol_info_t req>
struct symbol_get_info
{
  static T call(hsa_executable_symbol_t sym)
  {
    T res;
    hsa_status_t rc = hsa_executable_symbol_get_info(sym, req, &res);
    (void)rc;
    return res;
  }
};

inline hsa_symbol_kind_t symbol_get_info_type(hsa_executable_symbol_t sym)
{
  return symbol_get_info<hsa_symbol_kind_t,
                         HSA_EXECUTABLE_SYMBOL_INFO_TYPE>::call(sym);
}

inline uint32_t symbol_get_info_name_length(hsa_executable_symbol_t sym)
{
  return symbol_get_info<uint32_t,
                         HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH>::call(sym);
}

inline std::string symbol_get_info_name(hsa_executable_symbol_t sym)
{
  uint32_t size = symbol_get_info_name_length(sym);
  std::string res;
  res.resize(size + 1);

  hsa_status_t rc = hsa_executable_symbol_get_info(
      sym, HSA_EXECUTABLE_SYMBOL_INFO_NAME, static_cast<void*>(&res.front()));
  (void)rc;
  return res;
}

inline uint64_t symbol_get_info_variable_address(hsa_executable_symbol_t sym)
{
  // could assert that symbol kind is variable
  return symbol_get_info<
      uint64_t, HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS>::call(sym);
}

// The handle written to the kernel dispatch packet
inline uint64_t symbol_get_info_kernel_object(hsa_executable_symbol_t sym)
{
  return symbol_get_info<uint64_t,
                         HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT>::call(sym);
}

template <typename C>
hsa_status_t iterate_regions(hsa_agent_t agent, C cb)
{
  requires_invocable_r(hsa_status_t, C, hsa_region_t);

  auto L = [](hsa_region_t region, void* data) -> hsa_status_t {
    C* unwrapped = static_cast<C*>(data);
    return (*unwrapped)(region);
  };

  return hsa_agent_iterate_regions(agent, L, static_cast<void*>(&cb));
}

template <typename T, hsa_region_info_t req>
struct region_get_info
{
  static T call(hsa_region_t region)
  {
    T res;
    hsa_status_t r = hsa_region_get_info(region, req, static_cast<void*>(&res));
    (void)r;
    return res;
  }
};

#define REGION_GEN_INFO(suffix, type, req)                  \
  inline type region_get_info_##suffix(hsa_region_t region) \
  {                                                         \
    return region_get_info<type, req>::call(region);        \
  }

REGION_GEN_INFO(segment, hsa_region_segment_t, HSA_REGION_INFO_SEGMENT);
REGION_GEN_INFO(global_flags, hsa_region_global_flag_t,
                HSA_REGION_INFO_GLOBAL_FLAGS);
REGION_GEN_INFO(size, size_t, HSA_REGION_INFO_SIZE);
REGION_GEN_INFO(alloc_max_size, size_t, HSA_REGION_INFO_ALLOC_MAX_SIZE);
REGION_GEN_INFO(alloc_max_private_workgroup_size, uint32_t,
                HSA_REGION_INFO_ALLOC_MAX_PRIVATE_WORKGROUP_SIZE);
REGION_GEN_INFO(runtime_alloc_allowed, bool,
                HSA_REGION_INFO_RUNTIME_ALLOC_ALLOWED);
REGION_GEN_INFO(runtime_alloc_granule, size_t,
                HSA_REGION_INFO_RUNTIME_ALLOC_GRANULE);
REGION_GEN_INFO(runtime_alloc_alignment, size_t,
                HSA_REGION_INFO_RUNTIME_ALLOC_ALIGNMENT);

// {nullptr} on failure

namespace detail
{
template <hsa_region_global_flag_t Flag>
inline hsa_region_t global_region_with_flag(hsa_agent_t agent)
{
  hsa_region_t result;
  hsa_status_t r =
      hsa::iterate_regions(agent, [&](hsa_region_t region) -> hsa_status_t {
        hsa_region_segment_t segment = hsa::region_get_info_segment(region);
        if (segment != HSA_REGION_SEGMENT_GLOBAL)
          {
            return HSA_STATUS_SUCCESS;
          }

        if (!hsa::region_get_info_runtime_alloc_allowed(region))
          {
            return HSA_STATUS_SUCCESS;
          }

        if (hsa::region_get_info_global_flags(region) & Flag)
          {
            result = region;
            return HSA_STATUS_INFO_BREAK;
          }

        return HSA_STATUS_SUCCESS;
      });
  if (r == HSA_STATUS_INFO_BREAK)
    {
      return result;
    }
  else
    {
      return {reinterpret_cast<uint64_t>(nullptr)};
    }
}
}  // namespace detail

inline hsa_region_t region_kernarg(hsa_agent_t agent)
{
  return detail::global_region_with_flag<HSA_REGION_GLOBAL_FLAG_KERNARG>(agent);
}

inline hsa_region_t region_fine_grained(hsa_agent_t agent)
{
  return detail::global_region_with_flag<HSA_REGION_GLOBAL_FLAG_FINE_GRAINED>(
      agent);
}

inline hsa_region_t region_coarse_grained(hsa_agent_t agent)
{
  return detail::global_region_with_flag<HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED>(
      agent);
}

namespace detail
{
struct memory_deleter
{
  void operator()(void* data) { hsa_memory_free(data); }
};
}  // namespace detail
inline std::unique_ptr<void, detail::memory_deleter> allocate(
    hsa_region_t region, size_t size)
{
  void* res;
  // NB: hsa_memory_allocate is deprecated, should move to
  // hsa_amd_memory_pool_allocate with a fine grain pool
  hsa_status_t r = hsa_memory_allocate(region, size, &res);
  if (r == HSA_STATUS_SUCCESS)
    {
      return std::unique_ptr<void, detail::memory_deleter>(res);
    }
  else
    {
      return {nullptr};
    }
}

inline uint64_t sentinel() { return reinterpret_cast<uint64_t>(nullptr); }

struct executable
{
  // hsa expects executable management to be quite dynamic
  // one can load multiple shared libraries, which can probably reference
  // symbols from each other. It supports 'executable_global_variable_define'
  // which names some previously allocated memory. Or readonly equivalent. This
  // wrapper is

  operator hsa_executable_t() { return state; }

  bool valid() { return reinterpret_cast<void*>(state.handle) != nullptr; }

  ~executable()
  {
    hsa_executable_destroy(state);
    // reader needs to be destroyed after the executable
    hsa_code_object_reader_destroy(reader);
  }

  executable(hsa_agent_t agent, hsa_file_t file)
      : agent(agent), state({sentinel()})
  {
    if (HSA_STATUS_SUCCESS == init_state())
      {
        if (HSA_STATUS_SUCCESS == load_from_file(file))
          {
            if (HSA_STATUS_SUCCESS == freeze_and_validate())
              {
                return;
              }
          }
      }
    hsa_executable_destroy(state);
    state = {sentinel()};
  }

  executable(hsa_agent_t agent, const void* bytes, size_t size)
      : agent(agent), state({sentinel()})
  {
    if (HSA_STATUS_SUCCESS == init_state())
      {
        if (HSA_STATUS_SUCCESS == load_from_memory(bytes, size))
          {
            if (HSA_STATUS_SUCCESS == freeze_and_validate())
              {
                return;
              }
          }
      }
    hsa_executable_destroy(state);
    state = {sentinel()};
  }

  hsa_executable_symbol_t get_symbol_by_name(const char* symbol_name)
  {
    hsa_executable_symbol_t res;
    hsa_status_t rc =
        hsa_executable_get_symbol_by_name(state, symbol_name, &agent, &res);
    if (rc != HSA_STATUS_SUCCESS)
      {
        res = {sentinel()};
      }
    return res;
  }

  uint64_t get_symbol_address_by_name(const char* symbol_name)
  {
    hsa_executable_symbol_t symbol = get_symbol_by_name(symbol_name);
    if (symbol.handle == sentinel())
      {
        return 0;
      }

    hsa_symbol_kind_t kind = hsa::symbol_get_info_type(symbol);
    if (kind == HSA_SYMBOL_KIND_VARIABLE)
      {
        return hsa::symbol_get_info_variable_address(symbol);
      }

    if (kind == HSA_SYMBOL_KIND_KERNEL)
      {
        return hsa::symbol_get_info_kernel_object(symbol);
      }

    return 0;
  }

 private:
  hsa_status_t init_state()
  {
    hsa_profile_t profile =
        HSA_PROFILE_BASE;  // HIP uses full, vega claims 'base', unsure
    hsa_default_float_rounding_mode_t default_rounding_mode =
        HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT;
    const char* options = 0;
    hsa_executable_t e;
    hsa_status_t rc =
        hsa_executable_create_alt(profile, default_rounding_mode, options, &e);
    if (rc == HSA_STATUS_SUCCESS)
      {
        state = e;
      }
    return rc;
  }

  hsa_status_t load_from_file(hsa_file_t file)
  {
    hsa_status_t rc = hsa_code_object_reader_create_from_file(file, &reader);
    if (rc != HSA_STATUS_SUCCESS)
      {
        return rc;
      }

    hsa_loaded_code_object_t code;
    return hsa_executable_load_agent_code_object(state, agent, reader, NULL,
                                                 &code);
  }

  hsa_status_t load_from_memory(const void* bytes, size_t size)
  {
    hsa_status_t rc =
        hsa_code_object_reader_create_from_memory(bytes, size, &reader);
    if (rc != HSA_STATUS_SUCCESS)
      {
        return rc;
      }

    hsa_loaded_code_object_t code;
    return hsa_executable_load_agent_code_object(state, agent, reader, NULL,
                                                 &code);
  }

  hsa_status_t freeze_and_validate()
  {
    {
      hsa_status_t rc = hsa_executable_freeze(state, NULL);
      if (rc != HSA_STATUS_SUCCESS)
        {
          return rc;
        }
    }

    {
      uint32_t vres;
      hsa_status_t rc = hsa_executable_validate(state, &vres);
      if (rc != HSA_STATUS_SUCCESS)
        {
          return rc;
        }

      if (vres != 0)
        {
          return HSA_STATUS_ERROR;
        }
    }
    return HSA_STATUS_SUCCESS;
  }

  hsa_agent_t agent;
  hsa_executable_t state;
  hsa_code_object_reader_t reader;
};

inline hsa_agent_t find_a_gpu_or_exit()
{
  hsa_agent_t kernel_agent;
  if (HSA_STATUS_INFO_BREAK !=
      hsa::iterate_agents([&](hsa_agent_t agent) -> hsa_status_t {
        auto features = hsa::agent_get_info_feature(agent);
        if (features & HSA_AGENT_FEATURE_KERNEL_DISPATCH)
          {
            kernel_agent = agent;
            return HSA_STATUS_INFO_BREAK;
          }
        return HSA_STATUS_SUCCESS;
      }))
    {
      fprintf(stderr, "Failed to find a kernel agent\n");
      exit(1);
    }
  return kernel_agent;
}

inline uint64_t acquire_available_packet_id(hsa_queue_t* queue)
{
  uint64_t packet_id = hsa_queue_add_write_index_relaxed(queue, 1);
  bool full = true;
  while (full)
    {
      full =
          packet_id >= (queue->size + hsa_queue_load_read_index_acquire(queue));
    }
  return packet_id;
}

inline void initialize_packet_defaults(hsa_kernel_dispatch_packet_t* packet)
{
  // Reserved fields, private and group memory, and completion signal are all
  // set to 0.
  memset(((uint8_t*)packet) + 4, 0, sizeof(hsa_kernel_dispatch_packet_t) - 4);
  // These values should probably be read from the kernel
  // Currently they're copied from documentation
  // Launching a single wavefront makes for easier debugging
  packet->workgroup_size_x = 64;
  packet->workgroup_size_y = 1;
  packet->workgroup_size_z = 1;
  packet->grid_size_x = 64;
  packet->grid_size_y = 1;
  packet->grid_size_z = 1;

  // These definitely get overwritten by the caller
  packet->kernel_object = 0;  //  KERNEL_OBJECT;
  packet->kernarg_address = NULL;
}

// Maps a queue to an integer in [0, 1023] which survives CWSR, so can be used
// to index into device local structures. Inspired by the rocr.
inline uint16_t queue_to_index(hsa_queue_t* q)
{
  char* sig = reinterpret_cast<char*>(q->doorbell_signal.handle);
  int64_t kind;
  memcpy(&kind, sig, 8);
  // TODO: Work out if any hardware that works for openmp uses legacy doorbell
  assert(kind == -1);
  sig += 8;

  const uint64_t MAX_NUM_DOORBELLS = 0x400;

  uint64_t ptr;
  memcpy(&ptr, sig, 8);
  ptr >>= 3;
  ptr %= MAX_NUM_DOORBELLS;

  return static_cast<uint16_t>(ptr);
}

}  // namespace hsa

#endif
