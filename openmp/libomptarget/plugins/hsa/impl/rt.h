/*===--------------------------------------------------------------------------
 *              ATMI (Asynchronous Task and Memory Interface)
 *
 * This file is distributed under the MIT License. See LICENSE.txt for details.
 *===------------------------------------------------------------------------*/
#ifndef SRC_RUNTIME_INCLUDE_RT_H_
#define SRC_RUNTIME_INCLUDE_RT_H_

#include "atmi_runtime.h"
#include "hsa.h"
#include <cstdarg>
#include <string>

namespace core {

#define DEFAULT_MAX_SIGNALS 1024
#define DEFAULT_MAX_QUEUE_SIZE 4096
#define DEFAULT_MAX_KERNEL_TYPES 32
#define DEFAULT_NUM_GPU_QUEUES -1  // computed in code
#define DEFAULT_NUM_CPU_QUEUES -1  // computed in code
#define DEFAULT_DEBUG_MODE 0
#define DEFAULT_PROFILE_MODE 0
class Environment {
 public:
  Environment()
      : max_signals_(DEFAULT_MAX_SIGNALS),
        max_queue_size_(DEFAULT_MAX_QUEUE_SIZE),
        max_kernel_types_(DEFAULT_MAX_KERNEL_TYPES),
        num_gpu_queues_(DEFAULT_NUM_GPU_QUEUES),
        num_cpu_queues_(DEFAULT_NUM_CPU_QUEUES),
        debug_mode_(DEFAULT_DEBUG_MODE),
        profile_mode_(DEFAULT_PROFILE_MODE) {
    GetEnvAll();
  }

  ~Environment() {}

  void GetEnvAll();

  int getDepSyncType() const { return dep_sync_type_; }
  int getMaxSignals() const { return max_signals_; }
  int getMaxQueueSize() const { return max_queue_size_; }
  int getMaxKernelTypes() const { return max_kernel_types_; }
  int getNumGPUQueues() const { return num_gpu_queues_; }
  int getNumCPUQueues() const { return num_cpu_queues_; }
  // TODO(ashwinma): int may change to enum if we have more debug modes
  int getDebugMode() const { return debug_mode_; }
  // TODO(ashwinma): int may change to enum if we have more profile modes
  int getProfileMode() const { return profile_mode_; }

 private:
  std::string GetEnv(const char *name) {
    char *env = getenv(name);
    std::string ret;
    if (env) {
      ret = env;
    }
    return ret;
  }

  int dep_sync_type_;
  int max_signals_;
  int max_queue_size_;
  int max_kernel_types_;
  int num_gpu_queues_;
  int num_cpu_queues_;
  int debug_mode_;
  int profile_mode_;
};

class Runtime {
 public:
  static Runtime &getInstance() {
    static Runtime instance;
    return instance;
  }

  // init/finalize
  virtual atmi_status_t Initialize(atmi_devtype_t);
  virtual atmi_status_t Finalize();
  // machine info
  atmi_machine_t *GetMachineInfo();
  // modules
  atmi_status_t RegisterModuleFromMemory(void **, size_t *,
                                         atmi_platform_type_t *, const int,
                                         atmi_place_t);
  atmi_status_t RegisterModule(const char **, atmi_platform_type_t *, const int,
                               atmi_place_t);
  atmi_status_t RegisterModuleFromMemory(void **, size_t *,
                                         atmi_platform_type_t *, const int);
  atmi_status_t RegisterModule(const char **, atmi_platform_type_t *,
                               const int);
  // kernels
  virtual atmi_status_t CreateKernel(atmi_kernel_t *, const int, const size_t *,
                                     const int, va_list);
  virtual atmi_status_t ReleaseKernel(atmi_kernel_t);
  atmi_status_t CreateEmptyKernel(atmi_kernel_t *, const int, const size_t *);
  atmi_status_t AddGPUKernelImpl(atmi_kernel_t, const char *,
                                 const unsigned int);
  atmi_status_t AddCPUKernelImpl(atmi_kernel_t, atmi_generic_fp,
                                 const unsigned int);
  // sync
  atmi_status_t TaskWait(atmi_task_handle_t);
  // tasks
  atmi_task_handle_t LaunchTask(atmi_lparm_t *, atmi_kernel_t, void **);
  // taskgroups
  atmi_status_t TaskGroupCreate(atmi_taskgroup_handle_t *, bool ordered = false,
                                atmi_place_t place = ATMI_DEFAULT_PLACE);
  atmi_status_t TaskGroupRelease(atmi_taskgroup_handle_t);
  // data
  atmi_status_t Memcpy(void *, const void *, size_t);
  atmi_task_handle_t MemcpyAsync(atmi_cparm_t *, void *, const void *, size_t);
  atmi_status_t Memfree(void *);
  atmi_status_t Malloc(void **, size_t, atmi_mem_place_t);

  // environment variables
  const Environment &getEnvironment() const { return env_; }
  int getDepSyncType() const { return env_.getDepSyncType(); }
  int getMaxSignals() const { return env_.getMaxSignals(); }
  int getMaxQueueSize() const { return env_.getMaxQueueSize(); }
  int getMaxKernelTypes() const { return env_.getMaxKernelTypes(); }
  int getNumGPUQueues() const { return env_.getNumGPUQueues(); }
  int getNumCPUQueues() const { return env_.getNumCPUQueues(); }
  // TODO(ashwinma): int may change to enum if we have more debug modes
  int getDebugMode() const { return env_.getDebugMode(); }
  // TODO(ashwinma): int may change to enum if we have more profile modes
  int getProfileMode() const { return env_.getProfileMode(); }


 protected:
  Runtime() = default;
  ~Runtime() = default;
  Runtime(const Runtime &) = delete;
  Runtime &operator=(const Runtime &) = delete;

 protected:
  // variable to track environment variables
  Environment env_;
};

}  // namespace core

#endif  // SRC_RUNTIME_INCLUDE_RT_H_
