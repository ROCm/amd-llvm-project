//===-- ThreadElfCore.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ThreadElfCore_h_
#define liblldb_ThreadElfCore_h_

// C Includes
// C++ Includes
#include <string>

// Other libraries and framework includes
// Project includes
#include "lldb/Core/DataExtractor.h"
#include "lldb/Target/Thread.h"

struct compat_timeval {
  alignas(8) uint64_t tv_sec;
  alignas(8) uint64_t tv_usec;
};

// PRSTATUS structure's size differs based on architecture.
// This is the layout in the x86-64 arch.
// In the i386 case we parse it manually and fill it again
// in the same structure
// The gp registers are also a part of this struct, but they are handled
// separately

#undef si_signo
#undef si_code
#undef si_errno

struct ELFLinuxPrStatus {
  int32_t si_signo;
  int32_t si_code;
  int32_t si_errno;

  int16_t pr_cursig;

  alignas(8) uint64_t pr_sigpend;
  alignas(8) uint64_t pr_sighold;

  uint32_t pr_pid;
  uint32_t pr_ppid;
  uint32_t pr_pgrp;
  uint32_t pr_sid;

  compat_timeval pr_utime;
  compat_timeval pr_stime;
  compat_timeval pr_cutime;
  compat_timeval pr_cstime;

  ELFLinuxPrStatus();

  lldb_private::Error Parse(lldb_private::DataExtractor &data,
                            lldb_private::ArchSpec &arch);

  // Return the bytesize of the structure
  // 64 bit - just sizeof
  // 32 bit - hardcoded because we are reusing the struct, but some of the
  // members are smaller -
  // so the layout is not the same
  static size_t GetSize(lldb_private::ArchSpec &arch) {
    switch (arch.GetCore()) {
    case lldb_private::ArchSpec::eCore_s390x_generic:
    case lldb_private::ArchSpec::eCore_x86_64_x86_64:
      return sizeof(ELFLinuxPrStatus);
    case lldb_private::ArchSpec::eCore_x86_32_i386:
    case lldb_private::ArchSpec::eCore_x86_32_i486:
      return 72;
    default:
      return 0;
    }
  }
};

static_assert(sizeof(ELFLinuxPrStatus) == 112,
              "sizeof ELFLinuxPrStatus is not correct!");

// PRPSINFO structure's size differs based on architecture.
// This is the layout in the x86-64 arch case.
// In the i386 case we parse it manually and fill it again
// in the same structure
struct ELFLinuxPrPsInfo {
  char pr_state;
  char pr_sname;
  char pr_zomb;
  char pr_nice;
  alignas(8) uint64_t pr_flag;
  uint32_t pr_uid;
  uint32_t pr_gid;
  int32_t pr_pid;
  int32_t pr_ppid;
  int32_t pr_pgrp;
  int32_t pr_sid;
  char pr_fname[16];
  char pr_psargs[80];

  ELFLinuxPrPsInfo();

  lldb_private::Error Parse(lldb_private::DataExtractor &data,
                            lldb_private::ArchSpec &arch);

  // Return the bytesize of the structure
  // 64 bit - just sizeof
  // 32 bit - hardcoded because we are reusing the struct, but some of the
  // members are smaller -
  // so the layout is not the same
  static size_t GetSize(lldb_private::ArchSpec &arch) {
    switch (arch.GetCore()) {
    case lldb_private::ArchSpec::eCore_s390x_generic:
    case lldb_private::ArchSpec::eCore_x86_64_x86_64:
      return sizeof(ELFLinuxPrPsInfo);
    case lldb_private::ArchSpec::eCore_x86_32_i386:
    case lldb_private::ArchSpec::eCore_x86_32_i486:
      return 124;
    default:
      return 0;
    }
  }
};

static_assert(sizeof(ELFLinuxPrPsInfo) == 136,
              "sizeof ELFLinuxPrPsInfo is not correct!");

struct ThreadData {
  lldb_private::DataExtractor gpregset;
  lldb_private::DataExtractor fpregset;
  lldb_private::DataExtractor vregset;
  lldb::tid_t tid;
  int signo;
  std::string name;
};

class ThreadElfCore : public lldb_private::Thread {
public:
  ThreadElfCore(lldb_private::Process &process, const ThreadData &td);

  ~ThreadElfCore() override;

  void RefreshStateAfterStop() override;

  lldb::RegisterContextSP GetRegisterContext() override;

  lldb::RegisterContextSP
  CreateRegisterContextForFrame(lldb_private::StackFrame *frame) override;

  void ClearStackFrames() override;

  static bool ThreadIDIsValid(lldb::tid_t thread) { return thread != 0; }

  const char *GetName() override {
    if (m_thread_name.empty())
      return NULL;
    return m_thread_name.c_str();
  }

  void SetName(const char *name) override {
    if (name && name[0])
      m_thread_name.assign(name);
    else
      m_thread_name.clear();
  }

protected:
  //------------------------------------------------------------------
  // Member variables.
  //------------------------------------------------------------------
  std::string m_thread_name;
  lldb::RegisterContextSP m_thread_reg_ctx_sp;

  int m_signo;

  lldb_private::DataExtractor m_gpregset_data;
  lldb_private::DataExtractor m_fpregset_data;
  lldb_private::DataExtractor m_vregset_data;

  bool CalculateStopInfo() override;
};

#endif // liblldb_ThreadElfCore_h_
