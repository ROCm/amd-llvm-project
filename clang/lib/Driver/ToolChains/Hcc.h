//===--- Hcc.h - HCC ToolChain Implementations ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_HCC_H
#define LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_HCC_H

#include "clang/Driver/Action.h"
#include "clang/Driver/ToolChain.h"
#include "clang/Driver/Tool.h"
#include "llvm/Support/Compiler.h"

extern bool FunctionCallDefault;

namespace clang {
namespace driver {

class HCCInstallationDetector {
private:
  const Driver &D;
  bool IsValid = false;

  std::string IncPath;
  std::string LibPath;

  std::vector<const char *> SystemLibs = {"-ldl", "-lm", "-lpthread"};
  std::vector<const char *> RuntimeLibs = {"-lhc_am", "-lmcwamp"};

public:
  HCCInstallationDetector(const Driver &D, const llvm::opt::ArgList &Args);
      
  void AddHCCIncludeArgs(const llvm::opt::ArgList &DriverArgs, llvm::opt::ArgStringList &CC1Args) const;

  void AddHCCLibArgs(const llvm::opt::ArgList &Args, llvm::opt::ArgStringList &CmdArgs) const;
      
  bool isValid() const { return IsValid; }
      
  void print(raw_ostream &OS) const;
};

namespace tools {
namespace HCC {

/// \brief HC assembler tool.
class LLVM_LIBRARY_VISIBILITY Assembler : public Tool {
public:
  Assembler(const ToolChain &TC)
      : Tool("hc-assemble", "HC assembler", TC) {}

  bool hasGoodDiagnostics() const override { return true; }
  bool hasIntegratedAssembler() const override { return false; }
  bool hasIntegratedCPP() const override { return false; }

  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output,
                    const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOuput) const override;
};

// \brief C++AMP linker.
class LLVM_LIBRARY_VISIBILITY CXXAMPLink : public Tool {
public:
  CXXAMPLink(const ToolChain &TC) : Tool("clamp-link", "HC linker", TC) {}

  bool hasGoodDiagnostics() const override { return true; }
  bool hasIntegratedAssembler() const override { return false; }
  bool hasIntegratedCPP() const override { return false; }

  void ConstructJob(Compilation &C,
                    const JobAction &JA,
                    const InputInfo &Output,
                    const InputInfoList &Inputs,
                    const llvm::opt::ArgList &Args,
                    const char *LinkingOuput) const override;

  void ConstructLinkerJob(Compilation &C,
                          const JobAction &JA,
                          const InputInfo &Output,
                          const InputInfoList &Inputs,
                          const llvm::opt::ArgList &Args,
                          const char *LinkingOutput,
                          llvm::opt::ArgStringList &CmdArgs) const;
};

} // end namespace HCC
} // end namespace tools

namespace toolchains {

class LLVM_LIBRARY_VISIBILITY HCCToolChain : public ToolChain {
public:
  HCCToolChain(const Driver &D, const llvm::Triple &Triple,
               const ToolChain &HostTC, const llvm::opt::ArgList &Args);

  llvm::opt::DerivedArgList *
  TranslateArgs(const llvm::opt::DerivedArgList &Args,
                StringRef BoundArch,
                Action::OffloadKind DeviceOffloadKind) const override;
  void
  addClangTargetOptions(const llvm::opt::ArgList &DriverArgs,
                        llvm::opt::ArgStringList &CC1Args,
                        Action::OffloadKind DeviceOffloadKind) const override;

  void AddClangSystemIncludeArgs(const llvm::opt::ArgList &DriverArgs, llvm::opt::ArgStringList &CC1Args) const override;

  void AddClangCXXStdlibIncludeArgs(const llvm::opt::ArgList &Args, llvm::opt::ArgStringList &CC1Args) const override;

  void AddHCCIncludeArgs(const llvm::opt::ArgList &DriverArgs, llvm::opt::ArgStringList &CC1Args) const override;
  
  bool useIntegratedAs() const override { return false; }

  // HCC ToolChain use DWARF version 2 by default
  unsigned GetDefaultDwarfVersion() const override { return 2; }

  // HCC ToolChain doesn't support "-pg"-style profiling yet
  bool SupportsProfiling() const override { return false; }

  bool isPICDefault() const override { return false; }
  bool isPIEDefault() const override { return false; }
  bool isPICDefaultForced() const override { return false; }

  const ToolChain &HostTC;

protected:
  Tool *buildAssembler() const override;
  Tool *buildLinker() const override;
};

} // end namespace toolchains
} // end namespace driver
} // end namespace clang

#endif // LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_HCC_H
