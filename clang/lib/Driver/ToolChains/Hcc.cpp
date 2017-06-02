//===--- Hcc.cpp - HCC Tool and ToolChain Implementations -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Hcc.h"
#include "InputInfo.h"
#include "clang/Basic/VirtualFileSystem.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/Options.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/Path.h"
#include <system_error>

using namespace clang::driver;
using namespace clang::driver::toolchains;
using namespace clang::driver::tools;
using namespace clang;
using namespace llvm::opt;

static void HCPassOptions(const ArgList &Args, ArgStringList &CmdArgs) {

  for(auto A : Args) {
    Option ArgOpt = A->getOption();
    // Avoid passing options that have already been processed by the compilation stage or will be used for the linking stage
    bool hasOpts = ArgOpt.hasFlag(options::LinkerInput) || // omit linking options
                   ArgOpt.hasFlag(options::DriverOption) || // omit --driver-mode -### -hc -o -Xclang
                   ArgOpt.matches(options::OPT_L) || // omit -L
                   ArgOpt.matches(options::OPT_I_Group) || // omit -I
                   ArgOpt.matches(options::OPT_std_EQ) || // omit -std=
                   ArgOpt.matches(options::OPT_stdlib_EQ) || // omit -stdlib=
                   ArgOpt.matches(options::OPT_m_Group) || // omit -m
                   ArgOpt.getKind() == Option::InputClass; // omit <input>
    if (!hasOpts) {
      std::string str = A->getSpelling().str();

      // If this is a valued option
      ArrayRef<const char *> Vals = A->getValues();
      if(!Vals.empty()) {
        for(auto V : Vals) {
          str += V;
        }
      }
      CmdArgs.push_back(Args.MakeArgString(str));
    }
  }
}

void HCC::HCKernelAssemble::ConstructJob(Compilation &C, const JobAction &JA,
                                    const InputInfo &Output,
                                    const InputInfoList &Inputs,
                                    const ArgList &Args,
                                    const char *LinkingOutput) const {
  assert(Inputs.size() == 1 && "Unable to handle multiple inputs.");

  ArgStringList CmdArgs;
  for (InputInfoList::const_iterator
         it = Inputs.begin(), ie = Inputs.end(); it != ie; ++it) {
    const InputInfo &II = *it;
    if (II.isFilename())
      CmdArgs.push_back(II.getFilename());
    else
      II.getInputArg().renderAsInput(Args, CmdArgs);
  }

  if (Output.isFilename())
    CmdArgs.push_back(Output.getFilename());
  else
    Output.getInputArg().renderAsInput(Args, CmdArgs);

  // locate where the command is
  const char *Exec = Args.MakeArgString(getToolChain().GetProgramPath("hc-kernel-assemble"));

  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs, Inputs));
}

void HCC::HCHostAssemble::ConstructJob(Compilation &C, const JobAction &JA,
                                  const InputInfo &Output,
                                  const InputInfoList &Inputs,
                                  const ArgList &Args,
                                  const char *LinkingOutput) const {
  assert(Inputs.size() == 1 && "Unable to handle multiple inputs.");

  ArgStringList CmdArgs;
  for (InputInfoList::const_iterator
         it = Inputs.begin(), ie = Inputs.end(); it != ie; ++it) {
    const InputInfo &II = *it;
    if (II.isFilename())
      CmdArgs.push_back(II.getFilename());
    else
      II.getInputArg().renderAsInput(Args, CmdArgs);
  }

  if (Output.isFilename())
    CmdArgs.push_back(Output.getFilename());
  else
    Output.getInputArg().renderAsInput(Args, CmdArgs);

  // decide which options gets passed through
  HCPassOptions(Args, CmdArgs);

  const char *Exec = Args.MakeArgString(getToolChain().GetProgramPath("hc-host-assemble"));

  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs, Inputs));
}

void HCC::CXXAMPAssemble::ConstructJob(Compilation &C, const JobAction &JA,
                                  const InputInfo &Output,
                                  const InputInfoList &Inputs,
                                  const ArgList &Args,
                                  const char *LinkingOutput) const {
  assert(Inputs.size() == 1 && "Unable to handle multiple inputs.");

  ArgStringList CmdArgs;
  for (InputInfoList::const_iterator
         it = Inputs.begin(), ie = Inputs.end(); it != ie; ++it) {
    const InputInfo &II = *it;
    if (II.isFilename())
      CmdArgs.push_back(II.getFilename());
    else
      II.getInputArg().renderAsInput(Args, CmdArgs);
  }

  if (Output.isFilename())
    CmdArgs.push_back(Output.getFilename());
  else
    Output.getInputArg().renderAsInput(Args, CmdArgs);

  const char *Exec = Args.MakeArgString(getToolChain().GetProgramPath("clamp-assemble"));

  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs, Inputs));
}

std::string temporaryReplaceLongFormGFXIp(const Compilation& C, std::string l)
{ // Precondition: l = "AMD:AMDGPU:\d:\d:\d"
  // TODO: this should be removed once we have transitioned all users to using
  //       the short form. It is purposefully inefficient.
  const auto t = l;

  l.replace(0u, 3u, {'g', 'f', 'x'});
  l.erase(std::copy_if(l.begin() + 3u, l.end(), l.begin() + 3u, isdigit), l.end());
  C.getDriver().Diag(diag::warn_drv_deprecated_arg) << t << l;

  return l;
}

void HCC::CXXAMPLink::ConstructJob(Compilation &C,
                                   const JobAction &JA,
                                   const InputInfo &Output,
                                   const InputInfoList &Inputs,
                                   const ArgList &Args,
                                   const char *LinkingOutput) const {
  ArgStringList CmdArgs;

  // add verbose flag to linker script if clang++ is invoked with --verbose flag
  if (Args.hasArg(options::OPT_v))
    CmdArgs.push_back("--verbose");

  // specify AMDGPU target
  if (Args.hasArg(options::OPT_amdgpu_target_EQ)) {
    auto AMDGPUTargetVector = Args.getAllArgValues(options::OPT_amdgpu_target_EQ);

    for (auto&& AMDGPUTarget : AMDGPUTargetVector) {
      // TODO: the known GFXip list should probably reside in a constant
      //       global variable so as to allow easy extension in the future.
      static constexpr const char prefix[] = "--amdgpu-target=";
      static constexpr unsigned int discard_sz = 3u;
      static const std::string gfx_ip{"gfx"};
      static const std::string long_gfx_ip_prefix{"AMD:AMDGPU:"}; // Temporary.

      // TODO: this is temporary.
      if (std::search(AMDGPUTarget.cbegin(), AMDGPUTarget.cend(), long_gfx_ip_prefix.cbegin(), long_gfx_ip_prefix.cend()) != AMDGPUTarget.cend()) {
          AMDGPUTarget = temporaryReplaceLongFormGFXIp(C, AMDGPUTarget);
      }

      if (std::search(AMDGPUTarget.cbegin(), AMDGPUTarget.cend(), gfx_ip.cbegin(), gfx_ip.cend()) != AMDGPUTarget.cend()) {
        std::string t{prefix};
        switch (std::atoi(AMDGPUTarget.data() + discard_sz)) {
        case 700: t += "gfx700";  break;
        case 701: t += "gfx701";  break;
        case 801: t += "gfx801"; break;
        case 802: t += "gfx802";   break;
        case 803: t += "gfx803";    break;
        case 900: t += "gfx900";  break;
        case 901: t += "gfx901";  break;
        default:
          C.getDriver().Diag(diag::warn_amdgpu_target_invalid) << AMDGPUTarget;
        break;
        }
        CmdArgs.push_back(Args.MakeArgString(t));
      }
      else {
        C.getDriver().Diag(diag::warn_amdgpu_target_invalid) << AMDGPUTarget;
      }
    }
  }

  // pass inputs to gnu ld for initial processing
  Linker::ConstructLinkerJob(C, JA, Output, Inputs, Args, LinkingOutput, CmdArgs);

  auto ClampArgs = CmdArgs;
  if (Args.hasArg(options::OPT_hcc_extra_libs_EQ)) {
    auto HccExtraLibs = Args.getAllArgValues(options::OPT_hcc_extra_libs_EQ);
    std::string prefix{"--hcc-extra-libs="};
    for(auto&& Lib:HccExtraLibs) {
      ClampArgs.push_back(Args.MakeArgString(prefix + Lib));
    }
  }

  const char *Exec = Args.MakeArgString(getToolChain().GetProgramPath("clamp-link"));

  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, ClampArgs, Inputs));
}

/// HCC toolchain.
/// It may operate in 2 modes, depending on the Environment in Triple
/// - C++AMP mode:
///   - use clamp-assemble as assembler
///   - use clamp-link as linker
/// - HC mode:
///   - use hc-kernel-assemble as assembler for kernel path
///   - use hc-host-assemble as assembler for host path
///   - use clamp-link as linker

HCCToolChain::HCCToolChain(const Driver &D, const llvm::Triple &Triple,
                           const ArgList &Args)
    : Linux(D, Triple, Args) {
  llvm::Triple defaultTriple(llvm::sys::getDefaultTargetTriple());
  GCCInstallation.init(defaultTriple, Args);
}

void
HCCToolChain::addClangTargetOptions(const llvm::opt::ArgList &DriverArgs,
                                    llvm::opt::ArgStringList &CC1Args) const {
  Linux::addClangTargetOptions(DriverArgs, CC1Args);

  // TBD, depends on mode set correct arguments
}

llvm::opt::DerivedArgList *
HCCToolChain::TranslateArgs(const llvm::opt::DerivedArgList &Args,
                            StringRef BoundArch,
                            Action::OffloadKind DeviceOffloadKind) const {
  // TBD look into what should be properly implemented
  DerivedArgList *DAL = new DerivedArgList(Args.getBaseArgs());
  const OptTable &Opts = getDriver().getOpts();

  for (Arg *A : Args) {
    if (A->getOption().matches(options::OPT_Xarch__)) {
      // Skip this argument unless the architecture matches BoundArch
      if (BoundArch.empty() || A->getValue(0) != BoundArch)
        continue;

      unsigned Index = Args.getBaseArgs().MakeIndex(A->getValue(1));
      unsigned Prev = Index;
      std::unique_ptr<Arg> XarchArg(Opts.ParseOneArg(Args, Index));

      // If the argument parsing failed or more than one argument was
      // consumed, the -Xarch_ argument's parameter tried to consume
      // extra arguments. Emit an error and ignore.
      //
      // We also want to disallow any options which would alter the
      // driver behavior; that isn't going to work in our model. We
      // use isDriverOption() as an approximation, although things
      // like -O4 are going to slip through.
      if (!XarchArg || Index > Prev + 1) {
        getDriver().Diag(diag::err_drv_invalid_Xarch_argument_with_args)
            << A->getAsString(Args);
        continue;
      } else if (XarchArg->getOption().hasFlag(options::DriverOption)) {
        getDriver().Diag(diag::err_drv_invalid_Xarch_argument_isdriver)
            << A->getAsString(Args);
        continue;
      }
      XarchArg->setBaseArg(A);
      A = XarchArg.release();
      DAL->AddSynthesizedArg(A);
    }
    DAL->append(A);
  }

  if (!BoundArch.empty())
    DAL->AddJoinedArg(nullptr, Opts.getOption(options::OPT_march_EQ), BoundArch);
  return DAL;
}

extern bool IsHCHostAssembleJobAction(const JobAction* A);
extern bool IsHCKernelAssembleJobAction(const JobAction* A);
extern bool IsCXXAMPAssembleJobAction(const JobAction* A);
extern bool IsCXXAMPCPUAssembleJobAction(const JobAction* A);

Tool *HCCToolChain::SelectTool(const JobAction &JA) const {
  Action::ActionClass AC = JA.getKind();

  if (AC == Action::AssembleJobClass) {
    if (IsHCHostAssembleJobAction(&JA)) {
      if (!HCHostAssembler)
        HCHostAssembler.reset(new tools::HCC::HCHostAssemble(*this));
      return HCHostAssembler.get();
    }
    if (IsHCKernelAssembleJobAction(&JA)) {
      if (!HCKernelAssembler)
        HCKernelAssembler.reset(new tools::HCC::HCKernelAssemble(*this));
      return HCKernelAssembler.get();
    }
    if (IsCXXAMPAssembleJobAction(&JA) || IsCXXAMPCPUAssembleJobAction(&JA)) {
      if (!CXXAMPAssembler)
        CXXAMPAssembler.reset(new tools::HCC::CXXAMPAssemble(*this));
      return CXXAMPAssembler.get();
    }
  }

  return ToolChain::SelectTool(JA);
}

Tool *HCCToolChain::buildLinker() const {
  return new tools::HCC::CXXAMPLink(*this);
}
