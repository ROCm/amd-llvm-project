//===--- Hcc.cpp - HCC Tool and ToolChain Implementations -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Hcc.h"
#include "Gnu.h"
#include "InputInfo.h"
#include "clang/Basic/VirtualFileSystem.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/Options.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/Path.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <system_error>
#include <utility>

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace clang::driver;
using namespace clang::driver::toolchains;
using namespace clang::driver::tools;
using namespace clang;
using namespace llvm::opt;

HCCInstallationDetector::HCCInstallationDetector(const Driver &D, const llvm::opt::ArgList &Args) : D(D) {
  std::string BinPath = D.Dir;
  std::string InstallPath = D.InstalledDir;
  auto &FS = D.getVFS();
  SmallVector<std::string, 4> HCCPathCandidates;

  if (Args.hasArg(options::OPT_hcc_path_EQ))
    HCCPathCandidates.push_back(
      Args.getLastArgValue(options::OPT_hcc_path_EQ));
    
  HCCPathCandidates.push_back(InstallPath + "/..");
  HCCPathCandidates.push_back(BinPath + "/..");
  HCCPathCandidates.push_back(BinPath + "/../..");

  for (const auto &HCCPath: HCCPathCandidates) {
    if (HCCPath.empty() ||
        !(FS.exists(HCCPath + "/include/hc.hpp") || FS.exists(HCCPath + "/include/hcc/hc.hpp")) || 
        !FS.exists(HCCPath + "/lib/libmcwamp.a"))
      continue;

    IncPath = HCCPath;
    LibPath = HCCPath + "/lib";

    IsValid = true;
    break;
  }
}

void HCCInstallationDetector::AddHCCIncludeArgs(const llvm::opt::ArgList &DriverArgs, llvm::opt::ArgStringList &CC1Args) const {
  if (IsValid) {
    CC1Args.push_back(DriverArgs.MakeArgString("-I" + IncPath + "/include"));
    CC1Args.push_back(DriverArgs.MakeArgString("-I" + IncPath + "/hcc/include"));
  }
}

void HCCInstallationDetector::AddHCCLibArgs(const llvm::opt::ArgList &Args, llvm::opt::ArgStringList &CmdArgs) const {
  if (IsValid) {
    // add verbose flag to linker script if clang++ is invoked with --verbose flag
    if (Args.hasArg(options::OPT_v)) CmdArgs.push_back("--verbose");
        
    // Reverse translate the -lstdc++ option
    // Or add -lstdc++ when running on RHEL 7 or CentOS 7
    if (Args.hasArg(options::OPT_Z_reserved_lib_stdcxx) ||
      HCC_TOOLCHAIN_RHEL) {
      CmdArgs.push_back("-lstdc++");
    }

    CmdArgs.push_back(Args.MakeArgString("-L" + LibPath));
    CmdArgs.push_back(Args.MakeArgString("--rpath=" + LibPath));

    for (auto &lib: SystemLibs)
      CmdArgs.push_back(lib);
    
    for (auto &lib: RuntimeLibs)
      CmdArgs.push_back(lib);

    if (Args.hasArg(options::OPT_hcc_extra_libs_EQ)) {
      auto HccExtraLibs = Args.getAllArgValues(options::OPT_hcc_extra_libs_EQ);
      std::string prefix{"--hcc-extra-libs="};

      for(auto&& Lib:HccExtraLibs)
        CmdArgs.push_back(Args.MakeArgString(prefix + Lib));
    }
  }
}

void HCCInstallationDetector::print(raw_ostream &OS) const {
  if (IsValid)
    OS << "Found HCC installation: " << IncPath << "\n";
}

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

namespace
{
    std::string temporary_replace_long_form_GFXIp(
        const Compilation& c, std::string l)
    {   // Precondition: l = "AMD:AMDGPU:\d:\d:\d"
        // TODO: this should be removed once we have transitioned all users to
        //       the short form. It is purposefully inefficient.
        const auto t = l;

        l.replace(0u, 3u, {'g', 'f', 'x'});
        l.erase(std::copy_if(
            l.begin() + 3u, l.end(), l.begin() + 3u, isdigit), l.end());
        c.getDriver().Diag(diag::warn_drv_deprecated_arg) << t << l;

        return l;
    }

    struct Process_deleter {
        int status = EXIT_FAILURE;
        void operator()(std::FILE* p)
        {
            if (p) {
                status = pclose(p);
                status = WIFEXITED(status) ? WEXITSTATUS(status) : status;
            }
        }
    };

    std::vector<std::string> detect_gfxip(
        const Compilation& c, const ToolChain& tc)
    {   // Invariant: iff it executes correctly, rocm_agent_enumerator returns
        //            at least gfx000; returning only gfx000 signals the absence
        //            of valid GPU agents.
        // Invariant: iff it executes correctly, and iff there are valid GPU
        //            agents present rocm_agent_enumerator returns the set
        //            formed from their union, including gfx000.
        std::vector<std::string> r;

        const char* tmp = std::getenv("ROCM_ROOT");
        const Twine rocm = tmp ? tmp : "/opt/rocm";
        const Twine e = rocm + "/bin/rocm_agent_enumerator";

        if (!tc.getVFS().exists(e)) return r;

        Process_deleter d;
        std::unique_ptr<std::FILE, Process_deleter> pipe{
            popen((e.str() + " --type GPU").c_str(), "r"), d};

        if (!pipe) return r;

        static constexpr std::size_t buf_sz = 16u;
        std::array<char, buf_sz> buf = {{}};
        while (std::fgets(buf.data(), buf.size(), pipe.get())) {
            r.emplace_back(buf.data());
        }

        for (auto&& x : r) { // fgets copies the newline.
            x.erase(std::remove(x.begin(), x.end(), '\n'), x.end());
        }

        if (r.size() > 1) {
            std::sort(r.rbegin(), r.rend());
            r.pop_back(); // Remove null-agent.
        }

        return r;
    }

    std::vector<std::string> detect_and_add_targets(
        const Compilation& c, const ToolChain& tc)
    {
        constexpr const char null_agent[] = "gfx000";

        const auto detected_targets = detect_gfxip(c, tc);
        if (detected_targets.empty()) {
            c.getDriver().Diag(diag::warn_amdgpu_agent_detector_failed);
        }
        else if (detected_targets[0] == null_agent) {
            c.getDriver().Diag(diag::err_amdgpu_no_agent_available);
        }

        return detected_targets;
    }

    bool is_valid(const std::string& gfxip)
    {
        static constexpr std::array<const char*, 7u> valid = {
            { "gfx700", "gfx701", "gfx801", "gfx802", "gfx803", "gfx900",
              "gfx901" }};

        return std::find(valid.cbegin(), valid.cend(), gfxip) != valid.cend();
    }

    bool is_deprecated(const std::string& gfxip)
    {
        static constexpr std::array<const char*, 1u> deprecated = {{"gfx700"}};

        return std::find(
            deprecated.cbegin(), deprecated.cend(), gfxip) != deprecated.cend();
    }

    void validate_and_add_to_command(
        const std::string& gfxip,
        const Compilation& c,
        const ArgList& args,
        ArgStringList& cmd_args)
    {
        static constexpr const char prefix[] = "--amdgpu-target=";

        if (!is_valid(gfxip)) {
            c.getDriver().Diag(diag::warn_amdgpu_target_invalid) << gfxip;
            return;
        }

        if (is_deprecated(gfxip)) {
            c.getDriver().Diag(diag::warn_amdgpu_target_deprecated) << gfxip;
        }
        cmd_args.push_back(args.MakeArgString(prefix + gfxip));
    }

    template<typename T>
    void split(const std::string& s, char delim, T result)
    {
        std::stringstream ss;
        ss.str(s);
        std::string item;
        while (std::getline(ss, item, delim)) {
            *(result++) = item;
        }
    }

    std::vector<std::string> split_gfx_list(
        const std::string& gfx_list,
        char delim)
    {
        std::vector<std::string> elems;
        split(gfx_list, delim, std::back_inserter(elems));
        return elems;
    }

    template <typename T>
    void remove_duplicate_targets(std::vector<T>& TargetVec)
    {
        std::sort(TargetVec.begin(), TargetVec.end());
        TargetVec.erase(unique(TargetVec.begin(), TargetVec.end()), TargetVec.end());
    }
}

#ifndef HCC_TOOLCHAIN_RHEL
  #define HCC_TOOLCHAIN_RHEL false
#endif

void HCC::CXXAMPLink::ConstructLinkerJob(
    Compilation &C,
    const JobAction &JA,
    const InputInfo &Output,
    const InputInfoList &Inputs,
    const ArgList &Args,
    const char *LinkingOutput,
    ArgStringList &CmdArgs) const
{
    // specify AMDGPU target
    constexpr const char auto_tgt[] = "auto";

    const auto &TC = static_cast<const toolchains::Generic_ELF &>(getToolChain());
    TC.HCCInstallation.AddHCCLibArgs(Args, CmdArgs);

    #if !defined(HCC_AMDGPU_TARGET)
        #define HCC_AMDGPU_TARGET auto_tgt
    #endif

    auto AMDGPUTargetVector =
        Args.getAllArgValues(options::OPT_amdgpu_target_EQ);

    if (AMDGPUTargetVector.empty()) {
        // split HCC_AMDGPU_TARGET list up
        AMDGPUTargetVector = split_gfx_list(HCC_AMDGPU_TARGET, ' ');
    }

    const auto cnt = std::count(
        AMDGPUTargetVector.cbegin(), AMDGPUTargetVector.cend(), auto_tgt);

    if (cnt > 1) C.getDriver().Diag(diag::warn_amdgpu_target_auto_nonsingular);
    if (cnt == AMDGPUTargetVector.size()) {
        AMDGPUTargetVector = detect_and_add_targets(C, getToolChain());
    }
    AMDGPUTargetVector.erase(
        std::remove(
            AMDGPUTargetVector.begin(), AMDGPUTargetVector.end(), auto_tgt),
        AMDGPUTargetVector.end());

    remove_duplicate_targets(AMDGPUTargetVector);

    for (auto&& AMDGPUTarget : AMDGPUTargetVector) {
        // TODO: this is Temporary.
        static const std::string long_gfx_ip_prefix{"AMD:AMDGPU:"};
        if (std::search(
            AMDGPUTarget.cbegin(),
            AMDGPUTarget.cend(),
            long_gfx_ip_prefix.cbegin(),
            long_gfx_ip_prefix.cend()) != AMDGPUTarget.cend()) {
            AMDGPUTarget =
                temporary_replace_long_form_GFXIp(C, AMDGPUTarget);
        }
        validate_and_add_to_command(AMDGPUTarget, C, Args, CmdArgs);
    }
}

void HCC::CXXAMPLink::ConstructJob(Compilation &C,
                                   const JobAction &JA,
                                   const InputInfo &Output,
                                   const InputInfoList &Inputs,
                                   const ArgList &Args,
                                   const char *LinkingOutput) const {
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
                           const ToolChain &HostTC, const ArgList &Args)
    : ToolChain(D, Triple, Args), HostTC(HostTC) {
  getProgramPaths().push_back(getDriver().getInstalledDir());
  if (getDriver().getInstalledDir() != getDriver().Dir)
    getProgramPaths().push_back(getDriver().Dir);
}

void HCCToolChain::addClangTargetOptions(
    const llvm::opt::ArgList &DriverArgs, llvm::opt::ArgStringList &CC1Args,
    Action::OffloadKind DeviceOffloadKind) const {
  HostTC.addClangTargetOptions(DriverArgs, CC1Args, DeviceOffloadKind);

  // TBD, depends on mode set correct arguments
}

void HCCToolChain::AddClangSystemIncludeArgs(const ArgList &DriverArgs, ArgStringList &CC1Args) const {
  HostTC.AddClangSystemIncludeArgs(DriverArgs, CC1Args);
}

void HCCToolChain::AddClangCXXStdlibIncludeArgs(const ArgList &Args, ArgStringList &CC1Args) const {
  HostTC.AddClangCXXStdlibIncludeArgs(Args, CC1Args);
}

void HCCToolChain::AddHCCIncludeArgs(const ArgList &DriverArgs, ArgStringList &CC1Args) const {
  HostTC.AddHCCIncludeArgs(DriverArgs, CC1Args);
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
