//===- AMDGPUOpenMP.cpp - AMDGPUOpenMP ToolChain Implementation -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPUOpenMP.h"
#include "AMDGPU.h"
#include "CommonArgs.h"
#include "InputInfo.h"
#include "clang/Basic/Cuda.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/Options.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/TargetParser.h"

using namespace clang::driver;
using namespace clang::driver::toolchains;
using namespace clang::driver::tools;
using namespace clang;
using namespace llvm::opt;

#if defined(_WIN32) || defined(_WIN64)
#define NULL_FILE "nul"
#else
#define NULL_FILE "/dev/null"
#endif

namespace {

static void addBCLib(const Driver &D, const ArgList &Args,
                     ArgStringList &CmdArgs, ArgStringList LibraryPaths,
                     StringRef BCName, bool postClangLink) {
  StringRef FullName;
  for (std::string LibraryPath : LibraryPaths) {
    SmallString<128> Path(LibraryPath);
    llvm::sys::path::append(Path, BCName);
    FullName = Path;
    if (llvm::sys::fs::exists(FullName)) {
      if (postClangLink)
        CmdArgs.push_back("-mlink-builtin-bitcode");
      CmdArgs.push_back(Args.MakeArgString(FullName));
      return;
    }
  }
  D.Diag(diag::err_drv_no_such_file) << BCName;
}

} // namespace

// OpenMP needs a custom link tool to build select statement
const char *AMDGCN::OpenMPLinker::constructOmpExtraCmds(
    Compilation &C, const JobAction &JA, const InputInfoList &Inputs,
    const ArgList &Args, StringRef SubArchName,
    StringRef OutputFilePrefix) const {

  std::string TmpName;
  TmpName = C.getDriver().isSaveTempsEnabled()
                ? OutputFilePrefix.str() + "-select.bc"
                : C.getDriver().GetTemporaryPath(
                      OutputFilePrefix.str() + "-select", "bc");
  const char *OutputFileName =
      C.addTempFile(C.getArgs().MakeArgString(TmpName));
  ArgStringList CmdArgs;
  // CmdArgs.push_back("-v");
  llvm::SmallVector<std::string, 10> BCLibs;
  for (const auto &II : Inputs) {
    if (II.isFilename())
      CmdArgs.push_back(II.getFilename());
  }

  ArgStringList LibraryPaths;

  // If device debugging turned on, get bc files from lib-debug dir
  std::string lib_debug_path = FindDebugInLibraryPath();
  if (!lib_debug_path.empty()) {
    LibraryPaths.push_back(Args.MakeArgString(lib_debug_path + "/libdevice"));
    LibraryPaths.push_back(Args.MakeArgString(lib_debug_path));
  }

  addDirectoryList(Args, LibraryPaths, "", "HIP_DEVICE_LIB_PATH");

  // Add compiler path libdevice last as lowest priority search
  LibraryPaths.push_back(
      Args.MakeArgString(C.getDriver().Dir + "/../amdgcn/bitcode"));
  LibraryPaths.push_back(
      Args.MakeArgString(C.getDriver().Dir + "/../../amdgcn/bitcode"));
  LibraryPaths.push_back(
      Args.MakeArgString(C.getDriver().Dir + "/../lib/libdevice"));
  LibraryPaths.push_back(Args.MakeArgString(C.getDriver().Dir + "/../lib"));
  LibraryPaths.push_back(
      Args.MakeArgString(C.getDriver().Dir + "/../../lib/libdevice"));
  LibraryPaths.push_back(Args.MakeArgString(C.getDriver().Dir + "/../../lib"));

  // Add bitcode library in --hip-device-lib.
  for (auto Lib : Args.getAllArgValues(options::OPT_hip_device_lib_EQ)) {
    BCLibs.push_back(Args.MakeArgString(Lib));
  }

  llvm::StringRef WaveFrontSizeBC;
  std::string GFXVersion = SubArchName.drop_front(3).str();
  if (stoi(GFXVersion) < 1000)
    WaveFrontSizeBC = "oclc_wavefrontsize64_on.bc";
  else
    WaveFrontSizeBC = "oclc_wavefrontsize64_off.bc";

  // FIXME: remove double link of hip aompextras, ockl, and WaveFrontSizeBC
  if (Args.hasArg(options::OPT_cuda_device_only))
    BCLibs.append(
        {Args.MakeArgString("libomptarget-amdgcn-" + SubArchName + ".bc"),
         "hip.bc", "hc.bc", "ockl.bc",
         std::string(WaveFrontSizeBC)});
  else {
    BCLibs.append(
        {Args.MakeArgString("libomptarget-amdgcn-" + SubArchName + ".bc"),
         Args.MakeArgString("libaompextras-amdgcn-" + SubArchName + ".bc"),
         "hip.bc", "hc.bc", "ockl.bc",
         Args.MakeArgString("libbc-hostrpc-amdgcn.a"),
         std::string(WaveFrontSizeBC)});

    if (!Args.hasArg(options::OPT_nostdlibxx) &&
        !Args.hasArg(options::OPT_nostdlib))
      BCLibs.append({Args.MakeArgString("libm-amdgcn-" + SubArchName + ".bc")});
  }

  for (auto Lib : BCLibs)
    addBCLib(C.getDriver(), Args, CmdArgs, LibraryPaths, Lib,
             /* PostClang Link? */ false);

  // This will find .a and .bc files that match naming convention.
  AddStaticDeviceLibs(C, *this, JA, Inputs, Args, CmdArgs, "amdgcn",
                      SubArchName,
                      /* bitcode SDL?*/ true,
                      /* PostClang Link? */ false);

  CmdArgs.push_back("-o");
  CmdArgs.push_back(OutputFileName);
  C.addCommand(std::make_unique<Command>(
      JA, *this,
      Args.MakeArgString(C.getDriver().Dir + "/clang-build-select-link"),
      CmdArgs, Inputs));

#if 0
  // Save the output of our custom linker
  {
  ArgStringList cpArgs;
  cpArgs.push_back(OutputFileName);
  cpArgs.push_back("/tmp/select_out.bc");
  C.addCommand(std::make_unique<Command>(
      JA, *this, Args.MakeArgString("/bin/cp"), cpArgs, Inputs));
  }
#endif

  return OutputFileName;
}

const char *AMDGCN::OpenMPLinker::constructLLVMLinkCommand(
    Compilation &C, const JobAction &JA, const InputInfoList &Inputs,
    const ArgList &Args, StringRef SubArchName, StringRef OutputFilePrefix,
    StringRef overrideInputsFile) const {
  ArgStringList CmdArgs;
  // Add the input bc's created by compile step.
  if (overrideInputsFile.empty()) {
    for (const auto &II : Inputs)
      if (II.isFilename())
        CmdArgs.push_back(II.getFilename());
  } else
    CmdArgs.push_back(Args.MakeArgString(overrideInputsFile));

  // for OpenMP, we already did this in clang-build-select-link
  if (JA.getOffloadingDeviceKind() != Action::OFK_OpenMP)
     AddStaticDeviceLibs(C, *this, JA, Inputs, Args, CmdArgs, "amdgcn",
                      SubArchName,
                      /* bitcode SDL?*/ true,
                      /* PostClang Link? */ false);

  // Add an intermediate output file.
  CmdArgs.push_back("-o");
  std::string TmpName =
      C.getDriver().GetTemporaryPath(OutputFilePrefix.str() + "-linked", "bc");
  const char *OutputFileName =
      C.addTempFile(C.getArgs().MakeArgString(TmpName));
  CmdArgs.push_back(OutputFileName);
  SmallString<128> ExecPath(C.getDriver().Dir);
  llvm::sys::path::append(ExecPath, "llvm-link");
  const char *Exec = Args.MakeArgString(ExecPath);
  C.addCommand(std::make_unique<Command>(JA, *this, Exec, CmdArgs, Inputs));
  return OutputFileName;
}

const char *AMDGCN::OpenMPLinker::constructOptCommand(
    Compilation &C, const JobAction &JA, const InputInfoList &Inputs,
    const llvm::opt::ArgList &Args, llvm::StringRef SubArchName,
    llvm::StringRef OutputFilePrefix, const char *InputFileName) const {
  // Construct opt command.
  ArgStringList OptArgs;
  // The input to opt is the output from llvm-link.
  OptArgs.push_back(InputFileName);
  // Pass optimization arg to opt.
  if (Arg *A = Args.getLastArg(options::OPT_O_Group)) {
    StringRef OOpt = "3";
    if (A->getOption().matches(options::OPT_O4) ||
        A->getOption().matches(options::OPT_Ofast))
      OOpt = "3";
    else if (A->getOption().matches(options::OPT_O0))
      OOpt = "0";
    else if (A->getOption().matches(options::OPT_O)) {
      // -Os, -Oz, and -O(anything else) map to -O2
      OOpt = llvm::StringSwitch<const char *>(A->getValue())
                 .Case("1", "1")
                 .Case("2", "2")
                 .Case("3", "3")
                 .Case("s", "2")
                 .Case("z", "2")
                 .Default("2");
    }
    OptArgs.push_back(Args.MakeArgString("-O" + OOpt));
  }

  // Get the environment variable AOMP_OPT_ARGS and add opt to llc.
  Optional<std::string> OptEnv = llvm::sys::Process::GetEnv("AOMP_OPT_ARGS");
  if (OptEnv.hasValue()) {
    SmallVector<StringRef, 8> Envs;
    SplitString(OptEnv.getValue(), Envs);
    for (StringRef Env : Envs)
      OptArgs.push_back(Args.MakeArgString(Env.trim()));
  }

  OptArgs.push_back("-mtriple=amdgcn-amd-amdhsa");
  OptArgs.push_back(Args.MakeArgString("-mcpu=" + SubArchName));

  for (const Arg *A : Args.filtered(options::OPT_mllvm)) {
    OptArgs.push_back(A->getValue(0));
  }

  OptArgs.push_back("-o");
  std::string TmpFileName = C.getDriver().GetTemporaryPath(
      OutputFilePrefix.str() + "-optimized", "bc");
  const char *OutputFileName =
      C.addTempFile(C.getArgs().MakeArgString(TmpFileName));
  OptArgs.push_back(OutputFileName);
  SmallString<128> OptPath(C.getDriver().Dir);
  llvm::sys::path::append(OptPath, "opt");
  const char *OptExec = Args.MakeArgString(OptPath);
  C.addCommand(std::make_unique<Command>(JA, *this, OptExec, OptArgs, Inputs));
  return OutputFileName;
}

const char *AMDGCN::OpenMPLinker::constructLlcCommand(
    Compilation &C, const JobAction &JA, const InputInfoList &Inputs,
    const llvm::opt::ArgList &Args, llvm::StringRef SubArchName,
    llvm::StringRef OutputFilePrefix, const char *InputFileName) const {
  // Construct llc command.
  ArgStringList LlcArgs{InputFileName, "-mtriple=amdgcn-amd-amdhsa",
                        "-filetype=obj",
                        Args.MakeArgString("-mcpu=" + SubArchName)};

  // Get the environment variable AOMP_LLC_ARGS and add opt to llc.
  Optional<std::string> OptEnv = llvm::sys::Process::GetEnv("AOMP_LLC_ARGS");
  if (OptEnv.hasValue()) {
    SmallVector<StringRef, 8> Envs;
    SplitString(OptEnv.getValue(), Envs);
    for (StringRef Env : Envs)
      LlcArgs.push_back(Args.MakeArgString(Env.trim()));
  }

  if (Args.hasArg(options::OPT_v))
    LlcArgs.push_back(Args.MakeArgString("-amdgpu-dump-hsa-metadata"));

  // Extract all the -m options
  std::vector<llvm::StringRef> Features;
  handleTargetFeaturesGroup(
    Args, Features, options::OPT_m_amdgpu_Features_Group);

  // Add features to mattr such as xnack
  std::string MAttrString = "-mattr=";
  for(auto OneFeature : Features) {
    MAttrString.append(Args.MakeArgString(OneFeature));
    if (OneFeature != Features.back())
      MAttrString.append(",");
  }
  if(!Features.empty())
    LlcArgs.push_back(Args.MakeArgString(MAttrString));

  for (const Arg *A : Args.filtered(options::OPT_mllvm)) {
    LlcArgs.push_back(A->getValue(0));
  }

  // Add output filename
  LlcArgs.push_back("-o");
  std::string LlcOutputFileName =
      C.getDriver().GetTemporaryPath(OutputFilePrefix, "o");
  const char *LlcOutputFile =
      C.addTempFile(C.getArgs().MakeArgString(LlcOutputFileName));
  LlcArgs.push_back(LlcOutputFile);
  SmallString<128> LlcPath(C.getDriver().Dir);
  llvm::sys::path::append(LlcPath, "llc");
  const char *Llc = Args.MakeArgString(LlcPath);
  C.addCommand(std::make_unique<Command>(JA, *this, Llc, LlcArgs, Inputs));
  return LlcOutputFile;
}

void AMDGCN::OpenMPLinker::constructLldCommand(Compilation &C, const JobAction &JA,
                                          const InputInfoList &Inputs,
                                          const InputInfo &Output,
                                          const llvm::opt::ArgList &Args,
                                          const char *InputFileName) const {
  // Construct lld command.
  // The output from ld.lld is an HSA code object file.
  ArgStringList LldArgs{"-flavor",    "gnu", "--no-undefined",
                        "-shared",    "-o",  Output.getFilename(),
                        InputFileName};
  const char *Lld = Args.MakeArgString(getToolChain().GetProgramPath("lld"));
  C.addCommand(std::make_unique<Command>(JA, *this, Lld, LldArgs, Inputs));
}

// For amdgcn the inputs of the linker job are device bitcode and output is
// object file. It calls llvm-link, opt, llc, then lld steps.
void AMDGCN::OpenMPLinker::ConstructJob(Compilation &C, const JobAction &JA,
                                   const InputInfo &Output,
                                   const InputInfoList &Inputs,
                                   const ArgList &Args,
                                   const char *LinkingOutput) const {

  if (JA.getType() == types::TY_HIP_FATBIN) {
    llvm::report_fatal_error("OpenMP linker does not support hip_fatbin");
  }

  assert(getToolChain().getTriple().getArch() == llvm::Triple::amdgcn &&
         "Unsupported target");

  std::string SubArchName = JA.getOffloadingArch();
  assert(StringRef(SubArchName).startswith("gfx") && "Unsupported sub arch");

  // Prefix for temporary file name.
  std::string Prefix;
  for (const auto &II : Inputs)
    if (II.isFilename())
      Prefix =
          llvm::sys::path::stem(II.getFilename()).str() + "-" + SubArchName;
  assert(Prefix.length() && "no linker inputs are files ");

  // Each command outputs different files.
  bool DoOverride = JA.getOffloadingDeviceKind() == Action::OFK_OpenMP;
  const char *overrideInputs =
      DoOverride
          ? constructOmpExtraCmds(C, JA, Inputs, Args, SubArchName, Prefix)
          : "";
  const char *LLVMLinkCommand = constructLLVMLinkCommand(
      C, JA, Inputs, Args, SubArchName, Prefix, overrideInputs);
  const char *OptCommand = constructOptCommand(C, JA, Inputs, Args, SubArchName,
                                               Prefix, LLVMLinkCommand);
  const char *LlcCommand =
      constructLlcCommand(C, JA, Inputs, Args, SubArchName, Prefix, OptCommand);
  constructLldCommand(C, JA, Inputs, Output, Args, LlcCommand);
}

AMDGPUOpenMPToolChain::AMDGPUOpenMPToolChain(const Driver &D, const llvm::Triple &Triple,
                           const ToolChain &HostTC, const ArgList &Args,
                           const Action::OffloadKind OK)
    : ROCMToolChain(D, Triple, Args), HostTC(HostTC), OK(OK) {
  // Lookup binaries into the driver directory, this is used to
  // discover the clang-offload-bundler executable.
  getProgramPaths().push_back(getDriver().Dir);
}

void AMDGPUOpenMPToolChain::addClangTargetOptions(
    const llvm::opt::ArgList &DriverArgs,
    llvm::opt::ArgStringList &CC1Args,
    Action::OffloadKind DeviceOffloadingKind) const {
  HostTC.addClangTargetOptions(DriverArgs, CC1Args, DeviceOffloadingKind);

  StringRef GpuArch = DriverArgs.getLastArgValue(options::OPT_march_EQ);
  assert(!GpuArch.empty() && "Must have an explicit GPU arch.");
  (void) GpuArch;
  assert((DeviceOffloadingKind == Action::OFK_HIP ||
          DeviceOffloadingKind == Action::OFK_OpenMP) &&
         "Only HIP offloading kinds are supported for GPUs.");
  auto Kind = llvm::AMDGPU::parseArchAMDGCN(GpuArch);

  CC1Args.push_back("-target-cpu");
  CC1Args.push_back(DriverArgs.MakeArgStringRef(GpuArch));
  CC1Args.push_back("-fcuda-is-device");

  if (DriverArgs.hasFlag(options::OPT_fcuda_flush_denormals_to_zero,
                         options::OPT_fno_cuda_flush_denormals_to_zero, false))
    CC1Args.push_back("-fcuda-flush-denormals-to-zero");

  if (DriverArgs.hasFlag(options::OPT_fcuda_approx_transcendentals,
                         options::OPT_fno_cuda_approx_transcendentals, false))
    CC1Args.push_back("-fcuda-approx-transcendentals");

  if (DriverArgs.hasFlag(options::OPT_fgpu_rdc, options::OPT_fno_gpu_rdc,
                         false))
    CC1Args.push_back("-fgpu-rdc");

  CC1Args.push_back("-fcuda-allow-variadic-functions");

  // Default to "hidden" visibility, as object level linking will not be
  // supported for the foreseeable future.
  if (!DriverArgs.hasArg(options::OPT_fvisibility_EQ,
                         options::OPT_fvisibility_ms_compat) &&
      DeviceOffloadingKind != Action::OFK_OpenMP) {
    CC1Args.append({"-fvisibility", "hidden"});
    CC1Args.push_back("-fapply-global-visibility-to-externs");
  }
  ArgStringList LibraryPaths;

  // Find in --hip-device-lib-path and HIP_LIBRARY_PATH.
  for (auto Path :
       DriverArgs.getAllArgValues(options::OPT_hip_device_lib_path_EQ))
    LibraryPaths.push_back(DriverArgs.MakeArgString(Path));

  addDirectoryList(DriverArgs, LibraryPaths, "", "HIP_DEVICE_LIB_PATH");

  // If device debugging turned on, add specially built bc files
  std::string lib_debug_path = FindDebugInLibraryPath();
  if (!lib_debug_path.empty()) {
    LibraryPaths.push_back(
        DriverArgs.MakeArgString(lib_debug_path + "/libdevice"));
    LibraryPaths.push_back(DriverArgs.MakeArgString(lib_debug_path));
  }

  // Add compiler path libdevice last as lowest priority search
  LibraryPaths.push_back(
      DriverArgs.MakeArgString(getDriver().Dir + "/../amdgcn/bitcode"));
  LibraryPaths.push_back(
      DriverArgs.MakeArgString(getDriver().Dir + "/../../amdgcn/bitcode"));
  LibraryPaths.push_back(
      DriverArgs.MakeArgString(getDriver().Dir + "/../lib/libdevice"));
  LibraryPaths.push_back(DriverArgs.MakeArgString(getDriver().Dir + "/../lib"));
  LibraryPaths.push_back(
      DriverArgs.MakeArgString(getDriver().Dir + "/../../lib/libdevice"));
  LibraryPaths.push_back(DriverArgs.MakeArgString(getDriver().Dir + "/../../lib"));

  llvm::SmallVector<std::string, 10> BCLibs;

  // Add bitcode library in --hip-device-lib.
  for (auto Lib : DriverArgs.getAllArgValues(options::OPT_hip_device_lib_EQ)) {
    BCLibs.push_back(DriverArgs.MakeArgString(Lib));
  }

  // If --hip-device-lib is not set, add the default bitcode libraries.
  if (BCLibs.empty()) {
    // Get the bc lib file name for ISA version. For example,
    // gfx803 => oclc_isa_version_803.amdgcn.bc.
    std::string GFXVersion = GpuArch.drop_front(3).str();
    std::string ISAVerBC = "oclc_isa_version_" + GFXVersion + ".bc";

    bool FTZDAZ = DriverArgs.hasFlag(
      options::OPT_fcuda_flush_denormals_to_zero,
      options::OPT_fno_cuda_flush_denormals_to_zero,
      getDefaultDenormsAreZeroForTarget(Kind));

    std::string FlushDenormalControlBC = FTZDAZ ?
      "oclc_daz_opt_on.bc" :
      "oclc_daz_opt_off.bc";

    llvm::StringRef WaveFrontSizeBC;
    if (stoi(GFXVersion) < 1000)
      WaveFrontSizeBC = "oclc_wavefrontsize64_on.bc";
    else
      WaveFrontSizeBC = "oclc_wavefrontsize64_off.bc";

    // FIXME remove double link of aompextras and hip
    if (DriverArgs.hasArg(options::OPT_cuda_device_only))
      // FIXME: when building aompextras, we need to skip aompextras
      BCLibs.append({"hip.bc", "opencl.bc", "ocml.bc",
                     "ockl.bc", "oclc_finite_only_off.bc",
                     FlushDenormalControlBC,
                     "oclc_correctly_rounded_sqrt_on.bc",
                     "oclc_unsafe_math_off.bc", ISAVerBC,
                     std::string(WaveFrontSizeBC)});

    else
      BCLibs.append(
          {"hip.bc", "opencl.bc", "ocml.bc",
           "ockl.bc", "oclc_finite_only_off.bc",
           FlushDenormalControlBC,
           "oclc_correctly_rounded_sqrt_on.bc",
           "oclc_unsafe_math_off.bc", ISAVerBC,
           DriverArgs.MakeArgString("libaompextras-amdgcn-" + GpuArch + ".bc"),
           std::string(WaveFrontSizeBC)});
  }
  for (auto Lib : BCLibs)
    addBCLib(getDriver(), DriverArgs, CC1Args, LibraryPaths, Lib,
             /* PostClang Link? */ true);
}

llvm::opt::DerivedArgList *
AMDGPUOpenMPToolChain::TranslateArgs(const llvm::opt::DerivedArgList &Args,
                             StringRef BoundArch,
                             Action::OffloadKind DeviceOffloadKind) const {
  DerivedArgList *DAL =
      HostTC.TranslateArgs(Args, BoundArch, DeviceOffloadKind);
  if (!DAL)
    DAL = new DerivedArgList(Args.getBaseArgs());

  const OptTable &Opts = getDriver().getOpts();

  for (Arg *A : Args) {
    DAL->append(A);
  }

  if (!BoundArch.empty()) {
    DAL->eraseArg(options::OPT_march_EQ);
    DAL->AddJoinedArg(nullptr, Opts.getOption(options::OPT_march_EQ), BoundArch);
  }

  return DAL;
}

Tool *AMDGPUOpenMPToolChain::buildLinker() const {
  assert(getTriple().getArch() == llvm::Triple::amdgcn);
  return new tools::AMDGCN::OpenMPLinker(*this);
}

void AMDGPUOpenMPToolChain::addClangWarningOptions(ArgStringList &CC1Args) const {
  HostTC.addClangWarningOptions(CC1Args);
}

ToolChain::CXXStdlibType
AMDGPUOpenMPToolChain::GetCXXStdlibType(const ArgList &Args) const {
  return HostTC.GetCXXStdlibType(Args);
}

void AMDGPUOpenMPToolChain::AddClangSystemIncludeArgs(const ArgList &DriverArgs,
                                              ArgStringList &CC1Args) const {
  // HCC2: Get includes from compiler installation
  const Driver &D = HostTC.getDriver();
  CC1Args.push_back("-internal-isystem");
  CC1Args.push_back(DriverArgs.MakeArgString(D.Dir + "/../include"));
  CC1Args.push_back("-internal-isystem");
  CC1Args.push_back(DriverArgs.MakeArgString(D.Dir + "/../../include"));

  HostTC.AddClangSystemIncludeArgs(DriverArgs, CC1Args);
  // HIP headers need the cuda_wrappers in include path

  CC1Args.push_back("-internal-isystem");
  SmallString<128> P(HostTC.getDriver().ResourceDir);
  llvm::sys::path::append(P, "include/cuda_wrappers");
  CC1Args.push_back(DriverArgs.MakeArgString(P));
}

/// Convert path list to Fortran frontend argument
static void AddFlangSysIncludeArg(const ArgList &DriverArgs,
                                  ArgStringList &Flang1args,
                                  ToolChain::path_list IncludePathList) {
  std::string ArgValue; // Path argument value

  // Make up argument value consisting of paths separated by colons
  bool first = true;
  for (auto P : IncludePathList) {
    if (first) {
      first = false;
    } else {
      ArgValue += ":";
    }
    ArgValue += P;
  }

  // Add the argument
  Flang1args.push_back("-stdinc");
  Flang1args.push_back(DriverArgs.MakeArgString(ArgValue));
}

/// Currently only adding include dir from install directory
void AMDGPUOpenMPToolChain::AddFlangSystemIncludeArgs(const ArgList &DriverArgs,
                                             ArgStringList &Flang1args) const {
  path_list IncludePathList;
  const Driver &D = getDriver();

  if (DriverArgs.hasArg(options::OPT_nostdinc))
    return;

  {
    SmallString<128> P(D.InstalledDir);
    llvm::sys::path::append(P, "../include");
    IncludePathList.push_back(DriverArgs.MakeArgString(P.str()));
  }

  AddFlangSysIncludeArg(DriverArgs, Flang1args, IncludePathList);
  return;
}

void AMDGPUOpenMPToolChain::AddClangCXXStdlibIncludeArgs(const ArgList &Args,
                                                 ArgStringList &CC1Args) const {
  HostTC.AddClangCXXStdlibIncludeArgs(Args, CC1Args);
}

void AMDGPUOpenMPToolChain::AddIAMCUIncludeArgs(const ArgList &Args,
                                        ArgStringList &CC1Args) const {
  HostTC.AddIAMCUIncludeArgs(Args, CC1Args);
}

SanitizerMask AMDGPUOpenMPToolChain::getSupportedSanitizers() const {
  // The AMDGPUOpenMPToolChain only supports sanitizers in the sense that it allows
  // sanitizer arguments on the command line if they are supported by the host
  // toolchain. The AMDGPUOpenMPToolChain will actually ignore any command line
  // arguments for any of these "supported" sanitizers. That means that no
  // sanitization of device code is actually supported at this time.
  //
  // This behavior is necessary because the host and device toolchains
  // invocations often share the command line, so the device toolchain must
  // tolerate flags meant only for the host toolchain.
  return HostTC.getSupportedSanitizers();
}

VersionTuple AMDGPUOpenMPToolChain::computeMSVCVersion(const Driver *D,
                                               const ArgList &Args) const {
  return HostTC.computeMSVCVersion(D, Args);
}
