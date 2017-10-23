#include "HID.h"

#include <iostream>

#include "clang/Basic/VirtualFileSystem.h"
#include "llvm/Option/ArgList.h"

using namespace clang::driver;
    
HCCInstallationDetector::HCCInstallationDetector(const Driver &D, const llvm::Triple &HostTriple, const llvm::opt::ArgList &Args) : D(D) {
    BinPath = D.Dir;
    auto &FS = D.getVFS();

    if (BinPath != (ROCmPath + "/bin"))
        InstallMode = false;

    if (!InstallMode) {
        if (const char *p = getenv("HCC_HOME")) {
            IncPath = std::string(p) + "/include";
            LibPath = std::string(p) + "/lib";
        }
        else {
            IncPath = BinPath + "/../../include";
            LibPath = BinPath + "/../../lib";
        }
    }

    if (!FS.exists(IncPath + "/hc.hpp") || !FS.exists(LibPath + "/libmcwamp.a"))
        IsValid = false;
}

void HCCInstallationDetector::AddHCCIncludeArgs(const llvm::opt::ArgList &DriverArgs, llvm::opt::ArgStringList &CC1Args) const {
    if (IsValid) {
        CC1Args.push_back(DriverArgs.MakeArgString("-I" + IncPath));

        if (InstallMode)
            CC1Args.push_back(DriverArgs.MakeArgString(ROCmPath));
    }
}

void HCCInstallationDetector::AddHCCLinkArgs(const llvm::opt::ArgList &Args, llvm::opt::ArgStringList &CmdArgs) const {
    if (IsValid) {
        CmdArgs.push_back(Args.MakeArgString("-L" + LibPath));
        CmdArgs.push_back(Args.MakeArgString("--rpath=" + LibPath));

        CmdArgs.push_back("-ldl");
        CmdArgs.push_back("-lm");
        CmdArgs.push_back("-lpthread");
        CmdArgs.push_back("-lunwind");

        CmdArgs.push_back("-lhc_am");
        CmdArgs.push_back("-lmcwamp");
    }
}

void HCCInstallationDetector::print(raw_ostream &OS) const {
    if (IsValid) {
        OS << "Found HCC headers in " << IncPath << "\n";
        OS << "Found HCC libs in " << LibPath << "\n";
    } else
        OS << "Couldn't find a valid HCC installation" << "\n";
}
