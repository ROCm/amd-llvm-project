#ifndef HID
#define HID

#include "clang/Driver/Driver.h"

namespace clang {
    namespace driver {
        class HCCInstallationDetector {
        private:
            const Driver &D;
            bool IsValid = true;
            bool InstallMode = true;

            std::string ROCmPath = "/opt/rocm";
            std::string BinPath = ROCmPath + "/bin";
            std::string IncPath = ROCmPath + "/include/hcc";
            std::string LibPath = ROCmPath + "/lib";
            
        public:
            HCCInstallationDetector(const Driver &D, const llvm::Triple &HostTriple, const llvm::opt::ArgList &Args);
            
            void AddHCCIncludeArgs(const llvm::opt::ArgList &DriverArgs, llvm::opt::ArgStringList &CC1Args) const;

            void AddHCCLinkArgs(const llvm::opt::ArgList &Args, llvm::opt::ArgStringList &CmdArgs) const;
            
            bool isValid() const { return IsValid; }
            
            void print(raw_ostream &OS) const;
        };
    }
}

#endif
