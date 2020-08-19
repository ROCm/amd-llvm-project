##===----------------------------------------------------------------------===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is dual licensed under the MIT and the University of Illinois Open
# Source Licenses. See LICENSE.txt for details.
#
##===----------------------------------------------------------------------===##
#
# Build a plugin for an HSA machine if available.
#
##===----------------------------------------------------------------------===##

################################################################################
# Add check for required compiler

find_package(LLVM QUIET CONFIG PATHS ${LLVM_INSTALL_PREFIX} ${CMAKE_INSTALL_PREFIX} NO_DEFAULT_PATH)

if (LLVM_DIR)
  libomptarget_say("Found LLVM ${LLVM_PACKAGE_VERSION}. Configure: ${LLVM_DIR}/LLVMConfig.cmake")
else()
  libomptarget_say("Not building HSA plugin: AOMP llvm not found")
  return()
endif()
set(LLVM_LIBDIR ${LLVM_INSTALL_PREFIX}/lib)

find_package(Clang QUIET CONFIG PATHS ${LLVM_INSTALL_PREFIX} ${CMAKE_INSTALL_PREFIX} NO_DEFAULT_PATH)
if (CLANG_CMAKE_DIR)
  libomptarget_say("Found Clang ${LLVM_PACKAGE_VERSION}. Configure: ${CLANG_CMAKE_DIR}/ClangConfig.cmake")
else()
  libomptarget_say("Not building HSA plugin: Clang not found")
  return()
endif()

# Add check for required libraries
if(NOT LIBOMPTARGET_DEP_LIBELF_FOUND)
  libomptarget_say("Not building HSA plugin: LIBELF not found")
  return()
endif()

# rocr cmake uses DHSAKMT_INC_PATH, DHSAKMT_LIB_PATH to find roct
# following that, look for DHSA_INC_PATH, DHSA_LIB_PATH, which allows
# builds to use source and library files from various locations

if(ROCM_DIR)
  set(HSA_INC_PATH ${ROCM_DIR}/hsa/include ${ROCM_DIR}/hsa/include/hsa)
  set(HSA_LIB_PATH ${ROCM_DIR}/hsa/lib)
  set(HSAKMT_INC_PATH "")
  set(HSAKMT_LIB_PATH ${ROCM_DIR}/lib)
elseif(NOT (HSA_INC_PATH AND HSA_LIB_PATH AND HSAKMT_INC_PATH AND HSAKMT_LIB_PATH))
  libomptarget_say("Not building HSA plugin: ROCM library paths unspecified")
  return()
endif()

mark_as_advanced(HSA_INC_PATH HSA_LIB_PATH HSAKMT_INC_PATH HSAKMT_LIB_PATH)

if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(ppc64le)|(aarch64)$" AND CMAKE_SYSTEM_NAME MATCHES "Linux")
  libomptarget_say("Not building HSA plugin: only support HSA in Linux x86_64, ppc64le , or aarch64 hosts.")
  return()
endif()
libomptarget_say("Building HSA offloading plugin")

libomptarget_say("HSA plugin: HSA_INC_PATH: ${HSA_INC_PATH}")
libomptarget_say("HSA plugin: HSA_LIB_PATH: ${HSA_LIB_PATH}")
libomptarget_say("HSA plugin: HSAKMT_INC_PATH: ${HSAKMT_INC_PATH}")
libomptarget_say("HSA plugin: HSAKMT_LIB_PATH: ${HSAKMT_LIB_PATH}")

################################################################################
# Define the suffix for the runtime messaging dumps.
add_definitions(-DTARGET_NAME=HSA)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(ppc64le)|(aarch64)$")
   add_definitions(-DLITTLEENDIAN_CPU=1)
endif()

if(CMAKE_BUILD_TYPE MATCHES Debug)
  add_definitions(-DHSA_ERROR_REPORT)
  add_definitions(-DDEBUG)
endif()

include_directories(
  ${HSA_INC_PATH}
  ${CLANG_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/impl
)

add_library(omptarget.rtl.hsa SHARED
      impl/atmi.cpp
      impl/atmi_interop_hsa.cpp
      impl/data.cpp
      impl/machine.cpp
      impl/system.cpp
      impl/utils.cpp
      impl/msgpack.cpp
      src/rtl.cpp 
      )

add_definitions(${OPENMP_SOURCE_DEBUG_MAP})

# Install plugin under the lib destination folder.
# When we build for debug, OPENMP_LIBDIR_SUFFIX get set to -debug
install(TARGETS omptarget.rtl.hsa LIBRARY DESTINATION "lib${OPENMP_LIBDIR_SUFFIX}")

# We need the AOMP specific build of ATMI that has HSA_INTEROP turned on. 
# Also, the AOMP specific build of ATMI has seperate release and debug builds. 

add_dependencies(omptarget.rtl.hsa hsa-runtime64 hsakmt)
target_link_libraries(
  omptarget.rtl.hsa
  hostrpc_services
  -lpthread -ldl -Wl,-rpath,${LLVM_LIBDIR}${OPENMP_LIBDIR_SUFFIX}
  -L${HSA_LIB_PATH} -L${HSAKMT_LIB_PATH} -lhsa-runtime64 -lhsakmt -Wl,-rpath,${HSA_LIB_PATH},-rpath,${HSAKMT_LIB_PATH}
  -lelf
  "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/../exports"
  "-Wl,-z,defs"
  )

# Report to the parent scope that we are building a plugin for hsa
set(LIBOMPTARGET_SYSTEM_TARGETS "${LIBOMPTARGET_SYSTEM_TARGETS} amdgcn-amd-amdhsa" PARENT_SCOPE)

