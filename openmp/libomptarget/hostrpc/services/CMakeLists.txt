

add_library(hostrpc_services STATIC hostcall.cpp hostrpc_execute_service.c hostrpc_externs.c)
set_property(TARGET hostrpc_services PROPERTY POSITION_INDEPENDENT_CODE ON)
include_directories(${CMAKE_INSTALL_PREFIX}/include/hsa)
