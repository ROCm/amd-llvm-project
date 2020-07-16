
if(LIBOMP_OMPD_SUPPORT)
set (CMAKE_MODULE_PATH
    "${CMAKE_SOURCE_DIR}/libompd/"
    ${CMAKE_MODULE_PATH}
)

find_package (PythonInterp) #2.7 REQUIRED)
find_package (PythonLibs) #2.7 REQUIRED)

execute_process(COMMAND "${PYTHON_EXECUTABLE}"
	"-mpip"
	"--version"
	OUTPUT_VARIABLE PIP_VERSION_INFO
	RESULT_VARIABLE HAD_ERROR)
string(REGEX REPLACE " " ";" PIP_VERSION_INFO "${PIP_VERSION_INFO}")
list(GET PIP_VERSION_INFO 1 PIP_VERSION)
set(PYSYSFLAG "")
if(PIP_VERSION MATCHES "9.0.1")
  set(PYSYSFLAG "--system")
endif()

include_directories (${OMPD_INCLUDE_PATH})
include_directories (${LIBOMP_INCLUDE_DIR})
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/python-module/loadompd.py
                   COMMAND ${CMAKE_COMMAND} -E env LIBOMP_INCLUDE_DIR=${LIBOMP_INCLUDE_DIR}
                   ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/setup.py -v bdist_wheel -b ${CMAKE_CURRENT_BINARY_DIR}/build -d ${CMAKE_CURRENT_BINARY_DIR}
                   COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/setup.py clean --all
                   COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_SOURCE_DIR}/ompd.egg-info
                   COMMAND ${PYTHON_EXECUTABLE} -m pip install ${PYSYSFLAG} -U -t ${CMAKE_CURRENT_BINARY_DIR}/python-module --no-index -v --find-links=${CMAKE_CURRENT_BINARY_DIR} ompd
                   WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

add_custom_target(ompd_gdb_plugin ALL
                  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/python-module/loadompd.py
                  COMMENT "Building the OMPD GDB plugin")

install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/python-module/ompd  DESTINATION ${OPENMP_INSTALL_LIBDIR})
endif()
