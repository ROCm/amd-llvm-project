/*******************************************************************************
*
* University of Illinois/NCSA
* Open Source License
*
* Copyright (c) 2018 Advanced Micro Devices, Inc. All Rights Reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* with the Software without restriction, including without limitation the
* rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
* sell copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
*     * Redistributions of source code must retain the above copyright notice,
*       this list of conditions and the following disclaimers.
*
*     * Redistributions in binary form must reproduce the above copyright
*       notice, this list of conditions and the following disclaimers in the
*       documentation and/or other materials provided with the distribution.
*
*     * Neither the names of Advanced Micro Devices, Inc. nor the names of its
*       contributors may be used to endorse or promote products derived from
*       this Software without specific prior written permission.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
* THE SOFTWARE.
*
*******************************************************************************/

#ifndef OPENMP_AMD_COMGR_H_
#define OPENMP_AMD_COMGR_H_

/*******************************************************************************
 *
 * This runtime library works by parsing information from device images.
 * The implementation of this parsing varies with different distributions of the
 * toolchain. This file helps abstract the differences.
 *
 * ROCm:
 *  The ROCm toolchain uses a library named ROCm-CompilerSupport, which exports
 *  a header amd_comgr, to implement image parsing and various other tasks. This
 *  is the primary use case for setting HSA_USE_EXTERNAL_COMGR
 *
 * LLVM trunk:
 *   The llvm toolchain prefers to keep the openmp host runtime standalone. When
 *   compiling this library with trunk, the subset of amd_comgr.h is available
 *   in this header file, with a corresponding implementation in the library
 *
 * AOMP:
 *   The AOMP library follows LLVM trunk as above
 *
 * Other:
 *   Other toolchains may prefer to use ROCm-CompilerSupport, this local
 *   implementation or do something entirely different
 *
 ********************************************************************************/

#if HSA_USE_EXTERNAL_COMGR

#include "amd_comgr.h"

#else

// This is a the subset of amd_comgr.h used by this library
// It is a direct copy & paste so that diff can be used to inspect whether the
// version in ROCm-CompilerSupport has diverged, and so that the comments are
// available to users of this local implementation.

#include <stddef.h> /* size_t */
#include <stdint.h>

#ifndef __cplusplus
#include <stdbool.h>  /* bool */
#endif /* __cplusplus */

/* Placeholder for calling convention and import/export macros */
#ifndef AMD_CALL
#define AMD_CALL
#endif

#ifndef AMD_EXPORT_DECORATOR
#ifdef __GNUC__
#define AMD_EXPORT_DECORATOR __attribute__ ((visibility ("default")))
#else
#define AMD_EXPORT_DECORATOR __declspec(dllexport)
#endif
#endif

#ifndef AMD_IMPORT_DECORATOR
#ifdef __GNUC__
#define AMD_IMPORT_DECORATOR
#else
#define AMD_IMPORT_DECORATOR __declspec(dllimport)
#endif
#endif

#define AMD_API_EXPORT AMD_EXPORT_DECORATOR AMD_CALL
#define AMD_API_IMPORT AMD_IMPORT_DECORATOR AMD_CALL

#ifndef AMD_API
#ifdef AMD_EXPORT
#define AMD_API AMD_API_EXPORT
#else
#define AMD_API AMD_API_IMPORT
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif  /* __cplusplus */

/** \defgroup codeobjectmanager Code Object Manager
 *  @{
 *
 * @brief The code object manager is a callable library that provides
 * operations for creating and inspecting code objects.
 *
 * The library provides handles to various objects. Concurrent execution of
 * operations is supported provided all objects accessed by each concurrent
 * operation are disjoint. For example, the @p amd_comgr_data_set_t handles
 * passed to operations must be disjoint, together with all the @p
 * amd_comgr_data_t handles that have been added to it. The exception is that
 * the default device library data object handles can be non-disjoint as they
 * are imutable.
 *
 * The library supports generating and inspecting code objects that
 * contain machine code for a certain set of instruction set
 * arhitectures (isa). The set of isa supported and information about
 * the properties of the isa can be queried.
 *
 * The library supports performing an action that can take data
 * objects of one kind, and generate new data objects of another kind.
 *
 * Data objects are referenced using handles using @p
 * amd_comgr_data_t. The kinds of data objects are given
 * by @p amd_comgr_data_kind_t.
 *
 * To perform an action, two @p amd_comgr_data_set_t
 * objects are created. One is used to hold all the data objects
 * needed by an action, and other is updated by the action with all
 * the result data objects. In addition, an @p
 * amd_comgr_action_info_t is created to hold
 * information that controls the action. These are then passed to @p
 * amd_comgr_do_action to perform an action specified by
 * @p amd_comgr_action_kind_t.
 *
 * Some data objects can have associated metadata. There are
 * operations for querying this metadata.
 *
 * The default device library that satisfies the requirements of the
 * compiler action can be obtained.
 *
 * The library inspects some environment variables to aid in debugging. These
 * include:
 * - @p AMD_COMGR_SAVE_TEMPS: If this is set, and is not "0", the library does
 *   not delete temporary files generated while executing compilation actions.
 *   These files do not appear in the current working directory, but are
 *   instead left in a platform-specific temporary directory (/tmp on Linux and
 *   C:\Temp or the path found in the TEMP environment variable on Windows).
 * - @p AMD_COMGR_REDIRECT_LOGS: If this is not set, or is set to "0", logs are
 *   returned to the caller as normal. If this is set to "stdout"/"-" or
 *   "stderr", logs are instead redirected to the standard output or error
 *   stream, respectively. If this is set to any other value, it is interpreted
 *   as a filename which logs should be appended to. Logs may be redirected
 *   irrespective of whether logging is enabled.
 * - @p AMD_COMGR_EMIT_VERBOSE_LOGS: If this is set, and is not "0", logs will
 *   include additional Comgr-specific informational messages.
 */

/**
 * @brief Status codes.
 */
typedef enum amd_comgr_status_s {
  /**
   * The function has been executed successfully.
   */
  AMD_COMGR_STATUS_SUCCESS = 0x0,
  /**
   * A generic error has occurred.
   */
  AMD_COMGR_STATUS_ERROR = 0x1,
  /**
   * One of the actual arguments does not meet a precondition stated
   * in the documentation of the corresponding formal argument.
   */
  AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT = 0x2,
  /**
   * Failed to allocate the necessary resources.
   */
  AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES = 0x3,
} amd_comgr_status_t;


/**
 * @brief The kinds of data supported.
 */
typedef enum amd_comgr_data_kind_s {
  /**
   * No data is available.
   */
  AMD_COMGR_DATA_KIND_UNDEF = 0x0,
  /**
   * The data is a textual main source.
   */
  AMD_COMGR_DATA_KIND_SOURCE = 0x1,
  /**
   * The data is a textual source that is included in the main source
   * or other include source.
   */
  AMD_COMGR_DATA_KIND_INCLUDE = 0x2,
  /**
   * The data is a precompiled-header source that is included in the main
   * source or other include source.
   */
  AMD_COMGR_DATA_KIND_PRECOMPILED_HEADER = 0x3,
  /**
   * The data is a diagnostic output.
   */
  AMD_COMGR_DATA_KIND_DIAGNOSTIC = 0x4,
  /**
   * The data is a textual log output.
   */
  AMD_COMGR_DATA_KIND_LOG = 0x5,
  /**
   * The data is compiler LLVM IR bit code for a specific isa.
   */
  AMD_COMGR_DATA_KIND_BC = 0x6,
  /**
   * The data is a relocatable machine code object for a specific isa.
   */
  AMD_COMGR_DATA_KIND_RELOCATABLE = 0x7,
  /**
   * The data is an executable machine code object for a specific
   * isa. An executable is the kind of code object that can be loaded
   * and executed.
   */
  AMD_COMGR_DATA_KIND_EXECUTABLE = 0x8,
  /**
   * The data is a block of bytes.
   */
  AMD_COMGR_DATA_KIND_BYTES = 0x9,
  /**
   * The data is a fat binary (clang-offload-bundler output).
   */
  AMD_COMGR_DATA_KIND_FATBIN = 0x10,
  /**
   * Marker for last valid data kind.
   */
  AMD_COMGR_DATA_KIND_LAST = AMD_COMGR_DATA_KIND_FATBIN
} amd_comgr_data_kind_t;

/**
 * @brief A handle to a data object.
 *
 * Data objects are used to hold the data which is either an input or
 * output of a code object manager action.
 */
typedef struct amd_comgr_data_s {
  uint64_t handle;
} amd_comgr_data_t;

/**
 * @brief A handle to a metadata node.
 *
 * A metadata node handle is used to traverse the metadata associated
 * with a data node.
 */
typedef struct amd_comgr_metadata_node_s {
  uint64_t handle;
} amd_comgr_metadata_node_t;

/**
 * @brief Create a data object that can hold data of a specified kind.
 *
 * Data objects are reference counted and are destroyed when the
 * reference count reaches 0. When a data object is created its
 * reference count is 1, it has 0 bytes of data, it has an empty name,
 * and it has no metadata.
 *
 * @param[in] kind The kind of data the object is intended to hold.
 *
 * @param[out] data A handle to the data object created. Its reference
 * count is set to 1.
 *
 * @retval ::AMD_COMGR_STATUS_SUCCESS The function has
 * been executed successfully.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT @p
 * kind is an invalid data kind, or @p
 * AMD_COMGR_DATA_KIND_UNDEF. @p data is NULL.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
 * Unable to create the data object as out of resources.
 */
amd_comgr_status_t AMD_API
amd_comgr_create_data(
  amd_comgr_data_kind_t kind,
  amd_comgr_data_t *data);

 /**
 * @brief Indicate that no longer using a data object handle.
 *
 * The reference count of the associated data object is
 * decremented. If it reaches 0 it is destroyed.
 *
 * @param[in] data The data object to release.
 *
 * @retval ::AMD_COMGR_STATUS_SUCCESS The function has
 * been executed successfully.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT @p
 * data is an invalid data object, or has kind @p
 * AMD_COMGR_DATA_KIND_UNDEF.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
 * Unable to update the data object as out of resources.
 */
amd_comgr_status_t AMD_API
amd_comgr_release_data(
  amd_comgr_data_t data);


/**
 * @brief Set the data content of a data object to the specified
 * bytes.
 *
 * Any previous value of the data object is overwritten. Any metadata
 * associated with the data object is also replaced which invalidates
 * all metadata handles to the old metadata.
 *
 * @param[in] data The data object to update.
 *
 * @param[in] size The number of bytes in the data specified by @p bytes.
 *
 * @param[in] bytes The bytes to set the data object to. The bytes are
 * copied into the data object and can be freed after the call.
 *
 * @retval ::AMD_COMGR_STATUS_SUCCESS The function has
 * been executed successfully.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT @p
 * data is an invalid data object, or has kind @p
 * AMD_COMGR_DATA_KIND_UNDEF.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
 * Unable to update the data object as out of resources.
 */
amd_comgr_status_t AMD_API
amd_comgr_set_data(
  amd_comgr_data_t data,
  size_t size,
  const char* bytes);

/**
 * @brief Set the name associated with a data object.
 *
 * When compiling, the fle name of an include directive is used to
 * reference the contents of the include data object with the same
 * name. The name may also be used for other data objects in log and
 * diagnostic output.
 *
 * @param[in] data The data object to update.
 *
 * @param[in] name A null terminated string that specifies the name to
 * use for the data object. If NULL then the name is set to the empty
 * string.
 *
 * @retval ::AMD_COMGR_STATUS_SUCCESS The function has
 * been executed successfully.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT @p
 * data is an invalid data object, or has kind @p
 * AMD_COMGR_DATA_KIND_UNDEF.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
 * Unable to update the data object as out of resources.
 */
amd_comgr_status_t AMD_API
amd_comgr_set_data_name(
  amd_comgr_data_t data,
  const char* name);

/**
 * @brief Get the data contents, and/or the size of the data
 * associated with a data object.
 *
 * @param[in] data The data object to query.
 *
 * @param[in, out] size On entry, the size of @p bytes. On return, if @p bytes
 * is NULL, set to the size of the data object contents.
 *
 * @param[out] bytes If not NULL, then the first @p size bytes of the
 * data object contents is copied. If NULL, no data is copied, and
 * only @p size is updated (useful in order to find the size of buffer
 * required to copy the data).
 *
 * @retval ::AMD_COMGR_STATUS_SUCCESS The function has
 * been executed successfully.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT @p
 * data is an invalid data object, or has kind @p
 * AMD_COMGR_DATA_KIND_UNDEF. @p size is NULL.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
 * Unable to update the data object as out of resources.
 */
amd_comgr_status_t AMD_API
amd_comgr_get_data(
  amd_comgr_data_t data,
  size_t *size,
  char *bytes);

/**
 * @brief Get the data object name and/or name length.
 *
 * @param[in] data The data object to query.
 *
 * @param[in, out] size On entry, the size of @p name. On return, if @p name is
 * NULL, set to the size of the data object name including the terminating null
 * character.
 *
 * @param[out] name If not NULL, then the first @p size characters of the
 * data object name are copied. If NULL, no name is copied, and
 * only @p size is updated (useful in order to find the size of buffer
 * required to copy the name).
 *
 * @retval ::AMD_COMGR_STATUS_SUCCESS The function has
 * been executed successfully.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT @p
 * data is an invalid data object, or has kind @p
 * AMD_COMGR_DATA_KIND_UNDEF. @p size is NULL.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
 * Unable to update the data object as out of resources.
 */
amd_comgr_status_t AMD_API
amd_comgr_get_data_name(
  amd_comgr_data_t data,
  size_t *size,
  char *name);

 /**
 * @brief Get a handle to the metadata of a data object.
 *
 * @param[in] data The data object to query.
 *
 * @param[out] metadata A handle to the metadata of the data
 * object. If the data object has no metadata then the returned handle
 * has a kind of @p AMD_COMGR_METADATA_KIND_NULL. The
 * handle must be destroyed using @c amd_comgr_destroy_metadata.
 *
 * @retval ::AMD_COMGR_STATUS_SUCCESS The function has
 * been executed successfully.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT @p
 * data is an invalid data object, or has kind @p
 * AMD_COMGR_DATA_KIND_UNDEF. @p metadata is NULL.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
 * Unable to update the data object as out of resources.
 */
amd_comgr_status_t AMD_API
amd_comgr_get_data_metadata(
  amd_comgr_data_t data,
  amd_comgr_metadata_node_t *metadata);

/**
 * @brief Destroy a metadata handle.
 *
 * @param[in] metadata A metadata handle to destroy.
 *
 * @retval ::AMD_COMGR_STATUS_SUCCESS The function has been executed
 * successfully.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT @p metadata is an invalid
 * metadata handle.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES Unable to update metadata
 * handle as out of resources.
 */
amd_comgr_status_t AMD_API
amd_comgr_destroy_metadata(amd_comgr_metadata_node_t metadata);


/**
 * @brief The kinds of metadata nodes.
 */
typedef enum amd_comgr_metadata_kind_s {
  /**
   * The NULL metadata handle.
   */
  AMD_COMGR_METADATA_KIND_NULL = 0x0,
  /**
   * A sting value.
   */
  AMD_COMGR_METADATA_KIND_STRING = 0x1,
  /**
   * A map that consists of a set of key and value pairs.
   */
  AMD_COMGR_METADATA_KIND_MAP = 0x2,
  /**
   * A list that consists of a sequence of values.
   */
  AMD_COMGR_METADATA_KIND_LIST = 0x3,
  /**
   * Marker for last valid metadata kind.
   */
  AMD_COMGR_METADATA_KIND_LAST = AMD_COMGR_METADATA_KIND_LIST
} amd_comgr_metadata_kind_t;

/**
 * @brief Get the kind of the metadata node.
 *
 * @param[in] metadata The metadata node to query.
 *
 * @param[out] kind The kind of the metadata node.
 *
 * @retval ::AMD_COMGR_STATUS_SUCCESS The function has
 * been executed successfully.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT @p
 * metadata is an invalid metadata node. @p kind is NULL.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
 * Unable to create the data object as out of resources.
 */
amd_comgr_status_t AMD_API
amd_comgr_get_metadata_kind(
  amd_comgr_metadata_node_t metadata,
  amd_comgr_metadata_kind_t *kind);

/**
 * @brief Get the string and/or string length from a metadata string
 * node.
 *
 * @param[in] metadata The metadata node to query.
 *
 * @param[in, out] size On entry, the size of @p string. On return, if @p
 * string is NULL, set to the size of the string including the terminating null
 * character.
 *
 * @param[out] string If not NULL, then the first @p size characters
 * of the string are copied. If NULL, no string is copied, and only @p
 * size is updated (useful in order to find the size of buffer required
 * to copy the string).
 *
 * @retval ::AMD_COMGR_STATUS_SUCCESS The function has
 * been executed successfully.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT @p
 * metadata is an invalid metadata node, or does not have kind @p
 * AMD_COMGR_METADATA_KIND_STRING. @p size is NULL.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
 * Unable to update the data object as out of resources.
 */
amd_comgr_status_t AMD_API
amd_comgr_get_metadata_string(
  amd_comgr_metadata_node_t metadata,
  size_t *size,
  char *string);

/**
 * @brief Get the map size from a metadata map node.
 *
 * @param[in] metadata The metadata node to query.
 *
 * @param[out] size The number of entries in the map.
 *
 * @retval ::AMD_COMGR_STATUS_SUCCESS The function has
 * been executed successfully.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT @p
 * metadata is an invalid metadata node, or not of kind @p
 * AMD_COMGR_METADATA_KIND_MAP. @p size is NULL.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
 * Unable to update the data object as out of resources.
 */
amd_comgr_status_t AMD_API
amd_comgr_get_metadata_map_size(
  amd_comgr_metadata_node_t metadata,
  size_t *size);

/**
 * @brief Iterate over the elements a metadata map node.
 *
 * @warning The metadata nodes which are passed to the callback are not owned
 * by the callback, and are freed just after the callback returns. The callback
 * must not save any references to its parameters between iterations.
 *
 * @param[in] metadata The metadata node to query.
 *
 * @param[in] callback The function to call for each entry in the map. The
 * entry's key is passed in @p key, the entry's value is passed in @p value, and
 * @p user_data is passed as @p user_data. If the function returns with a status
 * other than @p AMD_COMGR_STATUS_SUCCESS then iteration is stopped.
 *
 * @param[in] user_data The value to pass to each invocation of @p
 * callback. Allows context to be passed into the call back function.
 *
 * @retval ::AMD_COMGR_STATUS_SUCCESS The function has
 * been executed successfully.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR An error was
 * reported by @p callback.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT @p
 * metadata is an invalid metadata node, or not of kind @p
 * AMD_COMGR_METADATA_KIND_MAP. @p callback is NULL.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
 * Unable to iterate the metadata as out of resources.
 */
amd_comgr_status_t AMD_API
amd_comgr_iterate_map_metadata(
  amd_comgr_metadata_node_t metadata,
  amd_comgr_status_t (*callback)(
    amd_comgr_metadata_node_t key,
    amd_comgr_metadata_node_t value,
    void *user_data),
  void *user_data);

/**
 * @brief Use a string key to lookup an element of a metadata map
 * node and return the entry value.
 *
 * @param[in] metadata The metadata node to query.
 *
 * @param[in] key A null terminated string that is the key to lookup.
 *
 * @param[out] value The metadata node of the @p key element of the
 * @p metadata map metadata node. The handle must be destroyed
 * using @c amd_comgr_destroy_metadata.
 *
 * @retval ::AMD_COMGR_STATUS_SUCCESS The function has
 * been executed successfully.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR The map has no entry
 * with a string key with the value @p key.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT @p
 * metadata is an invalid metadata node, or not of kind @p
 * AMD_COMGR_METADATA_KIND_MAP. @p key or @p value is
 * NULL.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
 * Unable to lookup metadata as out of resources.
 */
amd_comgr_status_t AMD_API
amd_comgr_metadata_lookup(
  amd_comgr_metadata_node_t metadata,
  const char *key,
  amd_comgr_metadata_node_t *value);

/**
 * @brief Get the list size from a metadata list node.
 *
 * @param[in] metadata The metadata node to query.
 *
 * @param[out] size The number of entries in the list.
 *
 * @retval ::AMD_COMGR_STATUS_SUCCESS The function has
 * been executed successfully.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT @p
 * metadata is an invalid metadata node, or does nopt have kind @p
 * AMD_COMGR_METADATA_KIND_LIST. @p size is NULL.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
 * Unable to update the data object as out of resources.
 */
amd_comgr_status_t AMD_API
amd_comgr_get_metadata_list_size(
  amd_comgr_metadata_node_t metadata,
  size_t *size);

/**
 * @brief Return the Nth metadata node of a list metadata node.
 *
 * @param[in] metadata The metadata node to query.
 *
 * @param[in] index The index being requested. The first list element
 * is index 0.
 *
 * @param[out] value The metadata node of the @p index element of the
 * @p metadata list metadata node. The handle must be destroyed
 * using @c amd_comgr_destroy_metadata.
 *
 * @retval ::AMD_COMGR_STATUS_SUCCESS The function has
 * been executed successfully.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT @p
 * metadata is an invalid metadata node or not of kind @p
 * AMD_COMGR_METADATA_INFO_LIST. @p index is greater
 * than the number of list elements. @p value is NULL.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
 * Unable to update action data object as out of resources.
 */
amd_comgr_status_t AMD_API
amd_comgr_index_list_metadata(
  amd_comgr_metadata_node_t metadata,
  size_t index,
  amd_comgr_metadata_node_t *value);

/** @} */

#ifdef __cplusplus
}  /* end extern "C" block */
#endif

#endif  /* HSA_USE_EXTERNAL_COMGR */
#endif  /* header guard */
