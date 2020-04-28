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
 ******************************************************************************/

#ifndef COMGR_DATA_H_
#define COMGR_DATA_H_
#if !HSA_USE_EXTERNAL_COMGR

#include "openmp_amd_comgr.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/MsgPackDocument.h"


namespace COMGR {
struct DataMeta;

/// Update @p Dest to point to a newly allocated C-style (null terminated)
/// string with the contents of @p Src, optionally updating @p Size with the
/// length of the string (not including the null terminator).
///
/// If @p Dest is non-null, it will first be freed.
///
/// @p Src may contain null bytes.
amd_comgr_status_t setCStr(char *&Dest, llvm::StringRef Src,
                           size_t *Size = nullptr);

/// Return `true` if the kind is valid, or false otherwise.
bool isDataKindValid(amd_comgr_data_kind_t DataKind);

struct DataObject {

  // Allocate a new DataObject and return a pointer to it.
  static DataObject *allocate(amd_comgr_data_kind_t DataKind);

  // Decrement the refcount of this DataObject, and free it when it reaches 0.
  void release();

  static amd_comgr_data_t convert(DataObject *Data) {
    amd_comgr_data_t Handle = {
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(Data))};
    return Handle;
  }

  static const amd_comgr_data_t convert(const DataObject *Data) {
    const amd_comgr_data_t Handle = {
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(Data))};
    return Handle;
  }

  static DataObject *convert(amd_comgr_data_t Data) {
    return reinterpret_cast<DataObject *>(Data.handle);
  }

  bool hasValidDataKind() { return isDataKindValid(DataKind); }

  amd_comgr_status_t setName(llvm::StringRef Name);
  amd_comgr_status_t setData(llvm::StringRef Data);
  void setMetadata(DataMeta *Metadata);

  amd_comgr_data_kind_t DataKind;
  char *Data;
  char *Name;
  size_t Size;
  int RefCount;

private:
  // We require this type be allocated via new, specifically through calling
  // allocate, because we want to be able to `delete this` in release. To make
  // sure the type is not constructed without new, or destructed without
  // checking the reference count, we mark the constructor and destructor
  // private.
  DataObject(amd_comgr_data_kind_t Kind);
  ~DataObject();
};



// Elements common to all DataMeta which refer to the same "document".
struct MetaDocument {
  // The MsgPack document, which owns all memory allocated during parsing.
  llvm::msgpack::Document Document;
  // The MsgPack parser is zero-copy, so we retain a copy of the input buffer.
  std::string RawDocument;
  // The old YAML parser would produce the strings "true" and "false" for
  // booleans, whereas the old MsgPack parser produced "0" and "1". The new
  // universal parser produces "true" and "false", but we need to remain
  // backwards compatible, so we set a flag when parsing MsgPack.
  bool EmitIntegerBooleans = false;
};

struct DataMeta {
  static amd_comgr_metadata_node_t convert(DataMeta *Meta) {
    amd_comgr_metadata_node_t Handle = {
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(Meta))};
    return Handle;
  }

  static const amd_comgr_metadata_node_t convert(const DataMeta *Meta) {
    const amd_comgr_metadata_node_t Handle = {
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(Meta))};
    return Handle;
  }

  static DataMeta *convert(amd_comgr_metadata_node_t Meta) {
    return reinterpret_cast<DataMeta *>(Meta.handle);
  }

  amd_comgr_metadata_kind_t getMetadataKind();
  // Get the canonical string representation of @p DocNode, assuming
  // it is a scalar node.
  std::string convertDocNodeToString(llvm::msgpack::DocNode DocNode);

  // This DataMeta's "meta document", shared by all instances derived from the
  // same metadata.
  std::shared_ptr<MetaDocument> MetaDoc;
  // This DataMeta's "view" into the shared llvm::msgpack::Document.
  llvm::msgpack::DocNode DocNode;
};


} // namespace COMGR

#endif // !HSA_USE_EXTERNAL_COMGR
#endif // header guard
