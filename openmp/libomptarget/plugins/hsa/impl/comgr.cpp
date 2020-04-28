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
#if !HSA_USE_EXTERNAL_COMGR
#include "comgr.h"
#include "comgr-metadata.h"
#include <string>

using namespace llvm;
using namespace COMGR;

bool COMGR::isDataKindValid(amd_comgr_data_kind_t DataKind) {
  return DataKind > AMD_COMGR_DATA_KIND_UNDEF &&
         DataKind <= AMD_COMGR_DATA_KIND_LAST;
}

amd_comgr_status_t COMGR::setCStr(char *&Dest, StringRef Src, size_t *Size) {
  free(Dest);
  Dest = reinterpret_cast<char *>(malloc(Src.size() + 1));
  if (!Dest)
    return AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES;
  memcpy(Dest, Src.data(), Src.size());
  Dest[Src.size()] = '\0';
  if (Size)
    *Size = Src.size();
  return AMD_COMGR_STATUS_SUCCESS;
}



DataObject::DataObject(amd_comgr_data_kind_t DataKind)
    : DataKind(DataKind), Data(nullptr), Name(nullptr), Size(0), RefCount(1)
 {}

DataObject::~DataObject() {
  DataKind = AMD_COMGR_DATA_KIND_UNDEF;
  free(Data);
  free(Name);
  Size = 0;

}

DataObject *DataObject::allocate(amd_comgr_data_kind_t DataKind) {
  return new (std::nothrow) DataObject(DataKind);
}

void DataObject::release() {
  if (--RefCount == 0)
    delete this;
}

amd_comgr_status_t DataObject::setName(llvm::StringRef Name) {
  return setCStr(this->Name, Name);
}

amd_comgr_status_t DataObject::setData(llvm::StringRef Data) {
  return setCStr(this->Data, Data, &Size);
}


amd_comgr_metadata_kind_t DataMeta::getMetadataKind() {
  if (DocNode.isScalar())
    return AMD_COMGR_METADATA_KIND_STRING;
  else if (DocNode.isArray())
    return AMD_COMGR_METADATA_KIND_LIST;
  else if (DocNode.isMap())
    return AMD_COMGR_METADATA_KIND_MAP;
  else
    // treat as NULL
    return AMD_COMGR_METADATA_KIND_NULL;
}

std::string DataMeta::convertDocNodeToString(msgpack::DocNode DocNode) {
  assert(DocNode.isScalar() && "cannot convert non-scalar DocNode to string");
  if (MetaDoc->EmitIntegerBooleans &&
      DocNode.getKind() == msgpack::Type::Boolean)
    return DocNode.getBool() ? "1" : "0";
  return DocNode.toString();
}

amd_comgr_status_t AMD_API
    // NOLINTNEXTLINE(readability-identifier-naming)
    amd_comgr_status_string
    //
    (amd_comgr_status_t Status, const char **StatusString) {
  if (!StatusString || Status < AMD_COMGR_STATUS_SUCCESS ||
      Status > AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  switch (Status) {
  case AMD_COMGR_STATUS_SUCCESS:
    *StatusString = "SUCCESS";
    break;
  case AMD_COMGR_STATUS_ERROR:
    *StatusString = "ERROR";
    break;
  case AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT:
    *StatusString = "INVALID_ARGUMENT";
    break;
  case AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES:
    *StatusString = "OUT_OF_RESOURCES";
    break;
  }

  return AMD_COMGR_STATUS_SUCCESS;
}

// API functions on Data Object

amd_comgr_status_t AMD_API
    // NOLINTNEXTLINE(readability-identifier-naming)
    amd_comgr_create_data
    //
    (amd_comgr_data_kind_t DataKind, amd_comgr_data_t *Data) {
  if (!Data || DataKind <= AMD_COMGR_DATA_KIND_UNDEF ||
      DataKind > AMD_COMGR_DATA_KIND_LAST)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  DataObject *DataP = DataObject::allocate(DataKind);
  if (!DataP)
    return AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES;

  *Data = DataObject::convert(DataP);

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_API
    // NOLINTNEXTLINE(readability-identifier-naming)
    amd_comgr_release_data
    //
    (amd_comgr_data_t Data) {
  DataObject *DataP = DataObject::convert(Data);

  if (!DataP || !DataP->hasValidDataKind())
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  DataP->release();

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_API
    // NOLINTNEXTLINE(readability-identifier-naming)
    amd_comgr_get_data_kind
    //
    (amd_comgr_data_t Data, amd_comgr_data_kind_t *DataKind) {
  DataObject *DataP = DataObject::convert(Data);

  if (!DataP || !DataP->hasValidDataKind() || !DataKind) {
    *DataKind = AMD_COMGR_DATA_KIND_UNDEF;
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  *DataKind = DataP->DataKind;

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_API
    // NOLINTNEXTLINE(readability-identifier-naming)
    amd_comgr_set_data
    //
    (amd_comgr_data_t Data, size_t Size, const char *Bytes) {
  DataObject *DataP = DataObject::convert(Data);

  if (!DataP || !DataP->hasValidDataKind() || !Size || !Bytes)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  return DataP->setData(StringRef(Bytes, Size));
}

amd_comgr_status_t AMD_API
    // NOLINTNEXTLINE(readability-identifier-naming)
    amd_comgr_get_data
    //
    (amd_comgr_data_t Data, size_t *Size, char *Bytes) {
  DataObject *DataP = DataObject::convert(Data);

  if (!DataP || !DataP->Data || !DataP->hasValidDataKind() || !Size)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  if (Bytes)
    memcpy(Bytes, DataP->Data, *Size);
  else
    *Size = DataP->Size;

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_API
    // NOLINTNEXTLINE(readability-identifier-naming)
    amd_comgr_set_data_name
    //
    (amd_comgr_data_t Data, const char *Name) {
  DataObject *DataP = DataObject::convert(Data);

  if (!DataP || !DataP->hasValidDataKind())
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  return DataP->setName(Name);
}

amd_comgr_status_t AMD_API
    // NOLINTNEXTLINE(readability-identifier-naming)
    amd_comgr_get_data_name
    //
    (amd_comgr_data_t Data, size_t *Size, char *Name) {
  DataObject *DataP = DataObject::convert(Data);

  if (!DataP || !DataP->hasValidDataKind() || !Size)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  if (Name)
    memcpy(Name, DataP->Name, *Size);
  else
    *Size = strlen(DataP->Name) + 1; // include terminating null

  return AMD_COMGR_STATUS_SUCCESS;
}



amd_comgr_status_t AMD_API
    // NOLINTNEXTLINE(readability-identifier-naming)
    amd_comgr_get_data_metadata
    //
    (amd_comgr_data_t Data, amd_comgr_metadata_node_t *MetadataNode) {
  DataObject *DataP = DataObject::convert(Data);

  if (!DataP || !DataP->hasValidDataKind() ||
      DataP->DataKind == AMD_COMGR_DATA_KIND_UNDEF || !MetadataNode)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  std::unique_ptr<DataMeta> MetaP(new (std::nothrow) DataMeta());
  if (!MetaP)
    return AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES;

  MetaDocument *MetaDoc = new (std::nothrow) MetaDocument();
  if (!MetaDoc)
    return AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES;

  MetaP->MetaDoc.reset(MetaDoc);
  MetaP->DocNode = MetaP->MetaDoc->Document.getRoot();

  if (auto Status = metadata::getMetadataRoot(DataP, MetaP.get()))
    return Status;

  // if no metadata found in this data object, still return SUCCESS but
  // with default NULL kind

  *MetadataNode = DataMeta::convert(MetaP.release());

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_API
    // NOLINTNEXTLINE(readability-identifier-naming)
    amd_comgr_destroy_metadata
    //
    (amd_comgr_metadata_node_t MetadataNode) {
  DataMeta *MetaP = DataMeta::convert(MetadataNode);
  delete MetaP;
  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_API
    // NOLINTNEXTLINE(readability-identifier-naming)
    amd_comgr_get_metadata_kind
    //
    (amd_comgr_metadata_node_t MetadataNode,
     amd_comgr_metadata_kind_t *MetadataKind) {
  DataMeta *MetaP = DataMeta::convert(MetadataNode);

  if (!MetadataKind)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  *MetadataKind = MetaP->getMetadataKind();

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_API
    // NOLINTNEXTLINE(readability-identifier-naming)
    amd_comgr_get_metadata_string
    //
    (amd_comgr_metadata_node_t MetadataNode, size_t *Size, char *String) {
  DataMeta *MetaP = DataMeta::convert(MetadataNode);

  if (MetaP->getMetadataKind() != AMD_COMGR_METADATA_KIND_STRING || !Size)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  std::string Str = MetaP->convertDocNodeToString(MetaP->DocNode);

  if (String)
    memcpy(String, Str.c_str(), *Size);
  else
    *Size = Str.size() + 1;

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_API
    // NOLINTNEXTLINE(readability-identifier-naming)
    amd_comgr_get_metadata_map_size
    //
    (amd_comgr_metadata_node_t MetadataNode, size_t *Size) {
  DataMeta *MetaP = DataMeta::convert(MetadataNode);

  if (MetaP->getMetadataKind() != AMD_COMGR_METADATA_KIND_MAP || !Size)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  *Size = MetaP->DocNode.getMap().size();

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_API
    // NOLINTNEXTLINE(readability-identifier-naming)
    amd_comgr_iterate_map_metadata
    //
    (amd_comgr_metadata_node_t MetadataNode,
     amd_comgr_status_t (*Callback)(amd_comgr_metadata_node_t,
                                    amd_comgr_metadata_node_t, void *),
     void *UserData) {
  DataMeta *MetaP = DataMeta::convert(MetadataNode);

  if (MetaP->getMetadataKind() != AMD_COMGR_METADATA_KIND_MAP || !Callback)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  auto Map = MetaP->DocNode.getMap();

  for (auto &KV : Map) {
    if (KV.first.isEmpty() || KV.second.isEmpty())
      return AMD_COMGR_STATUS_ERROR;
    std::unique_ptr<DataMeta> KeyP(new (std::nothrow) DataMeta());
    std::unique_ptr<DataMeta> ValueP(new (std::nothrow) DataMeta());
    if (!KeyP || !ValueP)
      return AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES;
    KeyP->MetaDoc = MetaP->MetaDoc;
    KeyP->DocNode = KV.first;
    ValueP->MetaDoc = MetaP->MetaDoc;
    ValueP->DocNode = KV.second;
    (*Callback)(DataMeta::convert(KeyP.get()), DataMeta::convert(ValueP.get()),
                UserData);
  }

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_API
    // NOLINTNEXTLINE(readability-identifier-naming)
    amd_comgr_metadata_lookup
    //
    (amd_comgr_metadata_node_t MetadataNode, const char *Key,
     amd_comgr_metadata_node_t *Value) {
  DataMeta *MetaP = DataMeta::convert(MetadataNode);

  if (MetaP->getMetadataKind() != AMD_COMGR_METADATA_KIND_MAP || !Key || !Value)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  for (auto Iter : MetaP->DocNode.getMap()) {
    if (!Iter.first.isScalar() ||
        StringRef(Key) != MetaP->convertDocNodeToString(Iter.first))
      continue;

    DataMeta *NewMetaP = new (std::nothrow) DataMeta();
    if (!NewMetaP)
      return AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES;

    NewMetaP->MetaDoc = MetaP->MetaDoc;
    NewMetaP->DocNode = Iter.second;
    *Value = DataMeta::convert(NewMetaP);

    return AMD_COMGR_STATUS_SUCCESS;
  }

  return AMD_COMGR_STATUS_ERROR;
}

amd_comgr_status_t AMD_API
    // NOLINTNEXTLINE(readability-identifier-naming)
    amd_comgr_get_metadata_list_size
    //
    (amd_comgr_metadata_node_t MetadataNode, size_t *Size) {
  DataMeta *MetaP = DataMeta::convert(MetadataNode);

  if (MetaP->getMetadataKind() != AMD_COMGR_METADATA_KIND_LIST || !Size)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  *Size = MetaP->DocNode.getArray().size();

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_API
    // NOLINTNEXTLINE(readability-identifier-naming)
    amd_comgr_index_list_metadata
    //
    (amd_comgr_metadata_node_t MetadataNode, size_t Index,
     amd_comgr_metadata_node_t *Value) {
  DataMeta *MetaP = DataMeta::convert(MetadataNode);

  if (MetaP->getMetadataKind() != AMD_COMGR_METADATA_KIND_LIST || !Value)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  auto List = MetaP->DocNode.getArray();

  if (Index >= List.size())
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  DataMeta *NewMetaP = new (std::nothrow) DataMeta();
  if (!NewMetaP)
    return AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES;

  NewMetaP->MetaDoc = MetaP->MetaDoc;
  NewMetaP->DocNode = List[Index];
  *Value = DataMeta::convert(NewMetaP);

  return AMD_COMGR_STATUS_SUCCESS;
}

#endif // !HSA_USE_EXTERNAL_COMGR
