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
#include "comgr-metadata.h"

#include "llvm/BinaryFormat/ELF.h"
#include "llvm/BinaryFormat/MsgPackDocument.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace llvm;
using namespace llvm::object;

namespace COMGR {
namespace metadata {

template <typename ELFT> using Elf_Note = typename ELFT::Note;

static Expected<std::unique_ptr<ELFObjectFileBase>>
getELFObjectFileBase(DataObject *DataP) {
  std::unique_ptr<MemoryBuffer> Buf =
      MemoryBuffer::getMemBuffer(StringRef(DataP->Data, DataP->Size));

  Expected<std::unique_ptr<ObjectFile>> ObjOrErr =
      ObjectFile::createELFObjectFile(*Buf);

  if (auto Err = ObjOrErr.takeError())
    return std::move(Err);

  return unique_dyn_cast<ELFObjectFileBase>(std::move(*ObjOrErr));
}

/// Process all notes in the given ELF object file, passing them each to @p
/// ProcessNote.
///
/// @p ProcessNote should return @c true when the desired note is found, which
/// signals to stop searching and return @c AMD_COMGR_STATUS_SUCCESS. It should
/// return @c false otherwise to continue iteration.
///
/// @returns @c AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT If all notes are
/// processed without @p ProcessNote returning @c true, otherwise
/// AMD_COMGR_STATUS_SUCCESS.
template <class ELFT, typename F>
static amd_comgr_status_t processElfNotes(const ELFObjectFile<ELFT> *Obj,
                                          F ProcessNote) {
  const ELFFile<ELFT> *ELFFile = Obj->getELFFile();

  bool Found = false;

  auto ProgramHeadersOrError = ELFFile->program_headers();
  if (errorToBool(ProgramHeadersOrError.takeError()))
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  for (const auto &Phdr : *ProgramHeadersOrError) {
    if (Phdr.p_type != ELF::PT_NOTE)
      continue;
    Error Err = Error::success();
    for (const auto &Note : ELFFile->notes(Phdr, Err))
      if (ProcessNote(Note)) {
        Found = true;
        break;
      }
    if (errorToBool(std::move(Err)))
      return AMD_COMGR_STATUS_ERROR;
    if (Found)
      return AMD_COMGR_STATUS_SUCCESS;
  }

  auto SectionsOrError = ELFFile->sections();
  if (errorToBool(SectionsOrError.takeError()))
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  for (const auto &Shdr : *SectionsOrError) {
    if (Shdr.sh_type != ELF::SHT_NOTE)
      continue;
    Error Err = Error::success();
    for (const auto &Note : ELFFile->notes(Shdr, Err))
      if (ProcessNote(Note)) {
        Found = true;
        break;
      }
    if (errorToBool(std::move(Err)))
      return AMD_COMGR_STATUS_ERROR;
    if (Found)
      return AMD_COMGR_STATUS_SUCCESS;
  }

  return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
}

// PAL currently produces MsgPack metadata in a note with this ID.
// FIXME: Unify with HSA note types?
#define PAL_METADATA_NOTE_TYPE 13

template <class ELFT>
static amd_comgr_status_t getElfMetadataRoot(const ELFObjectFile<ELFT> *Obj,
                                             DataMeta *MetaP) {
  amd_comgr_status_t NoteStatus = AMD_COMGR_STATUS_SUCCESS;
  auto ProcessNote = [&](const Elf_Note<ELFT> &Note) {
    auto DescString =
        StringRef(reinterpret_cast<const char *>(Note.getDesc().data()),
                  Note.getDesc().size());
    if (Note.getName() == "AMD" &&
        Note.getType() == ELF::NT_AMD_AMDGPU_HSA_METADATA) {
      MetaP->MetaDoc->EmitIntegerBooleans = false;
      MetaP->MetaDoc->RawDocument.clear();
      if (!MetaP->MetaDoc->Document.fromYAML(DescString))
        return false;
      MetaP->DocNode = MetaP->MetaDoc->Document.getRoot();
      return true;
    } else if (((Note.getName() == "AMD" || Note.getName() == "AMDGPU") &&
                Note.getType() == PAL_METADATA_NOTE_TYPE) ||
               (Note.getName() == "AMDGPU" &&
                Note.getType() == ELF::NT_AMDGPU_METADATA)) {
      MetaP->MetaDoc->EmitIntegerBooleans = true;
      MetaP->MetaDoc->RawDocument = std::string(DescString);
      if (!MetaP->MetaDoc->Document.readFromBlob(MetaP->MetaDoc->RawDocument,
                                                 false))
        return false;
      MetaP->DocNode = MetaP->MetaDoc->Document.getRoot();
      return true;
    }
    return false;
  };
  if (auto ElfStatus = processElfNotes(Obj, ProcessNote))
    return ElfStatus;
  if (NoteStatus)
    return NoteStatus;
  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t getMetadataRoot(DataObject *DataP, DataMeta *MetaP) {
  auto ObjOrErr = getELFObjectFileBase(DataP);
  if (errorToBool(ObjOrErr.takeError()))
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  auto Obj = ObjOrErr->get();

  if (auto ELF32LE = dyn_cast<ELF32LEObjectFile>(Obj))
    return getElfMetadataRoot(ELF32LE, MetaP);
  if (auto ELF64LE = dyn_cast<ELF64LEObjectFile>(Obj))
    return getElfMetadataRoot(ELF64LE, MetaP);
  if (auto ELF32BE = dyn_cast<ELF32BEObjectFile>(Obj))
    return getElfMetadataRoot(ELF32BE, MetaP);
  auto ELF64BE = dyn_cast<ELF64BEObjectFile>(Obj);
  return getElfMetadataRoot(ELF64BE, MetaP);
}

} // namespace metadata
} // namespace COMGR
#endif // !HSA_USE_EXTERNAL_COMGR
