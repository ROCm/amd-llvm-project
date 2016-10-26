//===- LinkerScript.cpp ---------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the parser/evaluator of the linker script.
// It parses a linker script and write the result to Config or ScriptConfig
// objects.
//
// If SECTIONS command is used, a ScriptConfig contains an AST
// of the command which will later be consumed by createSections() and
// assignAddresses().
//
//===----------------------------------------------------------------------===//

#include "LinkerScript.h"
#include "Config.h"
#include "Driver.h"
#include "InputSection.h"
#include "OutputSections.h"
#include "ScriptParser.h"
#include "Strings.h"
#include "Symbols.h"
#include "SymbolTable.h"
#include "Target.h"
#include "Writer.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/StringSaver.h"

using namespace llvm;
using namespace llvm::ELF;
using namespace llvm::object;
using namespace llvm::support::endian;
using namespace lld;
using namespace lld::elf;

LinkerScriptBase *elf::ScriptBase;
ScriptConfiguration *elf::ScriptConfig;

template <class ELFT> static void addRegular(SymbolAssignment *Cmd) {
  Symbol *Sym = Symtab<ELFT>::X->addRegular(Cmd->Name, STB_GLOBAL, STV_DEFAULT);
  Sym->Visibility = Cmd->Hidden ? STV_HIDDEN : STV_DEFAULT;
  Cmd->Sym = Sym->body();

  // If we have no SECTIONS then we don't have '.' and don't call
  // assignAddresses(). We calculate symbol value immediately in this case.
  if (!ScriptConfig->HasSections)
    cast<DefinedRegular<ELFT>>(Cmd->Sym)->Value = Cmd->Expression(0);
}

template <class ELFT> static void addSynthetic(SymbolAssignment *Cmd) {
  Symbol *Sym = Symtab<ELFT>::X->addSynthetic(
      Cmd->Name, nullptr, 0, Cmd->Hidden ? STV_HIDDEN : STV_DEFAULT);
  Cmd->Sym = Sym->body();
}

template <class ELFT> static void addSymbol(SymbolAssignment *Cmd) {
  if (Cmd->IsAbsolute)
    addRegular<ELFT>(Cmd);
  else
    addSynthetic<ELFT>(Cmd);
}
// If a symbol was in PROVIDE(), we need to define it only when
// it is an undefined symbol.
template <class ELFT> static bool shouldDefine(SymbolAssignment *Cmd) {
  if (Cmd->Name == ".")
    return false;
  if (!Cmd->Provide)
    return true;
  SymbolBody *B = Symtab<ELFT>::X->find(Cmd->Name);
  return B && B->isUndefined();
}

bool SymbolAssignment::classof(const BaseCommand *C) {
  return C->Kind == AssignmentKind;
}

bool OutputSectionCommand::classof(const BaseCommand *C) {
  return C->Kind == OutputSectionKind;
}

bool InputSectionDescription::classof(const BaseCommand *C) {
  return C->Kind == InputSectionKind;
}

bool AssertCommand::classof(const BaseCommand *C) {
  return C->Kind == AssertKind;
}

bool BytesDataCommand::classof(const BaseCommand *C) {
  return C->Kind == BytesDataKind;
}

template <class ELFT> static bool isDiscarded(InputSectionBase<ELFT> *S) {
  return !S || !S->Live;
}

template <class ELFT> LinkerScript<ELFT>::LinkerScript() {}
template <class ELFT> LinkerScript<ELFT>::~LinkerScript() {}

template <class ELFT>
bool LinkerScript<ELFT>::shouldKeep(InputSectionBase<ELFT> *S) {
  for (InputSectionDescription *ID : Opt.KeptSections) {
    StringRef Filename = S->getFile()->getName();
    if (!ID->FileRe.match(sys::path::filename(Filename)))
      continue;

    for (SectionPattern &P : ID->SectionPatterns)
      if (P.SectionRe.match(S->Name))
        return true;
  }
  return false;
}

static bool comparePriority(InputSectionData *A, InputSectionData *B) {
  return getPriority(A->Name) < getPriority(B->Name);
}

static bool compareName(InputSectionData *A, InputSectionData *B) {
  return A->Name < B->Name;
}

static bool compareAlignment(InputSectionData *A, InputSectionData *B) {
  // ">" is not a mistake. Larger alignments are placed before smaller
  // alignments in order to reduce the amount of padding necessary.
  // This is compatible with GNU.
  return A->Alignment > B->Alignment;
}

static std::function<bool(InputSectionData *, InputSectionData *)>
getComparator(SortSectionPolicy K) {
  switch (K) {
  case SortSectionPolicy::Alignment:
    return compareAlignment;
  case SortSectionPolicy::Name:
    return compareName;
  case SortSectionPolicy::Priority:
    return comparePriority;
  default:
    llvm_unreachable("unknown sort policy");
  }
}

template <class ELFT>
static bool matchConstraints(ArrayRef<InputSectionBase<ELFT> *> Sections,
                             ConstraintKind Kind) {
  if (Kind == ConstraintKind::NoConstraint)
    return true;
  bool IsRW = llvm::any_of(Sections, [=](InputSectionData *Sec2) {
    auto *Sec = static_cast<InputSectionBase<ELFT> *>(Sec2);
    return Sec->getFlags() & SHF_WRITE;
  });
  return (IsRW && Kind == ConstraintKind::ReadWrite) ||
         (!IsRW && Kind == ConstraintKind::ReadOnly);
}

static void sortSections(InputSectionData **Begin, InputSectionData **End,
                         SortSectionPolicy K) {
  if (K != SortSectionPolicy::Default && K != SortSectionPolicy::None)
    std::stable_sort(Begin, End, getComparator(K));
}

// Compute and remember which sections the InputSectionDescription matches.
template <class ELFT>
void LinkerScript<ELFT>::computeInputSections(InputSectionDescription *I) {
  // Collects all sections that satisfy constraints of I
  // and attach them to I.
  for (SectionPattern &Pat : I->SectionPatterns) {
    size_t SizeBefore = I->Sections.size();
    for (ObjectFile<ELFT> *F : Symtab<ELFT>::X->getObjectFiles()) {
      StringRef Filename = sys::path::filename(F->getName());
      if (!I->FileRe.match(Filename) || Pat.ExcludedFileRe.match(Filename))
        continue;

      for (InputSectionBase<ELFT> *S : F->getSections())
        if (!isDiscarded(S) && !S->OutSec && Pat.SectionRe.match(S->Name))
          I->Sections.push_back(S);
      if (Pat.SectionRe.match("COMMON"))
        I->Sections.push_back(InputSection<ELFT>::CommonInputSection);
    }

    // Sort sections as instructed by SORT-family commands and --sort-section
    // option. Because SORT-family commands can be nested at most two depth
    // (e.g. SORT_BY_NAME(SORT_BY_ALIGNMENT(.text.*))) and because the command
    // line option is respected even if a SORT command is given, the exact
    // behavior we have here is a bit complicated. Here are the rules.
    //
    // 1. If two SORT commands are given, --sort-section is ignored.
    // 2. If one SORT command is given, and if it is not SORT_NONE,
    //    --sort-section is handled as an inner SORT command.
    // 3. If one SORT command is given, and if it is SORT_NONE, don't sort.
    // 4. If no SORT command is given, sort according to --sort-section.
    InputSectionData **Begin = I->Sections.data() + SizeBefore;
    InputSectionData **End = I->Sections.data() + I->Sections.size();
    if (Pat.SortOuter != SortSectionPolicy::None) {
      if (Pat.SortInner == SortSectionPolicy::Default)
        sortSections(Begin, End, Config->SortSection);
      else
        sortSections(Begin, End, Pat.SortInner);
      sortSections(Begin, End, Pat.SortOuter);
    }
  }

  // We do not add duplicate input sections, so mark them with a dummy output
  // section for now.
  for (InputSectionData *S : I->Sections) {
    auto *S2 = static_cast<InputSectionBase<ELFT> *>(S);
    S2->OutSec = (OutputSectionBase<ELFT> *)-1;
  }
}

template <class ELFT>
void LinkerScript<ELFT>::discard(ArrayRef<InputSectionBase<ELFT> *> V) {
  for (InputSectionBase<ELFT> *S : V) {
    S->Live = false;
    reportDiscarded(S);
  }
}

template <class ELFT>
std::vector<InputSectionBase<ELFT> *>
LinkerScript<ELFT>::createInputSectionList(OutputSectionCommand &OutCmd) {
  std::vector<InputSectionBase<ELFT> *> Ret;

  for (const std::unique_ptr<BaseCommand> &Base : OutCmd.Commands) {
    auto *Cmd = dyn_cast<InputSectionDescription>(Base.get());
    if (!Cmd)
      continue;
    computeInputSections(Cmd);
    for (InputSectionData *S : Cmd->Sections)
      Ret.push_back(static_cast<InputSectionBase<ELFT> *>(S));
  }

  // After we created final list we should now set OutSec pointer to null,
  // instead of -1. Otherwise we may get a crash when writing relocs, in
  // case section is discarded by linker script
  for (InputSectionBase<ELFT> *S : Ret)
    S->OutSec = nullptr;

  return Ret;
}

template <class ELFT>
static SectionKey<ELFT::Is64Bits> createKey(InputSectionBase<ELFT> *C,
                                            StringRef OutsecName) {
  // When using linker script the merge rules are different.
  // Unfortunately, linker scripts are name based. This means that expressions
  // like *(.foo*) can refer to multiple input sections that would normally be
  // placed in different output sections. We cannot put them in different
  // output sections or we would produce wrong results for
  // start = .; *(.foo.*) end = .; *(.bar)
  // and a mapping of .foo1 and .bar1 to one section and .foo2 and .bar2 to
  // another. The problem is that there is no way to layout those output
  // sections such that the .foo sections are the only thing between the
  // start and end symbols.

  // An extra annoyance is that we cannot simply disable merging of the contents
  // of SHF_MERGE sections, but our implementation requires one output section
  // per "kind" (string or not, which size/aligment).
  // Fortunately, creating symbols in the middle of a merge section is not
  // supported by bfd or gold, so we can just create multiple section in that
  // case.
  typedef typename ELFT::uint uintX_t;
  uintX_t Flags = C->getFlags() & (SHF_MERGE | SHF_STRINGS);

  uintX_t Alignment = 0;
  if (isa<MergeInputSection<ELFT>>(C))
    Alignment = std::max<uintX_t>(C->Alignment, C->getEntsize());

  return SectionKey<ELFT::Is64Bits>{OutsecName, /*Type*/ 0, Flags, Alignment};
}

template <class ELFT>
void LinkerScript<ELFT>::addSection(OutputSectionFactory<ELFT> &Factory,
                                    InputSectionBase<ELFT> *Sec,
                                    StringRef Name) {
  OutputSectionBase<ELFT> *OutSec;
  bool IsNew;
  std::tie(OutSec, IsNew) = Factory.create(createKey(Sec, Name), Sec);
  if (IsNew)
    OutputSections->push_back(OutSec);
  OutSec->addSection(Sec);
}

template <class ELFT>
void LinkerScript<ELFT>::processCommands(OutputSectionFactory<ELFT> &Factory) {

  for (unsigned I = 0; I < Opt.Commands.size(); ++I) {
    auto Iter = Opt.Commands.begin() + I;
    const std::unique_ptr<BaseCommand> &Base1 = *Iter;
    if (auto *Cmd = dyn_cast<SymbolAssignment>(Base1.get())) {
      if (shouldDefine<ELFT>(Cmd))
        addRegular<ELFT>(Cmd);
      continue;
    }
    if (auto *Cmd = dyn_cast<AssertCommand>(Base1.get())) {
      // If we don't have SECTIONS then output sections have already been
      // created by Writer<ELFT>. The LinkerScript<ELFT>::assignAddresses
      // will not be called, so ASSERT should be evaluated now.
      if (!Opt.HasSections)
        Cmd->Expression(0);
      continue;
    }

    if (auto *Cmd = dyn_cast<OutputSectionCommand>(Base1.get())) {
      std::vector<InputSectionBase<ELFT> *> V = createInputSectionList(*Cmd);

      if (Cmd->Name == "/DISCARD/") {
        discard(V);
        continue;
      }

      if (!matchConstraints<ELFT>(V, Cmd->Constraint)) {
        for (InputSectionBase<ELFT> *S : V)
          S->OutSec = nullptr;
        Opt.Commands.erase(Iter);
        --I;
        continue;
      }

      for (const std::unique_ptr<BaseCommand> &Base : Cmd->Commands)
        if (auto *OutCmd = dyn_cast<SymbolAssignment>(Base.get()))
          if (shouldDefine<ELFT>(OutCmd))
            addSymbol<ELFT>(OutCmd);

      if (V.empty())
        continue;

      for (InputSectionBase<ELFT> *Sec : V) {
        addSection(Factory, Sec, Cmd->Name);
        if (uint32_t Subalign = Cmd->SubalignExpr ? Cmd->SubalignExpr(0) : 0)
          Sec->Alignment = Subalign;
      }
    }
  }
}

template <class ELFT>
void LinkerScript<ELFT>::createSections(OutputSectionFactory<ELFT> &Factory) {
  processCommands(Factory);
  // Add orphan sections.
  for (ObjectFile<ELFT> *F : Symtab<ELFT>::X->getObjectFiles())
    for (InputSectionBase<ELFT> *S : F->getSections())
      if (!isDiscarded(S) && !S->OutSec)
        addSection(Factory, S, getOutputSectionName(S->Name, Opt.Alloc));
}

// Sets value of a section-defined symbol. Two kinds of
// symbols are processed: synthetic symbols, whose value
// is an offset from beginning of section and regular
// symbols whose value is absolute.
template <class ELFT>
static void assignSectionSymbol(SymbolAssignment *Cmd,
                                OutputSectionBase<ELFT> *Sec,
                                typename ELFT::uint Off) {
  if (!Cmd->Sym)
    return;

  if (auto *Body = dyn_cast<DefinedSynthetic<ELFT>>(Cmd->Sym)) {
    Body->Section = Sec;
    Body->Value = Cmd->Expression(Sec->getVA() + Off) - Sec->getVA();
    return;
  }
  auto *Body = cast<DefinedRegular<ELFT>>(Cmd->Sym);
  Body->Value = Cmd->Expression(Sec->getVA() + Off);
}

template <class ELFT> static bool isTbss(OutputSectionBase<ELFT> *Sec) {
  return (Sec->getFlags() & SHF_TLS) && Sec->getType() == SHT_NOBITS;
}

template <class ELFT> void LinkerScript<ELFT>::output(InputSection<ELFT> *S) {
  if (!AlreadyOutputIS.insert(S).second)
    return;
  bool IsTbss = isTbss(CurOutSec);

  uintX_t Pos = IsTbss ? Dot + ThreadBssOffset : Dot;
  Pos = alignTo(Pos, S->Alignment);
  S->OutSecOff = Pos - CurOutSec->getVA();
  Pos += S->getSize();

  // Update output section size after adding each section. This is so that
  // SIZEOF works correctly in the case below:
  // .foo { *(.aaa) a = SIZEOF(.foo); *(.bbb) }
  CurOutSec->setSize(Pos - CurOutSec->getVA());

  if (IsTbss)
    ThreadBssOffset = Pos - Dot;
  else
    Dot = Pos;
}

template <class ELFT> void LinkerScript<ELFT>::flush() {
  if (!CurOutSec || !AlreadyOutputOS.insert(CurOutSec).second)
    return;
  if (auto *OutSec = dyn_cast<OutputSection<ELFT>>(CurOutSec)) {
    for (InputSection<ELFT> *I : OutSec->Sections)
      output(I);
  } else {
    Dot += CurOutSec->getSize();
  }
}

template <class ELFT>
void LinkerScript<ELFT>::switchTo(OutputSectionBase<ELFT> *Sec) {
  if (CurOutSec == Sec)
    return;
  if (AlreadyOutputOS.count(Sec))
    return;

  flush();
  CurOutSec = Sec;

  Dot = alignTo(Dot, CurOutSec->getAlignment());
  CurOutSec->setVA(isTbss(CurOutSec) ? Dot + ThreadBssOffset : Dot);

  // If neither AT nor AT> is specified for an allocatable section, the linker
  // will set the LMA such that the difference between VMA and LMA for the
  // section is the same as the preceding output section in the same region
  // https://sourceware.org/binutils/docs-2.20/ld/Output-Section-LMA.html
  CurOutSec->setLMAOffset(LMAOffset);
}

template <class ELFT> void LinkerScript<ELFT>::process(BaseCommand &Base) {
  // This handles the assignments to symbol or to a location counter (.)
  if (auto *AssignCmd = dyn_cast<SymbolAssignment>(&Base)) {
    if (AssignCmd->Name == ".") {
      // Update to location counter means update to section size.
      Dot = AssignCmd->Expression(Dot);
      CurOutSec->setSize(Dot - CurOutSec->getVA());
      return;
    }
    assignSectionSymbol<ELFT>(AssignCmd, CurOutSec, Dot - CurOutSec->getVA());
    return;
  }

  // Handle BYTE(), SHORT(), LONG(), or QUAD().
  if (auto *DataCmd = dyn_cast<BytesDataCommand>(&Base)) {
    DataCmd->Offset = Dot - CurOutSec->getVA();
    Dot += DataCmd->Size;
    CurOutSec->setSize(Dot - CurOutSec->getVA());
    return;
  }

  // It handles single input section description command,
  // calculates and assigns the offsets for each section and also
  // updates the output section size.
  auto &ICmd = cast<InputSectionDescription>(Base);
  for (InputSectionData *ID : ICmd.Sections) {
    auto *IB = static_cast<InputSectionBase<ELFT> *>(ID);
    switchTo(IB->OutSec);
    if (auto *I = dyn_cast<InputSection<ELFT>>(IB))
      output(I);
    else
      flush();
  }
}

template <class ELFT>
static std::vector<OutputSectionBase<ELFT> *>
findSections(StringRef Name,
             const std::vector<OutputSectionBase<ELFT> *> &Sections) {
  std::vector<OutputSectionBase<ELFT> *> Ret;
  for (OutputSectionBase<ELFT> *Sec : Sections)
    if (Sec->getName() == Name)
      Ret.push_back(Sec);
  return Ret;
}

template <class ELFT>
void LinkerScript<ELFT>::assignOffsets(OutputSectionCommand *Cmd) {
  if (Cmd->LMAExpr)
    LMAOffset = Cmd->LMAExpr(Dot) - Dot;
  std::vector<OutputSectionBase<ELFT> *> Sections =
      findSections(Cmd->Name, *OutputSections);
  if (Sections.empty())
    return;
  switchTo(Sections[0]);
  // Find the last section output location. We will output orphan sections
  // there so that end symbols point to the correct location.
  auto E = std::find_if(Cmd->Commands.rbegin(), Cmd->Commands.rend(),
                        [](const std::unique_ptr<BaseCommand> &Cmd) {
                          return !isa<SymbolAssignment>(*Cmd);
                        })
               .base();
  for (auto I = Cmd->Commands.begin(); I != E; ++I)
    process(**I);
  for (OutputSectionBase<ELFT> *Base : Sections)
    switchTo(Base);
  flush();
  std::for_each(E, Cmd->Commands.end(),
                [this](std::unique_ptr<BaseCommand> &B) { process(*B.get()); });
}

template <class ELFT> void LinkerScript<ELFT>::adjustSectionsBeforeSorting() {
  // It is common practice to use very generic linker scripts. So for any
  // given run some of the output sections in the script will be empty.
  // We could create corresponding empty output sections, but that would
  // clutter the output.
  // We instead remove trivially empty sections. The bfd linker seems even
  // more aggressive at removing them.
  auto Pos = std::remove_if(
      Opt.Commands.begin(), Opt.Commands.end(),
      [&](const std::unique_ptr<BaseCommand> &Base) {
        auto *Cmd = dyn_cast<OutputSectionCommand>(Base.get());
        if (!Cmd)
          return false;
        std::vector<OutputSectionBase<ELFT> *> Secs =
            findSections(Cmd->Name, *OutputSections);
        if (!Secs.empty())
          return false;
        for (const std::unique_ptr<BaseCommand> &I : Cmd->Commands)
          if (!isa<InputSectionDescription>(I.get()))
            return false;
        return true;
      });
  Opt.Commands.erase(Pos, Opt.Commands.end());

  // If the output section contains only symbol assignments, create a
  // corresponding output section. The bfd linker seems to only create them if
  // '.' is assigned to, but creating these section should not have any bad
  // consequeces and gives us a section to put the symbol in.
  uintX_t Flags = SHF_ALLOC;
  uint32_t Type = 0;
  for (const std::unique_ptr<BaseCommand> &Base : Opt.Commands) {
    auto *Cmd = dyn_cast<OutputSectionCommand>(Base.get());
    if (!Cmd)
      continue;
    std::vector<OutputSectionBase<ELFT> *> Secs =
        findSections(Cmd->Name, *OutputSections);
    if (!Secs.empty()) {
      Flags = Secs[0]->getFlags();
      Type = Secs[0]->getType();
      continue;
    }

    auto *OutSec = new OutputSection<ELFT>(Cmd->Name, Type, Flags);
    Out<ELFT>::Pool.emplace_back(OutSec);
    OutputSections->push_back(OutSec);
  }
}

// When placing orphan sections, we want to place them after symbol assignments
// so that an orphan after
//   begin_foo = .;
//   foo : { *(foo) }
//   end_foo = .;
// doesn't break the intended meaning of the begin/end symbols.
// We don't want to go over sections since Writer<ELFT>::sortSections is the
// one in charge of deciding the order of the sections.
// We don't want to go over alignments, since doing so in
//  rx_sec : { *(rx_sec) }
//  . = ALIGN(0x1000);
//  /* The RW PT_LOAD starts here*/
//  rw_sec : { *(rw_sec) }
// would mean that the RW PT_LOAD would become unaligned.
static bool shouldSkip(const BaseCommand &Cmd) {
  if (isa<OutputSectionCommand>(Cmd))
    return false;
  const auto *Assign = dyn_cast<SymbolAssignment>(&Cmd);
  if (!Assign)
    return true;
  return Assign->Name != ".";
}

template <class ELFT>
void LinkerScript<ELFT>::assignAddresses(std::vector<PhdrEntry<ELFT>> &Phdrs) {
  // Orphan sections are sections present in the input files which
  // are not explicitly placed into the output file by the linker script.
  // We place orphan sections at end of file.
  // Other linkers places them using some heuristics as described in
  // https://sourceware.org/binutils/docs/ld/Orphan-Sections.html#Orphan-Sections.

  // The OutputSections are already in the correct order.
  // This loops creates or moves commands as needed so that they are in the
  // correct order.
  int CmdIndex = 0;
  for (OutputSectionBase<ELFT> *Sec : *OutputSections) {
    StringRef Name = Sec->getName();

    // Find the last spot where we can insert a command and still get the
    // correct result.
    auto CmdIter = Opt.Commands.begin() + CmdIndex;
    auto E = Opt.Commands.end();
    while (CmdIter != E && shouldSkip(**CmdIter)) {
      ++CmdIter;
      ++CmdIndex;
    }

    auto Pos =
        std::find_if(CmdIter, E, [&](const std::unique_ptr<BaseCommand> &Base) {
          auto *Cmd = dyn_cast<OutputSectionCommand>(Base.get());
          return Cmd && Cmd->Name == Name;
        });
    if (Pos == E) {
      Opt.Commands.insert(CmdIter,
                          llvm::make_unique<OutputSectionCommand>(Name));
      ++CmdIndex;
      continue;
    }

    // Continue from where we found it.
    CmdIndex = (Pos - Opt.Commands.begin()) + 1;
    continue;
  }

  // Assign addresses as instructed by linker script SECTIONS sub-commands.
  Dot = 0;

  for (const std::unique_ptr<BaseCommand> &Base : Opt.Commands) {
    if (auto *Cmd = dyn_cast<SymbolAssignment>(Base.get())) {
      if (Cmd->Name == ".") {
        Dot = Cmd->Expression(Dot);
      } else if (Cmd->Sym) {
        cast<DefinedRegular<ELFT>>(Cmd->Sym)->Value = Cmd->Expression(Dot);
      }
      continue;
    }

    if (auto *Cmd = dyn_cast<AssertCommand>(Base.get())) {
      Cmd->Expression(Dot);
      continue;
    }

    auto *Cmd = cast<OutputSectionCommand>(Base.get());

    if (Cmd->AddrExpr)
      Dot = Cmd->AddrExpr(Dot);

    assignOffsets(Cmd);
  }

  uintX_t MinVA = std::numeric_limits<uintX_t>::max();
  for (OutputSectionBase<ELFT> *Sec : *OutputSections) {
    if (Sec->getFlags() & SHF_ALLOC)
      MinVA = std::min(MinVA, Sec->getVA());
    else
      Sec->setVA(0);
  }

  uintX_t HeaderSize = getHeaderSize();
  auto FirstPTLoad =
      std::find_if(Phdrs.begin(), Phdrs.end(), [](const PhdrEntry<ELFT> &E) {
        return E.H.p_type == PT_LOAD;
      });

  if (HeaderSize <= MinVA && FirstPTLoad != Phdrs.end()) {
    // If linker script specifies program headers and first PT_LOAD doesn't 
    // have both PHDRS and FILEHDR attributes then do nothing
    if (!Opt.PhdrsCommands.empty()) {
      size_t SegNum = std::distance(Phdrs.begin(), FirstPTLoad);
      if (!Opt.PhdrsCommands[SegNum].HasPhdrs ||
          !Opt.PhdrsCommands[SegNum].HasFilehdr)
        return;
    }
    // ELF and Program headers need to be right before the first section in
    // memory. Set their addresses accordingly.
    MinVA = alignDown(MinVA - HeaderSize, Target->PageSize);
    Out<ELFT>::ElfHeader->setVA(MinVA);
    Out<ELFT>::ProgramHeaders->setVA(Out<ELFT>::ElfHeader->getSize() + MinVA);
    FirstPTLoad->First = Out<ELFT>::ElfHeader;
    if (!FirstPTLoad->Last)
      FirstPTLoad->Last = Out<ELFT>::ProgramHeaders;
  } else if (!FirstPTLoad->First) {
    // Sometimes the very first PT_LOAD segment can be empty.
    // This happens if (all conditions met):
    //  - Linker script is used
    //  - First section in ELF image is not RO
    //  - Not enough space for program headers.
    // The code below removes empty PT_LOAD segment and updates
    // program headers size.
    Phdrs.erase(FirstPTLoad);
    Out<ELFT>::ProgramHeaders->setSize(sizeof(typename ELFT::Phdr) *
                                       Phdrs.size());
  }
}

// Creates program headers as instructed by PHDRS linker script command.
template <class ELFT>
std::vector<PhdrEntry<ELFT>> LinkerScript<ELFT>::createPhdrs() {
  std::vector<PhdrEntry<ELFT>> Ret;

  // Process PHDRS and FILEHDR keywords because they are not
  // real output sections and cannot be added in the following loop.
  std::vector<size_t> DefPhdrIds;
  for (const PhdrsCommand &Cmd : Opt.PhdrsCommands) {
    Ret.emplace_back(Cmd.Type, Cmd.Flags == UINT_MAX ? PF_R : Cmd.Flags);
    PhdrEntry<ELFT> &Phdr = Ret.back();

    if (Cmd.HasFilehdr)
      Phdr.add(Out<ELFT>::ElfHeader);
    if (Cmd.HasPhdrs)
      Phdr.add(Out<ELFT>::ProgramHeaders);

    if (Cmd.LMAExpr) {
      Phdr.H.p_paddr = Cmd.LMAExpr(0);
      Phdr.HasLMA = true;
    }

    // If output section command doesn't specify any segments,
    // and we haven't previously assigned any section to segment,
    // then we simply assign section to the very first load segment.
    // Below is an example of such linker script:
    // PHDRS { seg PT_LOAD; }
    // SECTIONS { .aaa : { *(.aaa) } }
    if (DefPhdrIds.empty() && Phdr.H.p_type == PT_LOAD)
      DefPhdrIds.push_back(Ret.size() - 1);
  }

  // Add output sections to program headers.
  for (OutputSectionBase<ELFT> *Sec : *OutputSections) {
    if (!(Sec->getFlags() & SHF_ALLOC))
      break;

    std::vector<size_t> PhdrIds = getPhdrIndices(Sec->getName());
    if (PhdrIds.empty())
      PhdrIds = std::move(DefPhdrIds);

    // Assign headers specified by linker script
    for (size_t Id : PhdrIds) {
      Ret[Id].add(Sec);
      if (Opt.PhdrsCommands[Id].Flags == UINT_MAX)
        Ret[Id].H.p_flags |= Sec->getPhdrFlags();
    }
    DefPhdrIds = std::move(PhdrIds);
  }
  return Ret;
}

template <class ELFT> bool LinkerScript<ELFT>::ignoreInterpSection() {
  // Ignore .interp section in case we have PHDRS specification
  // and PT_INTERP isn't listed.
  return !Opt.PhdrsCommands.empty() &&
         llvm::find_if(Opt.PhdrsCommands, [](const PhdrsCommand &Cmd) {
           return Cmd.Type == PT_INTERP;
         }) == Opt.PhdrsCommands.end();
}

template <class ELFT>
ArrayRef<uint8_t> LinkerScript<ELFT>::getFiller(StringRef Name) {
  for (const std::unique_ptr<BaseCommand> &Base : Opt.Commands)
    if (auto *Cmd = dyn_cast<OutputSectionCommand>(Base.get()))
      if (Cmd->Name == Name)
        return Cmd->Filler;
  return {};
}

template <class ELFT>
static void writeInt(uint8_t *Buf, uint64_t Data, uint64_t Size) {
  const endianness E = ELFT::TargetEndianness;

  switch (Size) {
  case 1:
    *Buf = (uint8_t)Data;
    break;
  case 2:
    write16<E>(Buf, Data);
    break;
  case 4:
    write32<E>(Buf, Data);
    break;
  case 8:
    write64<E>(Buf, Data);
    break;
  default:
    llvm_unreachable("unsupported Size argument");
  }
}

template <class ELFT>
void LinkerScript<ELFT>::writeDataBytes(StringRef Name, uint8_t *Buf) {
  int I = getSectionIndex(Name);
  if (I == INT_MAX)
    return;

  OutputSectionCommand *Cmd =
      dyn_cast<OutputSectionCommand>(Opt.Commands[I].get());
  for (const std::unique_ptr<BaseCommand> &Base2 : Cmd->Commands)
    if (auto *DataCmd = dyn_cast<BytesDataCommand>(Base2.get()))
      writeInt<ELFT>(&Buf[DataCmd->Offset], DataCmd->Data, DataCmd->Size);
}

template <class ELFT> bool LinkerScript<ELFT>::hasLMA(StringRef Name) {
  for (const std::unique_ptr<BaseCommand> &Base : Opt.Commands)
    if (auto *Cmd = dyn_cast<OutputSectionCommand>(Base.get()))
      if (Cmd->LMAExpr && Cmd->Name == Name)
        return true;
  return false;
}

// Returns the index of the given section name in linker script
// SECTIONS commands. Sections are laid out as the same order as they
// were in the script. If a given name did not appear in the script,
// it returns INT_MAX, so that it will be laid out at end of file.
template <class ELFT> int LinkerScript<ELFT>::getSectionIndex(StringRef Name) {
  int I = 0;
  for (std::unique_ptr<BaseCommand> &Base : Opt.Commands) {
    if (auto *Cmd = dyn_cast<OutputSectionCommand>(Base.get()))
      if (Cmd->Name == Name)
        return I;
    ++I;
  }
  return INT_MAX;
}

template <class ELFT> bool LinkerScript<ELFT>::hasPhdrsCommands() {
  return !Opt.PhdrsCommands.empty();
}

template <class ELFT>
uint64_t LinkerScript<ELFT>::getOutputSectionAddress(StringRef Name) {
  for (OutputSectionBase<ELFT> *Sec : *OutputSections)
    if (Sec->getName() == Name)
      return Sec->getVA();
  error("undefined section " + Name);
  return 0;
}

template <class ELFT>
uint64_t LinkerScript<ELFT>::getOutputSectionLMA(StringRef Name) {
  for (OutputSectionBase<ELFT> *Sec : *OutputSections)
    if (Sec->getName() == Name)
      return Sec->getLMA();
  error("undefined section " + Name);
  return 0;
}

template <class ELFT>
uint64_t LinkerScript<ELFT>::getOutputSectionSize(StringRef Name) {
  for (OutputSectionBase<ELFT> *Sec : *OutputSections)
    if (Sec->getName() == Name)
      return Sec->getSize();
  error("undefined section " + Name);
  return 0;
}

template <class ELFT>
uint64_t LinkerScript<ELFT>::getOutputSectionAlign(StringRef Name) {
  for (OutputSectionBase<ELFT> *Sec : *OutputSections)
    if (Sec->getName() == Name)
      return Sec->getAlignment();
  error("undefined section " + Name);
  return 0;
}

template <class ELFT> uint64_t LinkerScript<ELFT>::getHeaderSize() {
  return elf::getHeaderSize<ELFT>();
}

template <class ELFT> uint64_t LinkerScript<ELFT>::getSymbolValue(StringRef S) {
  if (SymbolBody *B = Symtab<ELFT>::X->find(S))
    return B->getVA<ELFT>();
  error("symbol not found: " + S);
  return 0;
}

template <class ELFT> bool LinkerScript<ELFT>::isDefined(StringRef S) {
  return Symtab<ELFT>::X->find(S) != nullptr;
}

// Returns indices of ELF headers containing specific section, identified
// by Name. Each index is a zero based number of ELF header listed within
// PHDRS {} script block.
template <class ELFT>
std::vector<size_t> LinkerScript<ELFT>::getPhdrIndices(StringRef SectionName) {
  for (const std::unique_ptr<BaseCommand> &Base : Opt.Commands) {
    auto *Cmd = dyn_cast<OutputSectionCommand>(Base.get());
    if (!Cmd || Cmd->Name != SectionName)
      continue;

    std::vector<size_t> Ret;
    for (StringRef PhdrName : Cmd->Phdrs)
      Ret.push_back(getPhdrIndex(PhdrName));
    return Ret;
  }
  return {};
}

template <class ELFT>
size_t LinkerScript<ELFT>::getPhdrIndex(StringRef PhdrName) {
  size_t I = 0;
  for (PhdrsCommand &Cmd : Opt.PhdrsCommands) {
    if (Cmd.Name == PhdrName)
      return I;
    ++I;
  }
  error("section header '" + PhdrName + "' is not listed in PHDRS");
  return 0;
}

class elf::ScriptParser : public ScriptParserBase {
  typedef void (ScriptParser::*Handler)();

public:
  ScriptParser(StringRef S, bool B) : ScriptParserBase(S), IsUnderSysroot(B) {}

  void readLinkerScript();
  void readVersionScript();

private:
  void addFile(StringRef Path);

  void readAsNeeded();
  void readEntry();
  void readExtern();
  void readGroup();
  void readInclude();
  void readOutput();
  void readOutputArch();
  void readOutputFormat();
  void readPhdrs();
  void readSearchDir();
  void readSections();
  void readVersion();
  void readVersionScriptCommand();

  SymbolAssignment *readAssignment(StringRef Name);
  BytesDataCommand *readBytesDataCommand(StringRef Tok);
  std::vector<uint8_t> readFill();
  OutputSectionCommand *readOutputSectionDescription(StringRef OutSec);
  std::vector<uint8_t> readOutputSectionFiller(StringRef Tok);
  std::vector<StringRef> readOutputSectionPhdrs();
  InputSectionDescription *readInputSectionDescription(StringRef Tok);
  Regex readFilePatterns();
  std::vector<SectionPattern> readInputSectionsList();
  InputSectionDescription *readInputSectionRules(StringRef FilePattern);
  unsigned readPhdrType();
  SortSectionPolicy readSortKind();
  SymbolAssignment *readProvideHidden(bool Provide, bool Hidden);
  SymbolAssignment *readProvideOrAssignment(StringRef Tok, bool MakeAbsolute);
  void readSort();
  Expr readAssert();

  Expr readExpr();
  Expr readExpr1(Expr Lhs, int MinPrec);
  StringRef readParenLiteral();
  Expr readPrimary();
  Expr readTernary(Expr Cond);
  Expr readParenExpr();

  // For parsing version script.
  void readExtern(std::vector<SymbolVersion> *Globals);
  void readVersionDeclaration(StringRef VerStr);
  void readGlobal(StringRef VerStr);
  void readLocal();

  ScriptConfiguration &Opt = *ScriptConfig;
  StringSaver Saver = {ScriptConfig->Alloc};
  bool IsUnderSysroot;
};

void ScriptParser::readVersionScript() {
  readVersionScriptCommand();
  if (!atEOF())
    setError("EOF expected, but got " + next());
}

void ScriptParser::readVersionScriptCommand() {
  if (consume("{")) {
    readVersionDeclaration("");
    return;
  }

  while (!atEOF() && !Error && peek() != "}") {
    StringRef VerStr = next();
    if (VerStr == "{") {
      setError("anonymous version definition is used in "
               "combination with other version definitions");
      return;
    }
    expect("{");
    readVersionDeclaration(VerStr);
  }
}

void ScriptParser::readVersion() {
  expect("{");
  readVersionScriptCommand();
  expect("}");
}

void ScriptParser::readLinkerScript() {
  while (!atEOF()) {
    StringRef Tok = next();
    if (Tok == ";")
      continue;

    if (Tok == "ASSERT") {
      Opt.Commands.emplace_back(new AssertCommand(readAssert()));
    } else if (Tok == "ENTRY") {
      readEntry();
    } else if (Tok == "EXTERN") {
      readExtern();
    } else if (Tok == "GROUP" || Tok == "INPUT") {
      readGroup();
    } else if (Tok == "INCLUDE") {
      readInclude();
    } else if (Tok == "OUTPUT") {
      readOutput();
    } else if (Tok == "OUTPUT_ARCH") {
      readOutputArch();
    } else if (Tok == "OUTPUT_FORMAT") {
      readOutputFormat();
    } else if (Tok == "PHDRS") {
      readPhdrs();
    } else if (Tok == "SEARCH_DIR") {
      readSearchDir();
    } else if (Tok == "SECTIONS") {
      readSections();
    } else if (Tok == "VERSION") {
      readVersion();
    } else if (SymbolAssignment *Cmd = readProvideOrAssignment(Tok, true)) {
      Opt.Commands.emplace_back(Cmd);
    } else {
      setError("unknown directive: " + Tok);
    }
  }
}

void ScriptParser::addFile(StringRef S) {
  if (IsUnderSysroot && S.startswith("/")) {
    SmallString<128> PathData;
    StringRef Path = (Config->Sysroot + S).toStringRef(PathData);
    if (sys::fs::exists(Path)) {
      Driver->addFile(Saver.save(Path));
      return;
    }
  }

  if (sys::path::is_absolute(S)) {
    Driver->addFile(S);
  } else if (S.startswith("=")) {
    if (Config->Sysroot.empty())
      Driver->addFile(S.substr(1));
    else
      Driver->addFile(Saver.save(Config->Sysroot + "/" + S.substr(1)));
  } else if (S.startswith("-l")) {
    Driver->addLibrary(S.substr(2));
  } else if (sys::fs::exists(S)) {
    Driver->addFile(S);
  } else {
    std::string Path = findFromSearchPaths(S);
    if (Path.empty())
      setError("unable to find " + S);
    else
      Driver->addFile(Saver.save(Path));
  }
}

void ScriptParser::readAsNeeded() {
  expect("(");
  bool Orig = Config->AsNeeded;
  Config->AsNeeded = true;
  while (!Error && !consume(")"))
    addFile(unquote(next()));
  Config->AsNeeded = Orig;
}

void ScriptParser::readEntry() {
  // -e <symbol> takes predecence over ENTRY(<symbol>).
  expect("(");
  StringRef Tok = next();
  if (Config->Entry.empty())
    Config->Entry = Tok;
  expect(")");
}

void ScriptParser::readExtern() {
  expect("(");
  while (!Error && !consume(")"))
    Config->Undefined.push_back(next());
}

void ScriptParser::readGroup() {
  expect("(");
  while (!Error && !consume(")")) {
    StringRef Tok = next();
    if (Tok == "AS_NEEDED")
      readAsNeeded();
    else
      addFile(unquote(Tok));
  }
}

void ScriptParser::readInclude() {
  StringRef Tok = next();
  auto MBOrErr = MemoryBuffer::getFile(unquote(Tok));
  if (!MBOrErr) {
    setError("cannot open " + Tok);
    return;
  }
  std::unique_ptr<MemoryBuffer> &MB = *MBOrErr;
  StringRef S = Saver.save(MB->getMemBufferRef().getBuffer());
  std::vector<StringRef> V = tokenize(S);
  Tokens.insert(Tokens.begin() + Pos, V.begin(), V.end());
}

void ScriptParser::readOutput() {
  // -o <file> takes predecence over OUTPUT(<file>).
  expect("(");
  StringRef Tok = next();
  if (Config->OutputFile.empty())
    Config->OutputFile = unquote(Tok);
  expect(")");
}

void ScriptParser::readOutputArch() {
  // Error checking only for now.
  expect("(");
  skip();
  expect(")");
}

void ScriptParser::readOutputFormat() {
  // Error checking only for now.
  expect("(");
  skip();
  StringRef Tok = next();
  if (Tok == ")")
    return;
  if (Tok != ",") {
    setError("unexpected token: " + Tok);
    return;
  }
  skip();
  expect(",");
  skip();
  expect(")");
}

void ScriptParser::readPhdrs() {
  expect("{");
  while (!Error && !consume("}")) {
    StringRef Tok = next();
    Opt.PhdrsCommands.push_back(
        {Tok, PT_NULL, false, false, UINT_MAX, nullptr});
    PhdrsCommand &PhdrCmd = Opt.PhdrsCommands.back();

    PhdrCmd.Type = readPhdrType();
    do {
      Tok = next();
      if (Tok == ";")
        break;
      if (Tok == "FILEHDR")
        PhdrCmd.HasFilehdr = true;
      else if (Tok == "PHDRS")
        PhdrCmd.HasPhdrs = true;
      else if (Tok == "AT")
        PhdrCmd.LMAExpr = readParenExpr();
      else if (Tok == "FLAGS") {
        expect("(");
        // Passing 0 for the value of dot is a bit of a hack. It means that
        // we accept expressions like ".|1".
        PhdrCmd.Flags = readExpr()(0);
        expect(")");
      } else
        setError("unexpected header attribute: " + Tok);
    } while (!Error);
  }
}

void ScriptParser::readSearchDir() {
  expect("(");
  StringRef Tok = next();
  if (!Config->Nostdlib)
    Config->SearchPaths.push_back(unquote(Tok));
  expect(")");
}

void ScriptParser::readSections() {
  Opt.HasSections = true;
  expect("{");
  while (!Error && !consume("}")) {
    StringRef Tok = next();
    BaseCommand *Cmd = readProvideOrAssignment(Tok, true);
    if (!Cmd) {
      if (Tok == "ASSERT")
        Cmd = new AssertCommand(readAssert());
      else
        Cmd = readOutputSectionDescription(Tok);
    }
    Opt.Commands.emplace_back(Cmd);
  }
}

static int precedence(StringRef Op) {
  return StringSwitch<int>(Op)
      .Cases("*", "/", 5)
      .Cases("+", "-", 4)
      .Cases("<<", ">>", 3)
      .Cases("<", "<=", ">", ">=", "==", "!=", 2)
      .Cases("&", "|", 1)
      .Default(-1);
}

Regex ScriptParser::readFilePatterns() {
  std::vector<StringRef> V;
  while (!Error && !consume(")"))
    V.push_back(next());
  return compileGlobPatterns(V);
}

SortSectionPolicy ScriptParser::readSortKind() {
  if (consume("SORT") || consume("SORT_BY_NAME"))
    return SortSectionPolicy::Name;
  if (consume("SORT_BY_ALIGNMENT"))
    return SortSectionPolicy::Alignment;
  if (consume("SORT_BY_INIT_PRIORITY"))
    return SortSectionPolicy::Priority;
  if (consume("SORT_NONE"))
    return SortSectionPolicy::None;
  return SortSectionPolicy::Default;
}

// Method reads a list of sequence of excluded files and section globs given in
// a following form: ((EXCLUDE_FILE(file_pattern+))? section_pattern+)+
// Example: *(.foo.1 EXCLUDE_FILE (*a.o) .foo.2 EXCLUDE_FILE (*b.o) .foo.3)
// The semantics of that is next:
// * Include .foo.1 from every file.
// * Include .foo.2 from every file but a.o
// * Include .foo.3 from every file but b.o
std::vector<SectionPattern> ScriptParser::readInputSectionsList() {
  std::vector<SectionPattern> Ret;
  while (!Error && peek() != ")") {
    Regex ExcludeFileRe;
    if (consume("EXCLUDE_FILE")) {
      expect("(");
      ExcludeFileRe = readFilePatterns();
    }

    std::vector<StringRef> V;
    while (!Error && peek() != ")" && peek() != "EXCLUDE_FILE")
      V.push_back(next());

    if (!V.empty())
      Ret.push_back({std::move(ExcludeFileRe), compileGlobPatterns(V)});
    else
      setError("section pattern is expected");
  }
  return Ret;
}

// Section pattern grammar can have complex expressions, for example:
// *(SORT(.foo.* EXCLUDE_FILE (*file1.o) .bar.*) .bar.* SORT(.zed.*))
// Generally is a sequence of globs and excludes that may be wrapped in a SORT()
// commands, like: SORT(glob0) glob1 glob2 SORT(glob4)
// This methods handles wrapping sequences of excluded files and section globs
// into SORT() if that needed and reads them all.
InputSectionDescription *
ScriptParser::readInputSectionRules(StringRef FilePattern) {
  auto *Cmd = new InputSectionDescription(FilePattern);
  expect("(");
  while (!HasError && !consume(")")) {
    SortSectionPolicy Outer = readSortKind();
    SortSectionPolicy Inner = SortSectionPolicy::Default;
    std::vector<SectionPattern> V;
    if (Outer != SortSectionPolicy::Default) {
      expect("(");
      Inner = readSortKind();
      if (Inner != SortSectionPolicy::Default) {
        expect("(");
        V = readInputSectionsList();
        expect(")");
      } else {
        V = readInputSectionsList();
      }
      expect(")");
    } else {
      V = readInputSectionsList();
    }

    for (SectionPattern &Pat : V) {
      Pat.SortInner = Inner;
      Pat.SortOuter = Outer;
    }

    std::move(V.begin(), V.end(), std::back_inserter(Cmd->SectionPatterns));
  }
  return Cmd;
}

InputSectionDescription *
ScriptParser::readInputSectionDescription(StringRef Tok) {
  // Input section wildcard can be surrounded by KEEP.
  // https://sourceware.org/binutils/docs/ld/Input-Section-Keep.html#Input-Section-Keep
  if (Tok == "KEEP") {
    expect("(");
    StringRef FilePattern = next();
    InputSectionDescription *Cmd = readInputSectionRules(FilePattern);
    expect(")");
    Opt.KeptSections.push_back(Cmd);
    return Cmd;
  }
  return readInputSectionRules(Tok);
}

void ScriptParser::readSort() {
  expect("(");
  expect("CONSTRUCTORS");
  expect(")");
}

Expr ScriptParser::readAssert() {
  expect("(");
  Expr E = readExpr();
  expect(",");
  StringRef Msg = unquote(next());
  expect(")");
  return [=](uint64_t Dot) {
    uint64_t V = E(Dot);
    if (!V)
      error(Msg);
    return V;
  };
}

// Reads a FILL(expr) command. We handle the FILL command as an
// alias for =fillexp section attribute, which is different from
// what GNU linkers do.
// https://sourceware.org/binutils/docs/ld/Output-Section-Data.html
std::vector<uint8_t> ScriptParser::readFill() {
  expect("(");
  std::vector<uint8_t> V = readOutputSectionFiller(next());
  expect(")");
  expect(";");
  return V;
}

OutputSectionCommand *
ScriptParser::readOutputSectionDescription(StringRef OutSec) {
  OutputSectionCommand *Cmd = new OutputSectionCommand(OutSec);

  // Read an address expression.
  // https://sourceware.org/binutils/docs/ld/Output-Section-Address.html#Output-Section-Address
  if (peek() != ":")
    Cmd->AddrExpr = readExpr();

  expect(":");

  if (consume("AT"))
    Cmd->LMAExpr = readParenExpr();
  if (consume("ALIGN"))
    Cmd->AlignExpr = readParenExpr();
  if (consume("SUBALIGN"))
    Cmd->SubalignExpr = readParenExpr();

  // Parse constraints.
  if (consume("ONLY_IF_RO"))
    Cmd->Constraint = ConstraintKind::ReadOnly;
  if (consume("ONLY_IF_RW"))
    Cmd->Constraint = ConstraintKind::ReadWrite;
  expect("{");

  while (!Error && !consume("}")) {
    StringRef Tok = next();
    if (SymbolAssignment *Assignment = readProvideOrAssignment(Tok, false))
      Cmd->Commands.emplace_back(Assignment);
    else if (BytesDataCommand *Data = readBytesDataCommand(Tok))
      Cmd->Commands.emplace_back(Data);
    else if (Tok == "FILL")
      Cmd->Filler = readFill();
    else if (Tok == "SORT")
      readSort();
    else if (peek() == "(")
      Cmd->Commands.emplace_back(readInputSectionDescription(Tok));
    else
      setError("unknown command " + Tok);
  }
  Cmd->Phdrs = readOutputSectionPhdrs();

  if (consume("="))
    Cmd->Filler = readOutputSectionFiller(next());
  else if (peek().startswith("="))
    Cmd->Filler = readOutputSectionFiller(next().drop_front());

  return Cmd;
}

// Read "=<number>" where <number> is an octal/decimal/hexadecimal number.
// https://sourceware.org/binutils/docs/ld/Output-Section-Fill.html
//
// ld.gold is not fully compatible with ld.bfd. ld.bfd handles
// hexstrings as blobs of arbitrary sizes, while ld.gold handles them
// as 32-bit big-endian values. We will do the same as ld.gold does
// because it's simpler than what ld.bfd does.
std::vector<uint8_t> ScriptParser::readOutputSectionFiller(StringRef Tok) {
  uint32_t V;
  if (Tok.getAsInteger(0, V)) {
    setError("invalid filler expression: " + Tok);
    return {};
  }
  return {uint8_t(V >> 24), uint8_t(V >> 16), uint8_t(V >> 8), uint8_t(V)};
}

SymbolAssignment *ScriptParser::readProvideHidden(bool Provide, bool Hidden) {
  expect("(");
  SymbolAssignment *Cmd = readAssignment(next());
  Cmd->Provide = Provide;
  Cmd->Hidden = Hidden;
  expect(")");
  expect(";");
  return Cmd;
}

SymbolAssignment *ScriptParser::readProvideOrAssignment(StringRef Tok,
                                                        bool MakeAbsolute) {
  SymbolAssignment *Cmd = nullptr;
  if (peek() == "=" || peek() == "+=") {
    Cmd = readAssignment(Tok);
    expect(";");
  } else if (Tok == "PROVIDE") {
    Cmd = readProvideHidden(true, false);
  } else if (Tok == "HIDDEN") {
    Cmd = readProvideHidden(false, true);
  } else if (Tok == "PROVIDE_HIDDEN") {
    Cmd = readProvideHidden(true, true);
  }
  if (Cmd && MakeAbsolute)
    Cmd->IsAbsolute = true;
  return Cmd;
}

static uint64_t getSymbolValue(StringRef S, uint64_t Dot) {
  if (S == ".")
    return Dot;
  return ScriptBase->getSymbolValue(S);
}

SymbolAssignment *ScriptParser::readAssignment(StringRef Name) {
  StringRef Op = next();
  bool IsAbsolute = false;
  Expr E;
  assert(Op == "=" || Op == "+=");
  if (consume("ABSOLUTE")) {
    // The RHS may be something like "ABSOLUTE(.) & 0xff".
    // Call readExpr1 to read the whole expression.
    E = readExpr1(readParenExpr(), 0);
    IsAbsolute = true;
  } else {
    E = readExpr();
  }
  if (Op == "+=")
    E = [=](uint64_t Dot) { return getSymbolValue(Name, Dot) + E(Dot); };
  return new SymbolAssignment(Name, E, IsAbsolute);
}

// This is an operator-precedence parser to parse a linker
// script expression.
Expr ScriptParser::readExpr() { return readExpr1(readPrimary(), 0); }

static Expr combine(StringRef Op, Expr L, Expr R) {
  if (Op == "*")
    return [=](uint64_t Dot) { return L(Dot) * R(Dot); };
  if (Op == "/") {
    return [=](uint64_t Dot) -> uint64_t {
      uint64_t RHS = R(Dot);
      if (RHS == 0) {
        error("division by zero");
        return 0;
      }
      return L(Dot) / RHS;
    };
  }
  if (Op == "+")
    return [=](uint64_t Dot) { return L(Dot) + R(Dot); };
  if (Op == "-")
    return [=](uint64_t Dot) { return L(Dot) - R(Dot); };
  if (Op == "<<")
    return [=](uint64_t Dot) { return L(Dot) << R(Dot); };
  if (Op == ">>")
    return [=](uint64_t Dot) { return L(Dot) >> R(Dot); };
  if (Op == "<")
    return [=](uint64_t Dot) { return L(Dot) < R(Dot); };
  if (Op == ">")
    return [=](uint64_t Dot) { return L(Dot) > R(Dot); };
  if (Op == ">=")
    return [=](uint64_t Dot) { return L(Dot) >= R(Dot); };
  if (Op == "<=")
    return [=](uint64_t Dot) { return L(Dot) <= R(Dot); };
  if (Op == "==")
    return [=](uint64_t Dot) { return L(Dot) == R(Dot); };
  if (Op == "!=")
    return [=](uint64_t Dot) { return L(Dot) != R(Dot); };
  if (Op == "&")
    return [=](uint64_t Dot) { return L(Dot) & R(Dot); };
  if (Op == "|")
    return [=](uint64_t Dot) { return L(Dot) | R(Dot); };
  llvm_unreachable("invalid operator");
}

// This is a part of the operator-precedence parser. This function
// assumes that the remaining token stream starts with an operator.
Expr ScriptParser::readExpr1(Expr Lhs, int MinPrec) {
  while (!atEOF() && !Error) {
    // Read an operator and an expression.
    StringRef Op1 = peek();
    if (Op1 == "?")
      return readTernary(Lhs);
    if (precedence(Op1) < MinPrec)
      break;
    skip();
    Expr Rhs = readPrimary();

    // Evaluate the remaining part of the expression first if the
    // next operator has greater precedence than the previous one.
    // For example, if we have read "+" and "3", and if the next
    // operator is "*", then we'll evaluate 3 * ... part first.
    while (!atEOF()) {
      StringRef Op2 = peek();
      if (precedence(Op2) <= precedence(Op1))
        break;
      Rhs = readExpr1(Rhs, precedence(Op2));
    }

    Lhs = combine(Op1, Lhs, Rhs);
  }
  return Lhs;
}

uint64_t static getConstant(StringRef S) {
  if (S == "COMMONPAGESIZE")
    return Target->PageSize;
  if (S == "MAXPAGESIZE")
    return Config->MaxPageSize;
  error("unknown constant: " + S);
  return 0;
}

// Parses Tok as an integer. Returns true if successful.
// It recognizes hexadecimal (prefixed with "0x" or suffixed with "H")
// and decimal numbers. Decimal numbers may have "K" (kilo) or
// "M" (mega) prefixes.
static bool readInteger(StringRef Tok, uint64_t &Result) {
  if (Tok.startswith("-")) {
    if (!readInteger(Tok.substr(1), Result))
      return false;
    Result = -Result;
    return true;
  }
  if (Tok.startswith_lower("0x"))
    return !Tok.substr(2).getAsInteger(16, Result);
  if (Tok.endswith_lower("H"))
    return !Tok.drop_back().getAsInteger(16, Result);

  int Suffix = 1;
  if (Tok.endswith_lower("K")) {
    Suffix = 1024;
    Tok = Tok.drop_back();
  } else if (Tok.endswith_lower("M")) {
    Suffix = 1024 * 1024;
    Tok = Tok.drop_back();
  }
  if (Tok.getAsInteger(10, Result))
    return false;
  Result *= Suffix;
  return true;
}

BytesDataCommand *ScriptParser::readBytesDataCommand(StringRef Tok) {
  int Size = StringSwitch<unsigned>(Tok)
                 .Case("BYTE", 1)
                 .Case("SHORT", 2)
                 .Case("LONG", 4)
                 .Case("QUAD", 8)
                 .Default(-1);
  if (Size == -1)
    return nullptr;

  expect("(");
  uint64_t Val = 0;
  StringRef S = next();
  if (!readInteger(S, Val))
    setError("unexpected value: " + S);
  expect(")");
  return new BytesDataCommand(Val, Size);
}

StringRef ScriptParser::readParenLiteral() {
  expect("(");
  StringRef Tok = next();
  expect(")");
  return Tok;
}

Expr ScriptParser::readPrimary() {
  if (peek() == "(")
    return readParenExpr();

  StringRef Tok = next();

  if (Tok == "~") {
    Expr E = readPrimary();
    return [=](uint64_t Dot) { return ~E(Dot); };
  }
  if (Tok == "-") {
    Expr E = readPrimary();
    return [=](uint64_t Dot) { return -E(Dot); };
  }

  // Built-in functions are parsed here.
  // https://sourceware.org/binutils/docs/ld/Builtin-Functions.html.
  if (Tok == "ADDR") {
    StringRef Name = readParenLiteral();
    return
        [=](uint64_t Dot) { return ScriptBase->getOutputSectionAddress(Name); };
  }
  if (Tok == "LOADADDR") {
    StringRef Name = readParenLiteral();
    return [=](uint64_t Dot) { return ScriptBase->getOutputSectionLMA(Name); };
  }
  if (Tok == "ASSERT")
    return readAssert();
  if (Tok == "ALIGN") {
    Expr E = readParenExpr();
    return [=](uint64_t Dot) { return alignTo(Dot, E(Dot)); };
  }
  if (Tok == "CONSTANT") {
    StringRef Name = readParenLiteral();
    return [=](uint64_t Dot) { return getConstant(Name); };
  }
  if (Tok == "DEFINED") {
    expect("(");
    StringRef Tok = next();
    expect(")");
    return [=](uint64_t Dot) { return ScriptBase->isDefined(Tok) ? 1 : 0; };
  }
  if (Tok == "SEGMENT_START") {
    expect("(");
    skip();
    expect(",");
    Expr E = readExpr();
    expect(")");
    return [=](uint64_t Dot) { return E(Dot); };
  }
  if (Tok == "DATA_SEGMENT_ALIGN") {
    expect("(");
    Expr E = readExpr();
    expect(",");
    readExpr();
    expect(")");
    return [=](uint64_t Dot) { return alignTo(Dot, E(Dot)); };
  }
  if (Tok == "DATA_SEGMENT_END") {
    expect("(");
    expect(".");
    expect(")");
    return [](uint64_t Dot) { return Dot; };
  }
  // GNU linkers implements more complicated logic to handle
  // DATA_SEGMENT_RELRO_END. We instead ignore the arguments and just align to
  // the next page boundary for simplicity.
  if (Tok == "DATA_SEGMENT_RELRO_END") {
    expect("(");
    readExpr();
    expect(",");
    readExpr();
    expect(")");
    return [](uint64_t Dot) { return alignTo(Dot, Target->PageSize); };
  }
  if (Tok == "SIZEOF") {
    StringRef Name = readParenLiteral();
    return [=](uint64_t Dot) { return ScriptBase->getOutputSectionSize(Name); };
  }
  if (Tok == "ALIGNOF") {
    StringRef Name = readParenLiteral();
    return
        [=](uint64_t Dot) { return ScriptBase->getOutputSectionAlign(Name); };
  }
  if (Tok == "SIZEOF_HEADERS")
    return [=](uint64_t Dot) { return ScriptBase->getHeaderSize(); };

  // Tok is a literal number.
  uint64_t V;
  if (readInteger(Tok, V))
    return [=](uint64_t Dot) { return V; };

  // Tok is a symbol name.
  if (Tok != "." && !isValidCIdentifier(Tok))
    setError("malformed number: " + Tok);
  return [=](uint64_t Dot) { return getSymbolValue(Tok, Dot); };
}

Expr ScriptParser::readTernary(Expr Cond) {
  skip();
  Expr L = readExpr();
  expect(":");
  Expr R = readExpr();
  return [=](uint64_t Dot) { return Cond(Dot) ? L(Dot) : R(Dot); };
}

Expr ScriptParser::readParenExpr() {
  expect("(");
  Expr E = readExpr();
  expect(")");
  return E;
}

std::vector<StringRef> ScriptParser::readOutputSectionPhdrs() {
  std::vector<StringRef> Phdrs;
  while (!Error && peek().startswith(":")) {
    StringRef Tok = next();
    Tok = (Tok.size() == 1) ? next() : Tok.substr(1);
    if (Tok.empty()) {
      setError("section header name is empty");
      break;
    }
    Phdrs.push_back(Tok);
  }
  return Phdrs;
}

// Read a program header type name. The next token must be a
// name of a program header type or a constant (e.g. "0x3").
unsigned ScriptParser::readPhdrType() {
  StringRef Tok = next();
  uint64_t Val;
  if (readInteger(Tok, Val))
    return Val;

  unsigned Ret = StringSwitch<unsigned>(Tok)
                     .Case("PT_NULL", PT_NULL)
                     .Case("PT_LOAD", PT_LOAD)
                     .Case("PT_DYNAMIC", PT_DYNAMIC)
                     .Case("PT_INTERP", PT_INTERP)
                     .Case("PT_NOTE", PT_NOTE)
                     .Case("PT_SHLIB", PT_SHLIB)
                     .Case("PT_PHDR", PT_PHDR)
                     .Case("PT_TLS", PT_TLS)
                     .Case("PT_GNU_EH_FRAME", PT_GNU_EH_FRAME)
                     .Case("PT_GNU_STACK", PT_GNU_STACK)
                     .Case("PT_GNU_RELRO", PT_GNU_RELRO)
                     .Case("PT_OPENBSD_RANDOMIZE", PT_OPENBSD_RANDOMIZE)
                     .Case("PT_OPENBSD_WXNEEDED", PT_OPENBSD_WXNEEDED)
                     .Default(-1);

  if (Ret == (unsigned)-1) {
    setError("invalid program header type: " + Tok);
    return PT_NULL;
  }
  return Ret;
}

void ScriptParser::readVersionDeclaration(StringRef VerStr) {
  // Identifiers start at 2 because 0 and 1 are reserved
  // for VER_NDX_LOCAL and VER_NDX_GLOBAL constants.
  size_t VersionId = Config->VersionDefinitions.size() + 2;
  Config->VersionDefinitions.push_back({VerStr, VersionId});

  if (consume("global:") || peek() != "local:")
    readGlobal(VerStr);
  if (consume("local:"))
    readLocal();
  expect("}");

  // Each version may have a parent version. For example, "Ver2" defined as
  // "Ver2 { global: foo; local: *; } Ver1;" has "Ver1" as a parent. This
  // version hierarchy is, probably against your instinct, purely for human; the
  // runtime doesn't care about them at all. In LLD, we simply skip the token.
  if (!VerStr.empty() && peek() != ";")
    skip();
  expect(";");
}

void ScriptParser::readLocal() {
  Config->DefaultSymbolVersion = VER_NDX_LOCAL;
  expect("*");
  expect(";");
}

void ScriptParser::readExtern(std::vector<SymbolVersion> *Globals) {
  expect("\"C++\"");
  expect("{");

  for (;;) {
    if (peek() == "}" || Error)
      break;
    bool HasWildcard = !peek().startswith("\"") && hasWildcard(peek());
    Globals->push_back({unquote(next()), true, HasWildcard});
    expect(";");
  }

  expect("}");
  expect(";");
}

void ScriptParser::readGlobal(StringRef VerStr) {
  std::vector<SymbolVersion> *Globals;
  if (VerStr.empty())
    Globals = &Config->VersionScriptGlobals;
  else
    Globals = &Config->VersionDefinitions.back().Globals;

  for (;;) {
    if (consume("extern"))
      readExtern(Globals);

    StringRef Cur = peek();
    if (Cur == "}" || Cur == "local:" || Error)
      return;
    skip();
    Globals->push_back({unquote(Cur), false, hasWildcard(Cur)});
    expect(";");
  }
}

static bool isUnderSysroot(StringRef Path) {
  if (Config->Sysroot == "")
    return false;
  for (; !Path.empty(); Path = sys::path::parent_path(Path))
    if (sys::fs::equivalent(Config->Sysroot, Path))
      return true;
  return false;
}

void elf::readLinkerScript(MemoryBufferRef MB) {
  StringRef Path = MB.getBufferIdentifier();
  ScriptParser(MB.getBuffer(), isUnderSysroot(Path)).readLinkerScript();
}

void elf::readVersionScript(MemoryBufferRef MB) {
  ScriptParser(MB.getBuffer(), false).readVersionScript();
}

template class elf::LinkerScript<ELF32LE>;
template class elf::LinkerScript<ELF32BE>;
template class elf::LinkerScript<ELF64LE>;
template class elf::LinkerScript<ELF64BE>;
