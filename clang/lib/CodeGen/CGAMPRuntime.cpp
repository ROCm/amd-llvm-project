//===----- CGAMPRuntime.cpp - Interface to C++ AMP Runtime ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This provides an abstract class for C++ AMP code generation.  Concrete
// subclasses of this implement code generation for specific C++ AMP
// runtime libraries.
//
//===----------------------------------------------------------------------===//

#include "CodeGenFunction.h"
#include "CGAMPRuntime.h"
#include "clang/AST/Decl.h"
#include "clang/AST/ExprCXX.h"
#include "CGCall.h"
#include "TargetInfo.h"

namespace clang {
namespace CodeGen {

CGAMPRuntime::~CGAMPRuntime() {}

/// Creates an instance of a C++ AMP runtime class.
CGAMPRuntime *CreateAMPRuntime(CodeGenModule &CGM) {
  return new CGAMPRuntime(CGM);
}
static CXXMethodDecl *findValidIndexType(QualType IndexTy) {
  CXXRecordDecl *IndexClass = (*IndexTy).getAsCXXRecordDecl();
  CXXMethodDecl *IndexConstructor = NULL;
  if (IndexClass) {
    for (CXXRecordDecl::method_iterator CtorIt = IndexClass->method_begin(),
        CtorE = IndexClass->method_end();
        CtorIt != CtorE; ++CtorIt) {
      if (CtorIt->hasAttr<AnnotateAttr>() &&
          CtorIt->getAttr<AnnotateAttr>()->getAnnotation() ==
            "__cxxamp_opencl_index") {
        IndexConstructor = *CtorIt;
      }
    }
  }
  return IndexConstructor;
}
/// Operations:
/// For each reference-typed members, construct temporary object
/// Invoke constructor of index
/// Invoke constructor of the class
/// Invoke operator(index)
namespace
{
    inline
    void addFunctorAttrToTrampoline(
        const CXXMethodDecl* callOp, FunctionDecl* trampoline)
    {
        // TODO: this is an awful hack which should be removed once we move to pfe_v2.
        if (callOp->hasAttr<AMDGPUFlatWorkGroupSizeAttr>()) {
            trampoline->addAttr(callOp->getAttr<AMDGPUFlatWorkGroupSizeAttr>());
        }
        if (callOp->hasAttr<AMDGPUWavesPerEUAttr>()) {
          trampoline->addAttr(callOp->getAttr<AMDGPUWavesPerEUAttr>());
        }
    }
}
void CGAMPRuntime::EmitTrampolineBody(CodeGenFunction &CGF,
  const FunctionDecl *Trampoline, FunctionArgList& Args) {
  const CXXRecordDecl *ClassDecl = dyn_cast<CXXMethodDecl>(Trampoline)->getParent();

  for (auto&& x : ClassDecl->methods()) {
    if (x->getOverloadedOperator() == OO_Call) {
      addFunctorAttrToTrampoline(x, const_cast<FunctionDecl*>(Trampoline));
      break;
    }
  }

  assert(ClassDecl);
  // Allocate "this"
  Address ai = CGF.CreateMemTemp(QualType(ClassDecl->getTypeForDecl(),0));
  // Locate the constructor to call
  if(ClassDecl->getCXXAMPDeserializationConstructor()==NULL) {
    return;
  }
  CXXConstructorDecl *DeserializeConstructor =
    dyn_cast<CXXConstructorDecl>(
      ClassDecl->getCXXAMPDeserializationConstructor());
  assert(DeserializeConstructor);
  CallArgList DeserializerArgs;
  // this
  DeserializerArgs.add(RValue::get(ai.getPointer()),
    DeserializeConstructor ->getThisType(CGF.getContext()));
  // the rest of constructor args. Create temporary objects for references
  // on stack
  CXXConstructorDecl::param_iterator CPI = DeserializeConstructor->param_begin(),
    CPE = DeserializeConstructor->param_end();
  for (FunctionArgList::iterator I = Args.begin();
       I != Args.end() && CPI != CPE; ++CPI) {
    // Reference types are only allowed to have one level; i.e. no
    // class base {&int}; class foo { bar &base; };
    QualType MemberType = (*CPI)->getType().getNonReferenceType();
    if (MemberType != (*CPI)->getType()) {
      if (!CGM.getLangOpts().HSAExtension) {
        assert(MemberType.getTypePtr()->isClassType() == true &&
               "Only supporting taking reference of classes");
        CXXRecordDecl *MemberClass = MemberType.getTypePtr()->getAsCXXRecordDecl();
        CXXConstructorDecl *MemberDeserializer =
  	dyn_cast<CXXConstructorDecl>(
            MemberClass->getCXXAMPDeserializationConstructor());
        assert(MemberDeserializer);
        std::vector<Expr*>MemberArgDeclRefs;
        for (CXXMethodDecl::param_iterator MCPI = MemberDeserializer->param_begin(),
          MCPE = MemberDeserializer->param_end(); MCPI!=MCPE; ++MCPI, ++I) {
          Expr *ArgDeclRef = DeclRefExpr::Create(CGM.getContext(),
                                                 NestedNameSpecifierLoc(),
                                                 SourceLocation(),
                                                 const_cast<VarDecl *>(*I),
                                                 false,
                                                 SourceLocation(),
                                                 (*MCPI)->getType(), VK_RValue);
  	  MemberArgDeclRefs.push_back(ArgDeclRef);
        }
        // Allocate "this" for member referenced objects
        Address mai = CGF.CreateMemTemp(MemberType);
        // Emit code to call the deserializing constructor of temp objects
        CXXConstructExpr *CXXCE = CXXConstructExpr::Create(
          CGM.getContext(), MemberType,
          SourceLocation(),
          MemberDeserializer,
          false,
          MemberArgDeclRefs,
          false, false, false, false,
          CXXConstructExpr::CK_Complete,
          SourceLocation());
        CGF.EmitCXXConstructorCall(MemberDeserializer,
          Ctor_Complete, false, false, mai, CXXCE);
        DeserializerArgs.add(RValue::get(mai.getPointer()), (*CPI)->getType());
      } else { // HSA extension check
        if (MemberType.getTypePtr()->isClassType()) {
          // hc::array should still be serialized as traditional C++AMP objects
          if (MemberType.getTypePtr()->isGPUArrayType()) {
            CXXRecordDecl *MemberClass = MemberType.getTypePtr()->getAsCXXRecordDecl();
            CXXConstructorDecl *MemberDeserializer =
            dyn_cast<CXXConstructorDecl>(
                MemberClass->getCXXAMPDeserializationConstructor());
            assert(MemberDeserializer);
            std::vector<Expr*>MemberArgDeclRefs;
            for (CXXMethodDecl::param_iterator MCPI = MemberDeserializer->param_begin(),
              MCPE = MemberDeserializer->param_end(); MCPI!=MCPE; ++MCPI, ++I) {
              Expr *ArgDeclRef = DeclRefExpr::Create(CGM.getContext(),
                                                     NestedNameSpecifierLoc(),
                                                     SourceLocation(),
                                                     const_cast<VarDecl *>(*I),
                                                     false,
                                                     SourceLocation(),
                                                     (*MCPI)->getType(), VK_RValue);
               MemberArgDeclRefs.push_back(ArgDeclRef);
            }
            // Allocate "this" for member referenced objects
            Address mai = CGF.CreateMemTemp(MemberType);
            // Emit code to call the deserializing constructor of temp objects
            CXXConstructExpr *CXXCE = CXXConstructExpr::Create(
              CGM.getContext(), MemberType,
              SourceLocation(),
              MemberDeserializer,
              false,
              MemberArgDeclRefs,
              false, false, false, false,
              CXXConstructExpr::CK_Complete,
              SourceLocation());
            CGF.EmitCXXConstructorCall(MemberDeserializer,
              Ctor_Complete, false, false, mai, CXXCE);
            DeserializerArgs.add(RValue::get(mai.getPointer()), (*CPI)->getType());
          } else {
            // capture by refernce for HSA
            Expr *ArgDeclRef = DeclRefExpr::Create(CGM.getContext(),
                                                   NestedNameSpecifierLoc(),
                                                   SourceLocation(),
                                                   const_cast<VarDecl *>(*I), false,
                                                   SourceLocation(),
                                                   (*I)->getType(), VK_RValue);
            RValue ArgRV = CGF.EmitAnyExpr(ArgDeclRef);
            DeserializerArgs.add(ArgRV, CGM.getContext().getPointerType(MemberType));
            ++I;
          }
        } else {
          // capture by refernce for HSA
          Expr *ArgDeclRef = DeclRefExpr::Create(CGM.getContext(),
              NestedNameSpecifierLoc(),
              SourceLocation(),
              const_cast<VarDecl *>(*I), false,
              SourceLocation(),
              (*I)->getType(), VK_RValue);
          RValue ArgRV = CGF.EmitAnyExpr(ArgDeclRef);
          DeserializerArgs.add(ArgRV, CGM.getContext().getPointerType(MemberType));
          ++I;
        }
      } // HSA extension check
    } else {
      Expr *ArgDeclRef = DeclRefExpr::Create(CGM.getContext(),
	  NestedNameSpecifierLoc(),
	  SourceLocation(),
	  const_cast<VarDecl *>(*I), false,
	  SourceLocation(),
	  (*I)->getType(), VK_RValue);
      RValue ArgRV = CGF.EmitAnyExpr(ArgDeclRef);
      DeserializerArgs.add(ArgRV, (*CPI)->getType());
      ++I;
    }
  }
  // Emit code to call the deserializing constructor
  {
    llvm::Constant *Callee = CGM.getAddrOfCXXStructor(
      DeserializeConstructor, StructorType::Complete);
    const FunctionProtoType *FPT =
      DeserializeConstructor->getType()->castAs<FunctionProtoType>();
    const CGFunctionInfo &DesFnInfo =
      CGM.getTypes().arrangeCXXStructorDeclaration(
          DeserializeConstructor, StructorType::Complete);
    for (unsigned I = 1, E = DeserializerArgs.size(); I != E; ++I) {
      auto T = FPT->getParamType(I-1);
      // EmitFromMemory is necessary in case function has bool parameter.
      if (T->isBooleanType()) {
        DeserializerArgs[I] = CallArg(RValue::get(
            CGF.EmitFromMemory(DeserializerArgs[I].RV.getScalarVal(), T)),
            T, false);
      }
    }
    CGF.EmitCall(DesFnInfo, CGCallee::forDirect(Callee), ReturnValueSlot(), DeserializerArgs);
  }
  // Locate the type of Concurrency::index<1>
  // Locate the operator to call
  CXXMethodDecl *KernelDecl = NULL;
  const FunctionType *MT = NULL;
  QualType IndexTy;
  for (CXXRecordDecl::method_iterator Method = ClassDecl->method_begin(),
      MethodEnd = ClassDecl->method_end();
      Method != MethodEnd; ++Method) {
    CXXMethodDecl *MethodDecl = *Method;
    if (MethodDecl->isOverloadedOperator() &&
        MethodDecl->getOverloadedOperator() == OO_Call &&
        MethodDecl->hasAttr<CXXAMPRestrictAMPAttr>()) {
      //Check types.
      if(MethodDecl->getNumParams() != 1)
	continue;
      ParmVarDecl *P = MethodDecl->getParamDecl(0);
      IndexTy = P->getType().getNonReferenceType();
      if (!findValidIndexType(IndexTy))
        continue;
      MT = dyn_cast<FunctionType>(MethodDecl->getType().getTypePtr());
      assert(MT);
      KernelDecl = MethodDecl;
      break;
    }
  }

  // in case we couldn't find any kernel declarator
  // raise error
  if (!KernelDecl) {
    CGF.CGM.getDiags().Report(ClassDecl->getLocation(), diag::err_amp_ill_formed_functor);
    return;
  }
  // Allocate Index
  Address index = CGF.CreateMemTemp(IndexTy);

  // Locate the constructor to call
  CXXMethodDecl *IndexConstructor = findValidIndexType(IndexTy);
  assert(IndexConstructor);
  // Emit code to call the Concurrency::index<1>::__cxxamp_opencl_index()
  if (!CGF.getLangOpts().AMPCPU) {
    if (CXXConstructorDecl *Constructor =
        dyn_cast <CXXConstructorDecl>(IndexConstructor)) {
      CXXConstructExpr *CXXCE = CXXConstructExpr::Create(
        CGM.getContext(), IndexTy,
        SourceLocation(),
        Constructor,
        false,
        ArrayRef<Expr*>(),
        false, false, false, false,
        CXXConstructExpr::CK_Complete,
        SourceLocation());
      CGF.EmitCXXConstructorCall(Constructor,
        Ctor_Complete, false, false, index, CXXCE);
    } else {
      llvm::FunctionType *indexInitType =
        CGM.getTypes().GetFunctionType(
          CGM.getTypes().arrangeCXXMethodDeclaration(IndexConstructor));
      llvm::Constant *indexInitAddr = CGM.GetAddrOfFunction(
        IndexConstructor, indexInitType);

      CGF.EmitCXXMemberOrOperatorCall(IndexConstructor, CGCallee::forDirect(indexInitAddr),
        ReturnValueSlot(), index.getPointer(), /*ImplicitParam=*/0, QualType(), /*CallExpr=*/nullptr, /*RtlArgs=*/nullptr);
    }
  }
  // Invoke this->operator(index)
  // Prepate the operator() to call
  llvm::FunctionType *fnType =
    CGM.getTypes().GetFunctionType(CGM.getTypes().arrangeCXXMethodDeclaration(KernelDecl));
  llvm::Constant *fnAddr = CGM.GetAddrOfFunction(KernelDecl, fnType);
  // Prepare argument
  CallArgList KArgs;
  // this
  KArgs.add(RValue::get(ai.getPointer()), KernelDecl ->getThisType(CGF.getContext()));
  // *index
  KArgs.add(RValue::getAggregate(index), IndexTy);

  const CGFunctionInfo &FnInfo = CGM.getTypes().arrangeFreeFunctionCall(KArgs, MT, false);
  CGF.EmitCall(FnInfo, CGCallee::forDirect(fnAddr), ReturnValueSlot(), KArgs);
  CGM.getTargetCodeGenInfo().setTargetAttributes(KernelDecl, CGF.CurFn, CGM);
}

void CGAMPRuntime::EmitTrampolineNameBody(CodeGenFunction &CGF,
  const FunctionDecl *Trampoline, FunctionArgList& Args) {
  const CXXRecordDecl *ClassDecl = dyn_cast<CXXMethodDecl>(Trampoline)->getParent();
  assert(ClassDecl);
  // Locate the trampoline
  // Locate the operator to call
  CXXMethodDecl *TrampolineDecl = NULL;
  for (CXXRecordDecl::method_iterator Method = ClassDecl->method_begin(),
      MethodEnd = ClassDecl->method_end();
      Method != MethodEnd; ++Method) {
    CXXMethodDecl *MethodDecl = *Method;
    if (Method->hasAttr<AnnotateAttr>() &&
        Method->getAttr<AnnotateAttr>()->getAnnotation() == "__cxxamp_trampoline") {
      TrampolineDecl = MethodDecl;
      break;
    }
  }
  assert(TrampolineDecl && "Trampoline not declared!");
  GlobalDecl GD(TrampolineDecl);
  llvm::Constant *S = llvm::ConstantDataArray::getString(CGM.getLLVMContext(),
    CGM.getMangledName(GD));
  llvm::GlobalVariable *GV = new llvm::GlobalVariable(CGM.getModule(), S->getType(),
    true, llvm::GlobalValue::PrivateLinkage, S, "__cxxamp_trampoline.kernelname");
  GV->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Global);

  //Create GetElementPtr(0, 0)
  std::vector<llvm::Constant*> indices;
  llvm::ConstantInt *zero = llvm::ConstantInt::get(CGM.getLLVMContext(), llvm::APInt(32, StringRef("0"), 10));
  indices.push_back(zero);
  indices.push_back(zero);
  llvm::Constant *const_ptr = llvm::ConstantExpr::getGetElementPtr(GV->getValueType(), GV, indices);
  CGF.Builder.CreateStore(const_ptr, CGF.ReturnValue);

}
} // namespace CodeGen
} // namespace clang
