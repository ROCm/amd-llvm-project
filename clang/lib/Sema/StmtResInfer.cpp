//===---- StmtResInfer.cpp - Inferring implementation for  auto-restricted FunctionDecl-- ===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Sema::TryCXXAMPRestrictionInferring method, which tries to 
// automatically infer CXXAMP specific auto-restricted FunctionDecl with any eligible 
// non-auto implicit restrictions onto it.
//
//===----------------------------------------------------------------------===//

#include "clang/Sema/SemaInternal.h"
#include "TypeLocBuilder.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/CharUnits.h"
#include "clang/AST/CommentDiagnostic.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/EvaluatedExprVisitor.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/StmtCXX.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Sema/Scope.h"
#include "clang/AST/Attr.h"
#include "clang/AST/CommentVisitor.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclVisitor.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Basic/Module.h"
#include "clang/AST/DeclVisitor.h"
#include "clang/AST/StmtVisitor.h"
using namespace clang::comments;

using namespace clang;
using namespace sema;


//===----------------------------------------------------------------------===//
// StmtResInfer Visitor
//===----------------------------------------------------------------------===//

namespace  {
  class StmtResInfer
      : public ConstDeclVisitor<StmtResInfer>, public ConstStmtVisitor<StmtResInfer> {
    const SourceManager *SM;
    unsigned CppAMPSpec;
    Sema& TheSema;

  private:
    inline void ClearResAMP() { CppAMPSpec &=~CPPAMP_AMP;}
    inline void ClearResCPU() { CppAMPSpec &=~CPPAMP_CPU;}

  public:
    StmtResInfer(Sema& S, unsigned& NonAutoSpec, const SourceManager *SM) 
      : SM(SM), CppAMPSpec(NonAutoSpec), TheSema(S) { }

    ~StmtResInfer() {
    }
    unsigned Infer(const Stmt* Node);

//private:
    void dumpDecl(const Decl *D);
    void dumpStmt(const Stmt *S);
    void dumpFullComment(const FullComment *C);

    // Formatting
    void indent();
    void unindent();
    void lastChild();
    bool hasMoreChildren();
    void setMoreChildren(bool Value);

    // Utilities
    void dumpPointer(const void *Ptr);
    void dumpSourceRange(SourceRange R);
    void dumpLocation(SourceLocation Loc);
    void dumpBareType(QualType T);
    void dumpType(QualType T);
    void dumpBareDeclRef(const Decl *Node);
    void dumpDeclRef(const Decl *Node, const char *Label = 0);
    void dumpName(const NamedDecl *D);
    bool hasNodes(const DeclContext *DC);
    void dumpDeclContext(const DeclContext *DC);
    void dumpAttr(const Attr *A);

    // C++ Utilities
    void dumpAccessSpecifier(AccessSpecifier AS);
    void dumpCXXCtorInitializer(const CXXCtorInitializer *Init);
    void dumpTemplateParameters(const TemplateParameterList *TPL);
    void dumpTemplateArgumentListInfo(const TemplateArgumentListInfo &TALI);
    void dumpTemplateArgumentLoc(const TemplateArgumentLoc &A);
    void dumpTemplateArgumentList(const TemplateArgumentList &TAL);
    void dumpTemplateArgument(const TemplateArgument &A,
                              SourceRange R = SourceRange());

    // Decls
    void VisitLabelDecl(const LabelDecl *D);
    void VisitTypedefDecl(const TypedefDecl *D);
    void VisitEnumDecl(const EnumDecl *D);
    void VisitRecordDecl(const RecordDecl *D);
    void VisitEnumConstantDecl(const EnumConstantDecl *D);
    void VisitIndirectFieldDecl(const IndirectFieldDecl *D);
    void VisitFunctionDecl(const FunctionDecl *D);
    void VisitFieldDecl(const FieldDecl *D);
    void VisitVarDecl(const VarDecl *D);
    void VisitFileScopeAsmDecl(const FileScopeAsmDecl *D);
    void VisitImportDecl(const ImportDecl *D);

    // C++ Decls
    void VisitNamespaceDecl(const NamespaceDecl *D);
    void VisitUsingDirectiveDecl(const UsingDirectiveDecl *D);
    void VisitNamespaceAliasDecl(const NamespaceAliasDecl *D);
    void VisitTypeAliasDecl(const TypeAliasDecl *D);
    void VisitTypeAliasTemplateDecl(const TypeAliasTemplateDecl *D);
    void VisitCXXRecordDecl(const CXXRecordDecl *D);
    void VisitStaticAssertDecl(const StaticAssertDecl *D);
    void VisitFunctionTemplateDecl(const FunctionTemplateDecl *D);
    void VisitClassTemplateDecl(const ClassTemplateDecl *D);
    void VisitClassTemplateSpecializationDecl(
        const ClassTemplateSpecializationDecl *D);
    void VisitClassTemplatePartialSpecializationDecl(
        const ClassTemplatePartialSpecializationDecl *D);
    void VisitClassScopeFunctionSpecializationDecl(
        const ClassScopeFunctionSpecializationDecl *D);
    void VisitTemplateTypeParmDecl(const TemplateTypeParmDecl *D);
    void VisitNonTypeTemplateParmDecl(const NonTypeTemplateParmDecl *D);
    void VisitTemplateTemplateParmDecl(const TemplateTemplateParmDecl *D);
    void VisitUsingDecl(const UsingDecl *D);
    void VisitUnresolvedUsingTypenameDecl(const UnresolvedUsingTypenameDecl *D);
    void VisitUnresolvedUsingValueDecl(const UnresolvedUsingValueDecl *D);
    void VisitUsingShadowDecl(const UsingShadowDecl *D);
    void VisitLinkageSpecDecl(const LinkageSpecDecl *D);
    void VisitAccessSpecDecl(const AccessSpecDecl *D);
    void VisitFriendDecl(const FriendDecl *D);

    // ObjC Decls
    void VisitObjCIvarDecl(const ObjCIvarDecl *D);
    void VisitObjCMethodDecl(const ObjCMethodDecl *D);
    void VisitObjCCategoryDecl(const ObjCCategoryDecl *D);
    void VisitObjCCategoryImplDecl(const ObjCCategoryImplDecl *D);
    void VisitObjCProtocolDecl(const ObjCProtocolDecl *D);
    void VisitObjCInterfaceDecl(const ObjCInterfaceDecl *D);
    void VisitObjCImplementationDecl(const ObjCImplementationDecl *D);
    void VisitObjCCompatibleAliasDecl(const ObjCCompatibleAliasDecl *D);
    void VisitObjCPropertyDecl(const ObjCPropertyDecl *D);
    void VisitObjCPropertyImplDecl(const ObjCPropertyImplDecl *D);
    void VisitBlockDecl(const BlockDecl *D);

    // Stmts.
    void VisitStmt(const Stmt *Node);
    void VisitDeclStmt(const DeclStmt *Node);
    void VisitAttributedStmt(const AttributedStmt *Node);
    void VisitLabelStmt(const LabelStmt *Node);
    void VisitGotoStmt(const GotoStmt *Node);
    void VisitCXXTryStmt(const CXXTryStmt* Node);
    
    // Exprs
    void VisitExpr(const Expr *Node);
    void VisitCastExpr(const CastExpr *Node);
    void VisitDeclRefExpr(const DeclRefExpr *Node);
    void VisitPredefinedExpr(const PredefinedExpr *Node);
    void VisitCharacterLiteral(const CharacterLiteral *Node);
    void VisitIntegerLiteral(const IntegerLiteral *Node);
    void VisitFloatingLiteral(const FloatingLiteral *Node);
    void VisitStringLiteral(const StringLiteral *Str);
    void VisitUnaryOperator(const UnaryOperator *Node);
    void VisitUnaryExprOrTypeTraitExpr(const UnaryExprOrTypeTraitExpr *Node);
    void VisitMemberExpr(const MemberExpr *Node);
    void VisitExtVectorElementExpr(const ExtVectorElementExpr *Node);
    void VisitBinaryOperator(const BinaryOperator *Node);
    void VisitCompoundAssignOperator(const CompoundAssignOperator *Node);
    void VisitAddrLabelExpr(const AddrLabelExpr *Node);
    void VisitBlockExpr(const BlockExpr *Node);
    void VisitOpaqueValueExpr(const OpaqueValueExpr *Node);

    // C++
    void VisitCXXNamedCastExpr(const CXXNamedCastExpr *Node);
    void VisitCXXBoolLiteralExpr(const CXXBoolLiteralExpr *Node);
    void VisitCXXThisExpr(const CXXThisExpr *Node);
    void VisitCXXThrowExpr(const CXXThrowExpr *Node);
    void VisitCXXFunctionalCastExpr(const CXXFunctionalCastExpr *Node);
    void VisitCXXConstructExpr(const CXXConstructExpr *Node);
    void VisitCXXBindTemporaryExpr(const CXXBindTemporaryExpr *Node);
    void VisitExprWithCleanups(const ExprWithCleanups *Node);
    void VisitUnresolvedLookupExpr(const UnresolvedLookupExpr *Node);
    void dumpCXXTemporary(const CXXTemporary *Temporary);
    void VisitCXXTypeidExpr(const CXXTypeidExpr* Node);
    void VisitCXXDynamicCastExpr(const CXXDynamicCastExpr* Node);

    // ObjC
    void VisitObjCAtCatchStmt(const ObjCAtCatchStmt *Node);
    void VisitObjCEncodeExpr(const ObjCEncodeExpr *Node);
    void VisitObjCMessageExpr(const ObjCMessageExpr *Node);
    void VisitObjCBoxedExpr(const ObjCBoxedExpr *Node);
    void VisitObjCSelectorExpr(const ObjCSelectorExpr *Node);
    void VisitObjCProtocolExpr(const ObjCProtocolExpr *Node);
    void VisitObjCPropertyRefExpr(const ObjCPropertyRefExpr *Node);
    void VisitObjCSubscriptRefExpr(const ObjCSubscriptRefExpr *Node);
    void VisitObjCIvarRefExpr(const ObjCIvarRefExpr *Node);
    void VisitObjCBoolLiteralExpr(const ObjCBoolLiteralExpr *Node);

    // Comments.
    const char *getCommandName(unsigned CommandID);
    void dumpComment(const Comment *C);

    // Inline comments.
    void visitTextComment(const TextComment *C);
    void visitInlineCommandComment(const InlineCommandComment *C);
    void visitHTMLStartTagComment(const HTMLStartTagComment *C);
    void visitHTMLEndTagComment(const HTMLEndTagComment *C);

    // Block comments.
    void visitBlockCommandComment(const BlockCommandComment *C);
    void visitParamCommandComment(const ParamCommandComment *C);
    void visitTParamCommandComment(const TParamCommandComment *C);
    void visitVerbatimBlockComment(const VerbatimBlockComment *C);
    void visitVerbatimBlockLineComment(const VerbatimBlockLineComment *C);
    void visitVerbatimLineComment(const VerbatimLineComment *C);
  };
}
void StmtResInfer::dumpPointer(const void *Ptr) {
}

void StmtResInfer::dumpLocation(SourceLocation Loc) {
}

void StmtResInfer::dumpSourceRange(SourceRange R) {
}

void StmtResInfer::dumpBareType(QualType T) {
}

void StmtResInfer::dumpType(QualType T) {
  dumpBareType(T);
}

void StmtResInfer::dumpBareDeclRef(const Decl *D) {
  {
    // C++AMP
    if(D->getKind() == Decl::Function){
      if(TheSema.getLangOpts().DevicePath && (CppAMPSpec & CPPAMP_AMP) &&
        !D->hasAttr<CXXAMPRestrictAMPAttr>())
        ClearResAMP();
      if(!TheSema.getLangOpts().DevicePath && (CppAMPSpec & CPPAMP_CPU) &&
        D->hasAttr<CXXAMPRestrictAMPAttr>() && !D->hasAttr<CXXAMPRestrictCPUAttr>())
        ClearResCPU();
      }
  }

  if (const ValueDecl *VD = dyn_cast<ValueDecl>(D))
    dumpType(VD->getType());
}

void StmtResInfer::dumpDeclRef(const Decl *D, const char *Label) {
  if (!D)
    return;

  dumpBareDeclRef(D);
}

void StmtResInfer::dumpName(const NamedDecl *ND) {
}

bool StmtResInfer::hasNodes(const DeclContext *DC) {
  if (!DC)
    return false;

  return DC->decls_begin() != DC->decls_end();
}

void StmtResInfer::dumpDeclContext(const DeclContext *DC) {
  if (!DC)
    return;
  for (DeclContext::decl_iterator I = DC->decls_begin(), E = DC->decls_end();
       I != E; ++I) {
    dumpDecl(*I);
  }
}

//===----------------------------------------------------------------------===//
//  Decl dumping methods.
//===----------------------------------------------------------------------===//

void StmtResInfer::dumpDecl(const Decl *D) {
  if (!D) {
    return;
  }

  // Decls within functions are visited by the body
  bool HasDeclContext = !isa<FunctionDecl>(*D) && !isa<ObjCMethodDecl>(*D) &&
                         hasNodes(dyn_cast<DeclContext>(D));

  ConstDeclVisitor<StmtResInfer>::Visit(D);
  if (HasDeclContext)
    dumpDeclContext(cast<DeclContext>(D));
}

void StmtResInfer::VisitLabelDecl(const LabelDecl *D) {
}

void StmtResInfer::VisitTypedefDecl(const TypedefDecl *D) {
  dumpType(D->getUnderlyingType());
}

void StmtResInfer::VisitEnumDecl(const EnumDecl *D) {
  if (D->isFixed())
    dumpType(D->getIntegerType());
}

void StmtResInfer::VisitRecordDecl(const RecordDecl *D) {
}

void StmtResInfer::VisitEnumConstantDecl(const EnumConstantDecl *D) {
  dumpType(D->getType());
  if (const Expr *Init = D->getInitExpr()) {
    dumpStmt(Init);
  }
}

void StmtResInfer::VisitIndirectFieldDecl(const IndirectFieldDecl *D) {
  dumpType(D->getType());
  for (IndirectFieldDecl::chain_iterator I = D->chain_begin(),
                                         E = D->chain_end();
       I != E; ++I) {
    if (I + 1 == E)
    dumpDeclRef(*I);
  }
}

void StmtResInfer::VisitFunctionDecl(const FunctionDecl *D) {
}

void StmtResInfer::VisitFieldDecl(const FieldDecl *D) {
  dumpName(D);
  dumpType(D->getType());
  // Resue in BuildMemInitializer for err_amp_unsupported_reference_or_pointer
  const Type* Ty  = D->getType().getTypePtrOrNull();
  QualType TheType = D->getType();

  if(Ty) {
    // Case by case
    if(Ty->isPointerType())
      TheType = Ty->getPointeeType();
    if(Ty->isArrayType())
      TheType = dyn_cast<ArrayType>(Ty)->getElementType();
    if(!TheType.isNull() && TheType->isRecordType()) {
      CXXRecordDecl* RDecl = TheType->getAsCXXRecordDecl();
        if (RDecl->getName() == "array")
          ClearResAMP();
    }
  }
  // Checke if it is array_view's reference or pointer
  if(Ty && (Ty->isPointerType() ||Ty->isReferenceType())) {
    const Type* TargetTy = Ty->getPointeeType().getTypePtrOrNull();
    if(const TemplateSpecializationType* TST = TargetTy->getAs<TemplateSpecializationType>()) {
      // Check if it is a TemplateSpecializationType
      // FIXME: should consider alias Template
      // Get its underlying template decl*
      if(ClassTemplateDecl* CTDecl = dyn_cast_or_null<ClassTemplateDecl>(
        TST->getTemplateName().getAsTemplateDecl())) {
        if(CXXRecordDecl* RDecl = CTDecl->getTemplatedDecl())
          if(RDecl->getName() == "array_view")
            ClearResAMP();
      }
    }
  }
  
  bool IsBitField = D->isBitField();
  Expr *Init = D->getInClassInitializer();
  bool HasInit = Init;

  if (IsBitField) {
    dumpStmt(D->getBitWidth());
  }
  if (HasInit) {
    dumpStmt(Init);
  }
}

void StmtResInfer::VisitVarDecl(const VarDecl *D) {
  if(TheSema.IsIncompatibleType(D->getType().getTypePtrOrNull(), false, true)) {
    ClearResAMP();
    return;
  }
  
  if(D->getType().isVolatileQualified())
    ClearResAMP();

  if(D->getType()->isCharType() || D->getType()->isWideCharType() || 
    D->getType()->isSpecificBuiltinType(BuiltinType::Short) || 
    D->getType()->isSpecificBuiltinType(BuiltinType::LongLong) || 
    D->getType()->isSpecificBuiltinType(BuiltinType::LongDouble))
   ClearResAMP();

  //var's type
  dumpType(D->getType());

  // TODO: Should infer if it is static
#if 0
  StorageClass SC = D->getStorageClass();
#endif

  if (D->hasInit()) {
    dumpStmt(D->getInit());
  }
}

void StmtResInfer::VisitFileScopeAsmDecl(const FileScopeAsmDecl *D) {
  dumpStmt(D->getAsmString());
}

void StmtResInfer::VisitImportDecl(const ImportDecl *D) {
}

//===----------------------------------------------------------------------===//
// C++ Declarations
//===----------------------------------------------------------------------===//

void StmtResInfer::VisitNamespaceDecl(const NamespaceDecl *D) {
  if (!D->isOriginalNamespace())
    dumpDeclRef(D->getOriginalNamespace(), "original");
}

void StmtResInfer::VisitUsingDirectiveDecl(const UsingDirectiveDecl *D) {
  dumpBareDeclRef(D->getNominatedNamespace());
}

void StmtResInfer::VisitNamespaceAliasDecl(const NamespaceAliasDecl *D) {
  dumpDeclRef(D->getAliasedNamespace());
}

void StmtResInfer::VisitTypeAliasDecl(const TypeAliasDecl *D) {
  dumpType(D->getUnderlyingType());
}

void StmtResInfer::VisitTypeAliasTemplateDecl(const TypeAliasTemplateDecl *D) {
  // TODO
  #if 0
  dumpTemplateParameters(D->getTemplateParameters());
  #endif
  dumpDecl(D->getTemplatedDecl());
}

void StmtResInfer::VisitCXXRecordDecl(const CXXRecordDecl *D) {
  VisitRecordDecl(D);
  if (!D->isCompleteDefinition())
    return;

  for (CXXRecordDecl::base_class_const_iterator I = D->bases_begin(),
                                                E = D->bases_end();
       I != E; ++I) {
    dumpType(I->getType());
  }
}

void StmtResInfer::VisitStaticAssertDecl(const StaticAssertDecl *D) {
  dumpStmt(D->getAssertExpr());
  dumpStmt(D->getMessage());
}

void StmtResInfer::VisitFunctionTemplateDecl(const FunctionTemplateDecl *D) {
}

void StmtResInfer::VisitClassTemplateDecl(const ClassTemplateDecl *D) {
}

void StmtResInfer::VisitClassTemplateSpecializationDecl(
    const ClassTemplateSpecializationDecl *D) {
  VisitCXXRecordDecl(D);
  // TODO
  #if 0
  dumpTemplateArgumentList(D->getTemplateArgs());
  #endif
}

void StmtResInfer::VisitClassTemplatePartialSpecializationDecl(
    const ClassTemplatePartialSpecializationDecl *D) {
  VisitClassTemplateSpecializationDecl(D);
  // TODO
  #if 0
  dumpTemplateParameters(D->getTemplateParameters());
  #endif
}

void StmtResInfer::VisitClassScopeFunctionSpecializationDecl(
    const ClassScopeFunctionSpecializationDecl *D) {
  dumpDeclRef(D->getSpecialization());
  // TODO
  #if 0
  if (D->hasExplicitTemplateArgs())
    dumpTemplateArgumentListInfo(D->templateArgs());
  #endif
}

void StmtResInfer::VisitTemplateTypeParmDecl(const TemplateTypeParmDecl *D) {
  if (D->hasDefaultArgument())
    dumpType(D->getDefaultArgument());
}

void StmtResInfer::VisitNonTypeTemplateParmDecl(const NonTypeTemplateParmDecl *D) {
  dumpType(D->getType());
  if (D->hasDefaultArgument())
    dumpStmt(D->getDefaultArgument());
}

void StmtResInfer::VisitTemplateTemplateParmDecl(
    const TemplateTemplateParmDecl *D) {
  // TODO
  #if 0
  dumpTemplateParameters(D->getTemplateParameters());
  if (D->hasDefaultArgument())
    dumpTemplateArgumentLoc(D->getDefaultArgument());
  #endif
}

void StmtResInfer::VisitUsingDecl(const UsingDecl *D) {
}

void StmtResInfer::VisitUnresolvedUsingTypenameDecl(
    const UnresolvedUsingTypenameDecl *D) {
}

void StmtResInfer::VisitUnresolvedUsingValueDecl(const UnresolvedUsingValueDecl *D) {
  dumpType(D->getType());
}

void StmtResInfer::VisitUsingShadowDecl(const UsingShadowDecl *D) {
  dumpBareDeclRef(D->getTargetDecl());
}

void StmtResInfer::VisitLinkageSpecDecl(const LinkageSpecDecl *D) {
}

void StmtResInfer::VisitAccessSpecDecl(const AccessSpecDecl *D) {
}

void StmtResInfer::VisitFriendDecl(const FriendDecl *D) {
  if (TypeSourceInfo *T = D->getFriendType())
    dumpType(T->getType());
  else
    dumpDecl(D->getFriendDecl());
}

//===----------------------------------------------------------------------===//
// Obj-C Declarations
//===----------------------------------------------------------------------===//

void StmtResInfer::VisitObjCIvarDecl(const ObjCIvarDecl *D) {
  dumpType(D->getType());
}

void StmtResInfer::VisitObjCMethodDecl(const ObjCMethodDecl *D) {
  dumpType(D->getReturnType());

  bool HasBody = D->hasBody();

  if (D->isThisDeclarationADefinition()) {
    dumpDeclContext(D);
  } else {
    for (ObjCMethodDecl::param_const_iterator I = D->param_begin(),
                                              E = D->param_end();
         I != E; ++I) {
      if (I + 1 == E)
      dumpDecl(*I);
    }
  }

  if (HasBody) {
    dumpStmt(D->getBody());
  }
}

void StmtResInfer::VisitObjCCategoryDecl(const ObjCCategoryDecl *D) {
  dumpDeclRef(D->getClassInterface());
  if (D->protocol_begin() == D->protocol_end())
  dumpDeclRef(D->getImplementation());
  for (ObjCCategoryDecl::protocol_iterator I = D->protocol_begin(),
                                           E = D->protocol_end();
       I != E; ++I) {
    if (I + 1 == E)
    dumpDeclRef(*I);
  }
}

void StmtResInfer::VisitObjCCategoryImplDecl(const ObjCCategoryImplDecl *D) {
  dumpDeclRef(D->getClassInterface());
  dumpDeclRef(D->getCategoryDecl());
}

void StmtResInfer::VisitObjCProtocolDecl(const ObjCProtocolDecl *D) {
  for (ObjCProtocolDecl::protocol_iterator I = D->protocol_begin(),
                                           E = D->protocol_end();
       I != E; ++I) {
    if (I + 1 == E)
    dumpDeclRef(*I);
  }
}

void StmtResInfer::VisitObjCInterfaceDecl(const ObjCInterfaceDecl *D) {
  dumpDeclRef(D->getSuperClass(), "super");
  if (D->protocol_begin() == D->protocol_end())
  dumpDeclRef(D->getImplementation());
  for (ObjCInterfaceDecl::protocol_iterator I = D->protocol_begin(),
                                            E = D->protocol_end();
       I != E; ++I) {
    if (I + 1 == E)
    dumpDeclRef(*I);
  }
}

void StmtResInfer::VisitObjCImplementationDecl(const ObjCImplementationDecl *D) {
  dumpDeclRef(D->getSuperClass(), "super");
  dumpDeclRef(D->getClassInterface());
  for (ObjCImplementationDecl::init_const_iterator I = D->init_begin(),
                                                   E = D->init_end();
       I != E; ++I) {
  }
}

void StmtResInfer::VisitObjCCompatibleAliasDecl(const ObjCCompatibleAliasDecl *D) {
  dumpDeclRef(D->getClassInterface());
}

void StmtResInfer::VisitObjCPropertyDecl(const ObjCPropertyDecl *D) {
  dumpType(D->getType());

  ObjCPropertyDecl::PropertyAttributeKind Attrs = D->getPropertyAttributes();
  if (Attrs != ObjCPropertyDecl::OBJC_PR_noattr) {
    if (Attrs & ObjCPropertyDecl::OBJC_PR_getter) {
      if (!(Attrs & ObjCPropertyDecl::OBJC_PR_setter))
      dumpDeclRef(D->getGetterMethodDecl(), "getter");
    }
    if (Attrs & ObjCPropertyDecl::OBJC_PR_setter) {
      dumpDeclRef(D->getSetterMethodDecl(), "setter");
    }
  }
}

void StmtResInfer::VisitObjCPropertyImplDecl(const ObjCPropertyImplDecl *D) {
  dumpDeclRef(D->getPropertyDecl());
  dumpDeclRef(D->getPropertyIvarDecl());
}

void StmtResInfer::VisitBlockDecl(const BlockDecl *D) {
  for (BlockDecl::param_const_iterator I = D->param_begin(), E = D->param_end();
       I != E; ++I)
    dumpDecl(*I);

  if (D->isVariadic()) {
  }

  if (D->capturesCXXThis()) {
  }
  for (BlockDecl::capture_const_iterator I = D->capture_begin(), E = D->capture_end();
       I != E; ++I) {
    if (I->getVariable()) {
      dumpBareDeclRef(I->getVariable());
    }
    if (I->hasCopyExpr())
      dumpStmt(I->getCopyExpr());
  }
  dumpStmt(D->getBody());
}

//===----------------------------------------------------------------------===//
//  Stmt dumping methods.
//===----------------------------------------------------------------------===//

void StmtResInfer::dumpStmt(const Stmt *S) {
  if (!S) {
    return;
  }

  if (const DeclStmt *DS = dyn_cast<DeclStmt>(S)) {
    VisitDeclStmt(DS);
    return;
  }

  ConstStmtVisitor<StmtResInfer>::Visit(S);
  for (Stmt::const_child_iterator CI = S->child_begin(); CI != S->child_end(); ++CI) {
    dumpStmt(*CI);
  }
}

// Perform the inferring
unsigned StmtResInfer::Infer(const Stmt* Node) {
  dumpStmt(Node);
  return CppAMPSpec;
}
void StmtResInfer::VisitStmt(const Stmt *Node) {
}

void StmtResInfer::VisitDeclStmt(const DeclStmt *Node) {
  VisitStmt(Node);
  for (DeclStmt::const_decl_iterator I = Node->decl_begin(),
                                     E = Node->decl_end();
       I != E; ++I) {
    dumpDecl(*I);
  }
}

void StmtResInfer::VisitAttributedStmt(const AttributedStmt *Node) {
  VisitStmt(Node);
}

void StmtResInfer::VisitLabelStmt(const LabelStmt *Node) {
  VisitStmt(Node);

  // label statement is not valid in C++AMP
  // but is valid in HSA extension mode
  if(!TheSema.getLangOpts().HSAExtension) {
    ClearResAMP();
  }
}

void StmtResInfer::VisitGotoStmt(const GotoStmt *Node) {
  VisitStmt(Node);

  // goto statement is not valid in C++AMP
  // but is valid in HSA extension mode
  if(!TheSema.getLangOpts().HSAExtension) {
    ClearResAMP();
  }
}
void StmtResInfer::VisitCXXTryStmt(const CXXTryStmt* Node) {
  VisitStmt(Node);
  ClearResAMP();
}
void StmtResInfer::VisitCXXTypeidExpr(const CXXTypeidExpr* Node) {
  VisitStmt(Node);
  ClearResAMP();
}
void StmtResInfer::VisitCXXDynamicCastExpr(const CXXDynamicCastExpr* Node) {
  VisitStmt(Node);
  ClearResAMP();
}


//===----------------------------------------------------------------------===//
//  Expr dumping methods.
//===----------------------------------------------------------------------===//

void StmtResInfer::VisitExpr(const Expr *Node) {
  VisitStmt(Node);
  dumpType(Node->getType());
}

void StmtResInfer::VisitCastExpr(const CastExpr *Node) {
  VisitExpr(Node);
  //TODO: infer if any
}

void StmtResInfer::VisitDeclRefExpr(const DeclRefExpr *Node) {
  //Format: DeclRefExpr 0x3eca4e8 <col:10> 'int (void)' lvalue
  VisitExpr(Node);
  dumpBareDeclRef(Node->getDecl());
  if (Node->getDecl() != Node->getFoundDecl()) {
    dumpBareDeclRef(Node->getFoundDecl());
  }
}

void StmtResInfer::VisitUnresolvedLookupExpr(const UnresolvedLookupExpr *Node) {
  VisitExpr(Node);
}

void StmtResInfer::VisitObjCIvarRefExpr(const ObjCIvarRefExpr *Node) {
  VisitExpr(Node);
}

void StmtResInfer::VisitPredefinedExpr(const PredefinedExpr *Node) {
  VisitExpr(Node);
}

void StmtResInfer::VisitCharacterLiteral(const CharacterLiteral *Node) {
  VisitExpr(Node);
}

void StmtResInfer::VisitIntegerLiteral(const IntegerLiteral *Node) {
  VisitExpr(Node);
}

void StmtResInfer::VisitFloatingLiteral(const FloatingLiteral *Node) {
  VisitExpr(Node);
}

void StmtResInfer::VisitStringLiteral(const StringLiteral *Str) {
  VisitExpr(Str);
}

void StmtResInfer::VisitUnaryOperator(const UnaryOperator *Node) {
  VisitExpr(Node);
}

void StmtResInfer::VisitUnaryExprOrTypeTraitExpr(
    const UnaryExprOrTypeTraitExpr *Node) {
  VisitExpr(Node);
  if (Node->isArgumentType())
    dumpType(Node->getArgumentType());
}

void StmtResInfer::VisitMemberExpr(const MemberExpr *Node) {
  VisitExpr(Node);
  ValueDecl* VD = Node->getMemberDecl();
  if((CppAMPSpec & CPPAMP_AMP) && !VD->hasAttr<CXXAMPRestrictAMPAttr>())
    ClearResAMP();
  if((CppAMPSpec & CPPAMP_CPU) && VD->hasAttr<CXXAMPRestrictAMPAttr>() &&
    !VD->hasAttr<CXXAMPRestrictCPUAttr>())
    ClearResCPU();;
}

void StmtResInfer::VisitExtVectorElementExpr(const ExtVectorElementExpr *Node) {
  VisitExpr(Node);
}

void StmtResInfer::VisitBinaryOperator(const BinaryOperator *Node) {
  VisitExpr(Node);
}

void StmtResInfer::VisitCompoundAssignOperator(
    const CompoundAssignOperator *Node) {
  VisitExpr(Node);
  dumpBareType(Node->getComputationLHSType());
  dumpBareType(Node->getComputationResultType());
}

void StmtResInfer::VisitBlockExpr(const BlockExpr *Node) {
  VisitExpr(Node);
  dumpDecl(Node->getBlockDecl());
}

void StmtResInfer::VisitOpaqueValueExpr(const OpaqueValueExpr *Node) {
  VisitExpr(Node);
  if (Expr *Source = Node->getSourceExpr()) {
    dumpStmt(Source);
  }
}

// GNU extensions.

void StmtResInfer::VisitAddrLabelExpr(const AddrLabelExpr *Node) {
  VisitExpr(Node);
}

//===----------------------------------------------------------------------===//
// C++ Expressions
//===----------------------------------------------------------------------===//

void StmtResInfer::VisitCXXNamedCastExpr(const CXXNamedCastExpr *Node) {
  VisitExpr(Node);
}

void StmtResInfer::VisitCXXBoolLiteralExpr(const CXXBoolLiteralExpr *Node) {
  VisitExpr(Node);
}

void StmtResInfer::VisitCXXThisExpr(const CXXThisExpr *Node) {
  VisitExpr(Node);
}

void StmtResInfer::VisitCXXThrowExpr(const CXXThrowExpr *Node) {
  VisitExpr(Node);
  ClearResAMP();
}

void StmtResInfer::VisitCXXFunctionalCastExpr(const CXXFunctionalCastExpr *Node) {
  VisitExpr(Node);
}

void StmtResInfer::VisitCXXConstructExpr(const CXXConstructExpr *Node) {
  VisitExpr(Node);
  // TODO: infer if any
#if 0
  CXXConstructorDecl *Ctor = Node->getConstructor();
#endif
}

void StmtResInfer::VisitCXXBindTemporaryExpr(const CXXBindTemporaryExpr *Node) {
  VisitExpr(Node);
  dumpCXXTemporary(Node->getTemporary());
}

void StmtResInfer::VisitExprWithCleanups(const ExprWithCleanups *Node) {
  VisitExpr(Node);
  for (unsigned i = 0, e = Node->getNumObjects(); i != e; ++i)
    dumpDeclRef(Node->getObject(i), "cleanup");
}

void StmtResInfer::dumpCXXTemporary(const CXXTemporary *Temporary) {
}

//===----------------------------------------------------------------------===//
// Obj-C Expressions
//===----------------------------------------------------------------------===//

void StmtResInfer::VisitObjCMessageExpr(const ObjCMessageExpr *Node) {
  VisitExpr(Node);
}

void StmtResInfer::VisitObjCBoxedExpr(const ObjCBoxedExpr *Node) {
  VisitExpr(Node);
}
void StmtResInfer::VisitObjCAtCatchStmt(const ObjCAtCatchStmt *Node) {
  VisitStmt(Node);
}

void StmtResInfer::VisitObjCEncodeExpr(const ObjCEncodeExpr *Node) {
  VisitExpr(Node);
}

void StmtResInfer::VisitObjCSelectorExpr(const ObjCSelectorExpr *Node) {
  VisitExpr(Node);
}

void StmtResInfer::VisitObjCProtocolExpr(const ObjCProtocolExpr *Node) {
  VisitExpr(Node);
}

void StmtResInfer::VisitObjCPropertyRefExpr(const ObjCPropertyRefExpr *Node) {
  VisitExpr(Node);
}

void StmtResInfer::VisitObjCSubscriptRefExpr(const ObjCSubscriptRefExpr *Node) {
  VisitExpr(Node);
}

void StmtResInfer::VisitObjCBoolLiteralExpr(const ObjCBoolLiteralExpr *Node) {
  VisitExpr(Node);
}

static void InferFunctionType(FunctionDecl* FD, unsigned& Spec) {
  //check function return type
  {
    QualType ResultType = FD->getReturnType();
    if(ResultType->isPointerType())
      ResultType = ResultType->getPointeeType();
    // double* is allowed, while double** is not allowed
    if(ResultType->isPointerType()) {
      Spec &=~CPPAMP_AMP;
      return;
    }
  }
  // check if there's incompatible parameters in the function declarator
  for (FunctionDecl::param_iterator PIt = FD->param_begin();
    PIt != FD->param_end(); ++PIt) {
    ParmVarDecl *pvDecl = (*PIt);
    if(!pvDecl)
      continue;

    QualType Ty = pvDecl->getOriginalType();
    if (Ty->isCharType() || Ty->isWideCharType() || Ty->isSpecificBuiltinType(BuiltinType::Short) || 
      Ty->isSpecificBuiltinType(BuiltinType::LongLong) ||
      Ty->isSpecificBuiltinType(BuiltinType::LongDouble) || Ty.isVolatileQualified()) {
      Spec &=~CPPAMP_AMP;
      return;
    }

    if (Ty->isEnumeralType()) {
      const EnumType* ETy = dyn_cast<EnumType>(Ty);
      if (ETy && ETy->getDecl()) {
        const Type* UTy = ETy->getDecl()->getIntegerType().getTypePtrOrNull();
        if (UTy->isCharType() || UTy->isWideCharType() || 
          UTy->isSpecificBuiltinType(BuiltinType::Short) || 
          UTy->isSpecificBuiltinType(BuiltinType::LongLong) || 
          UTy->isSpecificBuiltinType(BuiltinType::LongDouble)) {
           Spec &=~CPPAMP_AMP;
           return;
        }
      }
    }

   // Pointer's pointer
   QualType TheType = Ty;
    if(Ty->isPointerType())
      TheType = Ty->getPointeeType();
    // double* is allowed, while double** is not allowed
    if(TheType->isPointerType()) {
      Spec &=~CPPAMP_AMP;
      return;
    }
  }

  QualType ResultType = FD->getReturnType();
  // check if the return type is of incompatible type
  if (ResultType->isCharType() || ResultType->isSpecificBuiltinType(BuiltinType::Short)) {
    Spec &=~CPPAMP_AMP;
    return;
  }

  if(FD->getType().isVolatileQualified())
     Spec &=~CPPAMP_AMP;

    return;
}

// FIXME: Once all statements of the declaration are passed, the restricitons
// inferring can be performed. This is only allowed in auto-restricted declaration
// Top down
void Sema::TryCXXAMPRestrictionInferring(Decl *dcl, Stmt *S) {
  if (!getLangOpts().CPlusPlusAMP ||!dcl ||!dcl->hasAttr<CXXAMPRestrictAUTOAttr>())
    return;
  
  // Only allow on funtion definition
  assert(isa<FunctionDecl>(*dcl) && dcl->hasBody());

  unsigned OtherSpec = CPPAMP_AMP | CPPAMP_CPU;
  if(dcl->hasAttr<CXXAMPRestrictAMPAttr>())
    OtherSpec &= ~CPPAMP_AMP;
  if(dcl->hasAttr<CXXAMPRestrictCPUAttr>())
    OtherSpec &= ~CPPAMP_CPU;

  // Inferring process
  // skip method in a lambda class (ex: kernel function in parallel_for_each)
  if (isa<CXXMethodDecl>(dcl) && dyn_cast<CXXMethodDecl>(dcl)->getParent()->isLambda()) {
  } else if(OtherSpec & CPPAMP_AMP) {
    // Assuming that 'auto' has been already inferred in parent scope if any
    // Contained in any CPU only caller?
    if(!IsInAMPRestricted() && dcl->getParentFunctionOrMethod())
      OtherSpec &= ~CPPAMP_AMP;
    else if(FunctionDecl* FD = dyn_cast<FunctionDecl>(dcl))
     InferFunctionType(FD, OtherSpec);
  }
  
  if(OtherSpec) {
     StmtResInfer SRI(*this, OtherSpec, &this->getSourceManager());
     OtherSpec = SRI.Infer(S);
    }

  // Update non-auto restriction specifiers if any
  if(OtherSpec) {
    
    //Place all manually created Attr in where 'auto' physically is
    CXXAMPRestrictAUTOAttr *AUTOAttr = dcl->getAttr<CXXAMPRestrictAUTOAttr>();
    assert(AUTOAttr);
    if(OtherSpec & CPPAMP_AMP)
      dcl->addAttr(::new (Context) CXXAMPRestrictAMPAttr(AUTOAttr->getRange(), Context, 0));
    if(OtherSpec & CPPAMP_CPU)
      dcl->addAttr(::new (Context) CXXAMPRestrictCPUAttr(AUTOAttr->getRange(), Context, 0));
  }
  
  // The inferring process is done. Drop AUTO Attribute in this compilation path
  dcl->dropAttr<CXXAMPRestrictAUTOAttr>();

}

