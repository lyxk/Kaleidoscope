#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace llvm;

// Unknown tokens are represented by their ASCII code
enum Token {
    tok_eof = -1,
    tok_def = -2,
    tok_extern = -3,
    tok_identifier = -4,
    tok_number = -5
};

static std::string IdentifierStr;
static double NumVal;

static int gettok() {
    static int LastChar = ' ';

    // Skip spaces
    while (isspace(LastChar)) {
        LastChar = getchar();
    }

    // Get identifier
    if (isalpha(LastChar)) {
        IdentifierStr = LastChar;

        while (isalnum((LastChar = getchar())))
            IdentifierStr += LastChar;
        
        if (IdentifierStr == "def")
            return tok_def;
        
        if (IdentifierStr == "extern")
            return tok_extern;
        
        return tok_identifier;
    }
    
    // Get number
    if (isdigit(LastChar) || LastChar == '.') {
        std::string NumStr;

        do {
            NumStr += LastChar;
            LastChar = getchar();
        } while (isdigit(LastChar) || LastChar == '.');

        NumVal = stoi(NumStr);

        return tok_number;
    }

    // Skip comments
    if (LastChar == '#') {
        do {
            LastChar = getchar();
        } while (LastChar != EOF && LastChar != '\n' && LastChar != 'r');

        if (LastChar != EOF)
            return gettok();
    }

    // Get EOF
    if (LastChar == EOF)
        return tok_eof;

    // Keep moving if LastChar is not processed by any of the above cases
    int ThisChar = LastChar;
    LastChar = getchar();

    return ThisChar;
}

/*
int main() {
    int tok;

    do {
        tok = gettok();
        std::cout << "Got token: " << tok << std::endl;
    } while (tok != EOF);

    return 0;
}
*/

class ExprAST {
public:
    virtual ~ExprAST() {}
    virtual Value *codegen() = 0;
};

class NumberExprAST : public ExprAST {
private:
    double Val;
public:
    NumberExprAST(double Val) : Val(Val) {}
    virtual Value* codegen(); 
};

class VariableExprAST : public ExprAST {
private:
    std::string Name;
public:
    VariableExprAST(const std::string& Name) : Name(Name) {}
    virtual Value* codegen(); 
};

class BinaryExprAST : public ExprAST {
private:
    char Op;
    std::unique_ptr<ExprAST> LHS, RHS;
public:
    BinaryExprAST(
        char Op,
        std::unique_ptr<ExprAST> LHS,
        std::unique_ptr<ExprAST> RHS
    ) : Op(Op), LHS(std::move(LHS)), RHS(std::move(RHS)) {}
    virtual Value* codegen(); 
};

class CallExprAST : public ExprAST {
private:
    std::string Callee;
    std::vector<std::unique_ptr<ExprAST>> Args;
public:
    CallExprAST(
        const std::string& Callee, 
        std::vector<std::unique_ptr<ExprAST>> Args
    ) : Callee(Callee), Args(std::move(Args)) {}
    virtual Value* codegen(); 
};

class PrototypeAST {
private:
    std::string Name;
    std::vector<std::string> Args;
public: 
    PrototypeAST(
        const std::string& Name, 
        std::vector<std::string> Args
    ) : Name(Name), Args(Args) {}
    virtual Function* codegen(); 
    const std::string &getName() const { return Name; }
};

class FunctionAST {
private:
    std::unique_ptr<PrototypeAST> Proto;
    std::unique_ptr<ExprAST> Body;
public:
    FunctionAST(
        std::unique_ptr<PrototypeAST> Proto,
        std::unique_ptr<ExprAST> Body
    ) : Proto(std::move(Proto)), Body(std::move(Body)) {}
    virtual Function* codegen(); 
};

// Get the token returned by the gettok() function
static int CurTok;

static int getNextToken() {
    return CurTok = gettok();
}

std::unique_ptr<ExprAST> LogError(const char* Str) {
    fprintf(stderr, "LogError: %s\n", Str);
    return nullptr;
}

std::unique_ptr<PrototypeAST> LogErrorP(const char* Str) {
    LogError(Str);
    return nullptr;
}

static std::unique_ptr<ExprAST> ParseExpression();

static std::unique_ptr<ExprAST> ParseNumberExpr() {
    auto Result = std::make_unique<NumberExprAST>(NumVal);
    getNextToken();
    return std::move(Result);
}

static std::unique_ptr<ExprAST> ParseParenExpr() {
    getNextToken();
    auto V = ParseExpression();
    if (!V)
        return nullptr;
    
    if (CurTok != ')')
        return LogError("expected ')");
    
    getNextToken();

    return V;
}

static std::unique_ptr<ExprAST> ParseIdentifierOrCallExpr() {
    std::string IdName = IdentifierStr;

    getNextToken();

    if (CurTok != '(')
        return std::make_unique<VariableExprAST>(IdName);

    getNextToken();
    std::vector<std::unique_ptr<ExprAST>> Args;

    if (CurTok != ')') {
        while (true) {
            auto Arg = ParseExpression();

            if (Arg) Args.push_back(std::move(Arg));
            else return nullptr;

            if (CurTok == ')') {
                getNextToken();
                break;
            }

            if (CurTok == ',') {
                getNextToken();
                continue;
            }
            else
                return LogError("Expected ')' or ',' in ");
        }
        return std::make_unique<CallExprAST>(IdName, std::move(Args));
    } 

    return std::make_unique<VariableExprAST>(IdName);
}

static std::unique_ptr<ExprAST> ParsePrimary() {
    switch (CurTok) {
        case tok_identifier:
            return ParseIdentifierOrCallExpr();
        case tok_number:
            return ParseNumberExpr();
        case '(':
            return ParseParenExpr();
        default:
            return LogError("unknown token when expecting an expression");
    }
}

static std::map<char, int> BinopPrecedence;

static int GetTokPrecedence() {
    // The lowest precedence
    if (!isascii(CurTok))
        return -1;
    
    int TokPrec = BinopPrecedence[CurTok];
    if (TokPrec <= 0) return -1;

    return TokPrec;
}

static std::unique_ptr<ExprAST> ParseBinOpRHS(int ExprPrec, std::unique_ptr<ExprAST> LHS) {
    while (true) {
        int TokPrec = GetTokPrecedence();

        if (TokPrec < ExprPrec)
            return LHS;
        
        int BinOp = CurTok;
        getNextToken();

        // Parse the primary expression after the binary operator
        auto RHS = ParsePrimary();
        if (!RHS)
            return nullptr;

        // If BinOp binds less tightly with RHS than the operator after RHS, let
        // the pending operator take RHS as its LHS
        int NextPrec = GetTokPrecedence();
        if (TokPrec < NextPrec) {
            RHS = ParseBinOpRHS(TokPrec+1, std::move(RHS));
            if (!RHS)
                return nullptr;
        }

        // Merge LHS/RHS
        LHS = std::make_unique<BinaryExprAST>(BinOp, std::move(LHS), std::move(RHS));
    }
}

static std::unique_ptr<ExprAST> ParseExpression() {
    auto LHS = ParsePrimary();
    if (!LHS) {
        return nullptr;
    }
    return ParseBinOpRHS(0, std::move(LHS));
}

static std::unique_ptr<PrototypeAST> ParsePrototype() {
    if (CurTok != tok_identifier)
        return LogErrorP("Expected function name in prototype");

    std::string FcnName = IdentifierStr;
    getNextToken();

    if (CurTok != '(')
        return LogErrorP("Expected '(' in prototype");
    
    std::vector<std::string> ArgNames;
    while (getNextToken() == tok_identifier)
        ArgNames.push_back(IdentifierStr);
    
    if (CurTok != ')')
        return LogErrorP("Expected ')' in prototype");
    
    getNextToken();

    return std::make_unique<PrototypeAST>(FcnName, std::move(ArgNames));
}

static std::unique_ptr<FunctionAST> ParseDefinition() {
    getNextToken();
    auto Proto = ParsePrototype();
    if (!Proto) {
        return nullptr;
    }
    
    auto E = ParseExpression();
    if (E)
        return std::make_unique<FunctionAST>(std::move(Proto), std::move(E));
    else
        return nullptr;
}

static std::unique_ptr<PrototypeAST> ParseExtern() {
    getNextToken();
    return ParsePrototype();
}

static std::unique_ptr<FunctionAST> ParseTopLevelExpr() {
    if (auto E = ParseExpression()) {
        // Make an anoynomous proto
        auto Proto = std::make_unique<PrototypeAST>("", std::vector<std::string>());
        return std::make_unique<FunctionAST>(std::move(Proto), std::move(E));
    }
    return nullptr;
}

//===----------------------------------------------------------------------===//
// Code Generation
//===----------------------------------------------------------------------===//

static std::unique_ptr<LLVMContext> TheContext;
static std::unique_ptr<Module> TheModule;
static std::unique_ptr<IRBuilder<>> Builder;
static std::map<std::string, Value *> NamedValues;
static std::unique_ptr<legacy::FunctionPassManager> TheFPM;

Value *LogErrorV(const char *Str) {
  LogError(Str);
  return nullptr;
}

Value *NumberExprAST::codegen() {
    return ConstantFP::get(*TheContext, APFloat(Val));
}

Value *VariableExprAST::codegen() {
    // Look up the variable name
    Value *V = NamedValues[Name];
    if (!V)
        return LogErrorV("Unknown varibale name");
    return V;
}

Value *BinaryExprAST::codegen() {
    Value *L = LHS->codegen();
    Value *R = RHS->codegen();

    if (!L || !R)
        return nullptr;
    
    switch (Op) {
        case '+': return Builder->CreateFAdd(L, R, "addtmp");
        case '-': return Builder->CreateFSub(L, R, "subtmp");
        case '*': return Builder->CreateFMul(L, R, "multmp");
        case '<': 
            L = Builder->CreateFCmpULT(L, R, "cmptmp");
            // Convert bool 0/1  to double 0.0 or 1.0
            return Builder->CreateUIToFP(L, Type::getDoubleTy(*TheContext), "booltmp");
        default: return LogErrorV("invalid binary operator");
    }
}

Value *CallExprAST::codegen() {
    // Look up the function name in the global module table
    Function *CalleeF = TheModule->getFunction(Callee);
    if (!CalleeF)
        return LogErrorV("Unknown function referenced");

    // If argument size mismatch
    if (CalleeF->arg_size() != Args.size())
        return LogErrorV("Incorrect # arguments passed");

    std::vector<Value*> ArgsV;
    for (size_t i = 0; i < Args.size(); i++) {
        auto ArgV = Args[i]->codegen();
        if (!ArgV)
            return nullptr;
        ArgsV.push_back(ArgV);
    }

    return Builder->CreateCall(CalleeF, ArgsV, "calltmp");
}

Function *PrototypeAST::codegen() {
    std::vector<Type*> Doubles(Args.size(), Type::getDoubleTy(*TheContext));

    FunctionType *FT = FunctionType::get(Type::getDoubleTy(*TheContext), Doubles, false);

    Function *F = Function::Create(FT, Function::ExternalLinkage, Name, TheModule.get());

    unsigned Idx = 0;
    for (auto& Arg : F->args())
        Arg.setName(Args[Idx++]);

    return F;
}

Function *FunctionAST::codegen() {
    Function *TheFunction = TheModule->getFunction(Proto->getName());

    if (!TheFunction)
        TheFunction = Proto->codegen();
    
    if (!TheFunction)
        return nullptr;
    
    if (!TheFunction->empty())
        return (Function*)LogErrorV("Funciton cannot be redefined");

    // Create a new basic block to start insertion into.
    BasicBlock *BB = BasicBlock::Create(*TheContext, "entry", TheFunction);
    Builder->SetInsertPoint(BB);

    // Record the function arguments in the NamedValues map.
    NamedValues.clear();
    for (auto &Arg : TheFunction->args())
        NamedValues[Arg.getName()] = &Arg;
    
    auto RetVal = Body->codegen();
    if (RetVal) {
        Builder->CreateRet(RetVal);

        verifyFunction(*TheFunction);

        // Optimize the function.
        TheFPM->run(*TheFunction);

        return TheFunction;
    }
    
    TheFunction->eraseFromParent();
    return nullptr;
} 

//===----------------------------------------------------------------------===//
// Top-Level parsing and JIT Driver
//===----------------------------------------------------------------------===//

void InitializeModuleAndPassManager(void) {
  // Open a new context and module.
  TheContext = std::make_unique<LLVMContext>();
  TheModule = std::make_unique<Module>("my cool jit", *TheContext);

  // Create a new builder for the module.
  Builder = std::make_unique<IRBuilder<>>(*TheContext);

  // Create a new pass manager attached to it.
  TheFPM = std::make_unique<legacy::FunctionPassManager>(TheModule.get());

  // Do simple "peephole" optimizations and bit-twiddling optzns.
  TheFPM->add(createInstructionCombiningPass());
  // Reassociate expressions.
  TheFPM->add(createReassociatePass());
  // Eliminate Common SubExpressions.
  TheFPM->add(createGVNPass());
  // Simplify the control flow graph (deleting unreachable blocks, etc).
  TheFPM->add(createCFGSimplificationPass());

  TheFPM->doInitialization();
}

static void HandleDefinition() {
  if (auto FnAST = ParseDefinition()) {
    if (auto *FnIR = FnAST->codegen()) {
      fprintf(stderr, "Read function definition:");
      FnIR->print(errs());
      fprintf(stderr, "\n");
    }
  } else {
    // Skip token for error recovery.
    getNextToken();
  }
}

static void HandleExtern() {
  if (auto ProtoAST = ParseExtern()) {
    if (auto *FnIR = ProtoAST->codegen()) {
      fprintf(stderr, "Read extern: ");
      FnIR->print(errs());
      fprintf(stderr, "\n");
    }
  } else {
    // Skip token for error recovery.
    getNextToken();
  }
}

static void HandleTopLevelExpression() {
  // Evaluate a top-level expression into an anonymous function.
  if (auto FnAST = ParseTopLevelExpr()) {
    if (auto *FnIR = FnAST->codegen()) {
      fprintf(stderr, "Read top-level expression:");
      FnIR->print(errs());
      fprintf(stderr, "\n");

      // Remove the anonymous expression.
      FnIR->eraseFromParent();
    }
  } else {
    // Skip token for error recovery.
    getNextToken();
  }
}

/// top ::= definition | external | expression | ';'
static void MainLoop() {
  while (true) {
    fprintf(stderr, "ready> ");
    switch (CurTok) {
    case tok_eof:
      return;
    case ';': // ignore top-level semicolons.
      getNextToken();
      break;
    case tok_def:
      HandleDefinition();
      break;
    case tok_extern:
      HandleExtern();
      break;
    default:
      HandleTopLevelExpression();
      break;
    }
  }
}

//===----------------------------------------------------------------------===//
// Main driver code.
//===----------------------------------------------------------------------===//

int main() {
// Install standard binary operators.
  // 1 is lowest precedence.
  BinopPrecedence['<'] = 10;
  BinopPrecedence['+'] = 20;
  BinopPrecedence['-'] = 20;
  BinopPrecedence['*'] = 40; // highest.

  // Prime the first token.
  fprintf(stderr, "ready> ");
  getNextToken();

  // Make the module, which holds all the code.
  InitializeModuleAndPassManager();

  // Run the main "interpreter loop" now.
  MainLoop();

  // Print out all of the generated code.
  TheModule->print(errs(), nullptr);

  return 0;
}




