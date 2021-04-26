#include "KaleidoscopeJIT.h"
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
using namespace llvm::orc;

// Unknown tokens are represented by their ASCII code
enum Token {
    tok_eof = -1,
    tok_def = -2,
    tok_extern = -3,
    tok_identifier = -4,
    tok_number = -5,
    tok_if = -6,
    tok_then = -7,
    tok_else = -8,
    tok_for = -9,
    tok_in = -10,
    tok_binary = -11,
    tok_unary = -12
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
        
        if (IdentifierStr == "if")
            return tok_if;

        if (IdentifierStr == "then")
            return tok_then;
        
        if (IdentifierStr == "else")
            return tok_else;
        
        if (IdentifierStr == "for")
            return tok_for;

        if (IdentifierStr == "in")
            return tok_in;
        
        if (IdentifierStr == "binary")
            return tok_binary;

        if (IdentifierStr == "unary")
            return tok_unary;

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
    Value* codegen() override; 
};

class VariableExprAST : public ExprAST {
private:
    std::string Name;
public:
    VariableExprAST(const std::string& Name) : Name(Name) {}
    Value* codegen() override; 
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
    Value* codegen() override; 
};

class UnaryExprAST : public ExprAST {
private:
    char Op;
    std::unique_ptr<ExprAST> Operand;
public:
    UnaryExprAST(
        char Op, 
        std::unique_ptr<ExprAST> Operand
    ) : Op(Op), Operand(std::move(Operand)) {}
    Value *codegen() override;
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
    Value* codegen() override; 
};

class IfExprAST : public ExprAST {
private:
    std::unique_ptr<ExprAST> Cond, Then, Else;
public:
    IfExprAST(
        std::unique_ptr<ExprAST> Cond,
        std::unique_ptr<ExprAST> Then,
        std::unique_ptr<ExprAST> Else
    ) : Cond(std::move(Cond)), Then(std::move(Then)), Else(std::move(Else)) {}
    Value* codegen() override;
};

class ForExprAST : public ExprAST {
private:
    std::string VarName;
    std::unique_ptr<ExprAST> Start, End, Step, Body;
public:
    ForExprAST(
        const std::string &VarName, 
        std::unique_ptr<ExprAST> Start, 
        std::unique_ptr<ExprAST> End,
        std::unique_ptr<ExprAST> Step,
        std::unique_ptr<ExprAST> Body
    ) : VarName(VarName), Start(std::move(Start)), End(std::move(End)), Step(std::move(Step)), Body(std::move(Body)) {}
    Value *codegen() override;
};

class PrototypeAST {
private:
    std::string Name;
    std::vector<std::string> Args;
    bool IsOperator;
    unsigned Precendence;
public:
  PrototypeAST(
    const std::string &Name, 
    std::vector<std::string> Args, 
    bool IsOperator = false, 
    unsigned Precendence = 0
   ) : Name(Name), Args(Args), IsOperator(IsOperator), Precendence(Precendence) {}

  Function *codegen();
  const std::string &getName() const { return Name; }

  bool isUnaryOp() const { return IsOperator && Args.size() == 1; }
  bool isBinaryOp() const { return IsOperator && Args.size() == 2; }

  char getOperatororName() const {
    assert(isUnaryOp() || isBinaryOp());
    return Name[Name.size() - 1];
    }

    unsigned getBinaryPrecendence() const { return Precendence; }
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
    Function* codegen(); 
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

static std::unique_ptr<ExprAST> ParseIfExpr() {
    getNextToken(); // eat the if

    auto Cond = ParseExpression();
    if (!Cond) return nullptr;

    if (CurTok != tok_then)
        return LogError("expected then");
    getNextToken(); // eat the then

    auto Then = ParseExpression();
    if (!Then) return nullptr;

    if (CurTok != tok_else)
        return LogError("expected else");
    getNextToken(); // eat the else

    auto Else = ParseExpression();
    if (!Else)
        return nullptr;

    return std::make_unique<IfExprAST>(std::move(Cond), std::move(Then), std::move(Else));
}

static std::unique_ptr<ExprAST> ParseForExpr() {
    getNextToken(); // eat the for

    if (CurTok != tok_identifier)
        return LogError("expected identifier after for");

    std::string IdName = IdentifierStr;
    getNextToken(); // eat the identifier

    if (CurTok != '=')
        return LogError("expected '=' after for start value");
    getNextToken();

    auto Start = ParseExpression();
    if (!Start)
        return nullptr;
    if (CurTok != ',')
        return LogError("expected ',' after start value");
    getNextToken();

    auto End = ParseExpression();
    if (!End)
        return nullptr;
    
    // Step value is optional
    std::unique_ptr<ExprAST> Step;
    if (CurTok == ',') {
        getNextToken();
        Step = ParseExpression();
        if (!Step)
            return nullptr;
    }

    if (CurTok != tok_in)
        return LogError("expected 'in' after for");
    getNextToken();

    auto Body = ParseExpression();
    if (!Body)
        return nullptr;

    return std::make_unique<ForExprAST>(
        IdName, 
        std::move(Start), 
        std::move(End), 
        std::move(Step),
        std::move(Body)
    );
}

static std::unique_ptr<ExprAST> ParsePrimary() {
    switch (CurTok) {
        case tok_identifier:
            return ParseIdentifierOrCallExpr();
        case tok_number:
            return ParseNumberExpr();
        case tok_if:
            return ParseIfExpr();
        case tok_for:
            return ParseForExpr();
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


static std::unique_ptr<ExprAST> ParseUnary() {
    // If the current token is not an operator, parse it as an primary expression
    if (!isascii(CurTok) || CurTok == '(' || CurTok == ',') 
        return ParsePrimary();
    
    int Op = CurTok;
    // Eat the unary operator
    getNextToken();
    if (auto Operand = ParseUnary())
        return std::make_unique<UnaryExprAST>(Op, std::move(Operand));
    
    return nullptr;
}

// Shaunting yard algorithm for operator precedence
static std::unique_ptr<ExprAST> ParseBinOpRHS(int ExprPrec, std::unique_ptr<ExprAST> LHS) {
    while (true) {
        int TokPrec = GetTokPrecedence();

        if (TokPrec < ExprPrec)
            return LHS;
        
        int BinOp = CurTok;
        getNextToken();

        // Parse the primary expression after the binary operator
        auto RHS = ParseUnary();
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
    auto LHS = ParseUnary();
    if (!LHS) {
        return nullptr;
    }
    return ParseBinOpRHS(0, std::move(LHS));
}

static std::unique_ptr<PrototypeAST> ParsePrototype() {
    std::string FnName;

    unsigned Kind = 0, BinaryPrecedence = 30;

    switch (CurTok) {
        case tok_identifier:
            FnName = IdentifierStr;
            Kind = 0;
            getNextToken();
            break;
        
        case tok_unary:
            getNextToken();
            if (!isascii(CurTok))
                return LogErrorP("Expected unary operator");
            FnName = std::string("unary") + (char)CurTok;
            Kind = 1;
            getNextToken();
            break;

        case tok_binary:
            getNextToken();
            if (!isascii(CurTok))
                return LogErrorP("Expected binary operator");
            FnName = std::string("binary") + (char)CurTok;
            Kind = 2;
            getNextToken();

            // Read the precedence if necessary
            if (CurTok == tok_number) {
                if (NumVal < 1 || NumVal > 100)
                    return LogErrorP("Invalid precedence: must be between 1 and 100");
                BinaryPrecedence = (unsigned)NumVal;
                getNextToken();
            }
            break;

        default: 
            return LogErrorP("Expected function name in prototype");
    }

    if (CurTok != '(')
        return LogErrorP("Expected '(' in prototype");
    
    std::vector<std::string> ArgNames;
    while (getNextToken() == tok_identifier)
        ArgNames.push_back(IdentifierStr);
    
    if (CurTok != ')')
        return LogErrorP("Expected ')' in prototype");
    
    // Success
    getNextToken();

    // Verify right number of names for the operator
    if (Kind && ArgNames.size() != Kind)
        return LogErrorP("Invalid number of operands for operator");

    return std::make_unique<PrototypeAST>(FnName, std::move(ArgNames), Kind != 0, BinaryPrecedence);
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
        auto Proto = std::make_unique<PrototypeAST>("__anon_expr", std::vector<std::string>());
        return std::make_unique<FunctionAST>(std::move(Proto), std::move(E));
    }
    return nullptr;
}

//===----------------------------------------------------------------------===//
// Code Generation
//===----------------------------------------------------------------------===//

static LLVMContext TheContext;
static std::unique_ptr<Module> TheModule;
static std::unique_ptr<IRBuilder<>> Builder;
static std::map<std::string, Value *> NamedValues;
static std::unique_ptr<legacy::FunctionPassManager> TheFPM;
static std::unique_ptr<KaleidoscopeJIT> TheJIT;
static std::map<std::string, std::unique_ptr<PrototypeAST>> FunctionProtos;

Value *LogErrorV(const char *Str) {
  LogError(Str);
  return nullptr;
}

Value *NumberExprAST::codegen() {
    return ConstantFP::get(TheContext, APFloat(Val));
}

Value *VariableExprAST::codegen() {
    // Look up the variable name
    Value *V = NamedValues[Name];
    if (!V)
        return LogErrorV("Unknown variable name");
    return V;
}

Function *getFunction(std::string Name) {
  // First, check if the function has already been added to the current module
  if (auto *F = TheModule->getFunction(Name))
    return F;

  // If not, check if we can codegen the declaration from an existing prototype
  auto FI = FunctionProtos.find(Name);
  if (FI != FunctionProtos.end())
    return FI->second->codegen();

  // If no existing prototype exists, return null
  return nullptr;
}

Value *UnaryExprAST::codegen() {
    Value *OperandV = Operand->codegen();
    if (!OperandV) {
        return nullptr;
    }

    Function *F = getFunction(std::string("unary") + Op);
    if (!F)
        return LogErrorV("Unknown unary operator");
    
    return Builder->CreateCall(F, OperandV, "unop");
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
            return Builder->CreateUIToFP(L, Type::getDoubleTy(TheContext), "booltmp");
        default: 
            break;
    }

    Function *F = getFunction(std::string("binary") + Op);
    assert(F && "binary operator not found");

    Value *Ops[2] = {L, R};
    return Builder->CreateCall(F, Ops, "binop");
}

Value *CallExprAST::codegen() {
    // Look up the function name in the global module table
    Function *CalleeF = getFunction(Callee);
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

Value *IfExprAST::codegen() {
    Value *CondV = Cond->codegen();
    if (!CondV)
        return nullptr;
    
    CondV = Builder->CreateFCmpONE(CondV, ConstantFP::get(TheContext, APFloat(0.0)), "ifcond");

    Function *TheFunction = Builder->GetInsertBlock()->getParent();
    BasicBlock *ThenBB = BasicBlock::Create(TheContext, "then", TheFunction);
    BasicBlock *ElseBB = BasicBlock::Create(TheContext, "else");
    BasicBlock *MergeBB = BasicBlock::Create(TheContext, "ifcont");

    Builder->CreateCondBr(CondV, ThenBB, ElseBB);

    // ************Emit the code in then block*****************
    Builder->SetInsertPoint(ThenBB);

    Value *ThenV = Then->codegen();
    if (!ThenV)
        return nullptr;
    Builder->CreateBr(MergeBB);
    // The then expression may change the the current block, we need to get the current block again for the phi node
    ThenBB = Builder->GetInsertBlock();
    // ********************************************************

    // ************Emit the code in else block****************
    TheFunction->getBasicBlockList().push_back(ElseBB);
    Builder->SetInsertPoint(ElseBB);

    Value *ElseV = Else->codegen();
    if (!ElseV)
        return nullptr;
    Builder->CreateBr(MergeBB);
    // The else expression may change the the current block, we need to get the current block again for the phi node
    ElseBB = Builder->GetInsertBlock();
    // ********************************************************

    // ************Emit the code in merge block**************** 
    TheFunction->getBasicBlockList().push_back(MergeBB);
    Builder->SetInsertPoint(MergeBB);

    PHINode *PN = Builder->CreatePHI(Type::getDoubleTy(TheContext), 2, "iftmp");
    PN->addIncoming(ThenV, ThenBB);
    PN->addIncoming(ElseV, ElseBB);
    // ********************************************************

    return PN;
}

Value *ForExprAST::codegen() {
    Value *StartVal = Start->codegen();
    if (!StartVal)
        return nullptr;

    BasicBlock *EntryBB = Builder->GetInsertBlock();
    Function *TheFunction = EntryBB->getParent();
    BasicBlock *LoopBB = BasicBlock::Create(TheContext, "loop", TheFunction);

    Builder->CreateBr(LoopBB);

    Builder->SetInsertPoint(LoopBB);

    PHINode *Variable = Builder->CreatePHI(Type::getDoubleTy(TheContext), 2, VarName.c_str());
    Variable->addIncoming(StartVal, EntryBB);

    Value *OldVal = NamedValues[VarName];
    NamedValues[VarName] = Variable;

    if (!Body->codegen())
        return nullptr;
    
    Value *StepVal = nullptr;
    if (Step) {
        StepVal = Step->codegen();
        if (!StepVal)
            return nullptr;
    } else {
        StepVal = ConstantFP::get(TheContext, APFloat(1.0));
    }

    Value *NextVar = Builder->CreateFAdd(Variable, StepVal, "nextvar");

    Value *EndCond = End->codegen();
    if (!EndCond)
        return nullptr;

    EndCond = Builder->CreateFCmpONE(
        EndCond,
        ConstantFP::get(TheContext, APFloat(0.0)),
        "loopcond" 
    );

    BasicBlock *AfterBB = BasicBlock::Create(TheContext, "afterloop", TheFunction);
    Builder->CreateCondBr(EndCond, LoopBB, AfterBB);

    BasicBlock *LoopEndBB = Builder->GetInsertBlock();
    Variable->addIncoming(NextVar, LoopEndBB);

    Builder->SetInsertPoint(AfterBB);

    if (OldVal)
        NamedValues[VarName] = OldVal;
    else
        NamedValues.erase(VarName);

    return Constant::getNullValue(Type::getDoubleTy(TheContext));
}

Function *PrototypeAST::codegen() {
    // Creates a vector of "N" LLVM double types.
    std::vector<Type*> Doubles(Args.size(), Type::getDoubleTy(TheContext));

    // The call to FunctionType::get creates the FunctionType that should be used for a given Prototype. False indicates not vararg
    FunctionType *FT = FunctionType::get(Type::getDoubleTy(TheContext), Doubles, false);

    // Creates the IR function corresponding to the Prototype.
    // "external linkage" means that the function may be found outside the current module and/or that it is callable by functions outside the module
    Function *F = Function::Create(FT, Function::ExternalLinkage, Name, TheModule.get());

    unsigned Idx = 0;
    for (auto& Arg : F->args())
        Arg.setName(Args[Idx++]);

    return F;
}

Function *FunctionAST::codegen() {
    auto &P = *Proto;
    std::string ProtoName = Proto->getName();
    FunctionProtos[ProtoName] = std::move(Proto);
    Function *TheFunction = getFunction(ProtoName);

    if (!TheFunction)
        return nullptr;
    
    if (!TheFunction->empty())
        return (Function*)LogErrorV("Funciton cannot be redefined");

    if (P.isBinaryOp()) {
        BinopPrecedence[P.getOperatororName()] = P.getBinaryPrecendence();
    }

    // Create a new basic block to start insertion into.
    BasicBlock *BB = BasicBlock::Create(TheContext, "entry", TheFunction);
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
  TheModule = std::make_unique<Module>("my cool jit", TheContext);
  TheModule->setDataLayout(TheJIT->getTargetMachine().createDataLayout());

  // Create a new builder for the module.
  Builder = std::make_unique<IRBuilder<>>(TheContext);

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
      fprintf(stderr, "Read function definition:\n");
      FnIR->print(errs());
      fprintf(stderr, "\n");
      TheJIT->addModule(std::move(TheModule));
      InitializeModuleAndPassManager();
    }
  } else {
    // Skip token for error recovery.
    getNextToken();
  }
}

static void HandleExtern() {
  if (auto ProtoAST = ParseExtern()) {
    if (auto *FnIR = ProtoAST->codegen()) {
      fprintf(stderr, "Read extern:\n");
      FnIR->print(errs());
      fprintf(stderr, "\n");
      FunctionProtos[ProtoAST->getName()] = std::move(ProtoAST);
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
      fprintf(stderr, "Read top level expression:\n");
      FnIR->print(errs());
      fprintf(stderr, "\n");

      auto H = TheJIT->addModule(std::move(TheModule));

      InitializeModuleAndPassManager();

      // Search the JIT for the __anon_expr symbol
      auto ExprSymbol = TheJIT->findSymbol("__anon_expr");
      assert(ExprSymbol && "Function not found");

      // Get the symbol's address and cast it to a function pointer so that we can call it as a native function
      double (*FP)() = (double (*)())(intptr_t)cantFail(ExprSymbol.getAddress());
      fprintf(stderr, "Evaluated to %f\n", FP());

      // Delete the anoymous expression module from the JIT
      TheJIT->removeModule(H);
    }
  } else {
    // Skip token for error recovery.
    getNextToken();
  }
}

/// top ::= definition | external | expression | ';'
static void MainLoop() {
  while (true) {
    switch (CurTok) {
    case tok_eof:
      return;
    case ';': // ignore top-level semicolons.
      getNextToken();
      break;
    case tok_def:
      HandleDefinition();
      fprintf(stderr, "ready> ");
      break;
    case tok_extern:
      HandleExtern();
      fprintf(stderr, "ready> ");
      break;
    default:
      HandleTopLevelExpression();
      fprintf(stderr, "ready> ");
      break;
    }
    
  }
}

//===----------------------------------------------------------------------===//
// "Library" functions that can be "extern'd" from user code.
//===----------------------------------------------------------------------===//

#ifdef _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

/*
If no definition is found inside the JIT, it falls back to calling “dlsym("fun")” on the Kaleidoscope process itself.
*/

/// putchard - putchar that takes a double and returns 0.
extern "C" DLLEXPORT double putchard(double X) {
  fputc((char)X, stderr);
  return 0;
}

/// printd - printf that takes a double prints it as "%f\n", returning 0.
extern "C" DLLEXPORT double printd(double X) {
  fprintf(stderr, "%f\n", X);
  return 0;
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

  TheJIT = std::make_unique<KaleidoscopeJIT>();

  // Make the module, which holds all the code.
  InitializeModuleAndPassManager();
  
  // Run the main "interpreter loop" now.
  MainLoop();

  // Print out all of the generated code.
  TheModule->print(errs(), nullptr);

  return 0;
}




