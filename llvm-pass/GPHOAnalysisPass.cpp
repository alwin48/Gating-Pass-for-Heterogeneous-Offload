//===- GPHOAnalysisPass.cpp - Static IR features for GPU offload gating ----===//
//
// GPHO: Gating-Pass for Heterogeneous Offload
// Extracts static loop features from LLVM IR into JSON for asymmetric ML gating.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/DependenceAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <cmath>
#include <string>

using namespace llvm;

static cl::opt<std::string>
    GPHOBenchmarkId("gpho-benchmark-id", cl::desc("Benchmark id for JSON records"),
                    cl::init("unknown"));

static cl::opt<std::string>
    GPHOSuite("gpho-suite", cl::desc("Suite tag for LOBO-CV"),
              cl::init("synthetic"));

static cl::opt<std::string>
    GPHOOut("gpho-json-out", cl::desc("Write JSON array to this path (default: stdout)"),
            cl::init("-"));

namespace {

unsigned typeSizeBytes(Type *Ty) {
  if (!Ty)
    return 4;
  if (Ty->isFloatTy())
    return 4;
  if (Ty->isDoubleTy())
    return 8;
  if (Ty->isHalfTy())
    return 2;
  if (Ty->isIntegerTy())
    return (Ty->getIntegerBitWidth() + 7) / 8;
  if (Ty->isPointerTy())
    return 8;
  if (Ty->isVectorTy()) {
    auto *VT = cast<VectorType>(Ty);
    return typeSizeBytes(VT->getElementType()) *
           VT->getElementCount().getKnownMinValue();
  }
  return 4;
}

bool isFPOpcode(unsigned Op) {
  switch (Op) {
  case Instruction::FAdd:
  case Instruction::FSub:
  case Instruction::FMul:
  case Instruction::FDiv:
  case Instruction::FRem:
  case Instruction::FPExt:
  case Instruction::FPTrunc:
  case Instruction::SIToFP:
  case Instruction::UIToFP:
    return true;
  default:
    return false;
  }
}

bool isIntArithOpcode(unsigned Op) {
  switch (Op) {
  case Instruction::Add:
  case Instruction::Sub:
  case Instruction::Mul:
  case Instruction::UDiv:
  case Instruction::SDiv:
  case Instruction::URem:
  case Instruction::SRem:
  case Instruction::Shl:
  case Instruction::LShr:
  case Instruction::AShr:
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor:
    return true;
  default:
    return false;
  }
}

/// Conservative dependence check: any loop-carried ordered/confused dep ⇒ unsafe.
bool isGPUSafe(Loop *L, DependenceInfo &DI) {
  SmallVector<Instruction *, 16> MemInsts;
  for (BasicBlock *BB : L->blocks()) {
    for (Instruction &I : *BB) {
      if (isa<LoadInst>(I) || isa<StoreInst>(I))
        MemInsts.push_back(&I);
    }
  }
  for (size_t i = 0; i < MemInsts.size(); ++i) {
    for (size_t j = i; j < MemInsts.size(); ++j) {
      std::unique_ptr<Dependence> D = DI.depends(MemInsts[i], MemInsts[j], true);
      if (!D)
        continue;
      if (D->isLoopIndependent())
        continue;
      if (D->isConfused() || D->isOrdered())
        return false;
    }
  }
  return true;
}

struct LoopMetrics {
  unsigned TripCount = 0;
  bool TripUnknown = true;
  unsigned NestDepth = 1;
  unsigned FPOps = 0;
  unsigned IntOps = 0;
  unsigned Loads = 0;
  unsigned Stores = 0;
  unsigned BytesPerIter = 0;
  unsigned StrideOne = 0;
  unsigned Strided = 0;
  double Divergence = 0.0;
  uint64_t TransferBytes = 0;
  double TTICpu = 0.0;
  double TTIGpu = 0.0;
};

void classifyPointerStride(Value *Ptr, ScalarEvolution &SE, Loop *L,
                           unsigned &StrideOne, unsigned &Strided) {
  const SCEV *S = SE.getSCEV(Ptr);
  if (auto *AddRec = dyn_cast<SCEVAddRecExpr>(S)) {
    if (AddRec->getLoop() == L && AddRec->isAffine()) {
      if (auto *Step = dyn_cast<SCEVConstant>(AddRec->getStepRecurrence(SE))) {
        int64_t StepVal = Step->getAPInt().getSExtValue();
        if (StepVal == 0)
          Strided++;
        else if (std::llabs(StepVal) <= 8)
          StrideOne++;
        else
          Strided++;
        return;
      }
    }
  }
  Strided++;
}

LoopMetrics analyzeLoop(Loop *L, ScalarEvolution &SE, TargetTransformInfo &TTI,
                        DependenceInfo &DI) {
  (void)DI;
  LoopMetrics M;
  M.NestDepth = L->getLoopDepth();

  if (unsigned TC = SE.getSmallConstantTripCount(L)) {
    M.TripCount = TC;
    M.TripUnknown = false;
  } else {
    M.TripCount = 0;
    M.TripUnknown = true;
  }

  unsigned CondBranches = 0;
  unsigned TotalTerm = 0;

  for (BasicBlock *BB : L->blocks()) {
    for (Instruction &I : *BB) {
      unsigned Op = I.getOpcode();
      if (isFPOpcode(Op))
        M.FPOps++;
      if (isIntArithOpcode(Op))
        M.IntOps++;

      if (auto *LI = dyn_cast<LoadInst>(&I)) {
        M.Loads++;
        Type *Ty = LI->getType();
        M.BytesPerIter += typeSizeBytes(Ty);
        classifyPointerStride(LI->getPointerOperand(), SE, L, M.StrideOne,
                              M.Strided);
        InstructionCost Cost = TTI.getMemoryOpCost(
            Instruction::Load, Ty, LI->getAlign(), 0,
            TargetTransformInfo::TCK_RecipThroughput);
        if (auto V = Cost.getValue())
          M.TTICpu += (double)*V;
        else
          M.TTICpu += 1.0;
      } else if (auto *SI = dyn_cast<StoreInst>(&I)) {
        M.Stores++;
        Type *Ty = SI->getValueOperand()->getType();
        M.BytesPerIter += typeSizeBytes(Ty);
        classifyPointerStride(SI->getPointerOperand(), SE, L, M.StrideOne,
                              M.Strided);
        InstructionCost Cost = TTI.getMemoryOpCost(
            Instruction::Store, Ty, SI->getAlign(), 0,
            TargetTransformInfo::TCK_RecipThroughput);
        if (auto V = Cost.getValue())
          M.TTICpu += (double)*V;
        else
          M.TTICpu += 1.0;
      } else if (I.isBinaryOp()) {
        InstructionCost Cost = TTI.getArithmeticInstrCost(
            Op, I.getType(), TargetTransformInfo::TCK_RecipThroughput);
        if (auto V = Cost.getValue())
          M.TTICpu += (double)*V;
        else
          M.TTICpu += 1.0;
      }
    }
    if (auto *Br = dyn_cast<BranchInst>(BB->getTerminator())) {
      TotalTerm++;
      if (Br->isConditional())
        CondBranches++;
    }
  }

  M.Divergence = TotalTerm ? (double)CondBranches / (double)TotalTerm : 0.0;
  M.TTIGpu = M.TTICpu * 0.35;

  uint64_t Trip = M.TripUnknown ? 1024ull : (uint64_t)M.TripCount;
  M.TransferBytes = (uint64_t)M.BytesPerIter * Trip;
  return M;
}

json::Object metricsToJSON(const LoopMetrics &M, StringRef Bench, StringRef Suite,
                           StringRef LoopId, bool Safe) {
  unsigned Bytes = M.BytesPerIter ? M.BytesPerIter : 1;
  double AI = (double)(M.FPOps + M.IntOps) / (double)Bytes;

  uint64_t Ops = (uint64_t)(M.FPOps + M.IntOps);
  uint64_t Trip = M.TripUnknown ? 0 : M.TripCount;
  double TCR = 0.0;
  if (Ops > 0 && Trip > 0)
    TCR = (double)M.TransferBytes / (double)(Ops * Trip);
  else if (Ops > 0)
    TCR = (double)M.TransferBytes / (double)Ops;

  json::Object O;
  O["benchmark_id"] = Bench.str();
  O["suite"] = Suite.str();
  O["loop_id"] = LoopId.str();
  O["gpu_safe"] = Safe;
  O["trip_count_eval"] = (int64_t)M.TripCount;
  O["trip_count_unknown"] = M.TripUnknown;
  O["loop_nest_depth"] = (int64_t)M.NestDepth;
  O["fp_op_count"] = (int64_t)M.FPOps;
  O["int_op_count"] = (int64_t)M.IntOps;
  O["arithmetic_intensity"] = AI;
  O["load_inst_count"] = (int64_t)M.Loads;
  O["store_inst_count"] = (int64_t)M.Stores;
  O["stride_one_accesses"] = (int64_t)M.StrideOne;
  O["strided_accesses"] = (int64_t)M.Strided;
  O["divergence_score"] = M.Divergence;
  O["transfer_bytes_est"] = (int64_t)M.TransferBytes;
  O["transfer_compute_ratio"] = TCR;
  O["tti_cpu_cost"] = M.TTICpu;
  O["tti_gpu_cost"] = M.TTIGpu;
  O["tau"] = 1.2;
  O["t_cpu_ms"] = nullptr;
  O["t_gpu_ms"] = nullptr;
  O["speedup_actual"] = nullptr;
  O["label_profitable"] = nullptr;
  return O;
}

struct GPHOAnalysisPass : public PassInfoMixin<GPHOAnalysisPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM) {
    if (F.isDeclaration())
      return PreservedAnalyses::all();

    auto &LI = FAM.getResult<LoopAnalysis>(F);
    auto &SE = FAM.getResult<ScalarEvolutionAnalysis>(F);
    auto &TTI = FAM.getResult<TargetIRAnalysis>(F);
    auto &DI = FAM.getResult<DependenceAnalysis>(F);

    json::Array Records;
    unsigned Idx = 0;
    for (Loop *L : LI.getLoopsInPreorder()) {
      bool Safe = isGPUSafe(L, DI);
      LoopMetrics M = analyzeLoop(L, SE, TTI, DI);
      std::string LoopId = (F.getName() + ":loop" + Twine(Idx++)).str();
      Records.push_back(
          metricsToJSON(M, GPHOBenchmarkId, GPHOSuite, LoopId, Safe));
    }

    if (!Records.empty()) {
      std::string Buf;
      raw_string_ostream OS(Buf);
      OS << formatv("{0:2}", json::Value(std::move(Records)));
      OS.flush();
      if (GPHOOut == "-") {
        outs() << Buf << "\n";
      } else {
        std::error_code EC;
        raw_fd_ostream FOS(GPHOOut, EC);
        if (EC)
          errs() << "GPHO: failed to write " << GPHOOut << ": " << EC.message()
                 << "\n";
        else
          FOS << Buf << "\n";
      }
    }
    return PreservedAnalyses::all();
  }
};

} // namespace

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "GPHOPass", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, FunctionPassManager &FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "gpho-analysis") {
                    FPM.addPass(GPHOAnalysisPass());
                    return true;
                  }
                  return false;
                });
          }};
}
