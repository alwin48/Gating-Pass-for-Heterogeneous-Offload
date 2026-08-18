//===--- GPHOAnalysisPass.cpp - LLVM 18 Static IR Feature Extraction Pass -===//
//
// GPHO: Static-First ML-Gated GPU Offloading Framework for LLVM 18
//
// Computes 12 static IR metrics and safety properties for candidate loops,
// emitting structured JSON diagnostics for the downstream asymmetric ML gate.
//
//===----------------------------------------------------------------------===//

#include "GPHOAnalysisPass.h"

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/DependenceAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <cmath>
#include <cstdlib>
#include <string>

using namespace llvm;

// Command-line options for the GPHO analysis pass
static cl::opt<std::string>
    GPHOBenchmarkId("gpho-benchmark-id",
                    cl::desc("Benchmark identifier for JSON records"),
                    cl::init("unknown"));

static cl::opt<std::string>
    GPHOSuite("gpho-suite",
              cl::desc("Benchmark suite tag for LOBO-CV evaluation"),
              cl::init("synthetic"));

static cl::opt<std::string>
    GPHOJsonOut("gpho-json-out",
                cl::desc("Destination path for feature JSON array (default: stdout)"),
                cl::init("-"));

namespace llvm {

unsigned getGPHOTypeSizeBytes(Type *Ty) {
  if (!Ty)
    return 4;
  if (Ty->isFloatTy())
    return 4;
  if (Ty->isDoubleTy())
    return 8;
  if (Ty->isHalfTy())
    return 2;
  if (Ty->isIntegerTy())
    return std::max(1u, (Ty->getIntegerBitWidth() + 7) / 8);
  if (Ty->isPointerTy())
    return 8;
  if (Ty->isVectorTy()) {
    auto *VT = cast<VectorType>(Ty);
    return getGPHOTypeSizeBytes(VT->getElementType()) *
           VT->getElementCount().getKnownMinValue();
  }
  return 4;
}

unsigned getGPHOFPOps(const Instruction &I) {
  unsigned Op = I.getOpcode();
  if (isGPHOFPOpcode(Op))
    return 1;
  if (auto *CI = dyn_cast<CallInst>(&I)) {
    if (Function *CalledF = CI->getCalledFunction()) {
      StringRef Name = CalledF->getName();
      if (Name.starts_with("llvm.fmuladd") || Name.starts_with("llvm.fma"))
        return 2;
      if (Name.starts_with("llvm.sqrt") || Name.starts_with("llvm.sin") ||
          Name.starts_with("llvm.cos") || Name.starts_with("llvm.pow") ||
          Name.starts_with("llvm.exp") || Name.starts_with("llvm.fabs") ||
          Name.starts_with("llvm.maxnum") || Name.starts_with("llvm.minnum"))
        return 1;
    }
  }
  return 0;
}

bool isGPHOFPOpcode(unsigned Opcode) {
  switch (Opcode) {
  case Instruction::FAdd:
  case Instruction::FSub:
  case Instruction::FMul:
  case Instruction::FDiv:
  case Instruction::FRem:
  case Instruction::FPExt:
  case Instruction::FPTrunc:
  case Instruction::SIToFP:
  case Instruction::UIToFP:
  case Instruction::FCmp:
  case Instruction::FNeg:
    return true;
  default:
    return false;
  }
}

bool isGPHOIntArithOpcode(unsigned Opcode) {
  switch (Opcode) {
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

bool checkGPUSafety(Loop *L, DependenceInfo &DI) {
  SmallVector<Instruction *, 32> MemInsts;
  for (BasicBlock *BB : L->blocks()) {
    for (Instruction &I : *BB) {
      if (isa<LoadInst>(I) || isa<StoreInst>(I))
        MemInsts.push_back(&I);
    }
  }

  // Check all pairwise memory instructions for loop-carried dependencies
  for (size_t i = 0; i < MemInsts.size(); ++i) {
    for (size_t j = i; j < MemInsts.size(); ++j) {
      std::unique_ptr<Dependence> D = DI.depends(MemInsts[i], MemInsts[j], true);
      if (!D)
        continue;
      if (D->isLoopIndependent())
        continue;
      // Any loop-carried ordered dependence or confused dependence violates parallel safety
      if (D->isConfused() || D->isOrdered())
        return false;
    }
  }
  return true;
}

void classifyAccessStride(Value *Ptr, ScalarEvolution &SE, Loop *L,
                          unsigned &StrideOne, unsigned &Strided) {
  if (!Ptr) {
    Strided++;
    return;
  }

  const SCEV *S = SE.getSCEV(Ptr);
  if (auto *AddRec = dyn_cast<SCEVAddRecExpr>(S)) {
    if (AddRec->getLoop() == L && AddRec->isAffine()) {
      if (auto *Step = dyn_cast<SCEVConstant>(AddRec->getStepRecurrence(SE))) {
        int64_t StepVal = Step->getAPInt().getSExtValue();
        // Contiguous / unrolled stride (<= 64 bytes)
        if (StepVal == 0) {
          Strided++;
        } else if (std::llabs(StepVal) <= 64) {
          StrideOne++;
        } else {
          Strided++;
        }
        return;
      }
    }
  }
  // Unknown or complex non-affine pointer index
  Strided++;
}

GPHOLoopMetrics analyzeCandidateLoop(Loop *L, ScalarEvolution &SE,
                                     TargetTransformInfo &TTI,
                                     DependenceInfo &DI) {
  GPHOLoopMetrics M;
  M.NestDepth = L->getLoopDepth();
  M.IsGPUSafe = checkGPUSafety(L, DI);

  // SCEV Trip count estimation
  if (unsigned TC = SE.getSmallConstantTripCount(L)) {
    M.TripCount = TC;
    M.TripUnknown = false;
  } else if (unsigned MaxTC = SE.getSmallConstantMaxTripCount(L)) {
    M.TripCount = MaxTC;
    M.TripUnknown = false;
  } else {
    M.TripCount = 0;
    M.TripUnknown = true;
  }

  unsigned CondBranches = 0;
  unsigned TotalTerminators = 0;

  for (BasicBlock *BB : L->blocks()) {
    for (Instruction &I : *BB) {
      unsigned Op = I.getOpcode();
      M.FPOps += getGPHOFPOps(I);
      if (isGPHOIntArithOpcode(Op))
        M.IntOps++;

      if (auto *LI = dyn_cast<LoadInst>(&I)) {
        M.Loads++;
        Type *Ty = LI->getType();
        M.BytesPerIter += getGPHOTypeSizeBytes(Ty);
        classifyAccessStride(LI->getPointerOperand(), SE, L, M.StrideOne,
                             M.Strided);

        InstructionCost Cost = TTI.getMemoryOpCost(
            Instruction::Load, Ty, LI->getAlign(), 0,
            TargetTransformInfo::TCK_RecipThroughput);
        if (auto V = Cost.getValue())
          M.TTICpuCost += static_cast<double>(*V);
        else
          M.TTICpuCost += 1.0;
      } else if (auto *SI = dyn_cast<StoreInst>(&I)) {
        M.Stores++;
        Type *Ty = SI->getValueOperand()->getType();
        M.BytesPerIter += getGPHOTypeSizeBytes(Ty);
        classifyAccessStride(SI->getPointerOperand(), SE, L, M.StrideOne,
                             M.Strided);

        InstructionCost Cost = TTI.getMemoryOpCost(
            Instruction::Store, Ty, SI->getAlign(), 0,
            TargetTransformInfo::TCK_RecipThroughput);
        if (auto V = Cost.getValue())
          M.TTICpuCost += static_cast<double>(*V);
        else
          M.TTICpuCost += 1.0;
      } else if (I.isBinaryOp()) {
        InstructionCost Cost = TTI.getArithmeticInstrCost(
            Op, I.getType(), TargetTransformInfo::TCK_RecipThroughput);
        if (auto V = Cost.getValue())
          M.TTICpuCost += static_cast<double>(*V);
        else
          M.TTICpuCost += 1.0;
      }
    }

    if (auto *Br = dyn_cast<BranchInst>(BB->getTerminator())) {
      TotalTerminators++;
      if (Br->isConditional())
        CondBranches++;
    } else if (isa<SwitchInst>(BB->getTerminator())) {
      TotalTerminators++;
      CondBranches++;
    }
  }

  // Metric 5: Arithmetic Intensity = Total Operations / Bytes
  unsigned TotalOps = M.FPOps + M.IntOps;
  unsigned BytesPerIterSafe = M.BytesPerIter > 0 ? M.BytesPerIter : 1;
  M.ArithmeticIntensity = static_cast<double>(TotalOps) / static_cast<double>(BytesPerIterSafe);

  // Metric 10: SIMT Divergence Score
  M.DivergenceScore = TotalTerminators > 0
                          ? static_cast<double>(CondBranches) / static_cast<double>(TotalTerminators)
                          : 0.0;

  // Metric 11: Estimated Host-Device Transfer Bytes
  uint64_t EffectiveTrip = M.TripUnknown ? 1024ULL : (M.TripCount > 0 ? M.TripCount : 1ULL);
  M.TransferBytesEst = static_cast<uint64_t>(M.BytesPerIter) * EffectiveTrip;

  // Metric 12: Host-Device Transfer to Compute Ratio
  uint64_t TotalComputeWork = static_cast<uint64_t>(TotalOps) * EffectiveTrip;
  if (TotalComputeWork > 0) {
    M.TransferComputeRatio = static_cast<double>(M.TransferBytesEst) / static_cast<double>(TotalComputeWork);
  } else {
    M.TransferComputeRatio = static_cast<double>(M.TransferBytesEst);
  }

  // TTI GPU Cost Model approximation
  M.TTIGpuCost = M.TTICpuCost * 0.35;

  return M;
}

json::Object loopMetricsToJSON(const GPHOLoopMetrics &M, StringRef BenchmarkId,
                               StringRef Suite, StringRef LoopId,
                               StringRef FunctionName) {
  json::Object O;
  O["benchmark_id"] = BenchmarkId.str();
  O["suite"] = Suite.str();
  O["loop_id"] = LoopId.str();
  O["function_name"] = FunctionName.str();
  O["gpu_safe"] = M.IsGPUSafe;

  // 12 Core Extracted Metrics
  O["trip_count_eval"] = static_cast<int64_t>(M.TripCount);
  O["trip_count_unknown"] = M.TripUnknown;
  O["loop_nest_depth"] = static_cast<int64_t>(M.NestDepth);
  O["fp_op_count"] = static_cast<int64_t>(M.FPOps);
  O["int_op_count"] = static_cast<int64_t>(M.IntOps);
  O["arithmetic_intensity"] = M.ArithmeticIntensity;
  O["load_inst_count"] = static_cast<int64_t>(M.Loads);
  O["store_inst_count"] = static_cast<int64_t>(M.Stores);
  O["stride_one_accesses"] = static_cast<int64_t>(M.StrideOne);
  O["strided_accesses"] = static_cast<int64_t>(M.Strided);
  O["divergence_score"] = M.DivergenceScore;
  O["transfer_bytes_est"] = static_cast<int64_t>(M.TransferBytesEst);
  O["transfer_compute_ratio"] = M.TransferComputeRatio;

  // Target Transform Info Cost Properties
  O["tti_cpu_cost"] = M.TTICpuCost;
  O["tti_gpu_cost"] = M.TTIGpuCost;

  // Offload Decision Target Values
  O["tau"] = 1.20;
  O["t_cpu_ms"] = nullptr;
  O["t_gpu_ms"] = nullptr;
  O["speedup_actual"] = nullptr;
  O["label_profitable"] = nullptr;

  return O;
}

PreservedAnalyses GPHOAnalysisPass::run(Function &F, FunctionAnalysisManager &FAM) {
  if (F.isDeclaration())
    return PreservedAnalyses::all();

  auto &LI = FAM.getResult<LoopAnalysis>(F);
  auto &SE = FAM.getResult<ScalarEvolutionAnalysis>(F);
  auto &TTI = FAM.getResult<TargetIRAnalysis>(F);
  auto &DI = FAM.getResult<DependenceAnalysis>(F);

  json::Array Records;
  unsigned LoopIdx = 0;

  for (Loop *L : LI.getLoopsInPreorder()) {
    GPHOLoopMetrics M = analyzeCandidateLoop(L, SE, TTI, DI);
    std::string LoopId = (F.getName() + ":loop" + Twine(LoopIdx++)).str();
    Records.push_back(
        loopMetricsToJSON(M, GPHOBenchmarkId, GPHOSuite, LoopId, F.getName()));
  }

  if (!Records.empty()) {
    std::string OutputBuffer;
    raw_string_ostream SS(OutputBuffer);
    for (const auto &Rec : Records) {
      SS << formatv("{0}\n", Rec);
    }
    SS.flush();

    if (GPHOJsonOut == "-") {
      outs() << OutputBuffer;
    } else {
      std::error_code EC;
      raw_fd_ostream FOS(GPHOJsonOut, EC, sys::fs::OF_Append | sys::fs::OF_TextWithCRLF);
      if (EC) {
        errs() << "GPHO: Error writing output to " << GPHOJsonOut << ": "
               << EC.message() << "\n";
      } else {
        FOS << OutputBuffer;
      }
    }
  }

  return PreservedAnalyses::all();
}

} // namespace llvm

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
