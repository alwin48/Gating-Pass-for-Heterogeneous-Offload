# GPHO LLVM Pass Developer Reference Guide

This guide provides an exhaustive technical reference for the **GPHO LLVM 18 C++ Analysis Pass** (`pass/GPHOAnalysisPass.cpp` and `pass/GPHOAnalysisPass.h`). It documents the internal interaction with LLVM core analysis passes, the feature extraction algorithms, and instructions for extending or embedding the pass into custom compiler pipelines.

---

## 1. Pass Infrastructure & New Pass Manager Architecture

GPHO is built using LLVM's **New Pass Manager** (NPM) interface via `PassInfoMixin<GPHOAnalysisPass>`.

### Class Definition (`pass/GPHOAnalysisPass.h`)
```cpp
namespace llvm {

class GPHOAnalysisPass : public PassInfoMixin<GPHOAnalysisPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM);
  static bool isRequired() { return true; }
};

} // namespace llvm
```

### Analysis Dependencies Requested via `FunctionAnalysisManager`
When `run(Function &F, FunctionAnalysisManager &FAM)` executes, it requests cached analyses:
1. `LoopAnalysis` (`LoopInfo &LI = FAM.getResult<LoopAnalysis>(F);`): Discovers top-level and nested loop hierarchies.
2. `ScalarEvolutionAnalysis` (`ScalarEvolution &SE = FAM.getResult<ScalarEvolutionAnalysis>(F);`): Performs symbolic loop bound analysis and affine recurrence evaluation.
3. `TargetIRAnalysis` (`TargetTransformInfo &TTI = FAM.getResult<TargetIRAnalysis>(F);`): Queries target-calibrated instruction throughput and memory latency costs.
4. `DependenceAnalysis` (`DependenceInfo &DI = FAM.getResult<DependenceAnalysis>(F);`): Computes pairwise data dependences across memory instructions.

---

## 2. Core LLVM Analysis Interactions

### 2.1 DependenceAnalysis: Proving GPU Parallel Safety
GPU threads execute in lockstep warps (SIMT). Any loop-carried data dependency (RAW, WAR, WAW) across iterations will cause race conditions unless synchronized.

```cpp
bool checkGPUSafety(Loop *L, DependenceInfo &DI) {
  SmallVector<Instruction *, 32> MemInsts;
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
        continue; // Independent within same iteration is safe
      if (D->isConfused() || D->isOrdered())
        return false; // Loop-carried dependence detected!
    }
  }
  return true;
}
```

### 2.2 ScalarEvolution (SCEV): Trip Counts & Stride Classification
Memory coalescing is the single most critical performance factor on GPUs. GPHO queries SCEV to classify pointer address calculations into **Stride-1 (Contiguous)** or **Strided/Indirect**:

```cpp
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
        if (StepVal == 0) {
          Strided++; // Invariant/broadcast
        } else if (std::llabs(StepVal) <= 8) {
          StrideOne++; // Contiguous element access (<= 8 bytes)
        } else {
          Strided++; // Large non-unit stride
        }
        return;
      }
    }
  }
  Strided++; // Non-affine or complex indirect access
}
```

### 2.3 TargetTransformInfo (TTI): Architecture-Specific Cost Evaluation
TTI assigns cost values calibrated to target CPU/GPU instruction throughput:

```cpp
// Arithmetic Cost
InstructionCost Cost = TTI.getArithmeticInstrCost(
    Op, I.getType(), TargetTransformInfo::TCK_RecipThroughput);

// Memory Cost
InstructionCost Cost = TTI.getMemoryOpCost(
    Instruction::Load, Ty, LI->getAlign(), 0,
    TargetTransformInfo::TCK_RecipThroughput);
```

---

## 3. Extracted 12 Static IR Metrics Reference

| # | Metric Field | Type | LLVM Extraction Mechanism | Description |
|---|--------------|------|---------------------------|-------------|
| 1 | `trip_count_eval` | `int64` | `SE.getSmallConstantTripCount(L)` | Evaluated trip count or max trip count. |
| 2 | `loop_nest_depth` | `int64` | `L->getLoopDepth()` | Nesting depth level (1 for outermost). |
| 3 | `fp_op_count` | `int64` | `isGPHOFPOpcode(I.getOpcode())` | Count of `fadd`, `fsub`, `fmul`, `fdiv`, etc. |
| 4 | `int_op_count` | `int64` | `isGPHOIntArithOpcode(I.getOpcode())` | Count of `add`, `sub`, `mul`, bitwise ops. |
| 5 | `arithmetic_intensity` | `float` | `(FPOps + IntOps) / BytesPerIter` | Total computational ops per byte transferred. |
| 6 | `load_inst_count` | `int64` | `isa<LoadInst>(I)` | Number of memory loads in loop body. |
| 7 | `store_inst_count` | `int64` | `isa<StoreInst>(I)` | Number of memory stores in loop body. |
| 8 | `stride_one_accesses` | `int64` | `SCEVAddRecExpr` Step analysis | Number of contiguous unit-stride memory accesses. |
| 9 | `strided_accesses` | `int64` | Non-affine or step $>8$ bytes | Number of un-coalesced/strided memory accesses. |
| 10| `divergence_score` | `float` | `CondBranches / TotalTerminators` | Ratio of conditional branches in loop body. |
| 11| `transfer_bytes_est` | `int64` | `BytesPerIter * TripCount` | Total memory volume copied over PCIe. |
| 12| `transfer_compute_ratio`| `float` | `TransferBytes / TotalComputeOps` | Host-device communication overhead vs compute. |

---

## 4. Building and Running the Pass

### 4.1 Building `GPHOPass.so` with CMake
```bash
# Configure and build the loadable pass plugin
cmake -B build -S .
cmake --build build
```
The compiled plugin will be generated at `build/pass/GPHOPass.so`.

### 4.2 Executing via LLVM `opt`
To run the analysis pass on any `.ll` file and emit JSON diagnostics:

```bash
opt -load-pass-plugin=build/pass/GPHOPass.so \
    -passes=gpho-analysis \
    -gpho-benchmark-id=kernel_gemm \
    -gpho-suite=polybench \
    -gpho-json-out=gemm_features.json \
    -disable-output benchmarks/kernel_gemm.ll
```

### 4.3 Integrating into Production `opt` Pipelines
You can register GPHO directly inside `PassBuilder` callbacks in custom LLVM tools:

```cpp
PB.registerPipelineParsingCallback(
    [](StringRef Name, FunctionPassManager &FPM,
       ArrayRef<PassBuilder::PipelineElement>) {
      if (Name == "gpho-analysis") {
        FPM.addPass(GPHOAnalysisPass());
        return true;
      }
      return false;
    });
```

---

## 5. Pass Extension Guidelines

1. **Adding New Vector Instruction Metrics:**
   To track AVX-512 / NEON vector intrinsics, extend `analyzeCandidateLoop` to inspect `VectorType` operands:
   ```cpp
   if (auto *VT = dyn_cast<VectorType>(I.getType())) {
       M.VectorWidthMax = std::max(M.VectorWidthMax, VT->getElementCount().getKnownMinValue());
   }
   ```

2. **Refining Reduction Variable Detection:**
   Query `llvm::RecurrenceDescriptor::isReductionCandidate` to explicitly tag associative accumulation loops (e.g. `sum += ...`).
