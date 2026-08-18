//===--- GPHOAnalysisPass.h - Static IR Feature Extraction Pass ----------===//
//
// GPHO: Static-First ML-Gated GPU Offloading Framework for LLVM 18
//
// This header declares the GPHOAnalysisPass, the primary analysis pass that
// traverses loop nests, queries ScalarEvolution (SCEV), TargetTransformInfo
// (TTI), DependenceAnalysis (DA), and BlockFrequencyInfo (BFI) to extract
// 12 structural IR metrics and safety properties for ML-gated offloading.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_GPHO_ANALYSIS_PASS_H
#define LLVM_TRANSFORMS_GPHO_ANALYSIS_PASS_H

#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/DependenceAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <string>
#include <vector>

namespace llvm {

/// Data structure holding the 12 extracted static IR structural metrics
/// and hardware cost approximations for a single candidate loop.
struct GPHOLoopMetrics {
  // 1. Trip count evaluation & bounds
  uint64_t TripCount = 0;
  bool TripUnknown = true;

  // 2. Structural hierarchy
  unsigned NestDepth = 1;

  // 3 & 4. Arithmetic workload
  unsigned FPOps = 0;
  unsigned IntOps = 0;

  // 5. Arithmetic Intensity (Ops / Byte)
  double ArithmeticIntensity = 0.0;

  // 6 & 7. Memory instruction counts
  unsigned Loads = 0;
  unsigned Stores = 0;
  unsigned BytesPerIter = 0;

  // 8 & 9. Memory stride pattern classification
  unsigned StrideOne = 0;
  unsigned Strided = 0;

  // 10. SIMT branch divergence risk
  double DivergenceScore = 0.0;

  // 11. Estimated PCIe / memory bus transfer volume (Bytes)
  uint64_t TransferBytesEst = 0;

  // 12. Host-Device Transfer to Compute Ratio
  double TransferComputeRatio = 0.0;

  // Additional Target-specific metrics
  double TTICpuCost = 0.0;
  double TTIGpuCost = 0.0;

  // Correctness safety flag from DependenceAnalysis
  bool IsGPUSafe = true;
};

/// Helper utilities for instruction classification & stride evaluation
unsigned getGPHOTypeSizeBytes(Type *Ty);
unsigned getGPHOFPOps(const Instruction &I);
bool isGPHOFPOpcode(unsigned Opcode);
bool isGPHOIntArithOpcode(unsigned Opcode);

/// Conservative GPU safety verification via LLVM DependenceAnalysis.
/// Returns false if any loop-carried ordered or confused memory dependence exists.
bool checkGPUSafety(Loop *L, DependenceInfo &DI);

/// Memory stride classification using SCEV affine recurrence expressions.
void classifyAccessStride(Value *Ptr, ScalarEvolution &SE, Loop *L,
                          unsigned &StrideOne, unsigned &Strided);

/// Comprehensive loop analysis extracting all 12 structural metrics.
GPHOLoopMetrics analyzeCandidateLoop(Loop *L, ScalarEvolution &SE,
                                     TargetTransformInfo &TTI,
                                     DependenceInfo &DI);

/// Serializes extracted loop metrics to LLVM JSON object representation.
json::Object loopMetricsToJSON(const GPHOLoopMetrics &M, StringRef BenchmarkId,
                               StringRef Suite, StringRef LoopId,
                               StringRef FunctionName);

/// GPHO Analysis Pass implementing LLVM's New Pass Manager interface.
class GPHOAnalysisPass : public PassInfoMixin<GPHOAnalysisPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM);

  static bool isRequired() { return true; }
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_GPHO_ANALYSIS_PASS_H
