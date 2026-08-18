# GPHO System Architecture & Compiler Design Deep Dive

GPHO (**Gating-Pass for Heterogeneous Offload**) is an integrated compiler and machine learning framework designed to solve the *offloading regression problem* in modern heterogeneous computing. While modern parallelizing compilers (such as Clang/LLVM OpenMP, OpenACC, and Polly) possess sophisticated polyhedral analysis and transformation engines, they often apply static or greedy offloading heuristics that trigger severe runtime performance degradations. 

GPHO serves as an intelligent gatekeeper: candidate parallel loops are statically inspected at the LLVM Intermediate Representation (IR) level, encoded into structural and topological feature vectors, and evaluated against an asymmetric machine-learning gating engine and a deterministic hardware cost model before any device code generation or runtime dispatch is committed.

---

## 1. Architectural Overview

```
+-----------------------------------------------------------------------------------+
|                               Source Code (C / C++)                               |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼ (clang -O3 -emit-llvm)
+-----------------------------------------------------------------------------------+
|                              LLVM 18 IR (SSA Form)                                |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
|                           LLVM Analysis Passes (Opt)                              |
|  ┌───────────────────────┐ ┌──────────────────────┐ ┌──────────────────────────┐  |
|  │ ScalarEvolution (SCEV)│ │TargetTransformInfo(TTI)│ │DependenceAnalysis (DA) │  |
|  └───────────────────────┘ └──────────────────────┘ └──────────────────────────┘  |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
|               GPHO C++ Analysis Pass (`pass/GPHOAnalysisPass.cpp`)                |
|  • 12 Static IR Structural Metrics (Trip Counts, Strides, Divergence, AI, Bytes)  |
|  • Conservative GPU Safety Gate (Loop-Carried True/Anti/Output Dependences)       |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼ Structured Loop JSON Diagnostics
+-----------------------------------------------------------------------------------+
|           Hybrid Graph-Topology Encoder (`ml/graph_encoder.py`)                   |
|  • 4D Topological Embeddings: CFG Cyclomatic, DFG Fan-out, Locality, Path Entropy  |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼ Enriched 16D Feature Vector
+-----------------------------------------------------------------------------------+
|              Asymmetric ML Gate & Tree SHAP (`driver/gpho_gate.py`)               |
|  • XGBoost Asymmetric Classifier (alpha = 3.5 penalizing False Positives)         |
|  • Local Tree SHAP Feature Attribution (Lundberg Algorithm)                       |
|  • First-Principles Analytical Hardware Cost Model (SIMD vs PCIe + CUDA)          |
+-----------------------------------------------------------------------------------+
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼
        [✓] GPU ACCEPTED (>= 1.20x)               [✗] GPU REJECTED (< 1.20x)
  #pragma omp target teams distribute       #pragma omp parallel for
        parallel for if(target: N >= TC)            (Execute on Multicore CPU)
```

---

## 2. Why Static LLVM IR Over Raw ASTs?

Many machine-learning compiler tools operate directly on Abstract Syntax Trees (ASTs) or raw source tokens. GPHO deliberately operates on **canonicalized LLVM 18 Intermediate Representation (IR)** in Static Single Assignment (SSA) form. The rationale rests on three foundational pillars:

### 2.1 Optimization Canonicalization & Invariant Normalization
At the AST level, syntactically distinct constructs (e.g., `for`, `while`, pointer arithmetic, index calculations, macro unrolling) obscure the underlying computational workload. Clang's early optimization pipeline (`-O1` / `-O2` / `-O3`) canonicalizes these patterns into uniform SSA basic blocks and PHI nodes. LLVM IR provides a clean, machine-independent, target-informed substrate where dead code has been eliminated, subexpressions commoned, and induction variables simplified.

### 2.2 Deep Semantic Analysis Passes
LLVM provides mature, production-tested compiler analysis infrastructure unavailable at the AST level:
- **`ScalarEvolution` (SCEV):** Symbolically evaluates loop trip counts, computes closed-form expressions for induction variables, and detects affine recurrence steps (determining unit-stride vs strided/indirect pointer arithmetic).
- **`TargetTransformInfo` (TTI):** Provides architecture-calibrated instruction costs (reciprocal throughput and latency) across differing vector widths and ALU capabilities.
- **`DependenceAnalysis` (DA):** Proves memory independence across loop iterations using GCD, Banerjee, and polyhedral-like subscript tests.
- **`BlockFrequencyInfo` (BFI):** Weights basic block execution frequencies and identifies hot paths.

### 2.3 Language Agnosticism
Operating at the IR level allows GPHO to seamlessly analyze candidate loops originating from C, C++, Fortran, Rust, or domain-specific frontends without needing language-specific AST parsers.

---

## 3. Mathematical Definition of Offloading Profitability ($\tau = 1.20$)

Offloading a compute loop to an accelerator incurs non-trivial hardware latencies:
1. **Driver & Kernel Launch Latency ($T_{\text{launch}} \approx 10\text{--}25\,\mu\text{s}$):** Direct driver interaction and thread block dispatch.
2. **Host-to-Device (H2D) & Device-to-Host (D2H) Memory Transfer ($T_{\text{PCIe}}$):** Governed by PCIe bandwidth ($B_{\text{PCIe}} \approx 16\text{--}32\,\text{GB/s}$).
3. **Memory Coalescing & Branch Divergence Overhead:** Non-unit strides and divergent branches serialize GPU warps.

Let $T_{\text{CPU}}$ be the execution time of the loop executed across all host CPU cores with SIMD vectorization. Let $T_{\text{GPU}}$ be the end-to-end device execution time:

$$T_{\text{GPU}} = T_{\text{launch}} + T_{\text{H2D}} + T_{\text{DeviceCompute}} + T_{\text{D2H}}$$

### Profitability Criterion
A candidate loop is defined as **GPU-Profitable** if and only if the achieved speedup meets or exceeds the strict margin $\tau = 1.20$:

$$\text{Speedup} = \frac{T_{\text{CPU}}}{T_{\text{GPU}}} \ge \tau \quad \text{where } \tau = 1.20$$

$$\text{Label} = \begin{cases} 1 (\text{Profitable}), & \text{if } \text{Speedup} \ge 1.20 \\ 0 (\text{Unprofitable}), & \text{if } \text{Speedup} < 1.20 \end{cases}$$

### Why $\tau = 1.20$?
A $1.0\times$ threshold is insufficient in practice due to runtime power consumption, thermal headroom, memory allocation contention, and PCIe bus congestion. A $20\%$ minimum performance margin guarantees that offloading delivers net-positive throughput even under realistic system variability.

---

## 4. Candidate Region Identification Boundaries

GPHO inspects every loop structure discovered by LLVM's `LoopInfo`. However, loops must pass strict boundary criteria before ML inference:

```
                          ┌────────────────────────────┐
                          │    Candidate Loop Nest     │
                          └─────────────┬──────────────┘
                                        │
                         [Loop-Carried Dependences?]
                                       ╱ ╲
                                     YES  NO
                                     ╱     ╲
                                    ▼       ▼
                            [Force CPU]  [Known Trip Count?]
                            (Hard Gate)        ╱ ╲
                                             NO   YES
                                             ╱     ╲
                                            ▼       ▼
                                    [Force CPU]  [Extract 16D Features]
                                    (Conservative)  │
                                                    ▼
                                            [ML Gating Engine]
```

1. **GPU Correctness Hard Gate (`gpu_safe`):**
   GPHO queries `DependenceAnalysis`. If any loop-carried ordered dependence (RAW/WAR/WAW) or confused dependence exists, the loop cannot be parallelized without race conditions. It is immediately designated `CPU` (Safe) with confidence $1.0$, bypassing ML inference.

2. **Loop Bounds Evaluability (`trip_count_unknown`):**
   If `ScalarEvolution` cannot determine a constant bound or symbolic upper limit for the loop trip count, GPHO adopts a *conservative stance* and refuses GPU offload. Naively offloading loops with dynamic small bounds (e.g., $N=8$) causes devastating kernel launch overhead regressions.

3. **Innermost vs Outermost Nesting:**
   GPHO computes metrics for all levels of a loop nest. For nested loops ($L_{\text{outer}} \rightarrow L_{\text{inner}}$), the outermost parallel loop is analyzed as the candidate target for `teams distribute parallel for`, while inner loops inform arithmetic intensity and memory coalescing metrics.

---

## 5. LLVM Pass Positioning in the Optimization Pipeline

GPHO is architected as an LLVM **New Pass Manager** Analysis Pass (`PassInfoMixin<GPHOAnalysisPass>`).

### Optimal Pipeline Placement
To ensure high-fidelity feature extraction, GPHO must run **after** standard canonicalization and loop simplification passes, but **before** target-specific loop transformations:

```
[ Clang Frontend ]
        │
        ▼
[ Early Inlining & SROA (Scalar Replacement of Aggregates) ]
        │
        ▼
[ Loop Simplify & LCSSA (Loop-Closed SSA Form) ]
        │
        ▼
[ Loop Rotate & Induction Variable Simplification (IndVars) ]
        │
        ▼
============================================================
  GPHO ANALYSIS PASS (`gpho-analysis`)
  • Accesses canonical SCEV expressions & affine recurrences
  • Queries finalized basic blocks for accurate branch divergence
  • Measures unvectorized loop body instruction costs via TTI
============================================================
        │
        ▼
[ OpenMP / Offload Code Generation & Dynamic Pragma Injection ]
        │
        ▼
[ Target Vectorization (SLP / LoopVectorize for CPU fallback) ]
        │
        ▼
[ Machine Code Generation (LLVM Backend) ]
```

By placing GPHO immediately after `IndVars` and `LoopRotate`, `ScalarEvolution` can extract exact trip counts and stride distances, while `TargetTransformInfo` computes realistic execution costs on clean, rotated loop bodies.
