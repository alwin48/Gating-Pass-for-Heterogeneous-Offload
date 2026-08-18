# GPHO Feature Schema (Day-1 Frozen)
#
# Contract between Member 1 (LLVM pass) and Member 2 (ML/harness).
# Canonical machine-readable schema: schema/gpho_features.schema.json

## Profitability policy

- `tau = 1.20` (mandatory 20% margin over optimized multithreaded CPU)
- `label_profitable = (speedup_actual >= tau)` when timings exist
- `gpu_safe == false` ⇒ never offload (hard gate; ML is not consulted)
- `trip_count_unknown == true` ⇒ conservative CPU stance

## Required static features (12 + flags)

| Field | Type | LLVM source |
|-------|------|-------------|
| trip_count_eval | int | SCEV getSmallConstantTripCount (0 if unknown) |
| trip_count_unknown | bool | SCEV could not evaluate constant bound |
| loop_nest_depth | int | LoopInfo getLoopDepth |
| fp_op_count | int | IR opcode scan |
| int_op_count | int | IR opcode scan |
| arithmetic_intensity | float | (fp+int)/bytes per iter |
| load_inst_count | int | LoadInst count |
| store_inst_count | int | StoreInst count |
| stride_one_accesses | int | SCEV step == 1 |
| strided_accesses | int | SCEV step > 1 or non-affine |
| divergence_score | float | branch weight in body [0,1] |
| transfer_bytes_est | int | estimated H2D+D2H footprint |
| transfer_compute_ratio | float | transfer_bytes / (ops * trip) |

## Identity / LOBO tags

| Field | Type | Notes |
|-------|------|-------|
| benchmark_id | string | e.g. polybench_gemm |
| suite | string | polybench \| rodinia \| synthetic \| demo \| cpu_friendly |
| loop_id | string | func:line or IR name |
| gpu_safe | bool | DependenceAnalysis hard gate |

## Labels (filled by harness)

| Field | Type |
|-------|------|
| t_cpu_ms | float \| null |
| t_gpu_ms | float \| null |
| speedup_actual | float \| null |
| label_profitable | bool \| null |

## Example record

```json
{
  "benchmark_id": "demo_gemm",
  "suite": "demo",
  "loop_id": "gemm:kernel:0",
  "gpu_safe": true,
  "trip_count_eval": 1048576,
  "trip_count_unknown": false,
  "loop_nest_depth": 3,
  "fp_op_count": 2,
  "int_op_count": 6,
  "arithmetic_intensity": 0.083,
  "load_inst_count": 2,
  "store_inst_count": 1,
  "stride_one_accesses": 3,
  "strided_accesses": 0,
  "divergence_score": 0.0,
  "transfer_bytes_est": 25165824,
  "transfer_compute_ratio": 0.003,
  "t_cpu_ms": 142.0,
  "t_gpu_ms": 17.1,
  "speedup_actual": 8.30,
  "label_profitable": true,
  "tau": 1.2
}
```
