#!/usr/bin/env python3
"""LLVM IR Control-Flow (CFG) & Data-Flow (DFG) Graph Structure Encoder.

Inspired by TehraniJamsaz et al. (2024) - "Learning-driven Compiler Optimization
and Graph Embeddings for Heterogeneous Hardware".

Extracts topological graph representations from LLVM IR basic blocks, instruction
def-use chains, and loop control structures, outputting a 4-dimensional normalized
graph structural embedding vector:
    e_1: CFG Cyclomatic Branch Complexity & Edge-to-Node Ratio
    e_2: DFG Data-Flow Dependency Fan-out & ILP Pressure
    e_3: Spatial Memory Access Clustering & Coalescing Index
    e_4: Control Path Entropy & Execution Flow Uniformity
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class GraphEmbedding:
    """4-Dimensional topological graph embedding vector."""

    cfg_cyclomatic_density: float  # e_1: Branch complexity per BB
    dfg_fanout_ratio: float       # e_2: Def-use dependencies per instruction
    memory_clustering_idx: float  # e_3: Coalesced memory spatial locality
    control_path_entropy: float   # e_4: Control-flow divergence entropy

    def to_list(self) -> list[float]:
        return [
            round(self.cfg_cyclomatic_density, 4),
            round(self.dfg_fanout_ratio, 4),
            round(self.memory_clustering_idx, 4),
            round(self.control_path_entropy, 4),
        ]

    def to_dict(self) -> dict[str, float]:
        return {
            "graph_cfg_cyclomatic": round(self.cfg_cyclomatic_density, 4),
            "graph_dfg_fanout": round(self.dfg_fanout_ratio, 4),
            "graph_mem_clustering": round(self.memory_clustering_idx, 4),
            "graph_path_entropy": round(self.control_path_entropy, 4),
        }


class LLVMGraphEncoder:
    """Topological graph encoder for LLVM IR CFG and DFG structures."""

    def __init__(self) -> None:
        # Regex patterns for basic block headers, instructions, and branches
        self._bb_pattern = re.compile(r"^([\w\.\-]+):", re.MULTILINE)
        self._br_cond_pattern = re.compile(r"br\s+i1\s+[%@\w]+,\s+label\s+[%@\w]+,\s+label\s+[%@\w]+")
        self._br_uncond_pattern = re.compile(r"br\s+label\s+[%@\w]+")
        self._switch_pattern = re.compile(r"switch\s+i\d+\s+[%@\w]+,\s+label\s+[%@\w]+\s+\[(.*?)\]", re.DOTALL)
        self._inst_pattern = re.compile(r"^\s+(?:[%@][\w\.\-]+(?:\s*=\s*))?([a-z0-9\._]+)\s+.*", re.MULTILINE)
        self._load_store_pattern = re.compile(r"\b(load|store)\b")
        self._fp_pattern = re.compile(r"\b(fadd|fsub|fmul|fdiv|frem|fpext|fptrunc|sitofp|uitofp|fcmp)\b")
        self._int_pattern = re.compile(r"\b(add|sub|mul|udiv|sdiv|urem|srem|shl|lshr|ashr|and|or|xor|icmp)\b")

    def encode_ir(self, llvm_ir_text: str) -> GraphEmbedding:
        """Extract graph structural features directly from LLVM IR string."""
        bbs = self._bb_pattern.findall(llvm_ir_text)
        num_bbs = max(len(bbs), 1)

        # Count CFG Edges
        cond_brs = len(self._br_cond_pattern.findall(llvm_ir_text))
        uncond_brs = len(self._br_uncond_pattern.findall(llvm_ir_text))
        switches = self._switch_pattern.findall(llvm_ir_text)
        switch_cases = sum(len(re.findall(r"label\s+[%@\w]+", s)) for s in switches)

        total_cfg_edges = (cond_brs * 2) + uncond_brs + switch_cases
        if total_cfg_edges < num_bbs - 1:
            total_cfg_edges = num_bbs  # Fallback lower bound for connected loops

        # e_1: CFG Cyclomatic Density = (E - V + 2P) / V
        cyclomatic = max(0, total_cfg_edges - num_bbs + 2)
        e1 = min(cyclomatic / float(num_bbs), 5.0)

        # Count instructions and estimate DFG edges (def-use relations)
        all_lines = llvm_ir_text.splitlines()
        inst_lines = [l for l in all_lines if l.strip().startswith(("%", "store", "br", "ret", "call", "switch"))]
        num_insts = max(len(inst_lines), 1)

        # Approximate DFG edges by tracking SSA variable references (%1, %val)
        var_refs = len(re.findall(r"%[\w\.\-]+", llvm_ir_text))
        dfg_edges = max(var_refs, num_insts)
        # e_2: DFG Fanout = DFG_edges / num_instructions
        e2 = min(dfg_edges / float(num_insts), 6.0)

        # Memory operations & spatial clustering
        loads = len(re.findall(r"\bload\b", llvm_ir_text))
        stores = len(re.findall(r"\bstore\b", llvm_ir_text))
        total_mem = loads + stores

        # Estimate contiguous vs strided from getelementptr patterns (inbounds with stride-1 vs indirect)
        geps = re.findall(r"getelementptr.*", llvm_ir_text)
        contiguous_geps = sum(1 for g in geps if "i64 1" in g or "i32 1" in g or "nuw nsw" in g)
        strided_geps = max(0, len(geps) - contiguous_geps)

        if total_mem > 0:
            e3 = (contiguous_geps + 1.0) / float(total_mem + strided_geps + 1.0)
        else:
            e3 = 1.0  # Pure compute has no strided bottleneck
        e3 = min(max(e3, 0.0), 1.0)

        # e_4: Control Path Entropy
        # For loops with multiple branches, compute Shannon entropy over branch targets
        total_branches = cond_brs + switch_cases
        if total_branches > 0:
            # Branch probabilities estimated based on loop control backedge vs exit
            p_taken = 0.85 if cond_brs > 0 else 0.5
            p_not_taken = 1.0 - p_taken
            entropy = - (p_taken * math.log2(p_taken) + p_not_taken * math.log2(p_not_taken))
            # Scale by branching density
            e4 = min(entropy * (cond_brs / float(num_bbs)), 1.0)
        else:
            e4 = 0.0  # Straight-line loop body has 0 control path entropy

        return GraphEmbedding(
            cfg_cyclomatic_density=e1,
            dfg_fanout_ratio=e2,
            memory_clustering_idx=e3,
            control_path_entropy=e4,
        )

    def encode_metrics(self, rec: Mapping[str, Any]) -> GraphEmbedding:
        """Derive the 4D graph topological embedding from structured LLVM pass JSON metrics."""
        nest = max(int(rec.get("loop_nest_depth", 1)), 1)
        divergence = float(rec.get("divergence_score", 0.0))
        fp_ops = int(rec.get("fp_op_count", 0))
        int_ops = int(rec.get("int_op_count", 0))
        loads = int(rec.get("load_inst_count", 0))
        stores = int(rec.get("store_inst_count", 0))
        stride_one = int(rec.get("stride_one_accesses", 0))
        strided = int(rec.get("strided_accesses", 0))

        total_ops = max(fp_ops + int_ops, 1)
        total_mem = max(loads + stores, 1)

        # e_1: CFG Cyclomatic Branch Complexity
        # Proportional to divergence score and nest depth
        e1 = round(divergence * 2.5 + 0.5 * (nest - 1), 4)

        # e_2: DFG Dependency Fan-out Ratio
        # Arithmetic-heavy loops have high DFG internal chaining; memory loads feed arithmetic
        e2 = round(min(1.2 + (total_ops / float(total_mem)) * 0.4, 6.0), 4)

        # e_3: Memory Spatial Clustering Index
        total_accesses = stride_one + strided
        if total_accesses > 0:
            e3 = round(stride_one / float(total_accesses), 4)
        else:
            e3 = 1.0

        # e_4: Control Path Entropy
        # Derived from divergence score and branch distribution
        if divergence > 0:
            p = min(max(divergence, 0.01), 0.99)
            raw_entropy = - (p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))
            e4 = round(min(raw_entropy * divergence, 1.0), 4)
        else:
            e4 = 0.0

        return GraphEmbedding(
            cfg_cyclomatic_density=e1,
            dfg_fanout_ratio=e2,
            memory_clustering_idx=e3,
            control_path_entropy=e4,
        )

    def enrich_features(
        self,
        features_dict: dict[str, Any],
        raw_ir: str | None = None,
    ) -> dict[str, Any]:
        """Augment a loop feature dictionary with the 4D graph structural embedding."""
        if raw_ir:
            emb = self.encode_ir(raw_ir)
        else:
            emb = self.encode_metrics(features_dict)

        enriched = dict(features_dict)
        enriched.update(emb.to_dict())
        return enriched


# Standalone utility functions for easy import
_GLOBAL_ENCODER = LLVMGraphEncoder()


def extract_graph_embedding_from_ir(ir_text: str) -> list[float]:
    """Return [e1, e2, e3, e4] graph embedding vector from LLVM IR text."""
    return _GLOBAL_ENCODER.encode_ir(ir_text).to_list()


def extract_graph_embedding_from_metrics(rec: Mapping[str, Any]) -> list[float]:
    """Return [e1, e2, e3, e4] graph embedding vector from loop metrics dictionary."""
    return _GLOBAL_ENCODER.encode_metrics(rec).to_list()
