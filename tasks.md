# LLEGO Distillation Project - Task Tracker

## Project: Distilling Evolutionary Priors from Language Models

**Goal:** Use MI to understand what LLMs contribute to decision tree evolution, then distill into lightweight components.

**Deadline:** 2nd February 2026 (6 days remaining)

---

## Phase 1: Mechanistic Diagnosis ✅ COMPLETE

### Experiment 1.1: Semantic Ablation Study ✅
- [x] Create experiment script (`mi_analysis/semantic_ablation.py`)
- [x] Create config and utilities
- [x] Test dry-run locally
- [x] Run on cluster with Llama 3.1 8B (4-bit quantization)
- [x] Analyze results

**Key Finding:** Semantic names do NOT help! Arbitrary (X1, X2) performs equal/better.
- Semantic parse rate: 90%, Arbitrary: 95%
- **Implication:** LLM contribution is STRUCTURAL, not SEMANTIC

### Experiment 1.2: Fitness Attention Analysis ✅
- [x] Create experiment script (`mi_analysis/fitness_attention.py`)
- [x] Implement fitness swap test
- [x] Implement fitness ablation test
- [x] Run on cluster
- [x] Analyze results

**Key Finding:** LLM DOES use fitness information (0% identical when swapped)

### Experiment 1.3: Structural Prior Elicitation ✅ (via analysis)
- [x] Extract trees from Phase 1 results
- [x] Fit structural prior distribution
- **Learned Prior:** Depth=3.08±0.88, Size=10.68±3.17, Balance=0.75±0.26

---

## Phase 2: Distillation (Current)

### Component A: Structural Prior Model ✅
- [x] Design based on Phase 1 findings
- [x] Implement `StructuralPrior` class
- [x] Fit from LLM-generated trees (37 trees)
- [x] Save to `mi_analysis/results/phase2/structural_prior.pkl`

### Component B: Semantic Feature Graph ❌ SKIPPED
- Phase 1 showed semantics don't help → not needed!

### Integration: Distilled Evolution ✅
- [x] Implement `DistilledEvolution` class
- [x] Structure-guided crossover (no LLM calls)
- [x] Mutation operators
- [ ] Run validation experiments

---

## Phase 3: Validation & Writing ✅ COMPLETE

### Validation Experiments ✅
- [x] Create validation experiment script (`mi_analysis/validation.py`)
- [x] Run: GATree vs Distilled-LLEGO on breast/heart/iris datasets
- [x] Analyze results

**RESULTS - Distilled-LLEGO OUTPERFORMS GATree!**

| Dataset | GATree | Distilled-LLEGO | Improvement |
|---------|--------|-----------------|-------------|
| breast  | 86.55% | **88.89%** | +2.34% |
| iris    | 93.33% | **100.0%** | +6.67% |
| heart   | 86.67% | **87.78%** | +1.11% |

**Key Achievement:** 0 LLM calls at runtime, still better than baseline!

### Writing (TODO)
- [ ] Generate figures (fitness curves, tree visualizations)
- [ ] Write report (ICML/NeurIPS style)

---

## Infrastructure

### Files Created
- [x] `mi_analysis/__init__.py`
- [x] `mi_analysis/config.py`
- [x] `mi_analysis/utils.py`
- [x] `mi_analysis/semantic_ablation.py`
- [x] `mi_analysis/fitness_attention.py`
- [x] `mi_analysis/distillation.py` ← Phase 2 core
- [x] `mi_analysis/analyze_phase1.py` ← Phase 2 analysis
- [x] `mi_analysis/validation.py` ← Phase 3 experiments
- [x] `mi_analysis/submit_phase1.sh`
- [x] `mi_analysis/submit_semantic_fast.sh`
- [x] `mi_analysis/results/README.md`
- [x] `mi_analysis/results/phase2/structural_prior.pkl`
- [x] `mi_analysis/results/phase2/structural_prior.json`
- [x] `mi_analysis/results/phase2/phase1_analysis_report.txt`
- [x] `mi_analysis/results/phase3/validation_results_*.csv`
- [x] `imp_plan.md` (full implementation plan)

### Cluster Resources
- **Available:** Turing GPUs (28GB RAM, 2 CPUs per job)
- **Model:** Llama 3.1 8B Instruct with 4-bit quantization (~5GB VRAM)

---

## Log

### 2026-01-27 (Phase 2 & 3)
- **Phase 2 COMPLETE!** 🎉
  - Fitted structural prior from 37 LLM-generated trees
  - Prior: Depth=3.08±0.88, Size=10.68±3.17, Balance=0.75±0.26
  - Implemented DistilledEvolution (no LLM calls)
- **Phase 3 COMPLETE!** 🎉
  - Distilled-LLEGO beats GATree on ALL datasets
  - breast: +2.34%, iris: +6.67%, heart: +1.11%
  - **ZERO LLM calls at runtime!**
- Next: Generate figures and write report

### 2026-01-27 (Phase 1 Analysis)
- **Phase 1 COMPLETE!** 🎉
- Key insight: Semantics don't matter, structure does
- Fitted structural prior from 37 LLM-generated trees
- Implemented DistilledEvolution (no LLM calls at runtime)
- Created Phase 2 analysis pipeline

### 2026-01-26
- Created MI analysis infrastructure
- Implemented semantic ablation experiment
- Implemented fitness attention experiment
- Tested dry-runs locally - working
- **Issue:** Turing GPUs only have 11GB VRAM, Llama 8B needs ~16GB
- **Fix:** Added 4-bit quantization support (`--load-in-4bit` flag)
  - Llama 8B in 4-bit ≈ 5GB VRAM (fits on 11GB GPU)
  - Added bitsandbytes dependency
- Ran Phase 1 experiments on cluster (~4 hours total)
