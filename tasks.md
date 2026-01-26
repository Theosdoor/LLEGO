# LLEGO Distillation Project - Task Tracker

## Project: Distilling Evolutionary Priors from Language Models

**Goal:** Use MI to understand what LLMs contribute to decision tree evolution, then distill into lightweight components.

**Deadline:** 2nd February 2026

---

## Phase 1: Mechanistic Diagnosis (Current)

### Experiment 1.1: Semantic Ablation Study
- [x] Create experiment script (`mi_analysis/semantic_ablation.py`)
- [x] Create config and utilities
- [x] Test dry-run locally
- [ ] Run on cluster with Llama 3.1 8B
- [ ] Analyze results

**Status:** Ready to run on cluster

### Experiment 1.2: Fitness Attention Analysis
- [x] Create experiment script (`mi_analysis/fitness_attention.py`)
- [x] Implement fitness swap test
- [x] Implement fitness ablation test
- [ ] Run on cluster
- [ ] Analyze results

**Status:** Ready to run on cluster

### Experiment 1.3: Structural Prior Elicitation
- [ ] Create experiment script
- [ ] Generate trees with arbitrary features
- [ ] Analyze tree structure distributions

**Status:** Not started

---

## Phase 2: Distillation (After Phase 1)

### Component A: Structural Prior Model
- [ ] Design based on Phase 1 findings
- [ ] Implement distillation
- [ ] Validate

### Component B: Semantic Feature Graph
- [ ] Design based on Phase 1 findings
- [ ] Implement extraction
- [ ] Validate

### Integration: Distilled Evolution
- [ ] Combine components
- [ ] Run comparison experiments

---

## Phase 3: Validation & Writing

- [ ] Run MVP experiments (LLEGO vs Distilled-LLEGO vs GATree)
- [ ] Generate figures
- [ ] Write report (ICML/NeurIPS style)

---

## Infrastructure

### Files Created
- [x] `mi_analysis/__init__.py`
- [x] `mi_analysis/config.py`
- [x] `mi_analysis/utils.py`
- [x] `mi_analysis/semantic_ablation.py`
- [x] `mi_analysis/fitness_attention.py`
- [x] `mi_analysis/submit_phase1.sh`
- [x] `mi_analysis/submit_semantic_ablation.sh`
- [x] `mi_analysis/run_local_test.sh`
- [x] `mi_analysis/results/README.md`
- [x] `imp_plan.md` (full implementation plan)

### Cluster Resources
- **Available:** Turing GPUs (28GB RAM, 2 CPUs per job)
- **Model:** Llama 3.1 8B Instruct (fits in ~16-20GB VRAM)

---

## Log

### 2026-01-26
- Created MI analysis infrastructure
- Implemented semantic ablation experiment
- Implemented fitness attention experiment
- Tested dry-runs locally - working
- **Issue:** Turing GPUs only have 11GB VRAM, Llama 8B needs ~16GB
- **Fix:** Added 4-bit quantization support (`--load-in-4bit` flag)
  - Llama 8B in 4-bit ≈ 5GB VRAM (fits on 11GB GPU)
  - Added bitsandbytes dependency
- Ready to re-submit to cluster with quantization
