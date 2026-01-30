# LLEGO SAE Project - Task Tracker

## Goal
Use SAEs to extract semantic priors **once**, replacing expensive repeated LLM calls.

**Key Insight:** LLEGO paper shows semantics help (ablation in §5.3), but current approach requires LLM calls during optimization. We extract the same semantic knowledge upfront.

---

## Phase 1: SAE Extraction Pipeline
- [x] Fix `nb_main.py` ✅
- [x] Create `sae_semantic_prior.py` with `SAESemanticPrior` class ✅
- [x] Create `extract_sae_priors.py` extraction script ✅
- [ ] Test extraction on Heart Disease features (needs GPU)
- [ ] Generate similarity matrices for 3 datasets (needs GPU)

## Phase 2: Integration
- [x] Modify `DistilledEvolution` to accept semantic prior ✅
- [x] Add semantic coherence scoring to candidate selection ✅
- [ ] Test integration works (needs priors generated)

## Phase 3: Validation
- [x] Create `sae_validation.py` 4-way comparison script ✅
- [ ] Run experiments: GATree / Distilled-Struct / Distilled-SAE / Distilled-Full
- [ ] 3 datasets × 3 seeds
- [ ] Generate comparison table

## Phase 4: Write-up
- [ ] Figures: similarity heatmaps, convergence curves
- [ ] Paper sections

---

## Log

### 2026-01-30
- Fixed `nb_main.py` (dtype warning, missing function, hook_point detection)
- Reviewed LLEGO paper: semantics DO help, just modestly
- Created `sae_semantic_prior.py` with full SAE extraction pipeline
- Modified `distillation.py` to support semantic priors
- Created `sae_validation.py` for 4-way comparison
- Created `extract_sae_priors.py` for batch extraction
