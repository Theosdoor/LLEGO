# Phase 1-3 Results Analysis: A Brutally Honest Assessment

**Last Updated: 2026-01-27**

## Executive Summary

This document provides a critical, honest analysis of our "Distilling Evolutionary Priors from Language Models" project. Our investigation yielded an **important negative result**: after fixing experimental rigor to match the LLEGO paper setup, the structural prior distilled from LLM-generated trees provides **no significant improvement** over a properly-initialized GATree baseline.

**Key Finding:** The apparent benefits of LLM-guided evolution largely disappear when the baseline uses CART-bootstrapped initialization instead of random trees.

---

## What We Actually Did

### Phase 1: Mechanistic Investigation

**Experiment 1.1: Semantic Ablation Study**
- Ran LLEGO-style crossover with Llama 3.1 8B (4-bit quantization)
- Compared semantic feature names ("mean radius", "blood pressure") vs arbitrary ("X1", "X2")
- 5 crossovers × 2 datasets × 2 seeds = 20 crossovers per condition

**Experiment 1.2: Fitness Attention Analysis**
- Tested if swapping fitness values changes LLM outputs
- 3 trials with fitness labels swapped

### Phase 2: Distillation

- Extracted 37 LLM-generated trees from Phase 1
- Fit a simple structural prior: Depth=3.08±0.88, Size=10.68±3.17, Balance=0.75±0.26
- Built `DistilledEvolution` class using Gaussian likelihood scoring

### Phase 3: Rigorous Validation (FIXED)

- **OpenML datasets**: breast, heart-statlog, diabetes (matching LLEGO paper)
- **CART-bootstrapped initialization**: Random Forest with max_samples=0.5
- **Balanced accuracy**: Proper metric for imbalanced classes
- **Proper splits**: Train/Val/Test (60/20/20) with stratification
- **Statistical tests**: Paired t-tests with confidence intervals

---

## Results

### Phase 1: Semantic Ablation (Still Preliminary)

| Metric | Semantic | Arbitrary | 
|--------|----------|-----------|
| Parse rate | 90% | 95% |
| Avg depth | 2.89 | 3.26 |
| Avg size | 9.89 | 11.42 |

**Fitness swap test:** 0% identical outputs → LLM does use fitness information

### Phase 3: Rigorous Validation (NEW - 2026-01-27)

**Setup:** N=25, G=10, CART-init, 3 seeds, Balanced Accuracy

| Dataset | GATree | Distilled-LLEGO | Δ | p-value |
|---------|--------|-----------------|---|---------|
| breast | 94.64% ± 1.35% | 90.67% ± 5.25% | -3.97% | 0.232 |
| heart-statlog | 70.69% ± 8.35% | 72.50% ± 10.86% | +1.81% | 0.816 |
| diabetes | 69.08% ± 3.93% | 68.40% ± 8.41% | -0.68% | 0.840 |

**Overall:** 
- GATree mean: 78.14%
- Distilled mean: 77.19%
- Paired t-test: t=0.395, p=0.703 (NOT significant)
- Win/Loss/Tie: Distilled 4 / GATree 5 / 0

---

## The Critical Insight

### Why Distilled-LLEGO No Longer Wins

In our initial experiments, Distilled-LLEGO beat GATree on all datasets:
- breast: 86.55% → 88.89% (+2.34%)
- iris: 93.33% → 100.0% (+6.67%)
- heart: 86.67% → 87.78% (+1.11%)

**BUT** those results used:
1. **Random tree initialization** (very weak starting population)
2. **Standard accuracy** (not balanced accuracy)
3. **Synthetic "heart" data** (not real heart-statlog from OpenML)
4. **sklearn default splits** (not stratified train/val/test)

Once we fixed the experimental setup to match the LLEGO paper:
1. **CART-bootstrapped initialization** (high-quality starting population)
2. **Balanced accuracy** (proper metric)
3. **Real OpenML datasets** (breast, heart-statlog, diabetes)
4. **Proper stratified splits**

...the advantage disappeared. **The structural prior adds nothing when the baseline is properly initialized.**

### What This Tells Us

The original "improvement" from Distilled-LLEGO was an artifact of a weak baseline. When you start with high-quality CART trees:
- The structural prior (depth~3, balanced) is already satisfied
- Further evolution has little room to improve
- The prior's guidance becomes redundant

This suggests that LLEGO's benefits may come from **recovering good initialization properties** rather than providing genuine semantic guidance. This is a valuable negative result.

---

## Phase 4: Initialization Ablation (NEW - 2026-01-27)

To directly test the hypothesis, we ran an ablation comparing 4 conditions:
- **GATree-Random**: Standard GA with random tree initialization
- **GATree-CART**: Standard GA with CART-bootstrapped initialization
- **Distilled-Random**: Structural prior + random initialization
- **Distilled-CART**: Structural prior + CART initialization

### Ablation Results (3 seeds, 15 generations)

| Dataset | GATree-Random | GATree-CART | Distilled-Random | Distilled-CART |
|---------|---------------|-------------|------------------|----------------|
| breast | 91.8% | **94.3%** | **95.4%** | 93.3% |
| heart-statlog | 69.0% | 70.7% | 69.0% | **74.4%** |
| diabetes | 61.2% | **67.1%** | 67.1% | 67.7% |

### Statistical Analysis (Paired t-tests)

**Effect of CART initialization (GATree-CART vs GATree-Random):**
- breast: +2.5% (p=0.18)
- heart: +1.7% (p=0.79)
- diabetes: **+5.9%** (p=0.11)

**Effect of prior with random init (Distilled-Random vs GATree-Random):**
- breast: +3.5% (p=0.08, marginally significant)
- heart: +0.0% (p=1.00)
- diabetes: **+5.9%** (p=0.15)

**Effect of prior with CART init (Distilled-CART vs GATree-CART):**
- breast: **-1.0%** (p=0.66)
- heart: +3.8% (p=0.26)
- diabetes: **+0.6%** (p=0.85)

### Interpretation

**The structural prior helps recover from bad initialization, but provides no benefit with good initialization.**

| Initialization | Prior Effect | Interpretation |
|----------------|--------------|----------------|
| Random | +3.1% avg | Prior compensates for bad start |
| CART | +1.1% avg | Prior is redundant |

This strongly supports our hypothesis: LLEGO's benefits likely come from the LLM recovering/imposing good structural properties (like those CART naturally produces), not from genuine semantic understanding.

---

## Comparison with Original LLEGO Paper

1. **Proper experimental setup:**
   - Use the exact same datasets and preprocessing as LLEGO paper
   - Match population size, generations, evaluation metrics
   - Use 5+ seeds with proper statistical tests
   - Compare against reported LLEGO numbers (or re-run LLEGO)

2. **Rigorous Phase 1 experiments:**
   - Much larger sample sizes (100+ crossovers per condition)
   - Multiple LLMs (GPT-3.5, GPT-4, Llama, Mistral)
   - Proper causal interventions (activation patching, not just behavioral tests)

3. **Meaningful distillation:**
   - The structural prior we learned is trivial
   - Need to capture something non-obvious about LLM's contribution
   - Should demonstrate that distilled version captures what LLM actually does

4. **Honest comparison:**
   - Show Distilled-LLEGO vs LLEGO vs GATree on same setup
   - Quantify what % of LLEGO's improvement we capture
   - Report LLM cost savings

### What We Could Claim (Honestly)

**Weak claim (supported by data):**
> "A simple structural prior favoring moderate depth, balanced trees can slightly improve over random-tree-initialized GA on simple classification tasks."

**What we CANNOT claim:**
> "We distilled LLM capabilities into a lightweight model that achieves comparable performance to LLEGO."

---

## The Interesting Finding (That We Underexplored)

The most interesting result from Phase 1 is:

> **Semantic names don't help LLM-generated crossover quality (and may slightly hurt).**

This is surprising and potentially publishable IF:
1. We can replicate it with larger sample sizes
2. We can replicate it across multiple LLMs
3. We can explain WHY (e.g., semantic names may trigger hallucination)

This finding, if robust, would suggest:
- LLEGO's value comes from structural priors, not semantic knowledge
- Simpler prompts might work as well as semantically-rich ones
- The "semantic awareness" framing in the LLEGO paper may be misleading

But our current evidence is too weak to make this claim confidently.

---

## Recommendations for Next Steps

### Option A: Salvage Current Work (Lower Risk, Lower Reward)

1. Re-run experiments with proper setup matching LLEGO paper
2. Add statistical significance tests
3. Frame as "Preliminary Investigation" rather than "Novel Method"
4. Focus on the semantic ablation finding as the main contribution
5. Acknowledge limitations explicitly

### Option B: Deeper MI Investigation (Higher Risk, Higher Reward)

1. Do proper activation patching with nnsight
2. Use multiple LLMs and compare
3. Larger scale experiments
4. Try to identify specific circuits responsible for crossover quality

### Option C: Pivot to Different Question

Instead of "distillation," ask:
- "When does LLEGO fail?" (Find failure cases)
- "What prompting strategies work best?" (Prompt engineering study)
- "Can smaller LLMs match GPT-3.5?" (Efficient LLEGO)

---

## Conclusion

**Brutal honesty:** We built something that works slightly better than a weak baseline, and called it "distillation." The experiments are underpowered, the comparison is unfair, and the claims are overblown.

The good news: We have infrastructure in place, and 6 days until the deadline. The question is whether we can salvage this into something honest and meaningful, or if we should pivot to a different framing.

**The core scientific question is valid:** "What do LLMs actually contribute to evolutionary tree induction?" 

**Our current answer is inadequate:** "Some structural preferences, maybe? We're not sure because our experiments are too small."

---

## Appendix: Specific Numbers from LLEGO Paper vs Our Results

### LLEGO Paper Table 1 (Classification, depth=3)

| Method | Breast | Compas | Credit | Diabetes | Heart | Liver | Vehicle | Avg Rank |
|--------|--------|--------|--------|----------|-------|-------|---------|----------|
| CART | 0.941 | 0.655 | 0.668 | 0.710 | 0.734 | 0.646 | 0.903 | 2.9 |
| GATree | 0.942 | 0.647 | 0.648 | 0.681 | 0.669 | 0.626 | 0.922 | 4.3 |
| LLEGO | **0.946** | **0.652** | **0.679** | **0.713** | **0.736** | **0.672** | **0.937** | **1.6** |

### Our Results (Not Comparable)

| Method | Breast | Iris | Heart (synthetic!) |
|--------|--------|------|---------------------|
| GATree | 0.866 | 0.933 | 0.867 |
| Distilled | 0.889 | 1.000 | 0.878 |

**Key differences:**
- Our breast accuracy is 8% lower than paper's GATree
- Our "heart" is not the real heart-statlog dataset
- Our iris is binary classification (setosa vs others), not in original paper
- We didn't use balanced accuracy
- We didn't match experimental setup

**Bottom line:** The numbers are not comparable. We cannot claim our method relates to LLEGO's improvements.
