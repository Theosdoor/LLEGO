# Phase 1-3 Results Analysis: A Brutally Honest Assessment

## Executive Summary

This document provides a critical, honest analysis of our "Distilling Evolutionary Priors from Language Models" project. While we achieved some interesting results, **the findings do not yet constitute a publishable research contribution** and have significant limitations compared to the original LLEGO paper.

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

### Phase 3: Validation

- Compared GATree (baseline) vs Distilled-LLEGO on 3 datasets
- 3 seeds, 20 generations, population size 50

---

## Our Results

### Phase 1 Key Findings

| Metric | Semantic | Arbitrary | 
|--------|----------|-----------|
| Parse rate | 90% | 95% |
| Avg depth | 2.89 | 3.26 |
| Avg size | 9.89 | 11.42 |

**Fitness swap test:** 0% identical outputs → LLM does use fitness information

### Phase 3 Validation Results

| Dataset | GATree | Distilled-LLEGO | Δ |
|---------|--------|-----------------|---|
| breast | 86.55% | 88.89% | +2.34% |
| iris | 93.33% | 100.0% | +6.67% |
| heart | 86.67% | 87.78% | +1.11% |

---

## Critical Comparison with Original LLEGO Paper

### The Original LLEGO Results (Table 1 in paper)

**Classification tasks at depth=3:**

| Dataset | CART | GATree | LLEGO |
|---------|------|--------|-------|
| Breast | 94.1% | 94.2% | **94.6%** |
| Heart | 73.4% | 66.9% | **73.6%** |
| Diabetes | 71.0% | 68.1% | **71.3%** |
| Liver | 64.6% | 62.6% | **67.2%** |

**Key observations from the paper:**
- LLEGO achieves **average rank 1.6** across 7 datasets
- LLEGO beats GATree by small but consistent margins (~1-5%)
- LLEGO's advantage is most pronounced on harder datasets

### Why Our Results Don't Match Up

#### Problem 1: We're Testing the Wrong Hypothesis

**LLEGO paper claim:** LLMs provide semantic priors + fitness-guided search → better trees

**Our finding:** Semantic names don't help; arbitrary names work equally well

**The disconnect:** This is actually an interesting finding, BUT:
- The LLEGO paper uses GPT-3.5-turbo, we used Llama 3.1 8B (4-bit quantized)
- Different models may have different semantic capabilities
- Our sample size (37 trees) is too small to make strong claims
- We didn't test if semantics matter for the *final tree quality*, only for parsing success

#### Problem 2: Incomparable Experimental Setup

| Aspect | LLEGO Paper | Our Experiments |
|--------|-------------|-----------------|
| LLM | GPT-3.5-turbo | Llama 3.1 8B (4-bit) |
| Population | 25 | 50 |
| Generations | 25 | 20 |
| Seeds | 5 | 3 |
| Datasets | 7 classification + 5 regression | 3 (breast, iris, heart) |
| Evaluation | Balanced accuracy on proper test split | Accuracy on sklearn default split |
| Initialization | CART bootstrapped on 25% data | Random trees |
| Search time | 10 minutes wall clock | Unbounded |
| Hyperparameters | α=0.1, τ=10, ν=4 | Not applicable (no LLM at runtime) |

#### Problem 3: Our "Distilled-LLEGO" is NOT Comparable to LLEGO

The original LLEGO uses:
1. **Fitness-guided crossover**: Conditions generation on target fitness f*
2. **Diversity-guided mutation**: Uses log-probabilities to explore
3. **Higher arity**: 4 parents per operation
4. **Rich prompts**: Full task context, feature descriptions, fitness values

Our Distilled-LLEGO uses:
1. **Simple Gaussian scoring**: Score = weighted sum of (depth, size, balance, reuse) deviations
2. **Standard subtree crossover**: Random subtree exchange, no semantic guidance
3. **Binary arity**: Only 2 parents per crossover
4. **No LLM at all**: Just a fixed structural prior

**The structural prior we learned is essentially just:**
```
Prefer trees with:
- Depth ≈ 3
- Size ≈ 11 nodes
- Balanced structure
```

This is **trivial domain knowledge** that any decision tree practitioner would know. It's not a meaningful "distillation" of LLM capabilities.

#### Problem 4: Our Baseline (GATree) is Weak

Looking at our results:
- Breast: Our GATree = 86.55%, Paper's GATree = 94.2%
- Heart: Our "heart" is synthetic data, not the real heart-statlog dataset

**Why the gap?**
1. We initialized with random trees; paper uses CART-bootstrapped initialization
2. We used default sklearn split; paper uses proper cross-validation
3. Our "heart" dataset is synthetic (`np.random.randn`)
4. We didn't tune hyperparameters

#### Problem 5: Statistical Significance is Questionable

- 3 seeds is too few for reliable statistical inference
- Standard deviations on our results are large (e.g., heart: 10.84%)
- No statistical tests reported (t-tests, confidence intervals)

---

## What Would Make This a Valid Research Contribution?

### Minimum Bar for Publication

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
