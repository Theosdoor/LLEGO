# Distilling Evolutionary Priors from Language Models

## A Mechanistic Investigation of LLM-Guided Decision Tree Evolution

---

## Executive Summary

**Core Idea:** Use mechanistic interpretability (MI) to understand *what* LLMs contribute to evolutionary decision tree induction, then **distill** those capabilities into lightweight, interpretable components that work without LLM calls.

**Why This Matters:**
1. **Scientific Understanding:** We don't know *why* LLEGO works—understanding the mechanism advances the field
2. **Healthcare Deployment:** LLM black boxes are problematic for medical applications; distilled components are auditable
3. **Practical Impact:** If successful, we get LLEGO-level performance without LLEGO-level compute costs

**Risk Acknowledgment:** This is ambitious. Distillation may not fully capture LLM capabilities. But even partial success or well-characterized failure is a valid scientific contribution.

---

## Problem Statement

**The Scientific Gap:** LLEGO demonstrates that LLMs can enhance evolutionary search for decision trees, but we have no understanding of *what computational mechanisms* enable this. Is it:
- Semantic knowledge about feature relationships?
- Implicit structural priors about "good" trees?
- Better exploration via language model diversity?
- Domain knowledge from pretraining?

**Why This Matters Beyond Efficiency:**
- Without understanding, we can't predict when LLEGO will fail
- We can't identify potential biases from LLM pretraining
- We can't deploy in high-stakes settings requiring interpretability

**Research Questions:**
1. What do LLMs actually contribute to evolutionary tree induction? (Diagnostic)
2. Can we extract and distill these contributions into lightweight, interpretable components? (Methodological)
3. Does distilled evolution achieve comparable performance without LLM calls? (Empirical)

---

## Proposed Method: Three Phases

### Phase 1: Mechanistic Diagnosis (Days 1-4)

**Goal:** Identify what LLMs contribute through controlled experiments and activation analysis.

#### Experiment 1.1: Semantic Ablation Study
**Question:** Does LLM performance depend on meaningful feature names?

```python
# Run LLEGO on same dataset with two conditions:
# Condition A: Semantic features ("age", "blood_pressure", "cholesterol")  
# Condition B: Arbitrary features ("X1", "X2", "X3")

# If performance_A >> performance_B → semantics matter
# If performance_A ≈ performance_B → structural priors dominate
```

**What we learn:** Whether to distill semantic knowledge vs structural priors

#### Experiment 1.2: Fitness Attention Analysis  
**Question:** Does the LLM actually use fitness information in crossover?

```python
from nnsight import LanguageModel

model = LanguageModel("meta-llama/Llama-3.1-8B-Instruct")

def analyze_fitness_attention(prompt_with_fitness):
    with model.trace(prompt_with_fitness) as tracer:
        # Extract attention patterns from all layers
        for layer in range(32):
            attn = model.model.layers[layer].self_attn.attn_weights.save()
    
    # Find attention from generation tokens → fitness value tokens
    fitness_token_positions = find_token_positions(prompt, ["fitness=", "0.85"])
    attention_to_fitness = aggregate_attention(attn, fitness_token_positions)
    
    return attention_to_fitness  # Low = LLM ignores fitness!
```

**What we learn:** Whether fitness-aware prompting matters

#### Experiment 1.3: Structural Prior Elicitation
**Question:** What tree structures does the LLM prefer, independent of semantics?

```python
# Generate many trees using arbitrary feature names
# Analyze distribution of:
# - Tree depth
# - Balance (left vs right subtree size)
# - Branching factor
# - Feature reuse patterns

# Compare to: random trees, human-designed trees, optimal trees
```

**What we learn:** The implicit structural prior we need to distill

#### Experiment 1.4: Activation Patching (if time)
**Question:** Which model components are causally responsible for good crossovers?

```python
def patch_experiment(good_prompt, bad_prompt, layer):
    """
    Run bad_prompt, but patch in activations from good_prompt at layer.
    If output improves → this layer is causally important.
    """
    with model.trace(bad_prompt) as tracer:
        # Get activations from good prompt
        with model.trace(good_prompt) as good_tracer:
            good_acts = model.model.layers[layer].output.save()
        
        # Patch into bad prompt
        model.model.layers[layer].output = good_acts
        patched_output = model.generate()
    
    return evaluate_crossover_quality(patched_output)
```

**What we learn:** Which layers encode crossover-relevant knowledge

---

### Phase 2: Distillation (Days 5-8)

**Goal:** Extract identified mechanisms into lightweight, interpretable components.

Based on Phase 1 findings, we distill up to two components:

#### Component A: Structural Prior Model

**If MI shows structural priors matter:**

```python
class DistilledStructuralPrior:
    """
    A lightweight model that scores tree structures.
    Learned from LLM-generated trees, but requires no LLM at inference.
    """
    
    def __init__(self):
        # Simple MLP or decision rules learned from LLM outputs
        self.depth_preference = None  # Learned from LLM tree distribution
        self.balance_preference = None
        self.feature_reuse_penalty = None
    
    def score_structure(self, tree: DecisionTree) -> float:
        """Score how much this structure matches LLM's implicit prior."""
        depth_score = self.depth_preference.score(tree.depth)
        balance_score = self.balance_preference.score(tree.balance_ratio)
        reuse_score = self.feature_reuse_penalty.score(tree.feature_reuse)
        
        return depth_score + balance_score + reuse_score
    
    @classmethod
    def fit_from_llm_outputs(cls, llm_generated_trees: list[DecisionTree]):
        """Distill structural prior from corpus of LLM-generated trees."""
        prior = cls()
        prior.depth_preference = fit_distribution([t.depth for t in llm_generated_trees])
        prior.balance_preference = fit_distribution([t.balance_ratio for t in llm_generated_trees])
        # ... etc
        return prior
```

#### Component B: Semantic Feature Graph

**If MI shows semantic knowledge matters:**

```python
class DistilledSemanticGraph:
    """
    A graph capturing feature relationships extracted from LLM.
    Encodes: "age relates to heart_disease", "cholesterol predicts risk", etc.
    """
    
    def __init__(self, feature_names: list[str]):
        self.features = feature_names
        self.relationships = {}  # (feature_i, feature_j) → strength
        self.target_relevance = {}  # feature → target relevance
    
    def compatibility_score(self, features_in_subtree: list[str]) -> float:
        """Score how semantically coherent a set of features is."""
        # Features that "go together" should be in same subtree
        total = 0
        for f1, f2 in combinations(features_in_subtree, 2):
            total += self.relationships.get((f1, f2), 0)
        return total
    
    @classmethod
    def extract_from_llm(cls, model, feature_names: list[str], target: str):
        """
        Use LLM to build semantic graph (ONE-TIME extraction).
        
        Prompt LLM: "Rate relationship strength between {f1} and {f2} 
                     for predicting {target}. Score 0-10."
        """
        graph = cls(feature_names)
        
        for f1, f2 in combinations(feature_names, 2):
            prompt = f"For predicting {target}, how related are {f1} and {f2}? Score 0-10."
            with model.trace(prompt) as tracer:
                score = extract_score_from_generation(model.generate())
            graph.relationships[(f1, f2)] = score
        
        return graph
```

#### Combined: Distilled Evolution

```python
class DistilledEvolution:
    """
    Evolutionary algorithm using distilled components instead of LLM calls.
    """
    
    def __init__(
        self, 
        structural_prior: DistilledStructuralPrior,
        semantic_graph: DistilledSemanticGraph | None = None
    ):
        self.prior = structural_prior
        self.semantics = semantic_graph
    
    def crossover(self, parent1: DecisionTree, parent2: DecisionTree) -> DecisionTree:
        """
        Crossover guided by distilled priors, NO LLM CALL.
        """
        # Generate candidate children via standard subtree crossover
        candidates = self._generate_candidates(parent1, parent2, n=10)
        
        # Score each candidate using distilled components
        scores = []
        for child in candidates:
            structure_score = self.prior.score_structure(child)
            semantic_score = (
                self.semantics.compatibility_score(child.get_features())
                if self.semantics else 0
            )
            scores.append(structure_score + semantic_score)
        
        # Return best candidate
        return candidates[np.argmax(scores)]
    
    def _generate_candidates(self, p1, p2, n=10) -> list[DecisionTree]:
        """Generate n candidate children via standard crossover."""
        candidates = []
        for _ in range(n):
            # Random subtree exchange
            child = standard_subtree_crossover(p1, p2)
            candidates.append(child)
        return candidates
```

---

### Phase 3: MVP Validation (Days 9-11)

**Goal:** Empirically validate that distilled components capture LLM contribution.

#### Experimental Setup

| Method | Description | LLM Calls |
|--------|-------------|-----------|
| **GATree** | Standard GA, no LLM | 0 |
| **LLEGO** | Original LLM-guided evolution | Many |
| **Distilled-Struct** | GA + distilled structural prior | 0 (at runtime) |
| **Distilled-Full** | GA + structural prior + semantic graph | 0 (at runtime) |

#### Datasets

| Dataset | Semantic Richness | Expected Result |
|---------|-------------------|-----------------|
| Heart Disease | High | Distilled-Full ≈ LLEGO |
| Breast Cancer | High | Distilled-Full ≈ LLEGO |
| Iris | Medium | Distilled-Struct ≈ LLEGO |
| Iris (X1,X2...) | None | Distilled-Struct ≈ LLEGO > GATree? |

#### Success Metrics

| Metric | Success Criterion |
|--------|-------------------|
| **Accuracy Gap** | Distilled achieves ≥80% of (LLEGO - GATree) improvement |
| **LLM Calls** | Distilled uses 0 LLM calls at runtime |
| **Interpretability** | Distilled priors are human-readable |

---

## Fallback Positions (Risk Mitigation)

**If distillation doesn't work:**

| Failure Mode | What We Learn | Pivot To |
|--------------|---------------|----------|
| MI reveals no clear mechanism | "LLM contribution is holistic/emergent" | Report as finding; do activation steering instead |
| Structural prior helps but semantics don't distill | Semantics require full LLM | Report partial success; Distilled-Struct is still useful |
| Distilled version << LLEGO | LLM contribution isn't separable | Report as negative result with analysis of why |

**Key insight:** Even well-characterized failure advances understanding.

> "We attempted to distill LLM contributions but found that [specific capability] is not separable because [mechanistic reason]. This suggests LLM-evolution synergies arise from [emergent property], with implications for [future work]."

This is a valid research contribution—it tells the community what *doesn't* work and why.

---

## Technical Requirements

### Model Choice
**Primary:** Llama 3.1 8B Instruct (or Qwen 2.5 7B)
- Open weights = full MI access
- Small enough for local inference
- Capable enough for the task

### MI Tooling
**Primary:** nnsight
- Clean API for activation access
- Supports any HuggingFace model
- Optional: NDIF remote execution for larger models

```python
from nnsight import LanguageModel

model = LanguageModel("meta-llama/Llama-3.1-8B-Instruct")

# Simple activation extraction
with model.trace(prompt) as tracer:
    hidden = model.model.layers[16].output.save()
    attn = model.model.layers[16].self_attn.attn_weights.save()
```

### Compute Requirements
- Phase 1 (MI): GPU with 24GB+ VRAM (A100, RTX 4090) or NDIF remote
- Phase 2 (Distillation): CPU sufficient for fitting small models
- Phase 3 (Validation): CPU sufficient for GA runs

---

## Timeline (12 Days)

| Days | Phase | Deliverables | Risk Level |
|------|-------|--------------|------------|
| 1-2 | Setup | nnsight working; LLEGO baseline running | Low |
| 3-4 | MI: Semantic ablation | Clear finding: does semantics matter? | Low |
| 5-6 | MI: Fitness analysis + structural priors | Attention results; tree distribution | Medium |
| 7-8 | Distillation: Build components | Structural prior model; semantic graph | High |
| 9-10 | Validation: Run experiments | Comparison results | Medium |
| 11-12 | Writing | Complete paper-style report | Low |

---

## Project Structure

```
llego_distilled/
├── pyproject.toml
├── README.md
├── mi_analysis/
│   ├── __init__.py
│   ├── semantic_ablation.py    # Exp 1.1: semantic vs arbitrary
│   ├── fitness_attention.py    # Exp 1.2: attention to fitness
│   ├── structural_priors.py    # Exp 1.3: tree structure analysis
│   └── activation_patching.py  # Exp 1.4: causal analysis (optional)
├── distillation/
│   ├── __init__.py
│   ├── structural_prior.py     # Component A
│   ├── semantic_graph.py       # Component B
│   └── distilled_evolution.py  # Combined algorithm
├── experiments/
│   ├── run_mi_experiments.py
│   ├── run_distillation.py
│   └── run_comparison.py
├── results/
│   ├── mi_findings/
│   └── comparison/
└── report/
    └── figures/
```

---

## Dependencies

```toml
[project]
name = "llego-distilled"
version = "0.1.0"
description = "Distilling Evolutionary Priors from Language Models"
requires-python = ">=3.11"
dependencies = [
    # Core
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "scikit-learn>=1.3.0",
    
    # MI
    "nnsight>=0.3.0",
    "transformers>=4.40.0",
    "torch>=2.0.0",
    "accelerate>=0.27.0",
    
    # Visualization
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
]
```

---

## Success Criteria

### Minimum Viable Success
- [ ] Clear MI findings on what LLM contributes (semantic vs structural)
- [ ] At least one distilled component implemented
- [ ] Some evidence it captures part of LLM contribution
- [ ] Well-written report explaining findings and method

### Full Success
- [ ] Both components distilled (structural + semantic)
- [ ] Distilled-LLEGO achieves ≥80% of LLEGO's improvement over GATree
- [ ] Zero LLM calls at runtime
- [ ] Interpretable priors that can be inspected

### Valuable Failure (Still a Contribution)
- [ ] Clear MI findings on what LLM contributes
- [ ] Evidence that contribution is NOT distillable + explanation why
- [ ] Implications for future hybrid LLM+optimization systems

---

## Key Hypotheses

| ID | Hypothesis | Test | Implication if True |
|----|------------|------|---------------------|
| H1 | LLM contribution depends on semantic feature names | Semantic ablation | Distill semantic graph |
| H2 | LLM has structural tree priors independent of semantics | Arbitrary-feature analysis | Distill structural prior |
| H3 | LLM ignores fitness values in prompts | Attention analysis + ablation | Fitness prompting is wasted |
| H4 | Good crossovers activate specific attention patterns | Activation analysis | Can predict crossover quality |

---

## Report Outline (ICML/NeurIPS Style)

### Title
"Distilling Evolutionary Priors from Language Models: A Mechanistic Analysis of LLM-Guided Decision Tree Induction"

### Abstract (150 words)
- LLM-guided evolution works but we don't understand why
- We use MI to diagnose mechanisms
- We distill into lightweight components
- Result: [X]% of performance, [0] LLM calls, [interpretable]

### 1. Introduction
- LLM+evolution is promising but opaque
- Healthcare needs interpretability
- Our contribution: understand → extract → validate

### 2. Background & Related Work
- LLEGO and LLM-guided optimization
- Mechanistic interpretability
- Knowledge distillation

### 3. Mechanistic Analysis
- Methodology: ablations, attention, activation analysis
- Findings: what LLMs contribute

### 4. Distillation Method
- Structural prior model
- Semantic feature graph
- Distilled evolution algorithm

### 5. Experiments
- Setup: datasets, baselines, metrics
- Results: comparison table + figures
- Analysis: what distilled captures vs misses

### 6. Discussion
- Implications for LLM+optimization
- Healthcare deployment considerations
- Limitations

### 7. Conclusion & Future Work
