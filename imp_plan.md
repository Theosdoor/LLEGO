# LLEGO-Guided: Implementation Plan

## Semantically-Selective Evolutionary Operators for Decision Tree Induction

---

## Executive Summary

**Core Idea:** Use mechanistic analysis to design **improved genetic operators** that selectively use LLM capabilities only when semantically justified, achieving comparable performance with fewer LLM calls.

**Why This Matters:**
1. **Computational Cost:** LLM calls are expensive; reducing them makes the approach practical
2. **Healthcare Bias:** Understanding *when* LLMs help prevents silent failures on sensitive data
3. **Scientific Understanding:** Principled design over black-box optimization

---

## Problem Statement

**Current Limitation:** LLEGO uses LLM-based crossover/mutation for *every* genetic operation, regardless of whether semantic knowledge is beneficial. This is:
1. **Computationally wasteful** - LLM calls when standard operators would suffice
2. **Potentially harmful** - LLM biases may hurt performance on some problem types
3. **Poorly understood** - no principled way to know when LLMs help

**Research Question:** Can we design a *selective* hybrid system that uses LLM operators only when mechanistically justified, achieving comparable or better performance with fewer LLM calls?

---

## Mechanistic Interpretability: Practical Considerations

### The Open Source Question

**Short Answer:** Yes, we need open-source/open-weight models for deep MI analysis.

| Approach | Closed Models (GPT-4, Claude) | Open Models (Llama, Mistral, Qwen) |
|----------|-------------------------------|-------------------------------------|
| Logit lens / probing | ❌ No access | ✅ Full access |
| Attention analysis | ❌ No access | ✅ Full access |
| SAE features | ❌ No access | ✅ Can use pretrained or train |
| Activation patching | ❌ No access | ✅ Full access |
| Behavioral probing | ✅ API only | ✅ Full access |
| Prompt ablation | ✅ Works | ✅ Works |

**Recommendation:** Use **Llama 3.1 8B** or **Qwen 2.5 7B** as primary model
- Small enough to run locally (fits on A100 / good consumer GPU)
- Large enough to be capable at the task
- Open weights = full interpretability access
- Can compare behavioral results to closed models

### MI Tooling Options (No Training Required)

#### Tier 1: Zero Training Overhead

| Method | What It Reveals | Tools Available | Effort |
|--------|-----------------|-----------------|--------|
| **Logit Lens** | What model "believes" at each layer | TransformerLens | Low |
| **Attention Patterns** | What tokens attend to what | TransformerLens, BertViz | Low |
| **Activation Patching** | Causal importance of components | TransformerLens, pyvene | Medium |
| **Prompt Ablation** | Which prompt parts matter | Custom (easy) | Low |
| **Behavioral Probing** | Input-output relationships | Custom (easy) | Low |

#### Tier 2: Requires Some Training

| Method | What It Reveals | Tools Available | Effort |
|--------|-----------------|-----------------|--------|
| **Linear Probes** | What info is encoded where | sklearn + hooks | Low-Medium |
| **Concept Activation Vectors** | Direction of concepts in activation space | Custom | Medium |
| **Steering Vectors** | Can we control behavior? | Custom | Medium |

#### Tier 3: Pretrained SAEs Available

| Model | SAE Availability | Source |
|-------|------------------|--------|
| Llama 3.1 8B | ✅ EleutherAI SAEs | [sae-lens](https://github.com/jbloomAus/SAELens) |
| Gemma 2 9B | ✅ Google DeepMind | [gemma-scope](https://huggingface.co/google/gemma-scope) |
| GPT-2 | ✅ Many available | Various |
| Mistral 7B | ⚠️ Limited | Community |

**Recommendation for MVP:** Start with Tier 1 methods (zero training), add probes if time permits.

---

## Proposed Method: Three Components

### Component 1: Semantic Relevance Scorer

**Purpose:** Predict whether LLM-based operators will benefit a given problem instance.

**Intuition:** LLMs should help more when feature names carry semantic meaning (e.g., "blood_pressure") vs. arbitrary names (e.g., "X1").

```python
class SemanticRelevanceScorer:
    """
    Scores how much semantic knowledge an LLM might contribute.
    
    Uses sentence embeddings to measure "meaningfulness" of feature names.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(model_name)
        
        # Reference embeddings
        self.meaningful_examples = ["age", "blood_pressure", "income", "heart_rate"]
        self.arbitrary_examples = ["X1", "feature_0", "var_a", "col1"]
    
    def score_problem(self, problem: ProblemInstance) -> dict:
        """Returns semantic score in [0,1] and LLM recommendation."""
        # Embed feature names, compare to meaningful vs arbitrary centroids
        ...
```

### Component 2: Fitness-Conditioned Prompt Engineering

**Purpose:** Make fitness information more salient to LLMs during crossover.

**MI Insight:** Standard prompts may not effectively communicate which parent is "better."

```python
class FitnessConditionedPrompter:
    """
    Three prompting strategies based on MI insights:
    
    1. original: Baseline LLEGO-style
    2. enhanced: Explicit fitness comparison, labeled better/worse
    3. contrastive: Force LLM to reason about differences first
    """
    
    def create_crossover_prompt(self, parent1, parent2, style="enhanced"):
        if style == "enhanced":
            # Make fitness differential explicit
            better, worse = sorted([parent1, parent2], key=lambda x: -x.fitness)
            return f"""
            IMPORTANT: Parent A performs {fitness_diff:.2%} BETTER than Parent B.
            
            BETTER Parent A (fitness={better.fitness:.4f}):
            {better.structure}
            
            WORSE Parent B (fitness={worse.fitness:.4f}):
            {worse.structure}
            
            Generate a child that preserves effective splits from Parent A.
            """
```

### Component 3: Adaptive Hybrid Operator Selection

**Purpose:** Dynamically choose between LLM and standard operators.

```python
class LLEGOGuided:
    """
    Main algorithm: Adaptively selects operators based on:
    1. Semantic relevance of the problem (static)
    2. Runtime performance feedback (dynamic)
    """
    
    def __init__(self, problem, fitness_fn, config):
        # Initial LLM probability from semantic score
        self.llm_probability = self.semantic_scorer.score_problem(problem)
    
    def _select_operator(self):
        if random.random() < self.llm_probability:
            return self.llm_op, "llm"
        else:
            return self.standard_op, "standard"
    
    def _update_llm_probability(self):
        # Increase if LLM outperforming, decrease otherwise
        advantage = self.stats.llm_success_rate - self.stats.standard_success_rate
        self.llm_probability += 0.05 * np.sign(advantage)
```

---

## MI Analysis Plan (Lightweight Version)

### Phase 1: Behavioral Analysis (Days 1-3)

**No model internals needed - works with any LLM**

1. **Prompt Ablation Study**
   - Remove feature names → measure performance drop
   - Remove fitness values → measure performance drop
   - Remove task description → measure performance drop
   - **Goal:** Quantify contribution of each prompt component

2. **Semantic vs Arbitrary Features**
   - Same data, meaningful vs arbitrary feature names
   - **Goal:** Confirm LLMs leverage semantics

3. **Fitness Sensitivity Test**
   - Swap fitness labels (tell LLM worse parent is better)
   - **Goal:** Does LLM actually use fitness information?

### Phase 2: Activation Analysis (Days 4-6)

**Requires open-weight model (Llama/Qwen)**

1. **Attention Pattern Analysis**
   ```python
   # Using TransformerLens
   model = HookedTransformer.from_pretrained("meta-llama/Llama-3.1-8B")
   
   with model.hooks(fwd_hooks=[(f"blocks.{layer}.attn.hook_pattern", save_hook)]):
       model(prompt)
   
   # Visualize: Do later layers attend to fitness values? Feature names?
   ```

2. **Logit Lens**
   - What does model "predict" at intermediate layers?
   - Does it settle on tree structure early or late?

3. **Activation Patching (if time)**
   - Patch activations from "good crossover" into "bad crossover" run
   - Identify which layers/heads are causal

### Phase 3: Feature Analysis (Days 7-8, Optional)

**If pretrained SAEs available for chosen model**

```python
from sae_lens import SAE

# Load pretrained SAE for Llama
sae = SAE.load_from_pretrained("EleutherAI/llama-3.1-8b-sae-layer-16")

# Get activations during crossover
activations = model.run_with_cache(prompt)[1]["blocks.16.mlp.hook_post"]

# Decode to interpretable features
features = sae.encode(activations)

# Which features activate for "good" vs "bad" crossovers?
```

---

## Experimental Design

### Datasets

| Dataset | Semantic Richness | Domain | Expected LLM Benefit |
|---------|-------------------|--------|---------------------|
| Breast Cancer | High | Medical | High |
| Heart Disease | High | Medical | High |
| COMPAS | Medium | Criminal Justice | Medium (bias concern!) |
| Iris | Medium | Biology | Medium |
| Iris (X1,X2...) | Low | - | Low |
| Synthetic | None | - | None (control) |

### Baselines

1. **LLEGO-Full:** Always use LLM (original paper)
2. **GATree:** Never use LLM (standard GA)
3. **LLEGO-Random:** 50% LLM usage (random selection)
4. **LLEGO-Guided:** Our adaptive method

### Metrics

| Metric | What It Measures |
|--------|------------------|
| Test Accuracy | Task performance |
| LLM Calls | Computational cost |
| Efficiency Ratio | Accuracy / LLM Calls |
| Fitness Trajectory | Convergence speed |

### Key Hypotheses

1. **H1:** Semantic relevance score predicts when LLMs benefit tree evolution
2. **H2:** Fitness-conditioned prompts improve LLM crossover quality
3. **H3:** LLEGO-Guided achieves ≥90% of LLEGO's fitness with ≤50% of LLM calls

---

## Implementation Timeline

| Days | Phase | Deliverables |
|------|-------|--------------|
| 1-2 | Setup & Baseline | Replicate basic GA tree evolution; implement tree representation |
| 3-4 | Component 1 | Semantic scorer with embedding-based feature analysis |
| 5-6 | Component 2 | Fitness-conditioned prompt templates; LLM client |
| 7-8 | Component 3 | Adaptive hybrid operator; integration |
| 9-10 | MVP Experiments | Run comparisons on 3-4 UCI datasets |
| 11-12 | MI Analysis | Behavioral probing + attention analysis (if time) |
| 13-14 | Writing | Report with clear problem → method → results narrative |

---

## Project Structure

```
llego_guided/
├── pyproject.toml
├── README.md
├── llego_guided/
│   ├── __init__.py
│   ├── semantic_scorer.py      # Component 1
│   ├── prompt_engineering.py   # Component 2
│   ├── hybrid_evolution.py     # Component 3
│   ├── tree_utils.py           # Decision tree parsing/generation
│   └── llm_client.py           # LLM API wrapper (supports local + API)
├── mi_analysis/
│   ├── behavioral_probing.py   # Prompt ablation, sensitivity tests
│   ├── attention_analysis.py   # TransformerLens hooks
│   └── feature_analysis.py     # SAE analysis (optional)
├── experiments/
│   ├── main_experiment.py      # MVP validation
│   └── ablation_study.py       # Component contributions
├── results/
└── tests/
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| LLM too slow for evolution | Use smaller model (Llama 8B); batch prompts; cache responses |
| MI analysis inconclusive | Focus on behavioral probing (always works); MI is means not end |
| Results don't support hypothesis | Negative results are valid! "LLMs don't help on X" is useful finding |
| Time constraints | Core MVP (Components 1-3) is achievable; MI is optional depth |

---

## Dependencies

```toml
[project]
dependencies = [
    # Core
    "numpy>=1.24.0",
    "scikit-learn>=1.3.0",
    "pandas>=2.0.0",
    
    # Semantic scoring
    "sentence-transformers>=2.2.0",
    
    # LLM
    "openai>=1.0.0",           # For API models
    "transformers>=4.40.0",    # For local models
    "accelerate>=0.27.0",
    
    # Visualization
    "matplotlib>=3.7.0",
]

[project.optional-dependencies]
mi = [
    "transformer-lens>=1.0.0",
    "torch>=2.0.0",
    "sae-lens>=0.5.0",  # For pretrained SAEs
]
```

---

## Success Criteria

### Minimum Viable Success
- [ ] LLEGO-Guided runs on 3+ datasets
- [ ] Demonstrates efficiency gain (fewer LLM calls for similar accuracy)
- [ ] Clear visualization of semantic score → LLM benefit relationship

### Stretch Goals
- [ ] MI analysis reveals interpretable mechanism
- [ ] Novel prompting strategy improves crossover quality
- [ ] Framework generalizes to other LLM+evolution hybrid problems
