# MI Analysis Results

This directory contains results from mechanistic interpretability experiments.

## Directory Structure

```
results/
├── phase1/                    # Phase 1: Diagnostic experiments
│   ├── semantic_ablation_*.json
│   ├── fitness_attention_*.json
│   └── structural_priors_*.json
├── phase2/                    # Phase 2: Distillation (after Phase 1 findings)
│   └── ...
└── test/                      # Test/dry-run outputs
    └── ...
```

## Key Files

### Phase 1 Results

- `semantic_ablation_*.json`: Results from semantic vs arbitrary feature name experiment
  - Key metric: `analysis.semantic_advantage.parse_success_diff`
  - If positive: LLMs benefit from semantic feature names
  
- `fitness_attention_*.json`: Results from fitness swap/ablation experiments
  - Key metric: `swap_experiment.summary.identical_rate`
  - If high (>0.5): LLM ignores fitness values (major finding!)

## Interpreting Results

### Semantic Ablation

| Finding | Interpretation | Action |
|---------|----------------|--------|
| semantic >> arbitrary | LLM uses semantic knowledge | Distill semantic graph |
| semantic ≈ arbitrary | Structural priors dominate | Distill structural prior |

### Fitness Attention

| Finding | Interpretation | Action |
|---------|----------------|--------|
| High identical rate | LLM ignores fitness | Need fitness-conditioned prompting |
| Low identical rate | LLM uses fitness | Current prompts work |

## Analysis Commands

```python
# Load results
import json
from pathlib import Path

results_dir = Path("mi_analysis/results/phase1")
with open(results_dir / "semantic_ablation_TIMESTAMP.json") as f:
    semantic_results = json.load(f)

# Check semantic advantage
print(semantic_results["analysis"]["semantic_advantage"])
```
