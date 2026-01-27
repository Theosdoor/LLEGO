# Experiment Log

Keep this file up to date for reproducibility. Log all significant experiments with commands, outputs, and key results.

---

## 2026-01-27: GATree Baseline Reproducibility Check

**Objective:** Verify GATree results match paper Table 1 (depth=3)

**Bug Fix Applied:** Fixed `np.int64` breaking `ast.literal_eval` in `src/llego/custom/parsing_to_dict.py` (convert to native Python int)

**Command:**
```bash
cd experiments && for seed in 0 1 2 3 4; do
  PYTHONPATH=../src ../.venv/bin/python exp_gatree.py dataset=heart-statlog max_depth=3 seed=$seed log_wandb=False exp_name=test
done
```

**Results (heart-statlog, depth=3, balanced accuracy):**

| Seed | Test Acc |
|------|----------|
| 0    | 0.730    |
| 1    | 0.729    |
| 2    | 0.638    |
| 3    | 0.756    |
| 4    | 0.748    |
| **Mean** | **0.720** |

**Paper reports:** GATree = 0.669 (std 0.031)

**Discrepancy:** Our result (0.720) is ~5% higher than paper's reported GATree (0.669).

**Possible causes:**
1. Paper's experiments may have run with the numpy bug (0 valid trees initialized)
2. Different `max_samples` (code uses 0.5, paper claims 0.25)
3. Different random state handling

**Outputs:** `experiments/test/3/heart-statlog/GATREE/`

---

## 2026-01-27: GATree with max_samples=0.25

**Objective:** Test if using paper's claimed 25% bootstrap changes results

**Change:** Modified `src/llego/custom/population_initialization.py` to use `max_samples=0.25`

**Command:**
```bash
cd experiments && for seed in 0 1 2 3 4; do
  PYTHONPATH=../src ../.venv/bin/python exp_gatree.py dataset=heart-statlog max_depth=3 seed=$seed log_wandb=False exp_name=test_025
done
```

**Results (heart-statlog, depth=3):**

| Seed | Test BA |
|------|---------|
| 0    | 0.705   |
| 1    | 0.718   |
| 2    | 0.681   |
| 3    | 0.732   |
| 4    | 0.782   |
| **Mean** | **0.724** |

**Paper reports:** GATree = 0.669

---

## Summary: max_samples Ablation

| Configuration | Mean Test BA |
|---------------|--------------|
| max_samples=0.5 (original code) | 0.720 |
| max_samples=0.25 (paper claim) | 0.724 |
| **Paper reported** | **0.669** |

**Conclusion:** Changing max_samples made **no significant difference**. The discrepancy with paper is likely due to the numpy bug causing 0 valid trees → purely random initialization.

---
