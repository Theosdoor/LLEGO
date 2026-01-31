---
applyTo: '**'
---

# Repository Guidelines

**NOTE:** These guidelines apply to all files in this repository and should be followed by all AI assistants.

**KEY TASK:** https://raw.githubusercontent.com/Theosdoor/vanDerSchaarWork/refs/heads/main/context/vanderscharr_task.md?token=GHSAT0AAAAAADMXCD2DSEYZOESIZS3YAJVS2L3JWGQ
- ensure we are approaching this task

## Original Repository & Paper

**Paper:** "Decision Tree Induction Through LLMs via Semantically-Aware Evolution" (ICLR 2025)
- Authors: Tennison Liu, Nicolas Huynh, Mihaela van der Schaar (DAMTP, University of Cambridge)
- OpenReview: https://openreview.net/forum?id=UyhRtB4hjN
- Paper summary: https://raw.githubusercontent.com/Theosdoor/vanDerSchaarWork/refs/heads/main/context/llego_paper_short.md

**Original Repo:** This repo was forked from the official LLEGO implementation.
- First commit (original code): `e9bb2fe` - "Update README.md"
- Initial import: `16042dd` - "Initial commit"
- Use `git diff e9bb2fe..HEAD` to see all changes made since forking.

**Key Paper Parameters (Table 1, depth=3):**
- Population: N=25, Generations: G=25
- LLM: gpt-3.5-turbo version 0301
- Initialization: CART-bootstrapped on **25%** of training data (paper says 25%)
- Fitness: Balanced accuracy for classification, MSE for regression  
- Data split: [0.2, 0.4, 0.4] for train/val/test
- Datasets: credit-g, diabetes, compas, heart-statlog, liver, breast, vehicle

**Known Issues in Original Repo:**
1. `max_samples=0.5` in `population_initialization.py` - paper claims 25% but code uses 50%
2. `np.int64` types in tree dicts break `ast.literal_eval` in `tree_validation.py` (numpy 2.x compatibility issue) - **FIXED**
3. Tree parsing fails silently, causing 0 valid trees to be initialized

## SSH & Remote Compute

For running experiments on the compute cluster, use SSH with SLURM:

**SSH Connection:**
```bash
ssh nchw73@ncc1.clients.dur.ac.uk
```
- **Working directory:** `~/vanDerSchaarWork/LLEGO`
- **Job scheduler:** SLURM (use `sbatch`, `squeue`, `scancel`, etc.)
- **Important:** SSH requires password authentication. Establish one SSH session and reuse it for all commands to avoid repeated password prompts.
- Key results will likely be saved here, rather than on the local machine.

## Experiment Logging

Keep `EXPERIMENTS.md` up to date for reproducibility.

- When adding or modifying experiment scripts, ensure they log results to `EXPERIMENTS.md` on successful completion.
- When running one-off/manual experiments, either:
  - Add a short entry to `EXPERIMENTS.md` with the command, outputs directory, and key results.
  - Or use a logging script if available.

## Project Structure & Module Organization
- `src/`: primary library/package code.
  - `llego/`: core LLEGO algorithm implementation and operators.
    - `operators/`: genetic operators (crossover, mutation, selection, etc.).
    - `utils/`: utility functions for LLEGO.
    - `custom/`: custom components and extensions.
  - `baselines/`: baseline tree induction methods (CART, C4.5, DL8.5, GOSDT, GATree).
    - `utils/`: utility functions for baselines.
  - `external/`: external dependencies (bonsai-dt, gatree, pydl8.5).
  - `utils/`: shared utilities (data loading, evaluation, saving, extraction).
- `experiments/`: experiment runners and batch scripts.
  - `exp_llego.py`: main LLEGO experiment runner.
  - `exp_baselines.py`: baseline experiments runner.
  - `exp_gatree.py`: GATree experiment runner.
  - `run_*.sh`: batch experiment scripts for reproducibility.
- `configs/`: Hydra configuration files for experiments.
  - `dataset/`: dataset-specific configurations.
  - `baseline/`: baseline method configurations.
  - `crossover/`, `mutation/`, `selection/`: operator configurations.
  - `llm_api/`, `endpoint/`: LLM API and endpoint configurations.
  - `prompts/`: prompt templates for LLM-guided operators.
- `analysis/`: analysis scripts and result processing.
  - `artifacts/`: aggregated CSV results (classification/regression summaries).
  - `results/`: raw experimental results organized by method and dataset.
- `figures/`: visualizations and diagrams for the paper.
- `paper/`: LaTeX paper source (sections, bibliography, styles).
- `tests/`: automated tests mirroring `src/` structure.

## External Data & Run Outputs (Not In Git)
Keep large datasets and model outputs outside the repository to avoid bloating git history.
- Store large datasets externally or use data loaders that download on demand.
- Experimental run outputs should be organized under `analysis/results/` for small, reproducible results that support paper figures/tables.

## Experiment Logging (Required)
For reproducibility, maintain clear records of experiments:
- Document experiment commands, configurations, and results in a structured way.
- Use descriptive experiment names (`exp_name` parameter) to organize outputs.
- When running batch experiments via `run_*.sh` scripts, ensure outputs are properly logged.
- Consider using wandb for experiment tracking (configure credentials in `.env`).

## Build, Test, and Development Commands
- Install dependencies: `uv sync` (creates venv and installs all required packages)
- Install external baselines: `bash install_external.sh`
- Run unit tests: `python -m pytest tests/`
- Run a single experiment: `python3 experiments/exp_llego.py dataset=<dataset_name> seed=<seed>`
- Run batch experiments: Use the provided shell scripts (e.g., `bash experiments/run_llego.sh`)
- Build paper PDF (requires `latexmk`): `make -C paper`

## Paper Writing Conventions
- Avoid `\\emph{...}` in paper prose; prefer plain text.
- Avoid em-dashes (`---` or `—`) in paper prose; prefer commas, parentheses, or separate sentences.

## Coding Style & Naming Conventions
Follow language-standard style and enforce it with a formatter. If you add Python, prefer `black` + `ruff`; for JS/TS, prefer `prettier` + `eslint`. Use 4-space indentation for Python and 2-space for JS/TS. Name modules in lowercase with underscores (e.g., `feature_stats.py`) and classes in `PascalCase`.

## Testing Guidelines
Place tests in `tests/` and mirror module paths. Use `test_*.py` naming. Target meaningful coverage for new logic and include at least one failure-mode test for critical paths.

## Commit & Pull Request Guidelines
Use Conventional Commits (e.g., `feat: add feature extraction pipeline`). PRs should include: concise summary, test results, linked issue (if any), and screenshots or sample outputs for user-facing changes.

## Security & Configuration
Do not commit secrets or API tokens. Use `.env` for local configuration and document required variables in `README.md` when you add them.
