# Repository Guidelines

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
- Run a single experiment: `uv run python experiments/exp_llego.py dataset=<dataset_name> seed=<seed>`
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
No Git history is available in this repo. Use Conventional Commits (e.g., `feat: add feature extraction pipeline`). PRs should include: concise summary, test results, linked issue (if any), and screenshots or sample outputs for user-facing changes.

## Security & Configuration
Do not commit secrets or API tokens. Use `.env` for local configuration and document required variables in `README.md` when you add them.
