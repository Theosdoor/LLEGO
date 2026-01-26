# Mechanistic Interpretability Analysis for LLEGO
# 
# This module contains tools for understanding how LLMs contribute to
# evolutionary decision tree induction.
#
# Phase 1: Diagnostic experiments
# - semantic_ablation.py: Does LLM performance depend on meaningful feature names?
# - fitness_attention.py: Does the LLM attend to fitness information?
# - structural_priors.py: What tree structures does the LLM prefer?
#
# Phase 2: Distillation (after MI findings)
# - structural_prior.py: Extract structural preferences
# - semantic_graph.py: Extract feature relationship knowledge
