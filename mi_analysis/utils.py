"""
Utility functions for MI analysis.
"""
import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import (
    load_breast_cancer, 
    load_iris, 
    load_wine,
    load_diabetes as load_diabetes_sklearn
)

logger = logging.getLogger(__name__)


# =============================================================================
# Feature Name Mapping
# =============================================================================

def get_semantic_feature_names(dataset_name: str) -> list[str]:
    """Get original semantic feature names for a dataset."""
    
    if dataset_name == "breast":
        return list(load_breast_cancer().feature_names)
    elif dataset_name == "iris":
        return list(load_iris().feature_names)
    elif dataset_name == "wine":
        return list(load_wine().feature_names)
    elif dataset_name == "heart-statlog":
        return [
            "age", "sex", "chest_pain_type", "resting_blood_pressure",
            "serum_cholesterol", "fasting_blood_sugar", "resting_ecg",
            "max_heart_rate", "exercise_induced_angina", "st_depression",
            "st_slope", "num_major_vessels", "thalassemia"
        ]
    elif dataset_name == "diabetes":
        return [
            "pregnancies", "glucose", "blood_pressure", "skin_thickness",
            "insulin", "bmi", "diabetes_pedigree_function", "age"
        ]
    elif dataset_name == "credit-g":
        return [
            "checking_status", "duration", "credit_history", "purpose",
            "credit_amount", "savings_status", "employment", "installment_commitment",
            "personal_status", "other_parties", "residence_since", "property_magnitude",
            "age", "other_payment_plans", "housing", "existing_credits",
            "job", "num_dependents", "own_telephone", "foreign_worker"
        ]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_arbitrary_feature_names(n_features: int) -> list[str]:
    """Generate arbitrary feature names (X1, X2, ..., Xn)."""
    return [f"X{i+1}" for i in range(n_features)]


def create_feature_mapping(
    semantic_names: list[str], 
    arbitrary_names: list[str]
) -> dict[str, str]:
    """Create bidirectional mapping between semantic and arbitrary names."""
    return {
        "semantic_to_arbitrary": dict(zip(semantic_names, arbitrary_names)),
        "arbitrary_to_semantic": dict(zip(arbitrary_names, semantic_names)),
    }


# =============================================================================
# Prompt Construction
# =============================================================================

def build_crossover_prompt(
    parent1_tree: dict,
    parent2_tree: dict,
    parent1_fitness: float,
    parent2_fitness: float,
    feature_names: list[str],
    feature_ranges: dict[str, tuple],
    target_name: str,
    task_description: str,
    target_fitness: float | None = None,
) -> str:
    """
    Build a crossover prompt similar to LLEGO's format.
    
    Args:
        parent1_tree: First parent tree as dict
        parent2_tree: Second parent tree as dict  
        parent1_fitness: Fitness of first parent
        parent2_fitness: Fitness of second parent
        feature_names: List of feature names to use
        feature_ranges: Dict mapping feature names to (min, max) ranges
        target_name: Name of target variable
        task_description: Description of the classification task
        target_fitness: Optional target fitness for offspring
    
    Returns:
        Formatted prompt string
    """
    # Build feature semantics string
    feature_semantics = ", ".join([
        f"{name} (range: {feature_ranges.get(name, (0, 1))})"
        for name in feature_names
    ])
    
    # Build parent tree strings
    parent1_str = json.dumps(parent1_tree, indent=2)
    parent2_str = json.dumps(parent2_tree, indent=2)
    
    # Compute target fitness if not provided
    if target_fitness is None:
        target_fitness = max(parent1_fitness, parent2_fitness) + 0.05
    
    prompt = f"""{task_description}. The features are: {feature_semantics}. The target variable is {target_name}.

Generate a decision tree that combines the best aspects of the two parent trees below, aiming for fitness of {target_fitness:.4f}.

Parent 1 (fitness={parent1_fitness:.4f}):
{parent1_str}

Parent 2 (fitness={parent2_fitness:.4f}):  
{parent2_str}

Generate a child decision tree in JSON format. Return only the JSON surrounded by ## tree ##."""

    return prompt


def replace_feature_names_in_prompt(
    prompt: str,
    old_names: list[str],
    new_names: list[str]
) -> str:
    """Replace feature names in a prompt string."""
    result = prompt
    for old, new in zip(old_names, new_names):
        # Use word boundaries to avoid partial replacements
        result = re.sub(rf'\b{re.escape(old)}\b', new, result)
    return result


def replace_feature_names_in_tree(
    tree: dict,
    old_names: list[str],
    new_names: list[str]
) -> dict:
    """Replace feature names in a tree dictionary."""
    mapping = dict(zip(old_names, new_names))
    
    def replace_recursive(node):
        if isinstance(node, dict):
            new_node = {}
            for key, value in node.items():
                # Replace feature name if it's a key we recognize
                new_key = mapping.get(key, key)
                if key == "feature":
                    new_node[key] = mapping.get(value, value)
                else:
                    new_node[new_key] = replace_recursive(value)
            return new_node
        elif isinstance(node, list):
            return [replace_recursive(item) for item in node]
        elif isinstance(node, str) and node in mapping:
            return mapping[node]
        else:
            return node
    
    return replace_recursive(tree)


# =============================================================================
# Tree Analysis
# =============================================================================

def compute_tree_depth(tree: dict) -> int:
    """Compute depth of a decision tree."""
    if tree is None or not isinstance(tree, dict):
        return 0
    
    if "leaf" in tree or "class" in tree or "value" in tree:
        return 0
    
    left_depth = compute_tree_depth(tree.get("left") or tree.get("yes"))
    right_depth = compute_tree_depth(tree.get("right") or tree.get("no"))
    
    return 1 + max(left_depth, right_depth)


def compute_tree_size(tree: dict) -> int:
    """Compute number of nodes in a decision tree."""
    if tree is None or not isinstance(tree, dict):
        return 0
    
    if "leaf" in tree or "class" in tree or "value" in tree:
        return 1
    
    left_size = compute_tree_size(tree.get("left") or tree.get("yes"))
    right_size = compute_tree_size(tree.get("right") or tree.get("no"))
    
    return 1 + left_size + right_size


def compute_tree_balance(tree: dict) -> float:
    """
    Compute balance ratio of a tree.
    Returns ratio of min(left_size, right_size) / max(left_size, right_size).
    1.0 = perfectly balanced, closer to 0 = very unbalanced.
    """
    if tree is None or not isinstance(tree, dict):
        return 1.0
    
    if "leaf" in tree or "class" in tree or "value" in tree:
        return 1.0
    
    left_size = compute_tree_size(tree.get("left") or tree.get("yes"))
    right_size = compute_tree_size(tree.get("right") or tree.get("no"))
    
    if max(left_size, right_size) == 0:
        return 1.0
    
    return min(left_size, right_size) / max(left_size, right_size)


def extract_features_used(tree: dict) -> set[str]:
    """Extract set of features used in a tree."""
    features = set()
    
    def traverse(node):
        if node is None or not isinstance(node, dict):
            return
        
        if "feature" in node:
            features.add(node["feature"])
        
        traverse(node.get("left") or node.get("yes"))
        traverse(node.get("right") or node.get("no"))
    
    traverse(tree)
    return features


# =============================================================================
# Results Saving
# =============================================================================

def save_results(results: dict, output_path: Path, name: str):
    """Save results to JSON and create summary CSV if applicable."""
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save full results as JSON
    json_path = output_path / f"{name}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Saved results to {json_path}")
    
    # If results contain tabular data, save as CSV too
    if "summary" in results and isinstance(results["summary"], (list, dict)):
        csv_path = output_path / f"{name}_summary.csv"
        df = pd.DataFrame(results["summary"])
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved summary to {csv_path}")


def load_results(output_path: Path, name: str) -> dict:
    """Load results from JSON."""
    json_path = output_path / f"{name}.json"
    with open(json_path, "r") as f:
        return json.load(f)
