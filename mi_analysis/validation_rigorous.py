"""
Phase 3: Rigorous Validation Experiments

This validation script matches the LLEGO paper experimental setup:
- OpenML datasets (breast, heart-statlog, diabetes)
- Balanced accuracy metric
- CART-bootstrapped initialization (Random Forest with max_samples=0.5)
- Proper train/val/test splits
- N=25 population, G=25 generations
- 5 seeds with statistical significance tests
"""

import argparse
import copy
import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import openml
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier as SKDecisionTreeClassifier

# Import our distillation module
from distillation import DistilledEvolution, StructuralPrior

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# OpenML Dataset Loading (Matches LLEGO Paper)
# =============================================================================

OPENML_DATASETS = {
    # Dataset name: (OpenML ID, target_attribute)
    "breast": 15,  # Breast Cancer Wisconsin
    "heart-statlog": 53,  # Heart Disease (Statlog)
    "diabetes": 37,  # Pima Indians Diabetes
}


def load_openml_dataset(
    name: str,
    test_size: float = 0.2,
    val_size: float = 0.2,
    seed: int = 42,
) -> Dict:
    """
    Load dataset from OpenML with proper preprocessing.
    
    Returns dict with:
        X_train, y_train: Training data (60%)
        X_val, y_val: Validation data (20%) - for fitness evaluation
        X_test, y_test: Test data (20%) - final evaluation
        feature_names: List of feature names
    """
    if name not in OPENML_DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(OPENML_DATASETS.keys())}")
    
    dataset_id = OPENML_DATASETS[name]
    
    logger.info(f"Loading {name} from OpenML (ID: {dataset_id})...")
    
    # Fetch from OpenML
    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, categorical_mask, feature_names = dataset.get_data(
        target=dataset.default_target_attribute
    )
    
    # Convert to numpy
    X = X.values if hasattr(X, 'values') else np.array(X)
    y = y.values if hasattr(y, 'values') else np.array(y)
    
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)
    
    # Encode labels if needed
    if y.dtype == object or isinstance(y[0], str):
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    # Ensure binary classification
    unique_classes = np.unique(y)
    if len(unique_classes) != 2:
        # Take first two classes only
        mask = np.isin(y, unique_classes[:2])
        X, y = X[mask], y[mask]
        y = LabelEncoder().fit_transform(y)
    
    # Normalize features to [0, 100] range for threshold compatibility
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # Convert to 0-100 range
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8) * 100
    
    # Train/val/test split
    # First split: train+val (80%) vs test (20%)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    
    # Second split: train (75% of trainval = 60%) vs val (25% of trainval = 20%)
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_ratio, random_state=seed, stratify=y_trainval
    )
    
    # Clean feature names
    feature_names = [f.replace(" ", "_").replace("-", "_") for f in feature_names]
    
    logger.info(f"  Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
    logger.info(f"  Class balance - Train: {y_train.mean():.2f}, Val: {y_val.mean():.2f}, Test: {y_test.mean():.2f}")
    
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "feature_names": feature_names,
    }


# =============================================================================
# Decision Tree Representation (JSON format)
# =============================================================================

def compute_depth(tree: dict) -> int:
    """Compute tree depth."""
    if "class" in tree:
        return 0
    return 1 + max(
        compute_depth(tree.get("left", {"class": 0})),
        compute_depth(tree.get("right", {"class": 0})),
    )


def compute_size(tree: dict) -> int:
    """Compute number of nodes in tree."""
    if "class" in tree:
        return 1
    return 1 + compute_size(tree.get("left", {"class": 0})) + compute_size(tree.get("right", {"class": 0}))


class TreeClassifier:
    """Decision tree classifier from JSON representation."""
    
    def __init__(self, tree_dict: dict):
        self.tree = tree_dict
    
    def _predict_single(self, x: np.ndarray, feature_names: list) -> int:
        node = self.tree
        while "class" not in node:
            feature = node["feature"]
            threshold = node["threshold"]
            
            try:
                idx = feature_names.index(feature)
            except ValueError:
                return 0
            
            if x[idx] <= threshold:
                node = node.get("left", {"class": 0})
            else:
                node = node.get("right", {"class": 0})
        
        return node["class"]
    
    def predict(self, X: np.ndarray, feature_names: list) -> np.ndarray:
        return np.array([self._predict_single(x, feature_names) for x in X])
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, feature_names: list) -> float:
        """Evaluate using BALANCED ACCURACY (matching LLEGO paper)."""
        preds = self.predict(X, feature_names)
        return balanced_accuracy_score(y, preds)


# =============================================================================
# CART-Bootstrapped Initialization (Matching LLEGO Paper)
# =============================================================================

def sklearn_tree_to_dict(
    sk_tree: SKDecisionTreeClassifier,
    feature_names: list,
    node_id: int = 0,
) -> dict:
    """Convert sklearn DecisionTreeClassifier to dict format."""
    tree = sk_tree.tree_
    
    # Leaf node
    if tree.children_left[node_id] == tree.children_right[node_id]:
        # Get majority class
        class_counts = tree.value[node_id][0]
        majority_class = int(np.argmax(class_counts))
        return {"class": majority_class}
    
    # Internal node
    feature_idx = tree.feature[node_id]
    threshold = tree.threshold[node_id]
    
    # Scale threshold to 0-100 range (assuming data was normalized)
    threshold_scaled = float(threshold)
    
    return {
        "feature": feature_names[feature_idx],
        "threshold": round(threshold_scaled, 2),
        "left": sklearn_tree_to_dict(sk_tree, feature_names, tree.children_left[node_id]),
        "right": sklearn_tree_to_dict(sk_tree, feature_names, tree.children_right[node_id]),
    }


def cart_bootstrap_initialization(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list,
    pop_size: int,
    max_depth: int = 4,
    seed: int = 42,
) -> List[dict]:
    """
    CART-bootstrapped initialization matching LLEGO paper.
    
    Uses Random Forest with max_samples=0.5 to create diverse population
    of decision trees bootstrapped from training data.
    """
    logger.info(f"Initializing population with CART-bootstrap (N={pop_size}, max_depth={max_depth})")
    
    # Create Random Forest to generate diverse population
    # max_samples=0.5 means each tree sees 50% of training data (bootstrap)
    rf = RandomForestClassifier(
        n_estimators=pop_size,
        max_depth=max_depth,
        max_samples=0.5,  # Key parameter from LLEGO
        random_state=seed,
        bootstrap=True,
    )
    rf.fit(X_train, y_train)
    
    # Convert each tree to dict format
    population = []
    for sk_tree in rf.estimators_:
        tree_dict = sklearn_tree_to_dict(sk_tree, feature_names)
        population.append(tree_dict)
    
    return population


# =============================================================================
# GA Components
# =============================================================================

def tournament_selection(population: list[dict], fitnesses: list[float], k: int = 3) -> dict:
    """Select individual via tournament selection."""
    indices = random.sample(range(len(population)), min(k, len(population)))
    best_idx = max(indices, key=lambda i: fitnesses[i])
    return copy.deepcopy(population[best_idx])


def standard_crossover(parent1: dict, parent2: dict) -> dict:
    """Standard subtree crossover (no structural prior)."""
    child = copy.deepcopy(parent1)
    
    if random.random() < 0.5:
        child = copy.deepcopy(parent2)
    
    if "class" not in child and random.random() < 0.5:
        if "class" not in parent2:
            if random.random() < 0.5:
                child["left"] = copy.deepcopy(parent2.get("left", {"class": 0}))
            else:
                child["right"] = copy.deepcopy(parent2.get("right", {"class": 0}))
    
    return child


def standard_mutation(
    tree: dict,
    feature_names: list[str],
    threshold_ranges: Dict[str, Tuple[float, float]],
    mutation_rate: float = 0.1,
) -> dict:
    """Standard mutation with proper threshold ranges."""
    tree = copy.deepcopy(tree)
    
    if random.random() > mutation_rate:
        return tree
    
    if "class" in tree:
        if random.random() < 0.5:
            tree["class"] = 1 - tree["class"]
        else:
            feature = random.choice(feature_names)
            low, high = threshold_ranges.get(feature, (0, 100))
            tree = {
                "feature": feature,
                "threshold": random.uniform(low, high),
                "left": {"class": 0},
                "right": {"class": 1},
            }
    else:
        choice = random.choice(["feature", "threshold", "prune", "recurse"])
        
        if choice == "feature":
            tree["feature"] = random.choice(feature_names)
        elif choice == "threshold":
            low, high = threshold_ranges.get(tree["feature"], (0, 100))
            tree["threshold"] = random.uniform(low, high)
        elif choice == "prune":
            tree = {"class": random.choice([0, 1])}
        else:
            if random.random() < 0.5:
                tree["left"] = standard_mutation(tree["left"], feature_names, threshold_ranges, mutation_rate)
            else:
                tree["right"] = standard_mutation(tree["right"], feature_names, threshold_ranges, mutation_rate)
    
    return tree


# =============================================================================
# Evolution Algorithms
# =============================================================================

def run_gatree(
    data: Dict,
    pop_size: int = 25,
    n_generations: int = 25,
    max_depth: int = 4,
    use_cart_init: bool = True,
    seed: int = 42,
) -> dict:
    """
    Run standard GATree (no LLM, no structural prior).
    
    Matches LLEGO paper setup:
    - CART-bootstrapped initialization
    - Balanced accuracy fitness
    - N=25, G=25
    """
    logger.info("Running GATree (baseline)...")
    start_time = time.time()
    
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    feature_names = data["feature_names"]
    
    # Compute threshold ranges from training data
    threshold_ranges = {}
    for i, fname in enumerate(feature_names):
        threshold_ranges[fname] = (X_train[:, i].min(), X_train[:, i].max())
    
    # Initialize population (CART-bootstrap or random)
    if use_cart_init:
        population = cart_bootstrap_initialization(
            X_train, y_train, feature_names, pop_size, max_depth, seed
        )
    else:
        population = [
            random_tree(feature_names, threshold_ranges, max_depth)
            for _ in range(pop_size)
        ]
    
    # Evaluation function (on VALIDATION set, matching LLEGO)
    def evaluate(tree: dict) -> float:
        clf = TreeClassifier(tree)
        return clf.evaluate(X_val, y_val, feature_names)
    
    fitnesses = [evaluate(t) for t in population]
    
    best_fitness_history = []
    avg_fitness_history = []
    
    for gen in range(n_generations):
        new_population = []
        
        # Elitism: keep best
        best_idx = np.argmax(fitnesses)
        new_population.append(copy.deepcopy(population[best_idx]))
        
        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            
            child = standard_crossover(parent1, parent2)
            child = standard_mutation(child, feature_names, threshold_ranges)
            
            new_population.append(child)
        
        population = new_population
        fitnesses = [evaluate(t) for t in population]
        
        best_fitness = max(fitnesses)
        avg_fitness = np.mean(fitnesses)
        best_fitness_history.append(best_fitness)
        avg_fitness_history.append(avg_fitness)
        
        if (gen + 1) % 5 == 0:
            logger.info(f"  Gen {gen+1}: best_val_bacc={best_fitness:.4f}, avg={avg_fitness:.4f}")
    
    # Get best tree
    best_idx = np.argmax(fitnesses)
    best_tree = population[best_idx]
    
    # Final evaluation on TEST set
    clf = TreeClassifier(best_tree)
    test_bacc = clf.evaluate(X_test, y_test, feature_names)
    train_bacc = clf.evaluate(X_train, y_train, feature_names)
    
    elapsed = time.time() - start_time
    
    return {
        "method": "GATree",
        "train_bacc": train_bacc,
        "val_bacc": max(fitnesses),
        "test_bacc": test_bacc,
        "best_tree": best_tree,
        "tree_depth": compute_depth(best_tree),
        "tree_size": compute_size(best_tree),
        "n_generations": n_generations,
        "pop_size": pop_size,
        "elapsed_seconds": elapsed,
        "llm_calls": 0,
        "fitness_history": best_fitness_history,
    }


def random_tree(
    feature_names: list[str],
    threshold_ranges: Dict[str, Tuple[float, float]],
    max_depth: int = 3,
    current_depth: int = 0,
) -> dict:
    """Generate a random decision tree with proper threshold ranges."""
    if current_depth >= max_depth or (current_depth > 0 and random.random() < 0.3):
        return {"class": random.choice([0, 1])}
    
    feature = random.choice(feature_names)
    low, high = threshold_ranges.get(feature, (0, 100))
    threshold = random.uniform(low, high)
    
    return {
        "feature": feature,
        "threshold": threshold,
        "left": random_tree(feature_names, threshold_ranges, max_depth, current_depth + 1),
        "right": random_tree(feature_names, threshold_ranges, max_depth, current_depth + 1),
    }


def run_distilled_llego(
    data: Dict,
    structural_prior: StructuralPrior,
    pop_size: int = 25,
    n_generations: int = 25,
    max_depth: int = 4,
    use_cart_init: bool = True,
    seed: int = 42,
) -> dict:
    """
    Run Distilled-LLEGO (structural prior, no LLM calls).
    
    Same setup as GATree but uses learned structural prior for crossover.
    """
    logger.info("Running Distilled-LLEGO...")
    start_time = time.time()
    
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    feature_names = data["feature_names"]
    
    threshold_ranges = {}
    for i, fname in enumerate(feature_names):
        threshold_ranges[fname] = (X_train[:, i].min(), X_train[:, i].max())
    
    # Initialize distilled evolution
    distilled_evo = DistilledEvolution(
        structural_prior=structural_prior,
        n_candidates=10,
        fitness_weight=0.7,
        structure_weight=0.3,
    )
    
    # Initialize population
    if use_cart_init:
        population = cart_bootstrap_initialization(
            X_train, y_train, feature_names, pop_size, max_depth, seed
        )
    else:
        population = [
            random_tree(feature_names, threshold_ranges, max_depth)
            for _ in range(pop_size)
        ]
    
    def evaluate(tree: dict) -> float:
        clf = TreeClassifier(tree)
        return clf.evaluate(X_val, y_val, feature_names)
    
    fitnesses = [evaluate(t) for t in population]
    
    best_fitness_history = []
    avg_fitness_history = []
    
    for gen in range(n_generations):
        new_population = []
        
        # Elitism
        best_idx = np.argmax(fitnesses)
        new_population.append(copy.deepcopy(population[best_idx]))
        
        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            f1 = evaluate(parent1)
            f2 = evaluate(parent2)
            
            # DISTILLED crossover (uses structural prior)
            child = distilled_evo.crossover(
                parent1, parent2, f1, f2,
                evaluate_fn=evaluate,
            )
            
            # Distilled mutation
            if random.random() < 0.1:
                child = distilled_evo.mutation(child, feature_names, threshold_ranges)
            
            new_population.append(child)
        
        population = new_population
        fitnesses = [evaluate(t) for t in population]
        
        best_fitness = max(fitnesses)
        avg_fitness = np.mean(fitnesses)
        best_fitness_history.append(best_fitness)
        avg_fitness_history.append(avg_fitness)
        
        if (gen + 1) % 5 == 0:
            logger.info(f"  Gen {gen+1}: best_val_bacc={best_fitness:.4f}, avg={avg_fitness:.4f}")
    
    # Get best tree
    best_idx = np.argmax(fitnesses)
    best_tree = population[best_idx]
    
    # Final evaluation
    clf = TreeClassifier(best_tree)
    test_bacc = clf.evaluate(X_test, y_test, feature_names)
    train_bacc = clf.evaluate(X_train, y_train, feature_names)
    
    elapsed = time.time() - start_time
    
    return {
        "method": "Distilled-LLEGO",
        "train_bacc": train_bacc,
        "val_bacc": max(fitnesses),
        "test_bacc": test_bacc,
        "best_tree": best_tree,
        "tree_depth": compute_depth(best_tree),
        "tree_size": compute_size(best_tree),
        "n_generations": n_generations,
        "pop_size": pop_size,
        "elapsed_seconds": elapsed,
        "llm_calls": 0,
        "fitness_history": best_fitness_history,
    }


# =============================================================================
# Statistical Tests
# =============================================================================

def paired_ttest(scores1: List[float], scores2: List[float]) -> Tuple[float, float]:
    """Paired t-test for comparing two methods across seeds."""
    t_stat, p_value = stats.ttest_rel(scores1, scores2)
    return t_stat, p_value


def compute_confidence_interval(scores: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """Compute confidence interval for mean."""
    n = len(scores)
    mean = np.mean(scores)
    se = stats.sem(scores)
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean - h, mean + h


# =============================================================================
# Main Validation
# =============================================================================

def run_validation(
    datasets: List[str],
    prior_path: Path,
    n_seeds: int = 5,
    pop_size: int = 25,
    n_generations: int = 25,
    output_dir: Path = Path("mi_analysis/results/phase3"),
) -> Tuple[pd.DataFrame, List[dict]]:
    """Run rigorous validation experiments matching LLEGO paper setup."""
    
    # Load structural prior
    if prior_path.exists():
        prior = StructuralPrior.load(prior_path)
        logger.info(f"Loaded structural prior from {prior_path}")
        logger.info(f"  Prior - depth: {prior.depth_mean:.2f}±{prior.depth_std:.2f}, "
                   f"size: {prior.size_mean:.2f}±{prior.size_std:.2f}, "
                   f"balance: {prior.balance_mean:.2f}±{prior.balance_std:.2f}")
    else:
        logger.warning(f"Prior not found at {prior_path}, using default")
        prior = StructuralPrior()
    
    results = []
    
    for dataset in datasets:
        logger.info(f"\n{'='*70}")
        logger.info(f"Dataset: {dataset}")
        logger.info(f"{'='*70}")
        
        for seed in range(n_seeds):
            logger.info(f"\n--- Seed: {seed} ---")
            random.seed(seed)
            np.random.seed(seed)
            
            # Load data with this seed
            data = load_openml_dataset(dataset, seed=seed)
            
            # Run GATree (baseline)
            ga_result = run_gatree(
                data,
                pop_size=pop_size,
                n_generations=n_generations,
                use_cart_init=True,
                seed=seed,
            )
            ga_result["dataset"] = dataset
            ga_result["seed"] = seed
            results.append(ga_result)
            
            # Run Distilled-LLEGO
            distilled_result = run_distilled_llego(
                data,
                structural_prior=prior,
                pop_size=pop_size,
                n_generations=n_generations,
                use_cart_init=True,
                seed=seed,
            )
            distilled_result["dataset"] = dataset
            distilled_result["seed"] = seed
            results.append(distilled_result)
    
    # Create results dataframe
    df = pd.DataFrame([{
        "dataset": r["dataset"],
        "seed": r["seed"],
        "method": r["method"],
        "train_bacc": r["train_bacc"],
        "val_bacc": r["val_bacc"],
        "test_bacc": r["test_bacc"],
        "tree_depth": r["tree_depth"],
        "tree_size": r["tree_size"],
        "elapsed_seconds": r["elapsed_seconds"],
        "llm_calls": r["llm_calls"],
    } for r in results])
    
    return df, results


def print_results_with_stats(df: pd.DataFrame, datasets: List[str]):
    """Print results with statistical significance tests."""
    
    print("\n" + "=" * 80)
    print("RIGOROUS VALIDATION RESULTS")
    print("(Matching LLEGO Paper Setup: N=25, G=25, CART-init, Balanced Accuracy)")
    print("=" * 80)
    
    # Summary table
    summary = df.groupby(["dataset", "method"]).agg({
        "test_bacc": ["mean", "std"],
        "tree_depth": ["mean", "std"],
        "elapsed_seconds": "mean",
    }).round(4)
    
    print("\nSummary Statistics:")
    print(summary.to_string())
    
    # Statistical significance tests
    print("\n" + "-" * 80)
    print("STATISTICAL SIGNIFICANCE TESTS (Paired t-test)")
    print("-" * 80)
    
    for dataset in datasets:
        ga_scores = df[(df["dataset"] == dataset) & (df["method"] == "GATree")]["test_bacc"].values
        dist_scores = df[(df["dataset"] == dataset) & (df["method"] == "Distilled-LLEGO")]["test_bacc"].values
        
        ga_mean = np.mean(ga_scores)
        dist_mean = np.mean(dist_scores)
        
        ga_ci = compute_confidence_interval(ga_scores)
        dist_ci = compute_confidence_interval(dist_scores)
        
        t_stat, p_value = paired_ttest(ga_scores, dist_scores)
        
        improvement = dist_mean - ga_mean
        sig_marker = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
        
        print(f"\n{dataset}:")
        print(f"  GATree:      {ga_mean:.4f} ± {np.std(ga_scores):.4f}  95%CI: [{ga_ci[0]:.4f}, {ga_ci[1]:.4f}]")
        print(f"  Distilled:   {dist_mean:.4f} ± {np.std(dist_scores):.4f}  95%CI: [{dist_ci[0]:.4f}, {dist_ci[1]:.4f}]")
        print(f"  Difference:  {improvement:+.4f} (t={t_stat:.3f}, p={p_value:.4f}) {sig_marker}")
    
    # Overall comparison
    print("\n" + "-" * 80)
    print("OVERALL COMPARISON")
    print("-" * 80)
    
    all_ga = df[df["method"] == "GATree"]["test_bacc"].values
    all_dist = df[df["method"] == "Distilled-LLEGO"]["test_bacc"].values
    
    overall_t, overall_p = stats.ttest_rel(all_ga, all_dist)
    
    print(f"Overall GATree mean:      {np.mean(all_ga):.4f}")
    print(f"Overall Distilled mean:   {np.mean(all_dist):.4f}")
    print(f"Overall paired t-test:    t={overall_t:.3f}, p={overall_p:.4f}")
    
    # Win/loss/tie counts
    wins = np.sum(all_dist > all_ga)
    losses = np.sum(all_dist < all_ga)
    ties = np.sum(all_dist == all_ga)
    
    print(f"\nWin/Loss/Tie:  Distilled wins {wins}, GATree wins {losses}, Ties {ties}")


def main():
    parser = argparse.ArgumentParser(description="Phase 3 Rigorous Validation")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["breast", "heart-statlog", "diabetes"],
        choices=list(OPENML_DATASETS.keys()),
        help="Datasets to evaluate on (from OpenML)",
    )
    parser.add_argument(
        "--prior-path",
        type=Path,
        default=Path("mi_analysis/results/phase2/structural_prior.pkl"),
        help="Path to fitted structural prior",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=5,
        help="Number of random seeds",
    )
    parser.add_argument(
        "--pop-size",
        type=int,
        default=25,
        help="Population size (default: 25 to match LLEGO paper)",
    )
    parser.add_argument(
        "--n-generations",
        type=int,
        default=25,
        help="Number of generations (default: 25 to match LLEGO paper)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("mi_analysis/results/phase3"),
        help="Output directory",
    )
    
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("Phase 3: RIGOROUS Validation Experiments")
    logger.info("=" * 80)
    logger.info(f"Datasets: {args.datasets}")
    logger.info(f"Seeds: {args.n_seeds}")
    logger.info(f"Population: {args.pop_size}, Generations: {args.n_generations}")
    logger.info(f"Setup: CART-init, Balanced Accuracy, OpenML datasets")
    
    # Run validation
    df, results = run_validation(
        datasets=args.datasets,
        prior_path=args.prior_path,
        n_seeds=args.n_seeds,
        pop_size=args.pop_size,
        n_generations=args.n_generations,
        output_dir=args.output_dir,
    )
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(args.output_dir / f"rigorous_validation_{timestamp}.csv", index=False)
    
    with open(args.output_dir / f"rigorous_validation_full_{timestamp}.json", "w") as f:
        serializable_results = []
        for r in results:
            sr = {k: v for k, v in r.items() if k != "best_tree"}
            sr["best_tree_depth"] = r["tree_depth"]
            sr["best_tree_size"] = r["tree_size"]
            serializable_results.append(sr)
        json.dump(serializable_results, f, indent=2)
    
    # Print results with statistics
    print_results_with_stats(df, args.datasets)
    
    print(f"\n✅ Rigorous validation complete!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
