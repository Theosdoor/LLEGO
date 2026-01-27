"""
Phase 3: Validation Experiments

Compare:
1. GATree (baseline - standard GA, no LLM)
2. Distilled-LLEGO (our method - GA + distilled structural prior)
3. LLEGO (original - full LLM calls) [optional, expensive]

Success Criteria:
- Distilled achieves ≥80% of (LLEGO - GATree) improvement
- Distilled uses 0 LLM calls at runtime
"""

import argparse
import json
import logging
import random
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable
import copy

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Local imports
import sys
_this_dir = Path(__file__).parent
sys.path.insert(0, str(_this_dir.parent / "src"))
sys.path.insert(0, str(_this_dir.parent))

from mi_analysis.distillation import (
    StructuralPrior,
    DistilledEvolution,
    compute_depth,
    compute_size,
    compute_balance,
    get_features_used,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Decision Tree Implementation (for evaluation)
# =============================================================================

class DecisionTreeClassifier:
    """
    Simple decision tree classifier that can be built from dict representation.
    """
    
    def __init__(self, tree_dict: dict):
        self.tree = tree_dict
    
    def predict_one(self, x: np.ndarray, feature_names: list[str]) -> int:
        """Predict for a single sample."""
        node = self.tree
        
        while "class" not in node:
            feature = node.get("feature")
            threshold = node.get("threshold")
            
            # Find feature index
            if feature in feature_names:
                idx = feature_names.index(feature)
            else:
                # Handle arbitrary names (X1, X2, etc.)
                try:
                    idx = int(feature.replace("X", "")) - 1
                except:
                    # Default to class 0 if feature not found
                    return 0
            
            if idx >= len(x):
                return 0
            
            if x[idx] <= threshold:
                node = node.get("left", {"class": 0})
            else:
                node = node.get("right", {"class": 0})
        
        return node.get("class", 0)
    
    def predict(self, X: np.ndarray, feature_names: list[str]) -> np.ndarray:
        """Predict for multiple samples."""
        return np.array([self.predict_one(x, feature_names) for x in X])
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> float:
        """Compute accuracy on data."""
        preds = self.predict(X, feature_names)
        return accuracy_score(y, preds)


# =============================================================================
# GA Components
# =============================================================================

def random_tree(
    feature_names: list[str],
    max_depth: int = 3,
    current_depth: int = 0,
) -> dict:
    """Generate a random decision tree."""
    # Stop at max depth or randomly become leaf
    if current_depth >= max_depth or (current_depth > 0 and random.random() < 0.3):
        return {"class": random.choice([0, 1])}
    
    feature = random.choice(feature_names)
    threshold = random.uniform(0, 100)  # Assuming normalized features
    
    return {
        "feature": feature,
        "threshold": threshold,
        "left": random_tree(feature_names, max_depth, current_depth + 1),
        "right": random_tree(feature_names, max_depth, current_depth + 1),
    }


def tournament_selection(population: list[dict], fitnesses: list[float], k: int = 3) -> dict:
    """Select individual via tournament selection."""
    indices = random.sample(range(len(population)), min(k, len(population)))
    best_idx = max(indices, key=lambda i: fitnesses[i])
    return copy.deepcopy(population[best_idx])


def standard_crossover(parent1: dict, parent2: dict) -> dict:
    """Standard subtree crossover (no structural prior)."""
    # Simple implementation: swap subtrees
    child = copy.deepcopy(parent1)
    
    # 50% chance to take structure from parent2
    if random.random() < 0.5:
        child = copy.deepcopy(parent2)
    
    # Random subtree replacement
    if "class" not in child and random.random() < 0.5:
        if random.random() < 0.5 and "class" not in parent2:
            if random.random() < 0.5:
                child["left"] = copy.deepcopy(parent2.get("left", {"class": 0}))
            else:
                child["right"] = copy.deepcopy(parent2.get("right", {"class": 0}))
    
    return child


def standard_mutation(
    tree: dict,
    feature_names: list[str],
    mutation_rate: float = 0.1,
) -> dict:
    """Standard mutation (random changes)."""
    tree = copy.deepcopy(tree)
    
    if random.random() > mutation_rate:
        return tree
    
    if "class" in tree:
        # Flip class or grow
        if random.random() < 0.5:
            tree["class"] = 1 - tree["class"]
        else:
            tree = {
                "feature": random.choice(feature_names),
                "threshold": random.uniform(0, 100),
                "left": {"class": 0},
                "right": {"class": 1},
            }
    else:
        # Change feature, threshold, or prune
        choice = random.choice(["feature", "threshold", "prune", "recurse"])
        
        if choice == "feature":
            tree["feature"] = random.choice(feature_names)
        elif choice == "threshold":
            tree["threshold"] = random.uniform(0, 100)
        elif choice == "prune":
            tree = {"class": random.choice([0, 1])}
        else:
            if random.random() < 0.5:
                tree["left"] = standard_mutation(tree["left"], feature_names, mutation_rate)
            else:
                tree["right"] = standard_mutation(tree["right"], feature_names, mutation_rate)
    
    return tree


# =============================================================================
# Evolution Algorithms
# =============================================================================

def run_gatree(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    pop_size: int = 50,
    n_generations: int = 30,
    max_depth: int = 4,
) -> dict:
    """
    Run standard GATree (no LLM, no structural prior).
    """
    logger.info("Running GATree (baseline)...")
    start_time = time.time()
    
    # Initialize population
    population = [random_tree(feature_names, max_depth) for _ in range(pop_size)]
    
    # Evaluate
    def evaluate(tree: dict) -> float:
        clf = DecisionTreeClassifier(tree)
        return clf.evaluate(X_train, y_train, feature_names)
    
    fitnesses = [evaluate(t) for t in population]
    
    best_fitness_history = []
    avg_fitness_history = []
    
    for gen in range(n_generations):
        # Selection and reproduction
        new_population = []
        
        # Elitism: keep best
        best_idx = np.argmax(fitnesses)
        new_population.append(copy.deepcopy(population[best_idx]))
        
        while len(new_population) < pop_size:
            # Select parents
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            
            # Crossover
            child = standard_crossover(parent1, parent2)
            
            # Mutation
            child = standard_mutation(child, feature_names)
            
            new_population.append(child)
        
        population = new_population
        fitnesses = [evaluate(t) for t in population]
        
        best_fitness = max(fitnesses)
        avg_fitness = np.mean(fitnesses)
        best_fitness_history.append(best_fitness)
        avg_fitness_history.append(avg_fitness)
        
        if (gen + 1) % 10 == 0:
            logger.info(f"  Gen {gen+1}: best={best_fitness:.4f}, avg={avg_fitness:.4f}")
    
    # Get best tree
    best_idx = np.argmax(fitnesses)
    best_tree = population[best_idx]
    
    # Evaluate on test
    clf = DecisionTreeClassifier(best_tree)
    test_acc = clf.evaluate(X_test, y_test, feature_names)
    
    elapsed = time.time() - start_time
    
    return {
        "method": "GATree",
        "train_acc": max(fitnesses),
        "test_acc": test_acc,
        "best_tree": best_tree,
        "tree_depth": compute_depth(best_tree),
        "tree_size": compute_size(best_tree),
        "n_generations": n_generations,
        "elapsed_seconds": elapsed,
        "llm_calls": 0,
        "fitness_history": best_fitness_history,
    }


def run_distilled_llego(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    structural_prior: StructuralPrior,
    pop_size: int = 50,
    n_generations: int = 30,
    max_depth: int = 4,
) -> dict:
    """
    Run Distilled-LLEGO (structural prior, no LLM calls).
    """
    logger.info("Running Distilled-LLEGO...")
    start_time = time.time()
    
    # Initialize distilled evolution
    distilled_evo = DistilledEvolution(
        structural_prior=structural_prior,
        n_candidates=10,
        fitness_weight=0.7,
        structure_weight=0.3,
    )
    
    # Threshold ranges for mutation
    threshold_ranges = {f: (0, 100) for f in feature_names}
    
    # Initialize population
    population = [random_tree(feature_names, max_depth) for _ in range(pop_size)]
    
    # Evaluate
    def evaluate(tree: dict) -> float:
        clf = DecisionTreeClassifier(tree)
        return clf.evaluate(X_train, y_train, feature_names)
    
    fitnesses = [evaluate(t) for t in population]
    
    best_fitness_history = []
    avg_fitness_history = []
    
    for gen in range(n_generations):
        new_population = []
        
        # Elitism
        best_idx = np.argmax(fitnesses)
        new_population.append(copy.deepcopy(population[best_idx]))
        
        while len(new_population) < pop_size:
            # Select parents
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            f1 = evaluate(parent1)
            f2 = evaluate(parent2)
            
            # DISTILLED crossover (uses structural prior, NO LLM)
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
        
        if (gen + 1) % 10 == 0:
            logger.info(f"  Gen {gen+1}: best={best_fitness:.4f}, avg={avg_fitness:.4f}")
    
    # Get best tree
    best_idx = np.argmax(fitnesses)
    best_tree = population[best_idx]
    
    # Evaluate on test
    clf = DecisionTreeClassifier(best_tree)
    test_acc = clf.evaluate(X_test, y_test, feature_names)
    
    elapsed = time.time() - start_time
    
    return {
        "method": "Distilled-LLEGO",
        "train_acc": max(fitnesses),
        "test_acc": test_acc,
        "best_tree": best_tree,
        "tree_depth": compute_depth(best_tree),
        "tree_size": compute_size(best_tree),
        "n_generations": n_generations,
        "elapsed_seconds": elapsed,
        "llm_calls": 0,  # Key metric: ZERO LLM calls!
        "fitness_history": best_fitness_history,
    }


# =============================================================================
# Datasets
# =============================================================================

def load_dataset(name: str) -> tuple:
    """Load a dataset for validation."""
    if name == "breast":
        data = load_breast_cancer()
        X, y = data.data, data.target
        # Normalize to 0-100
        X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8) * 100
        feature_names = list(data.feature_names)
        # Simplify feature names
        feature_names = [f.replace(" ", "_") for f in feature_names]
        
    elif name == "iris":
        data = load_iris()
        X, y = data.data, data.target
        # Binary classification: setosa vs others
        y = (y == 0).astype(int)
        X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8) * 100
        feature_names = list(data.feature_names)
        
    elif name == "heart":
        # Use simple synthetic heart-like data
        np.random.seed(42)
        n = 300
        X = np.random.randn(n, 5) * 20 + 50
        # Simple rule: age > 50 and cholesterol > 60 → disease
        y = ((X[:, 0] > 50) & (X[:, 1] > 60)).astype(int)
        feature_names = ["age", "cholesterol", "blood_pressure", "heart_rate", "bmi"]
        
    else:
        raise ValueError(f"Unknown dataset: {name}")
    
    return X, y, feature_names


# =============================================================================
# Main Validation
# =============================================================================

def run_validation(
    datasets: list[str],
    prior_path: Path,
    n_seeds: int = 5,
    pop_size: int = 50,
    n_generations: int = 30,
    output_dir: Path = Path("mi_analysis/results/phase3"),
) -> pd.DataFrame:
    """Run validation experiments across datasets and seeds."""
    
    # Load structural prior
    if prior_path.exists():
        prior = StructuralPrior.load(prior_path)
        logger.info(f"Loaded structural prior from {prior_path}")
    else:
        logger.warning(f"Prior not found at {prior_path}, using default")
        prior = StructuralPrior()
    
    results = []
    
    for dataset in datasets:
        logger.info(f"\n{'='*60}")
        logger.info(f"Dataset: {dataset}")
        logger.info(f"{'='*60}")
        
        X, y, feature_names = load_dataset(dataset)
        
        for seed in range(n_seeds):
            logger.info(f"\nSeed: {seed}")
            random.seed(seed)
            np.random.seed(seed)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=seed
            )
            
            # Run GATree (baseline)
            ga_result = run_gatree(
                X_train, y_train, X_test, y_test, feature_names,
                pop_size=pop_size, n_generations=n_generations,
            )
            ga_result["dataset"] = dataset
            ga_result["seed"] = seed
            results.append(ga_result)
            
            # Run Distilled-LLEGO
            distilled_result = run_distilled_llego(
                X_train, y_train, X_test, y_test, feature_names,
                structural_prior=prior,
                pop_size=pop_size, n_generations=n_generations,
            )
            distilled_result["dataset"] = dataset
            distilled_result["seed"] = seed
            results.append(distilled_result)
    
    # Create results dataframe
    df = pd.DataFrame([{
        "dataset": r["dataset"],
        "seed": r["seed"],
        "method": r["method"],
        "train_acc": r["train_acc"],
        "test_acc": r["test_acc"],
        "tree_depth": r["tree_depth"],
        "tree_size": r["tree_size"],
        "elapsed_seconds": r["elapsed_seconds"],
        "llm_calls": r["llm_calls"],
    } for r in results])
    
    return df, results


def main():
    parser = argparse.ArgumentParser(description="Phase 3 Validation")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["breast", "iris", "heart"],
        help="Datasets to evaluate on",
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
        default=50,
        help="Population size for GA",
    )
    parser.add_argument(
        "--n-generations",
        type=int,
        default=30,
        help="Number of generations",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("mi_analysis/results/phase3"),
        help="Output directory",
    )
    
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Phase 3: Validation Experiments")
    logger.info(f"Datasets: {args.datasets}")
    logger.info(f"Seeds: {args.n_seeds}")
    logger.info(f"Population: {args.pop_size}, Generations: {args.n_generations}")
    
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
    df.to_csv(args.output_dir / f"validation_results_{timestamp}.csv", index=False)
    
    with open(args.output_dir / f"validation_full_{timestamp}.json", "w") as f:
        # Remove non-serializable items
        serializable_results = []
        for r in results:
            sr = {k: v for k, v in r.items() if k != "best_tree"}
            sr["best_tree_depth"] = r["tree_depth"]
            sr["best_tree_size"] = r["tree_size"]
            serializable_results.append(sr)
        json.dump(serializable_results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 70)
    print("VALIDATION RESULTS SUMMARY")
    print("=" * 70)
    
    summary = df.groupby(["dataset", "method"]).agg({
        "test_acc": ["mean", "std"],
        "elapsed_seconds": "mean",
        "llm_calls": "mean",
    }).round(4)
    
    print(summary.to_string())
    
    # Compute improvement
    print("\n" + "-" * 70)
    print("DISTILLED vs GATREE IMPROVEMENT")
    print("-" * 70)
    
    for dataset in args.datasets:
        ga_acc = df[(df["dataset"] == dataset) & (df["method"] == "GATree")]["test_acc"].mean()
        dist_acc = df[(df["dataset"] == dataset) & (df["method"] == "Distilled-LLEGO")]["test_acc"].mean()
        
        improvement = dist_acc - ga_acc
        print(f"{dataset}: GATree={ga_acc:.4f}, Distilled={dist_acc:.4f}, Δ={improvement:+.4f}")
    
    print("\n✅ Validation complete!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
