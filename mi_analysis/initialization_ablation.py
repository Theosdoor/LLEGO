"""
Phase 4: Ablation Study - Initialization Effects

This experiment directly tests whether LLEGO's benefits come from:
A) Semantic understanding of the problem
B) Simply recovering good initialization properties

We compare:
1. GATree-Random: Standard GA with random tree initialization
2. GATree-CART: Standard GA with CART-bootstrapped initialization  
3. Distilled-Random: Our structural prior + random initialization
4. Distilled-CART: Our structural prior + CART initialization

If hypothesis (B) is correct, we should see:
- GATree-CART ≈ Distilled-CART (both start well, prior is redundant)
- GATree-Random < Distilled-Random (prior helps recover from bad start)
- GATree-CART >> GATree-Random (initialization is critical)
"""

import argparse
import copy
import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

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

from distillation import DistilledEvolution, StructuralPrior

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Dataset Loading (from validation_rigorous.py)
# =============================================================================

OPENML_DATASETS = {
    "breast": 15,
    "heart-statlog": 53,
    "diabetes": 37,
}


def load_openml_dataset(name: str, seed: int = 42) -> Dict:
    """Load dataset from OpenML with proper preprocessing."""
    dataset_id = OPENML_DATASETS[name]
    
    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, _, feature_names = dataset.get_data(target=dataset.default_target_attribute)
    
    X = X.values if hasattr(X, 'values') else np.array(X)
    y = y.values if hasattr(y, 'values') else np.array(y)
    
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)
    
    if y.dtype == object or isinstance(y[0], str):
        y = LabelEncoder().fit_transform(y)
    
    unique_classes = np.unique(y)
    if len(unique_classes) != 2:
        mask = np.isin(y, unique_classes[:2])
        X, y = X[mask], y[mask]
        y = LabelEncoder().fit_transform(y)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8) * 100
    
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, random_state=seed, stratify=y_trainval
    )
    
    feature_names = [f.replace(" ", "_").replace("-", "_") for f in feature_names]
    
    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "feature_names": feature_names,
    }


# =============================================================================
# Tree Utilities
# =============================================================================

def compute_depth(tree: dict) -> int:
    if "class" in tree:
        return 0
    return 1 + max(
        compute_depth(tree.get("left", {"class": 0})),
        compute_depth(tree.get("right", {"class": 0})),
    )


def compute_size(tree: dict) -> int:
    if "class" in tree:
        return 1
    return 1 + compute_size(tree.get("left", {"class": 0})) + compute_size(tree.get("right", {"class": 0}))


class TreeClassifier:
    def __init__(self, tree_dict: dict):
        self.tree = tree_dict
    
    def _predict_single(self, x: np.ndarray, feature_names: list) -> int:
        node = self.tree
        while "class" not in node:
            try:
                idx = feature_names.index(node["feature"])
            except ValueError:
                return 0
            if x[idx] <= node["threshold"]:
                node = node.get("left", {"class": 0})
            else:
                node = node.get("right", {"class": 0})
        return node["class"]
    
    def predict(self, X: np.ndarray, feature_names: list) -> np.ndarray:
        return np.array([self._predict_single(x, feature_names) for x in X])
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, feature_names: list) -> float:
        return balanced_accuracy_score(y, self.predict(X, feature_names))


# =============================================================================
# Initialization Methods
# =============================================================================

def sklearn_tree_to_dict(sk_tree, feature_names, node_id=0):
    tree = sk_tree.tree_
    if tree.children_left[node_id] == tree.children_right[node_id]:
        return {"class": int(np.argmax(tree.value[node_id][0]))}
    return {
        "feature": feature_names[tree.feature[node_id]],
        "threshold": round(float(tree.threshold[node_id]), 2),
        "left": sklearn_tree_to_dict(sk_tree, feature_names, tree.children_left[node_id]),
        "right": sklearn_tree_to_dict(sk_tree, feature_names, tree.children_right[node_id]),
    }


def cart_init(X_train, y_train, feature_names, pop_size, max_depth, seed):
    """CART-bootstrapped initialization (matching LLEGO paper: 25% of training data)."""
    rf = RandomForestClassifier(
        n_estimators=pop_size,
        max_depth=max_depth,
        max_samples=0.25,  # LLEGO paper uses 25%
        random_state=seed,
        bootstrap=True,
    )
    rf.fit(X_train, y_train)
    return [sklearn_tree_to_dict(t, feature_names) for t in rf.estimators_]


def random_tree(feature_names, threshold_ranges, max_depth=3, current_depth=0):
    """Generate random tree (low quality)."""
    if current_depth >= max_depth or (current_depth > 0 and random.random() < 0.3):
        return {"class": random.choice([0, 1])}
    
    feature = random.choice(feature_names)
    low, high = threshold_ranges.get(feature, (0, 100))
    
    return {
        "feature": feature,
        "threshold": random.uniform(low, high),
        "left": random_tree(feature_names, threshold_ranges, max_depth, current_depth + 1),
        "right": random_tree(feature_names, threshold_ranges, max_depth, current_depth + 1),
    }


def random_init(X_train, feature_names, pop_size, max_depth, seed):
    """Random tree initialization (low quality)."""
    random.seed(seed)
    threshold_ranges = {f: (X_train[:, i].min(), X_train[:, i].max()) 
                       for i, f in enumerate(feature_names)}
    return [random_tree(feature_names, threshold_ranges, max_depth) for _ in range(pop_size)]


# =============================================================================
# GA Components
# =============================================================================

def tournament_selection(population, fitnesses, k=3):
    indices = random.sample(range(len(population)), min(k, len(population)))
    best_idx = max(indices, key=lambda i: fitnesses[i])
    return copy.deepcopy(population[best_idx])


def standard_crossover(parent1, parent2):
    child = copy.deepcopy(parent1 if random.random() < 0.5 else parent2)
    if "class" not in child and "class" not in parent2 and random.random() < 0.5:
        if random.random() < 0.5:
            child["left"] = copy.deepcopy(parent2.get("left", {"class": 0}))
        else:
            child["right"] = copy.deepcopy(parent2.get("right", {"class": 0}))
    return child


def standard_mutation(tree, feature_names, threshold_ranges, rate=0.1):
    tree = copy.deepcopy(tree)
    if random.random() > rate:
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
            if random.random() < 0.5 and "left" in tree:
                tree["left"] = standard_mutation(tree["left"], feature_names, threshold_ranges, rate)
            elif "right" in tree:
                tree["right"] = standard_mutation(tree["right"], feature_names, threshold_ranges, rate)
    return tree


# =============================================================================
# Experiment Runner
# =============================================================================

def run_experiment(
    data: Dict,
    method: str,  # "GATree-Random", "GATree-CART", "Distilled-Random", "Distilled-CART"
    structural_prior: StructuralPrior = None,
    pop_size: int = 25,
    n_generations: int = 25,
    max_depth: int = 4,
    seed: int = 42,
) -> dict:
    """Run a single experiment with specified method."""
    
    start_time = time.time()
    
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    feature_names = data["feature_names"]
    
    threshold_ranges = {f: (X_train[:, i].min(), X_train[:, i].max()) 
                       for i, f in enumerate(feature_names)}
    
    # Parse method
    use_distilled = method.startswith("Distilled")
    use_cart = method.endswith("CART")
    
    # Initialize population
    if use_cart:
        population = cart_init(X_train, y_train, feature_names, pop_size, max_depth, seed)
    else:
        population = random_init(X_train, feature_names, pop_size, max_depth, seed)
    
    # Set up distilled evolution if needed
    distilled_evo = None
    if use_distilled and structural_prior:
        distilled_evo = DistilledEvolution(
            structural_prior=structural_prior,
            n_candidates=10,
            fitness_weight=0.7,
            structure_weight=0.3,
        )
    
    # Evaluation function
    def evaluate(tree):
        clf = TreeClassifier(tree)
        return clf.evaluate(X_val, y_val, feature_names)
    
    fitnesses = [evaluate(t) for t in population]
    
    best_history = []
    
    for gen in range(n_generations):
        new_population = []
        
        # Elitism
        best_idx = np.argmax(fitnesses)
        new_population.append(copy.deepcopy(population[best_idx]))
        
        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            
            if use_distilled and distilled_evo:
                f1, f2 = evaluate(parent1), evaluate(parent2)
                child = distilled_evo.crossover(parent1, parent2, f1, f2, evaluate)
                if random.random() < 0.1:
                    child = distilled_evo.mutation(child, feature_names, threshold_ranges)
            else:
                child = standard_crossover(parent1, parent2)
                child = standard_mutation(child, feature_names, threshold_ranges)
            
            new_population.append(child)
        
        population = new_population
        fitnesses = [evaluate(t) for t in population]
        best_history.append(max(fitnesses))
    
    # Final evaluation
    best_idx = np.argmax(fitnesses)
    best_tree = population[best_idx]
    clf = TreeClassifier(best_tree)
    
    return {
        "method": method,
        "test_bacc": clf.evaluate(X_test, y_test, feature_names),
        "val_bacc": max(fitnesses),
        "train_bacc": clf.evaluate(X_train, y_train, feature_names),
        "tree_depth": compute_depth(best_tree),
        "tree_size": compute_size(best_tree),
        "elapsed": time.time() - start_time,
        "best_history": best_history,
    }


def run_ablation(
    datasets: List[str],
    prior_path: Path,
    n_seeds: int = 5,
    pop_size: int = 25,
    n_generations: int = 25,
    output_dir: Path = Path("mi_analysis/results/phase4"),
):
    """Run full ablation study."""
    
    # Load prior
    prior = StructuralPrior.load(prior_path) if prior_path.exists() else StructuralPrior()
    
    methods = ["GATree-Random", "GATree-CART", "Distilled-Random", "Distilled-CART"]
    results = []
    
    for dataset in datasets:
        logger.info(f"\n{'='*60}\nDataset: {dataset}\n{'='*60}")
        
        for seed in range(n_seeds):
            logger.info(f"  Seed {seed}")
            random.seed(seed)
            np.random.seed(seed)
            
            data = load_openml_dataset(dataset, seed=seed)
            
            for method in methods:
                result = run_experiment(
                    data, method, prior, pop_size, n_generations, seed=seed
                )
                result["dataset"] = dataset
                result["seed"] = seed
                results.append(result)
                logger.info(f"    {method}: test_bacc={result['test_bacc']:.4f}")
    
    # Create DataFrame
    df = pd.DataFrame([{
        "dataset": r["dataset"],
        "seed": r["seed"],
        "method": r["method"],
        "test_bacc": r["test_bacc"],
        "tree_depth": r["tree_depth"],
        "elapsed": r["elapsed"],
    } for r in results])
    
    return df, results


def print_ablation_results(df: pd.DataFrame, datasets: List[str]):
    """Print ablation results with statistical analysis."""
    
    print("\n" + "=" * 80)
    print("ABLATION STUDY: INITIALIZATION EFFECTS")
    print("=" * 80)
    
    # Summary table
    summary = df.groupby(["dataset", "method"])["test_bacc"].agg(["mean", "std"]).round(4)
    print("\nSummary (Test Balanced Accuracy):")
    print(summary.to_string())
    
    # Key comparisons
    print("\n" + "-" * 80)
    print("KEY HYPOTHESIS TESTS")
    print("-" * 80)
    
    for dataset in datasets:
        print(f"\n{dataset}:")
        
        ga_random = df[(df["dataset"] == dataset) & (df["method"] == "GATree-Random")]["test_bacc"]
        ga_cart = df[(df["dataset"] == dataset) & (df["method"] == "GATree-CART")]["test_bacc"]
        dist_random = df[(df["dataset"] == dataset) & (df["method"] == "Distilled-Random")]["test_bacc"]
        dist_cart = df[(df["dataset"] == dataset) & (df["method"] == "Distilled-CART")]["test_bacc"]
        
        # Test 1: Does CART init help? (GA-CART vs GA-Random)
        t1, p1 = stats.ttest_rel(ga_cart, ga_random)
        print(f"  CART vs Random init (GATree): Δ={ga_cart.mean()-ga_random.mean():+.4f}, p={p1:.4f}")
        
        # Test 2: Does prior help with random init? (Dist-Random vs GA-Random)
        t2, p2 = stats.ttest_rel(dist_random, ga_random)
        print(f"  Prior effect (Random init): Δ={dist_random.mean()-ga_random.mean():+.4f}, p={p2:.4f}")
        
        # Test 3: Does prior help with CART init? (Dist-CART vs GA-CART) 
        t3, p3 = stats.ttest_rel(dist_cart, ga_cart)
        print(f"  Prior effect (CART init):   Δ={dist_cart.mean()-ga_cart.mean():+.4f}, p={p3:.4f}")
    
    print("\n" + "-" * 80)
    print("INTERPRETATION")
    print("-" * 80)
    print("If prior effect disappears with CART init, LLEGO's benefit is")
    print("likely recovering good initialization properties, not semantic understanding.")


def main():
    parser = argparse.ArgumentParser(description="Phase 4: Initialization Ablation")
    parser.add_argument("--datasets", nargs="+", default=["breast", "heart-statlog", "diabetes"])
    parser.add_argument("--prior-path", type=Path, default=Path("mi_analysis/results/phase2/structural_prior.pkl"))
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--pop-size", type=int, default=25)
    parser.add_argument("--n-generations", type=int, default=25)
    parser.add_argument("--output-dir", type=Path, default=Path("mi_analysis/results/phase4"))
    
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Phase 4: Initialization Ablation Study")
    
    df, results = run_ablation(
        args.datasets,
        args.prior_path,
        args.n_seeds,
        args.pop_size,
        args.n_generations,
        args.output_dir,
    )
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(args.output_dir / f"ablation_{timestamp}.csv", index=False)
    
    # Print analysis
    print_ablation_results(df, args.datasets)
    
    print(f"\n✅ Ablation study complete! Results in {args.output_dir}")


if __name__ == "__main__":
    main()
