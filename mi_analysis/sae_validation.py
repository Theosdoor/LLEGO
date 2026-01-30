"""
SAE Validation Experiments

4-way comparison:
1. GATree - Standard GA, no priors
2. Distilled-Struct - Structural prior only  
3. Distilled-SAE - SAE semantic prior only
4. Distilled-Full - Both priors

This validates whether SAE-extracted semantic priors add value beyond structural priors.
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm

# Add paths
_this_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(_this_dir.parent))
sys.path.insert(0, str(_this_dir.parent / "src"))
sys.path.insert(0, str(_this_dir.parent / "sae_project"))

from mi_analysis.distillation import (
    StructuralPrior,
    DistilledEvolution,
    compute_depth,
    compute_size,
    get_features_used,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Dataset Loading
# =============================================================================

def load_dataset(name: str) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Load dataset from sklearn or OpenML.
    
    Returns:
        X: Feature DataFrame
        y: Target Series
        feature_names: List of feature names
    """
    if name == "breast":
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target)
        feature_names = list(data.feature_names)
        
    elif name == "heart":
        try:
            from sklearn.datasets import fetch_openml
            data = fetch_openml("heart-statlog", version=1, as_frame=True)
            X = data.data
            y = (data.target == "present").astype(int)
            feature_names = list(X.columns)
        except Exception:
            # Fallback to UCI heart disease
            from ucimlrepo import fetch_ucirepo
            heart = fetch_ucirepo(id=45)
            X = heart.data.features
            y = (heart.data.targets.values.ravel() > 0).astype(int)
            feature_names = list(X.columns)
    
    elif name == "diabetes":
        from sklearn.datasets import load_diabetes
        data = load_diabetes()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        # Convert to classification (above/below median)
        y = pd.Series((data.target > np.median(data.target)).astype(int))
        feature_names = list(data.feature_names)
    
    elif name == "liver":
        from sklearn.datasets import fetch_openml
        data = fetch_openml("liver-disorders", version=1, as_frame=True)
        X = data.data
        # Target: selector field (1 or 2)
        y = pd.Series((data.target.astype(int) == 2).astype(int))
        feature_names = list(X.columns)
    
    elif name == "credit-g":
        from sklearn.datasets import fetch_openml
        data = fetch_openml("credit-g", version=1, as_frame=True)
        X = data.data
        # Handle categorical columns - convert to numeric
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                X[col] = pd.Categorical(X[col]).codes
        y = pd.Series((data.target == 'good').astype(int))
        feature_names = list(X.columns)
    
    else:
        raise ValueError(f"Unknown dataset: {name}")
    
    return X, y, feature_names


# =============================================================================
# Tree Evaluation
# =============================================================================

class TreeClassifier:
    """Simple decision tree classifier from dict representation."""
    
    def __init__(self, tree_dict: dict):
        self.tree = tree_dict
    
    def predict_one(self, x: np.ndarray, feature_names: List[str]) -> int:
        node = self.tree
        while "class" not in node:
            feature = node.get("feature", "")
            threshold = node.get("threshold", 0)
            
            try:
                idx = feature_names.index(feature)
                if x[idx] <= threshold:
                    node = node.get("left", {"class": 0})
                else:
                    node = node.get("right", {"class": 1})
            except (ValueError, IndexError):
                return 0
        
        return node.get("class", 0)
    
    def predict(self, X: np.ndarray, feature_names: List[str]) -> np.ndarray:
        return np.array([self.predict_one(x, feature_names) for x in X])
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> float:
        preds = self.predict(X, feature_names)
        return balanced_accuracy_score(y, preds)


# =============================================================================
# Random Tree Generation (for initialization)
# =============================================================================

def random_tree(
    feature_names: List[str],
    threshold_ranges: Dict[str, Tuple[float, float]],
    max_depth: int = 3,
    current_depth: int = 0,
) -> dict:
    """Generate a random decision tree."""
    import random
    
    # Leaf node probability increases with depth
    leaf_prob = current_depth / max_depth
    
    if current_depth >= max_depth or random.random() < leaf_prob * 0.5:
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


def get_threshold_ranges(X: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    """Get min/max ranges for thresholds."""
    return {col: (X[col].min(), X[col].max()) for col in X.columns}


# =============================================================================
# GA Components
# =============================================================================

def tournament_selection(population: List[dict], fitnesses: List[float], k: int = 3) -> dict:
    """Select individual via tournament selection."""
    import random
    indices = random.sample(range(len(population)), min(k, len(population)))
    best_idx = max(indices, key=lambda i: fitnesses[i])
    return population[best_idx]


def standard_crossover(parent1: dict, parent2: dict) -> dict:
    """Standard subtree crossover (no prior guidance)."""
    import random
    import copy
    
    from mi_analysis.distillation import get_all_subtrees, replace_subtree
    
    subtrees1 = get_all_subtrees(parent1)
    subtrees2 = get_all_subtrees(parent2)
    
    if len(subtrees1) < 2 or len(subtrees2) < 1:
        return copy.deepcopy(parent1)
    
    replace_point = random.choice(subtrees1[1:])
    donor_subtree = random.choice(subtrees2)
    
    return replace_subtree(parent1, replace_point, donor_subtree)


def standard_mutation(
    tree: dict,
    feature_names: List[str],
    threshold_ranges: Dict,
    mutation_rate: float = 0.1,
) -> dict:
    """Standard random mutation."""
    import random
    import copy
    
    tree = copy.deepcopy(tree)
    
    if "class" in tree:
        if random.random() < mutation_rate:
            tree["class"] = 1 - tree["class"]
        return tree
    
    if random.random() < mutation_rate:
        tree["feature"] = random.choice(feature_names)
        low, high = threshold_ranges.get(tree["feature"], (0, 100))
        tree["threshold"] = random.uniform(low, high)
    
    tree["left"] = standard_mutation(tree["left"], feature_names, threshold_ranges, mutation_rate)
    tree["right"] = standard_mutation(tree["right"], feature_names, threshold_ranges, mutation_rate)
    
    return tree


# =============================================================================
# Evolution Runners
# =============================================================================

def run_gatree(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    feature_names: List[str],
    pop_size: int = 25,
    n_generations: int = 15,
    max_depth: int = 4,
    seed: int = 42,
) -> Tuple[dict, float, List[float]]:
    """Run standard GATree (no priors)."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    
    threshold_ranges = {name: (X_train[:, i].min(), X_train[:, i].max()) 
                        for i, name in enumerate(feature_names)}
    
    def evaluate(tree: dict) -> float:
        clf = TreeClassifier(tree)
        return clf.evaluate(X_train, y_train, feature_names)
    
    # Initialize population
    population = [random_tree(feature_names, threshold_ranges, max_depth) for _ in range(pop_size)]
    fitnesses = [evaluate(tree) for tree in population]
    
    history = [max(fitnesses)]
    
    for gen in range(n_generations):
        new_pop = []
        
        # Elitism: keep best
        best_idx = np.argmax(fitnesses)
        new_pop.append(population[best_idx])
        
        while len(new_pop) < pop_size:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            
            child = standard_crossover(parent1, parent2)
            child = standard_mutation(child, feature_names, threshold_ranges)
            
            if compute_depth(child) <= max_depth:
                new_pop.append(child)
        
        population = new_pop[:pop_size]
        fitnesses = [evaluate(tree) for tree in population]
        history.append(max(fitnesses))
    
    best_idx = np.argmax(fitnesses)
    best_tree = population[best_idx]
    
    # Evaluate on validation set
    clf = TreeClassifier(best_tree)
    val_score = clf.evaluate(X_val, y_val, feature_names)
    
    return best_tree, val_score, history


def run_distilled(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    feature_names: List[str],
    structural_prior: Optional[StructuralPrior] = None,
    semantic_prior: Optional[pd.DataFrame] = None,
    pop_size: int = 25,
    n_generations: int = 15,
    max_depth: int = 4,
    seed: int = 42,
) -> Tuple[dict, float, List[float]]:
    """
    Run distilled evolution (with optional structural and/or semantic priors).
    
    Args:
        structural_prior: Tree structure preferences
        semantic_prior: Feature similarity matrix
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    
    threshold_ranges = {name: (X_train[:, i].min(), X_train[:, i].max()) 
                        for i, name in enumerate(feature_names)}
    
    def evaluate(tree: dict) -> float:
        clf = TreeClassifier(tree)
        return clf.evaluate(X_train, y_train, feature_names)
    
    # Create evolution object
    struct_prior = structural_prior or StructuralPrior()
    
    # Adjust weights based on what priors are available
    if structural_prior is not None and semantic_prior is not None:
        evo = DistilledEvolution(struct_prior, semantic_prior, 
                                  fitness_weight=0.5, structure_weight=0.3, semantic_weight=0.2)
    elif structural_prior is not None:
        evo = DistilledEvolution(struct_prior, None,
                                  fitness_weight=0.6, structure_weight=0.4, semantic_weight=0.0)
    elif semantic_prior is not None:
        evo = DistilledEvolution(struct_prior, semantic_prior,
                                  fitness_weight=0.6, structure_weight=0.0, semantic_weight=0.4)
    else:
        evo = DistilledEvolution(struct_prior, None,
                                  fitness_weight=0.8, structure_weight=0.2, semantic_weight=0.0)
    
    # Initialize population
    population = [random_tree(feature_names, threshold_ranges, max_depth) for _ in range(pop_size)]
    fitnesses = [evaluate(tree) for tree in population]
    
    history = [max(fitnesses)]
    
    for gen in range(n_generations):
        new_pop = []
        
        # Elitism
        best_idx = np.argmax(fitnesses)
        new_pop.append(population[best_idx])
        
        while len(new_pop) < pop_size:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            
            fitness1 = fitnesses[population.index(parent1)] if parent1 in population else 0.5
            fitness2 = fitnesses[population.index(parent2)] if parent2 in population else 0.5
            
            # Guided crossover
            child = evo.crossover(parent1, parent2, fitness1, fitness2, evaluate_fn=evaluate)
            
            # Mutation
            child = evo.mutation(child, feature_names, threshold_ranges)
            
            if compute_depth(child) <= max_depth:
                new_pop.append(child)
        
        population = new_pop[:pop_size]
        fitnesses = [evaluate(tree) for tree in population]
        history.append(max(fitnesses))
    
    best_idx = np.argmax(fitnesses)
    best_tree = population[best_idx]
    
    clf = TreeClassifier(best_tree)
    val_score = clf.evaluate(X_val, y_val, feature_names)
    
    return best_tree, val_score, history


# =============================================================================
# Main Validation
# =============================================================================

@dataclass
class ExperimentResult:
    dataset: str
    method: str
    seed: int
    val_accuracy: float
    train_accuracy: float
    tree_depth: int
    tree_size: int
    features_used: int
    convergence_gen: int  # Generation when reached 90% of final accuracy
    
    def to_dict(self):
        return asdict(self)


def run_validation(
    datasets: List[str],
    structural_prior_path: Optional[Path] = None,
    sae_prior_dir: Optional[Path] = None,
    n_seeds: int = 5,
    pop_size: int = 25,
    n_generations: int = 15,
    max_depth: int = 3,
    output_dir: Path = Path("mi_analysis/results/sae_validation"),
) -> pd.DataFrame:
    """
    Run full validation experiments.
    
    Compares 4 methods:
    1. GATree - No priors
    2. Distilled-Struct - Structural prior only
    3. Distilled-SAE - SAE semantic prior only
    4. Distilled-Full - Both priors
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load structural prior if available
    structural_prior = None
    if structural_prior_path and structural_prior_path.exists():
        structural_prior = StructuralPrior.load(structural_prior_path)
        logger.info(f"Loaded structural prior from {structural_prior_path}")
    else:
        logger.warning("No structural prior found, using default")
        structural_prior = StructuralPrior()
    
    results = []
    
    for dataset in datasets:
        logger.info(f"\n{'='*60}")
        logger.info(f"Dataset: {dataset}")
        logger.info(f"{'='*60}")
        
        # Load data
        X, y, feature_names = load_dataset(dataset)
        
        # Load SAE semantic prior if available
        semantic_prior = None
        if sae_prior_dir:
            prior_path = sae_prior_dir / dataset / "similarity_matrix.csv"
            if prior_path.exists():
                semantic_prior = pd.read_csv(prior_path, index_col=0)
                logger.info(f"Loaded SAE prior from {prior_path}")
            else:
                logger.warning(f"No SAE prior found at {prior_path}")
        
        for seed in range(n_seeds):
            logger.info(f"\n  Seed {seed}")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X.values, y.values, test_size=0.3, random_state=seed, stratify=y
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=seed, stratify=y_train
            )
            
            # Method 1: GATree
            tree, val_acc, history = run_gatree(
                X_train, y_train, X_val, y_val, feature_names,
                pop_size=pop_size, n_generations=n_generations, max_depth=max_depth, seed=seed
            )
            train_acc = TreeClassifier(tree).evaluate(X_train, y_train, feature_names)
            convergence_gen = next((i for i, h in enumerate(history) if h >= 0.9 * max(history)), len(history))
            
            results.append(ExperimentResult(
                dataset=dataset, method="GATree", seed=seed,
                val_accuracy=val_acc, train_accuracy=train_acc,
                tree_depth=compute_depth(tree), tree_size=compute_size(tree),
                features_used=len(get_features_used(tree)), convergence_gen=convergence_gen
            ))
            logger.info(f"    GATree: {val_acc:.3f}")
            
            # Method 2: Distilled-Struct
            tree, val_acc, history = run_distilled(
                X_train, y_train, X_val, y_val, feature_names,
                structural_prior=structural_prior, semantic_prior=None,
                pop_size=pop_size, n_generations=n_generations, max_depth=max_depth, seed=seed
            )
            train_acc = TreeClassifier(tree).evaluate(X_train, y_train, feature_names)
            convergence_gen = next((i for i, h in enumerate(history) if h >= 0.9 * max(history)), len(history))
            
            results.append(ExperimentResult(
                dataset=dataset, method="Distilled-Struct", seed=seed,
                val_accuracy=val_acc, train_accuracy=train_acc,
                tree_depth=compute_depth(tree), tree_size=compute_size(tree),
                features_used=len(get_features_used(tree)), convergence_gen=convergence_gen
            ))
            logger.info(f"    Distilled-Struct: {val_acc:.3f}")
            
            # Method 3: Distilled-SAE (only if semantic prior available)
            if semantic_prior is not None:
                tree, val_acc, history = run_distilled(
                    X_train, y_train, X_val, y_val, feature_names,
                    structural_prior=None, semantic_prior=semantic_prior,
                    pop_size=pop_size, n_generations=n_generations, max_depth=max_depth, seed=seed
                )
                train_acc = TreeClassifier(tree).evaluate(X_train, y_train, feature_names)
                convergence_gen = next((i for i, h in enumerate(history) if h >= 0.9 * max(history)), len(history))
                
                results.append(ExperimentResult(
                    dataset=dataset, method="Distilled-SAE", seed=seed,
                    val_accuracy=val_acc, train_accuracy=train_acc,
                    tree_depth=compute_depth(tree), tree_size=compute_size(tree),
                    features_used=len(get_features_used(tree)), convergence_gen=convergence_gen
                ))
                logger.info(f"    Distilled-SAE: {val_acc:.3f}")
                
                # Method 4: Distilled-Full
                tree, val_acc, history = run_distilled(
                    X_train, y_train, X_val, y_val, feature_names,
                    structural_prior=structural_prior, semantic_prior=semantic_prior,
                    pop_size=pop_size, n_generations=n_generations, max_depth=max_depth, seed=seed
                )
                train_acc = TreeClassifier(tree).evaluate(X_train, y_train, feature_names)
                convergence_gen = next((i for i, h in enumerate(history) if h >= 0.9 * max(history)), len(history))
                
                results.append(ExperimentResult(
                    dataset=dataset, method="Distilled-Full", seed=seed,
                    val_accuracy=val_acc, train_accuracy=train_acc,
                    tree_depth=compute_depth(tree), tree_size=compute_size(tree),
                    features_used=len(get_features_used(tree)), convergence_gen=convergence_gen
                ))
                logger.info(f"    Distilled-Full: {val_acc:.3f}")
    
    # Convert to DataFrame and save
    df = pd.DataFrame([r.to_dict() for r in results])
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"sae_validation_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"\nResults saved to {csv_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    summary = df.groupby(["dataset", "method"])["val_accuracy"].agg(["mean", "std"])
    print(summary.round(3).to_string())
    
    # Overall comparison
    print("\n" + "-"*40)
    print("OVERALL MEAN ACCURACY BY METHOD:")
    print("-"*40)
    overall = df.groupby("method")["val_accuracy"].mean().sort_values(ascending=False)
    for method, acc in overall.items():
        print(f"  {method}: {acc:.3f}")
    
    return df


# =============================================================================
# Main
# =============================================================================

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    parser = argparse.ArgumentParser(description="SAE Validation Experiments")
    # Match LLEGO paper: credit-g, heart-statlog, liver, breast
    parser.add_argument("--datasets", nargs="+", default=["breast", "heart", "liver", "credit-g"])
    parser.add_argument("--n-seeds", type=int, default=5)  # LLEGO uses 5 seeds
    parser.add_argument("--pop-size", type=int, default=25)
    parser.add_argument("--n-generations", type=int, default=15)
    parser.add_argument("--max-depth", type=int, default=3)  # Also test with 4
    parser.add_argument("--structural-prior", type=Path, default=Path("mi_analysis/results/phase2/structural_prior.pkl"))
    parser.add_argument("--sae-prior-dir", type=Path, default=None, help="Directory containing SAE priors per dataset")
    parser.add_argument("--output-dir", type=Path, default=Path("mi_analysis/results/sae_validation"))
    parser.add_argument("--quick-test", action="store_true", help="Quick test with minimal settings")
    
    args = parser.parse_args()
    
    if args.quick_test:
        args.n_seeds = 1
        args.pop_size = 10
        args.n_generations = 5
        args.datasets = ["breast"]
    
    run_validation(
        datasets=args.datasets,
        structural_prior_path=args.structural_prior,
        sae_prior_dir=args.sae_prior_dir,
        n_seeds=args.n_seeds,
        pop_size=args.pop_size,
        n_generations=args.n_generations,
        max_depth=args.max_depth,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
