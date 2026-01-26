"""
Semantic Ablation Study

Research Question: Does LLM performance depend on meaningful feature names?

Experiment Design:
- Run LLEGO crossover on datasets with two conditions:
  - Condition A (Semantic): Original feature names ("age", "blood_pressure", etc.)
  - Condition B (Arbitrary): Arbitrary names ("X1", "X2", etc.)
- Same underlying data, same parent trees, only names differ
- Compare: fitness of generated offspring, tree structure quality

Key Insight: If performance_A >> performance_B → LLM leverages semantic knowledge
             If performance_A ≈ performance_B → Structural priors dominate

This experiment uses nnsight for optional activation analysis but can run
in behavioral-only mode without GPU access.
"""

import argparse
import json
import logging
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

# Local imports
import sys
from pathlib import Path

# Add parent directory to path for imports
_this_dir = Path(__file__).parent
sys.path.insert(0, str(_this_dir.parent / "src"))
sys.path.insert(0, str(_this_dir.parent))

from mi_analysis.config import SemanticAblationConfig
from mi_analysis.utils import (
    get_semantic_feature_names,
    get_arbitrary_feature_names,
    build_crossover_prompt,
    replace_feature_names_in_prompt,
    replace_feature_names_in_tree,
    compute_tree_depth,
    compute_tree_size,
    compute_tree_balance,
    extract_features_used,
    save_results,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Sample Trees for Testing
# =============================================================================

def get_sample_parent_trees(feature_names: list[str], seed: int = 42) -> list[dict]:
    """
    Generate sample parent trees for crossover experiments.
    
    These are simplified tree structures that could be parents in LLEGO.
    In a full experiment, we'd use actual trees from LLEGO runs.
    """
    random.seed(seed)
    
    trees = []
    
    # Tree 1: Simple depth-2 tree
    tree1 = {
        "feature": feature_names[0],
        "threshold": 50.0,
        "left": {"class": 0},
        "right": {
            "feature": feature_names[1],
            "threshold": 30.0,
            "left": {"class": 1},
            "right": {"class": 0}
        }
    }
    trees.append(tree1)
    
    # Tree 2: Different structure
    tree2 = {
        "feature": feature_names[1],
        "threshold": 40.0,
        "left": {
            "feature": feature_names[2] if len(feature_names) > 2 else feature_names[0],
            "threshold": 25.0,
            "left": {"class": 1},
            "right": {"class": 0}
        },
        "right": {"class": 1}
    }
    trees.append(tree2)
    
    # Tree 3: Deeper tree
    tree3 = {
        "feature": feature_names[0],
        "threshold": 45.0,
        "left": {
            "feature": feature_names[1],
            "threshold": 35.0,
            "left": {"class": 0},
            "right": {
                "feature": feature_names[2] if len(feature_names) > 2 else feature_names[0],
                "threshold": 20.0,
                "left": {"class": 1},
                "right": {"class": 0}
            }
        },
        "right": {
            "feature": feature_names[3] if len(feature_names) > 3 else feature_names[1],
            "threshold": 60.0,
            "left": {"class": 1},
            "right": {"class": 0}
        }
    }
    trees.append(tree3)
    
    return trees


def get_sample_feature_ranges(feature_names: list[str]) -> dict[str, tuple]:
    """Generate plausible feature ranges."""
    ranges = {}
    for i, name in enumerate(feature_names):
        # Generate different ranges based on typical medical data
        if any(term in name.lower() for term in ["age", "year"]):
            ranges[name] = (0, 100)
        elif any(term in name.lower() for term in ["pressure", "rate"]):
            ranges[name] = (0, 200)
        elif any(term in name.lower() for term in ["glucose", "cholesterol"]):
            ranges[name] = (0, 500)
        else:
            ranges[name] = (0, 100)
    return ranges


# =============================================================================
# LLM Interface (supports both API and local models)
# =============================================================================

class LLMInterface:
    """
    Unified interface for LLM generation.
    Supports: OpenAI API, local HuggingFace models, nnsight-wrapped models.
    """
    
    def __init__(
        self,
        model_name: str,
        use_local: bool = True,
        use_nnsight: bool = False,
        device: str = "cuda",
        cache_dir: Optional[Path] = None,
    ):
        self.model_name = model_name
        self.use_local = use_local
        self.use_nnsight = use_nnsight
        self.device = device
        self.cache_dir = cache_dir
        
        self.model = None
        self.tokenizer = None
        
        if use_local:
            self._load_local_model()
    
    def _load_local_model(self):
        """Load model locally with HuggingFace or nnsight."""
        logger.info(f"Loading model: {self.model_name}")
        
        if self.use_nnsight:
            from nnsight import LanguageModel
            self.model = LanguageModel(
                self.model_name,
                device_map=self.device,
                torch_dtype=torch.bfloat16,
                cache_dir=self.cache_dir,
            )
            self.tokenizer = self.model.tokenizer
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=self.device,
                torch_dtype=torch.bfloat16,
                cache_dir=self.cache_dir,
            )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"Model loaded successfully")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        do_sample: bool = True,
        return_activations: bool = False,
        layers_to_save: Optional[list[int]] = None,
    ) -> dict:
        """
        Generate response from LLM.
        
        Returns:
            dict with keys: "text", "input_tokens", "output_tokens"
            If return_activations=True, also includes "activations" dict
        """
        if self.use_nnsight and return_activations:
            return self._generate_with_nnsight(
                prompt, max_new_tokens, temperature, do_sample, layers_to_save
            )
        else:
            return self._generate_standard(
                prompt, max_new_tokens, temperature, do_sample
            )
    
    def _generate_standard(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        do_sample: bool,
    ) -> dict:
        """Standard generation without activation capture."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_length = inputs.input_ids.shape[1]
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        generated_text = self.tokenizer.decode(
            outputs[0][input_length:], 
            skip_special_tokens=True
        )
        
        return {
            "text": generated_text,
            "input_tokens": input_length,
            "output_tokens": outputs.shape[1] - input_length,
        }
    
    def _generate_with_nnsight(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        do_sample: bool,
        layers_to_save: Optional[list[int]] = None,
    ) -> dict:
        """Generation with nnsight activation capture."""
        if layers_to_save is None:
            layers_to_save = [0, 8, 16, 24, 31]
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_length = inputs.input_ids.shape[1]
        
        activations = {}
        
        with self.model.trace(prompt) as tracer:
            # Save activations from specified layers
            for layer_idx in layers_to_save:
                if hasattr(self.model.model, 'layers'):
                    layer = self.model.model.layers[layer_idx]
                    activations[f"layer_{layer_idx}_output"] = layer.output[0].save()
                    if hasattr(layer, 'self_attn'):
                        # Try to get attention weights if available
                        pass
            
            # Generate
            output = self.model.generate(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
            )
        
        generated_text = self.tokenizer.decode(
            output[0][input_length:],
            skip_special_tokens=True
        )
        
        # Convert saved activations to numpy
        activations_np = {}
        for key, val in activations.items():
            if hasattr(val, 'value'):
                activations_np[key] = val.value.cpu().numpy()
        
        return {
            "text": generated_text,
            "input_tokens": input_length,
            "output_tokens": len(output[0]) - input_length,
            "activations": activations_np,
        }


# =============================================================================
# Tree Parsing
# =============================================================================

def parse_tree_from_response(response: str) -> Optional[dict]:
    """
    Parse decision tree JSON from LLM response.
    Handles various output formats used by LLMs.
    """
    # Try to extract from ## tree ## markers
    import re
    
    tree_match = re.search(r'##\s*tree\s*##\s*(.*?)\s*##\s*tree\s*##', response, re.DOTALL | re.IGNORECASE)
    if tree_match:
        tree_str = tree_match.group(1).strip()
    else:
        # Try to find JSON block
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
        if json_match:
            tree_str = json_match.group(0)
        else:
            return None
    
    try:
        return json.loads(tree_str)
    except json.JSONDecodeError:
        # Try to fix common issues
        tree_str = tree_str.replace("'", '"')
        tree_str = re.sub(r'(\w+):', r'"\1":', tree_str)  # Add quotes to keys
        try:
            return json.loads(tree_str)
        except:
            return None


# =============================================================================
# Main Experiment
# =============================================================================

def run_single_crossover(
    llm: LLMInterface,
    parent1: dict,
    parent2: dict,
    parent1_fitness: float,
    parent2_fitness: float,
    feature_names: list[str],
    feature_ranges: dict,
    target_name: str,
    task_description: str,
    condition: str,  # "semantic" or "arbitrary"
    return_activations: bool = False,
) -> dict:
    """Run a single crossover operation and collect metrics."""
    
    # Build prompt
    prompt = build_crossover_prompt(
        parent1_tree=parent1,
        parent2_tree=parent2,
        parent1_fitness=parent1_fitness,
        parent2_fitness=parent2_fitness,
        feature_names=feature_names,
        feature_ranges=feature_ranges,
        target_name=target_name,
        task_description=task_description,
    )
    
    # Generate
    result = llm.generate(
        prompt=prompt,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True,
        return_activations=return_activations,
    )
    
    # Parse tree
    generated_tree = parse_tree_from_response(result["text"])
    
    # Compute metrics
    metrics = {
        "condition": condition,
        "prompt_length": len(prompt),
        "response_length": len(result["text"]),
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
        "tree_parsed_successfully": generated_tree is not None,
    }
    
    if generated_tree:
        metrics.update({
            "tree_depth": compute_tree_depth(generated_tree),
            "tree_size": compute_tree_size(generated_tree),
            "tree_balance": compute_tree_balance(generated_tree),
            "features_used": list(extract_features_used(generated_tree)),
            "n_features_used": len(extract_features_used(generated_tree)),
        })
    
    # Include activations if captured
    if "activations" in result:
        metrics["activations"] = result["activations"]
    
    metrics["prompt"] = prompt
    metrics["response"] = result["text"]
    metrics["generated_tree"] = generated_tree
    
    return metrics


def run_semantic_ablation_experiment(
    config: SemanticAblationConfig,
    llm: LLMInterface,
) -> dict:
    """
    Run the full semantic ablation experiment.
    
    For each dataset:
    1. Get semantic feature names
    2. Generate arbitrary feature names
    3. Run crossovers with both conditions
    4. Compare results
    """
    all_results = []
    
    for dataset in config.datasets:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing dataset: {dataset}")
        logger.info(f"{'='*60}")
        
        # Get feature names
        semantic_names = get_semantic_feature_names(dataset)
        arbitrary_names = get_arbitrary_feature_names(len(semantic_names))
        feature_ranges = get_sample_feature_ranges(semantic_names)
        
        # Create arbitrary feature ranges (same values, different names)
        arbitrary_ranges = {
            arb: feature_ranges[sem] 
            for sem, arb in zip(semantic_names, arbitrary_names)
        }
        
        # Task descriptions
        semantic_task = f"Classify whether a patient has the condition based on medical measurements"
        arbitrary_task = f"Classify the target variable based on input features"
        
        for seed in config.seeds:
            logger.info(f"  Seed: {seed}")
            
            # Get sample parent trees
            semantic_trees = get_sample_parent_trees(semantic_names, seed=seed)
            arbitrary_trees = [
                replace_feature_names_in_tree(t, semantic_names, arbitrary_names)
                for t in semantic_trees
            ]
            
            # Run crossovers
            for i in range(config.n_crossovers_per_condition):
                # Select two parents
                idx1, idx2 = random.sample(range(len(semantic_trees)), 2)
                parent1_fitness = 0.75 + random.random() * 0.1
                parent2_fitness = 0.70 + random.random() * 0.1
                
                # Condition A: Semantic
                try:
                    result_semantic = run_single_crossover(
                        llm=llm,
                        parent1=semantic_trees[idx1],
                        parent2=semantic_trees[idx2],
                        parent1_fitness=parent1_fitness,
                        parent2_fitness=parent2_fitness,
                        feature_names=semantic_names,
                        feature_ranges=feature_ranges,
                        target_name="disease_status",
                        task_description=semantic_task,
                        condition="semantic",
                        return_activations=config.save_activations,
                    )
                    result_semantic.update({
                        "dataset": dataset,
                        "seed": seed,
                        "crossover_idx": i,
                    })
                    all_results.append(result_semantic)
                    logger.info(f"    Crossover {i+1} (semantic): parsed={result_semantic['tree_parsed_successfully']}")
                except Exception as e:
                    logger.error(f"    Crossover {i+1} (semantic) failed: {e}")
                
                # Condition B: Arbitrary
                try:
                    result_arbitrary = run_single_crossover(
                        llm=llm,
                        parent1=arbitrary_trees[idx1],
                        parent2=arbitrary_trees[idx2],
                        parent1_fitness=parent1_fitness,
                        parent2_fitness=parent2_fitness,
                        feature_names=arbitrary_names,
                        feature_ranges=arbitrary_ranges,
                        target_name="Y",
                        task_description=arbitrary_task,
                        condition="arbitrary",
                        return_activations=config.save_activations,
                    )
                    result_arbitrary.update({
                        "dataset": dataset,
                        "seed": seed,
                        "crossover_idx": i,
                    })
                    all_results.append(result_arbitrary)
                    logger.info(f"    Crossover {i+1} (arbitrary): parsed={result_arbitrary['tree_parsed_successfully']}")
                except Exception as e:
                    logger.error(f"    Crossover {i+1} (arbitrary) failed: {e}")
    
    return {"raw_results": all_results}


def analyze_results(results: dict) -> dict:
    """Analyze semantic ablation results."""
    raw = results["raw_results"]
    
    # Filter to remove activations for summary (too large)
    summary_data = []
    for r in raw:
        summary_row = {k: v for k, v in r.items() 
                       if k not in ["activations", "prompt", "response", "generated_tree"]}
        summary_data.append(summary_row)
    
    df = pd.DataFrame(summary_data)
    
    # Compute aggregated statistics
    analysis = {
        "overall": {},
        "by_condition": {},
        "by_dataset": {},
        "by_dataset_condition": {},
    }
    
    # Overall stats
    analysis["overall"] = {
        "total_crossovers": len(df),
        "parse_success_rate": df["tree_parsed_successfully"].mean(),
        "avg_tree_depth": df[df["tree_parsed_successfully"]]["tree_depth"].mean() if "tree_depth" in df.columns else None,
        "avg_tree_size": df[df["tree_parsed_successfully"]]["tree_size"].mean() if "tree_size" in df.columns else None,
    }
    
    # By condition
    for condition in ["semantic", "arbitrary"]:
        cond_df = df[df["condition"] == condition]
        analysis["by_condition"][condition] = {
            "n": len(cond_df),
            "parse_success_rate": cond_df["tree_parsed_successfully"].mean(),
            "avg_tree_depth": cond_df[cond_df["tree_parsed_successfully"]]["tree_depth"].mean() if "tree_depth" in cond_df.columns and len(cond_df[cond_df["tree_parsed_successfully"]]) > 0 else None,
            "avg_tree_size": cond_df[cond_df["tree_parsed_successfully"]]["tree_size"].mean() if "tree_size" in cond_df.columns and len(cond_df[cond_df["tree_parsed_successfully"]]) > 0 else None,
            "avg_tree_balance": cond_df[cond_df["tree_parsed_successfully"]]["tree_balance"].mean() if "tree_balance" in cond_df.columns and len(cond_df[cond_df["tree_parsed_successfully"]]) > 0 else None,
        }
    
    # By dataset
    for dataset in df["dataset"].unique():
        ds_df = df[df["dataset"] == dataset]
        analysis["by_dataset"][dataset] = {
            "n": len(ds_df),
            "parse_success_rate": ds_df["tree_parsed_successfully"].mean(),
        }
        
        # By dataset and condition
        for condition in ["semantic", "arbitrary"]:
            key = f"{dataset}_{condition}"
            cond_df = ds_df[ds_df["condition"] == condition]
            if len(cond_df) > 0:
                analysis["by_dataset_condition"][key] = {
                    "n": len(cond_df),
                    "parse_success_rate": cond_df["tree_parsed_successfully"].mean(),
                    "avg_tree_depth": cond_df[cond_df["tree_parsed_successfully"]]["tree_depth"].mean() if "tree_depth" in cond_df.columns and len(cond_df[cond_df["tree_parsed_successfully"]]) > 0 else None,
                }
    
    # Compute semantic advantage
    sem = analysis["by_condition"].get("semantic", {})
    arb = analysis["by_condition"].get("arbitrary", {})
    
    analysis["semantic_advantage"] = {
        "parse_success_diff": (sem.get("parse_success_rate", 0) or 0) - (arb.get("parse_success_rate", 0) or 0),
        "tree_depth_diff": ((sem.get("avg_tree_depth") or 0) - (arb.get("avg_tree_depth") or 0)) if sem.get("avg_tree_depth") and arb.get("avg_tree_depth") else None,
    }
    
    return {
        "raw_results": results["raw_results"],
        "summary": summary_data,
        "analysis": analysis,
    }


def main():
    parser = argparse.ArgumentParser(description="Semantic Ablation Study")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Model to use")
    parser.add_argument("--datasets", type=str, nargs="+", 
                        default=["breast", "heart-statlog"],
                        help="Datasets to test")
    parser.add_argument("--n-crossovers", type=int, default=5,
                        help="Number of crossovers per condition per seed")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0],
                        help="Random seeds")
    parser.add_argument("--output-dir", type=str, default="mi_analysis/results",
                        help="Output directory")
    parser.add_argument("--use-nnsight", action="store_true",
                        help="Use nnsight for activation capture")
    parser.add_argument("--no-local", action="store_true",
                        help="Use API instead of local model")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run without LLM (for testing)")
    
    args = parser.parse_args()
    
    # Setup config
    config = SemanticAblationConfig(
        model_name=args.model,
        device=args.device,
        output_dir=Path(args.output_dir),
        datasets=args.datasets,
        n_crossovers_per_condition=args.n_crossovers,
        seeds=args.seeds,
        save_activations=args.use_nnsight,
    )
    
    logger.info("Semantic Ablation Study")
    logger.info(f"Config: {asdict(config)}")
    
    if args.dry_run:
        logger.info("DRY RUN - skipping LLM initialization")
        # Create mock results for testing
        results = {"raw_results": []}
        for dataset in config.datasets:
            for seed in config.seeds:
                for i in range(config.n_crossovers_per_condition):
                    for condition in ["semantic", "arbitrary"]:
                        results["raw_results"].append({
                            "dataset": dataset,
                            "seed": seed,
                            "crossover_idx": i,
                            "condition": condition,
                            "tree_parsed_successfully": random.random() > 0.2,
                            "tree_depth": random.randint(2, 5),
                            "tree_size": random.randint(3, 15),
                            "tree_balance": random.random(),
                        })
    else:
        # Initialize LLM
        llm = LLMInterface(
            model_name=config.model_name,
            use_local=not args.no_local,
            use_nnsight=args.use_nnsight,
            device=config.device,
            cache_dir=config.cache_dir,
        )
        
        # Run experiment
        results = run_semantic_ablation_experiment(config, llm)
    
    # Analyze
    results = analyze_results(results)
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_results(
        results=results,
        output_path=config.output_dir,
        name=f"semantic_ablation_{timestamp}"
    )
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("RESULTS SUMMARY")
    logger.info("="*60)
    
    analysis = results["analysis"]
    logger.info(f"\nOverall:")
    logger.info(f"  Total crossovers: {analysis['overall']['total_crossovers']}")
    logger.info(f"  Parse success rate: {analysis['overall']['parse_success_rate']:.2%}")
    
    logger.info(f"\nBy Condition:")
    for cond, stats in analysis["by_condition"].items():
        logger.info(f"  {cond}:")
        logger.info(f"    Parse success: {stats['parse_success_rate']:.2%}")
        if stats.get('avg_tree_depth'):
            logger.info(f"    Avg depth: {stats['avg_tree_depth']:.2f}")
        if stats.get('avg_tree_balance'):
            logger.info(f"    Avg balance: {stats['avg_tree_balance']:.2f}")
    
    logger.info(f"\nSemantic Advantage:")
    adv = analysis["semantic_advantage"]
    logger.info(f"  Parse success diff (sem - arb): {adv['parse_success_diff']:+.2%}")
    if adv.get('tree_depth_diff') is not None:
        logger.info(f"  Tree depth diff: {adv['tree_depth_diff']:+.2f}")
    
    logger.info(f"\nResults saved to: {config.output_dir}")


if __name__ == "__main__":
    main()
