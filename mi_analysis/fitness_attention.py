"""
Fitness Attention Analysis

Research Question: Does the LLM actually attend to fitness information in crossover prompts?

Experiment Design:
1. Attention Analysis: Measure attention weights from generation tokens to fitness tokens
2. Fitness Swap Test: Give LLM wrong fitness labels, see if output changes
3. Fitness Ablation: Remove fitness values entirely, compare behavior

Key Insight: If attention to fitness is low OR swapping fitness doesn't change output,
             then the LLM is ignoring fitness information (major finding!)
             This would motivate fitness-conditioned prompting improvements.

This experiment REQUIRES nnsight for attention extraction.
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
import torch

# Local imports
import sys
from pathlib import Path

# Add parent directory to path for imports
_this_dir = Path(__file__).parent
sys.path.insert(0, str(_this_dir.parent / "src"))
sys.path.insert(0, str(_this_dir.parent))

from mi_analysis.config import FitnessAttentionConfig
from mi_analysis.utils import (
    get_semantic_feature_names,
    build_crossover_prompt,
    compute_tree_depth,
    compute_tree_size,
    save_results,
)
from mi_analysis.semantic_ablation import (
    get_sample_parent_trees,
    get_sample_feature_ranges,
    parse_tree_from_response,
    LLMInterface,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Attention Analysis with nnsight
# =============================================================================

def find_token_positions(tokenizer, prompt: str, search_terms: list[str]) -> dict[str, list[int]]:
    """
    Find token positions for specific terms in a prompt.
    
    Returns dict mapping search_term -> list of token positions
    """
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    token_strings = [tokenizer.decode([t]) for t in tokens]
    
    positions = {}
    for term in search_terms:
        term_positions = []
        # Check each token
        for i, tok_str in enumerate(token_strings):
            if term.lower() in tok_str.lower():
                term_positions.append(i)
        positions[term] = term_positions
    
    return positions


def analyze_attention_to_fitness(
    model,  # nnsight LanguageModel
    tokenizer,
    prompt: str,
    fitness_values: list[str],  # e.g., ["0.7823", "0.7456"]
    layers_to_analyze: list[int],
) -> dict:
    """
    Analyze how much attention the model pays to fitness values.
    
    Uses nnsight to extract attention patterns and measure attention
    from later tokens (where generation happens) to fitness tokens.
    """
    from nnsight import LanguageModel
    
    # Find fitness token positions
    fitness_positions = find_token_positions(tokenizer, prompt, fitness_values)
    all_fitness_positions = []
    for positions in fitness_positions.values():
        all_fitness_positions.extend(positions)
    
    if not all_fitness_positions:
        logger.warning("Could not find fitness tokens in prompt")
        return {"error": "fitness_tokens_not_found"}
    
    # Get total sequence length
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    seq_len = input_ids.shape[1]
    
    # Extract attention patterns
    attention_to_fitness = {}
    
    with model.trace(prompt) as tracer:
        for layer_idx in layers_to_analyze:
            try:
                # Access attention weights
                # Note: The exact attribute path depends on the model architecture
                attn_module = model.model.layers[layer_idx].self_attn
                
                # For Llama models, we need to access the attention output
                # This captures the attention-weighted values
                attn_output = attn_module.output[0].save()
                
            except Exception as e:
                logger.warning(f"Could not extract attention from layer {layer_idx}: {e}")
    
    # Compute attention statistics
    results = {
        "fitness_token_positions": fitness_positions,
        "total_positions": all_fitness_positions,
        "seq_length": seq_len,
        "attention_by_layer": attention_to_fitness,
    }
    
    return results


def run_fitness_swap_experiment(
    llm: LLMInterface,
    parent1: dict,
    parent2: dict,
    feature_names: list[str],
    feature_ranges: dict,
    n_trials: int = 5,
) -> dict:
    """
    Test if LLM output changes when fitness labels are swapped.
    
    Design:
    1. Run crossover with correct fitness: parent1=0.85, parent2=0.70
    2. Run crossover with swapped fitness: parent1=0.70, parent2=0.85
    3. Compare generated trees
    
    If trees are similar despite swapped fitness -> LLM ignores fitness
    """
    results = {
        "trials": [],
        "summary": {},
    }
    
    correct_fitness_1, correct_fitness_2 = 0.85, 0.70
    
    for trial in range(n_trials):
        trial_result = {"trial": trial}
        
        # Condition A: Correct fitness labels
        prompt_correct = build_crossover_prompt(
            parent1_tree=parent1,
            parent2_tree=parent2,
            parent1_fitness=correct_fitness_1,
            parent2_fitness=correct_fitness_2,
            feature_names=feature_names,
            feature_ranges=feature_ranges,
            target_name="disease",
            task_description="Classify patient diagnosis",
        )
        
        response_correct = llm.generate(prompt_correct, temperature=0.3)
        tree_correct = parse_tree_from_response(response_correct["text"])
        
        # Condition B: Swapped fitness labels
        prompt_swapped = build_crossover_prompt(
            parent1_tree=parent1,
            parent2_tree=parent2,
            parent1_fitness=correct_fitness_2,  # Swapped!
            parent2_fitness=correct_fitness_1,  # Swapped!
            feature_names=feature_names,
            feature_ranges=feature_ranges,
            target_name="disease",
            task_description="Classify patient diagnosis",
        )
        
        response_swapped = llm.generate(prompt_swapped, temperature=0.3)
        tree_swapped = parse_tree_from_response(response_swapped["text"])
        
        # Compare
        trial_result["correct_parsed"] = tree_correct is not None
        trial_result["swapped_parsed"] = tree_swapped is not None
        
        if tree_correct and tree_swapped:
            # Compare tree structures
            trial_result["trees_identical"] = json.dumps(tree_correct) == json.dumps(tree_swapped)
            trial_result["correct_depth"] = compute_tree_depth(tree_correct)
            trial_result["swapped_depth"] = compute_tree_depth(tree_swapped)
            trial_result["correct_size"] = compute_tree_size(tree_correct)
            trial_result["swapped_size"] = compute_tree_size(tree_swapped)
        
        trial_result["response_correct"] = response_correct["text"][:500]
        trial_result["response_swapped"] = response_swapped["text"][:500]
        
        results["trials"].append(trial_result)
        logger.info(f"  Trial {trial+1}: correct_parsed={trial_result['correct_parsed']}, "
                   f"swapped_parsed={trial_result['swapped_parsed']}, "
                   f"identical={trial_result.get('trees_identical', 'N/A')}")
    
    # Summarize
    n_both_parsed = sum(1 for t in results["trials"] 
                       if t.get("correct_parsed") and t.get("swapped_parsed"))
    n_identical = sum(1 for t in results["trials"] if t.get("trees_identical", False))
    
    results["summary"] = {
        "n_trials": n_trials,
        "n_both_parsed": n_both_parsed,
        "n_trees_identical": n_identical,
        "identical_rate": n_identical / n_both_parsed if n_both_parsed > 0 else None,
        "interpretation": (
            "LLM IGNORES fitness" if n_identical / max(n_both_parsed, 1) > 0.5
            else "LLM uses fitness information"
        ) if n_both_parsed > 0 else "Insufficient data"
    }
    
    return results


def run_fitness_ablation_experiment(
    llm: LLMInterface,
    parent1: dict,
    parent2: dict,
    feature_names: list[str],
    feature_ranges: dict,
    n_trials: int = 5,
) -> dict:
    """
    Test effect of removing fitness values entirely.
    
    Compare:
    1. With fitness values
    2. Without fitness values (just tree structures)
    """
    results = {
        "trials": [],
        "summary": {},
    }
    
    for trial in range(n_trials):
        trial_result = {"trial": trial}
        
        # Condition A: With fitness
        prompt_with_fitness = build_crossover_prompt(
            parent1_tree=parent1,
            parent2_tree=parent2,
            parent1_fitness=0.82,
            parent2_fitness=0.75,
            feature_names=feature_names,
            feature_ranges=feature_ranges,
            target_name="disease",
            task_description="Classify patient diagnosis",
        )
        
        response_with = llm.generate(prompt_with_fitness, temperature=0.5)
        tree_with = parse_tree_from_response(response_with["text"])
        
        # Condition B: Without fitness (modify prompt)
        prompt_without_fitness = prompt_with_fitness.replace(
            "fitness=0.8200", ""
        ).replace(
            "fitness=0.7500", ""
        ).replace(
            ", aiming for fitness of", ", aiming to improve"
        )
        
        response_without = llm.generate(prompt_without_fitness, temperature=0.5)
        tree_without = parse_tree_from_response(response_without["text"])
        
        trial_result["with_fitness_parsed"] = tree_with is not None
        trial_result["without_fitness_parsed"] = tree_without is not None
        
        if tree_with:
            trial_result["with_depth"] = compute_tree_depth(tree_with)
            trial_result["with_size"] = compute_tree_size(tree_with)
        if tree_without:
            trial_result["without_depth"] = compute_tree_depth(tree_without)
            trial_result["without_size"] = compute_tree_size(tree_without)
        
        results["trials"].append(trial_result)
        logger.info(f"  Trial {trial+1}: with={trial_result['with_fitness_parsed']}, "
                   f"without={trial_result['without_fitness_parsed']}")
    
    # Summarize
    with_parsed = [t for t in results["trials"] if t.get("with_fitness_parsed")]
    without_parsed = [t for t in results["trials"] if t.get("without_fitness_parsed")]
    
    results["summary"] = {
        "n_trials": n_trials,
        "with_fitness_parse_rate": len(with_parsed) / n_trials,
        "without_fitness_parse_rate": len(without_parsed) / n_trials,
        "avg_depth_with": np.mean([t["with_depth"] for t in with_parsed]) if with_parsed else None,
        "avg_depth_without": np.mean([t["without_depth"] for t in without_parsed]) if without_parsed else None,
    }
    
    return results


def run_fitness_attention_experiment(config: FitnessAttentionConfig, llm: LLMInterface) -> dict:
    """Run the full fitness attention experiment suite."""
    
    results = {
        "config": asdict(config),
        "swap_experiment": {},
        "ablation_experiment": {},
        "attention_analysis": {},
    }
    
    # Setup
    feature_names = get_semantic_feature_names("breast")
    feature_ranges = get_sample_feature_ranges(feature_names)
    parent_trees = get_sample_parent_trees(feature_names, seed=42)
    
    parent1, parent2 = parent_trees[0], parent_trees[1]
    
    # Experiment 1: Fitness Swap
    logger.info("\n" + "="*50)
    logger.info("Experiment 1: Fitness Swap Test")
    logger.info("="*50)
    
    results["swap_experiment"] = run_fitness_swap_experiment(
        llm=llm,
        parent1=parent1,
        parent2=parent2,
        feature_names=feature_names,
        feature_ranges=feature_ranges,
        n_trials=config.n_samples // 5,
    )
    
    # Experiment 2: Fitness Ablation
    logger.info("\n" + "="*50)
    logger.info("Experiment 2: Fitness Ablation Test")
    logger.info("="*50)
    
    results["ablation_experiment"] = run_fitness_ablation_experiment(
        llm=llm,
        parent1=parent1,
        parent2=parent2,
        feature_names=feature_names,
        feature_ranges=feature_ranges,
        n_trials=config.n_samples // 5,
    )
    
    # Experiment 3: Attention Analysis (if nnsight available)
    # This requires direct model access which we'll implement if needed
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Fitness Attention Analysis")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--n-samples", type=int, default=25)
    parser.add_argument("--output-dir", type=str, default="mi_analysis/results")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use-nnsight", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true",
                        help="Load model in 8-bit quantization")
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Load model in 4-bit quantization")
    
    args = parser.parse_args()
    
    config = FitnessAttentionConfig(
        model_name=args.model,
        device=args.device,
        n_samples=args.n_samples,
        output_dir=Path(args.output_dir),
    )
    
    logger.info("Fitness Attention Analysis")
    logger.info(f"Config: {asdict(config)}")
    
    if args.dry_run:
        logger.info("DRY RUN - creating mock results")
        results = {
            "swap_experiment": {
                "summary": {
                    "n_trials": 5,
                    "n_both_parsed": 4,
                    "n_trees_identical": 3,
                    "identical_rate": 0.75,
                    "interpretation": "LLM IGNORES fitness"
                }
            },
            "ablation_experiment": {
                "summary": {
                    "with_fitness_parse_rate": 0.8,
                    "without_fitness_parse_rate": 0.8,
                }
            }
        }
    else:
        llm = LLMInterface(
            model_name=config.model_name,
            use_local=True,
            use_nnsight=args.use_nnsight,
            device=config.device,
            cache_dir=config.cache_dir,
            load_in_8bit=args.load_in_8bit,
            load_in_4bit=args.load_in_4bit,
        )
        
        results = run_fitness_attention_experiment(config, llm)
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_results(
        results=results,
        output_path=config.output_dir,
        name=f"fitness_attention_{timestamp}"
    )
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("RESULTS SUMMARY")
    logger.info("="*60)
    
    swap_summary = results.get("swap_experiment", {}).get("summary", {})
    logger.info(f"\nFitness Swap Test:")
    logger.info(f"  Identical rate: {swap_summary.get('identical_rate', 'N/A')}")
    logger.info(f"  Interpretation: {swap_summary.get('interpretation', 'N/A')}")
    
    ablation_summary = results.get("ablation_experiment", {}).get("summary", {})
    logger.info(f"\nFitness Ablation Test:")
    logger.info(f"  Parse rate with fitness: {ablation_summary.get('with_fitness_parse_rate', 'N/A')}")
    logger.info(f"  Parse rate without: {ablation_summary.get('without_fitness_parse_rate', 'N/A')}")


if __name__ == "__main__":
    main()
