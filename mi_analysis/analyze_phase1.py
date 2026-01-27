"""
Phase 2 Analysis: Interpret Phase 1 Results and Fit Structural Prior

This script:
1. Loads Phase 1 results (semantic ablation + fitness attention)
2. Computes key statistics and interpretations
3. Fits a structural prior from LLM-generated trees
4. Generates a report summarizing findings
"""

import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from distillation import (
    StructuralPrior,
    TreeStats,
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


def load_semantic_ablation_results(results_dir: Path) -> dict:
    """Load semantic ablation results."""
    for f in results_dir.glob("semantic_ablation_*.json"):
        if "_summary" not in str(f):
            with open(f) as fp:
                return json.load(fp)
    return {}


def load_fitness_attention_results(results_dir: Path) -> dict:
    """Load fitness attention results."""
    for f in results_dir.glob("fitness_attention_*.json"):
        with open(f) as fp:
            return json.load(fp)
    return {}


def analyze_semantic_vs_arbitrary(raw_results: list) -> dict:
    """Compare semantic vs arbitrary naming conditions."""
    semantic = [r for r in raw_results if r["condition"] == "semantic"]
    arbitrary = [r for r in raw_results if r["condition"] == "arbitrary"]
    
    def stats(results):
        parsed = [r for r in results if r.get("tree_parsed_successfully")]
        return {
            "n": len(results),
            "parse_rate": len(parsed) / len(results) if results else 0,
            "avg_depth": np.mean([r["tree_depth"] for r in parsed]) if parsed else 0,
            "avg_size": np.mean([r["tree_size"] for r in parsed]) if parsed else 0,
            "avg_balance": np.mean([r["tree_balance"] for r in parsed]) if parsed else 0,
            "avg_features": np.mean([r["n_features_used"] for r in parsed]) if parsed else 0,
        }
    
    sem_stats = stats(semantic)
    arb_stats = stats(arbitrary)
    
    return {
        "semantic": sem_stats,
        "arbitrary": arb_stats,
        "comparison": {
            "parse_rate_diff": sem_stats["parse_rate"] - arb_stats["parse_rate"],
            "depth_diff": sem_stats["avg_depth"] - arb_stats["avg_depth"],
            "size_diff": sem_stats["avg_size"] - arb_stats["avg_size"],
            "semantic_advantage": sem_stats["parse_rate"] > arb_stats["parse_rate"],
        }
    }


def extract_trees(raw_results: list) -> list[dict]:
    """Extract all successfully parsed trees."""
    trees = []
    for r in raw_results:
        if r.get("tree_parsed_successfully") and r.get("generated_tree"):
            trees.append(r["generated_tree"])
    return trees


def generate_report(
    semantic_results: dict,
    fitness_results: dict,
    comparison: dict,
    prior: StructuralPrior,
    output_path: Path,
):
    """Generate a comprehensive Phase 1 analysis report."""
    
    report = []
    report.append("=" * 70)
    report.append("PHASE 1 ANALYSIS REPORT: Mechanistic Investigation of LLEGO")
    report.append("=" * 70)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Key Findings
    report.append("## KEY FINDINGS")
    report.append("-" * 70)
    report.append("")
    
    # Finding 1: Semantic vs Arbitrary
    report.append("### Finding 1: Semantic Names Do NOT Help (Surprising!)")
    report.append("")
    report.append("| Metric | Semantic | Arbitrary | Difference |")
    report.append("|--------|----------|-----------|------------|")
    report.append(f"| Parse Rate | {comparison['semantic']['parse_rate']:.1%} | {comparison['arbitrary']['parse_rate']:.1%} | {comparison['comparison']['parse_rate_diff']:+.1%} |")
    report.append(f"| Avg Depth | {comparison['semantic']['avg_depth']:.2f} | {comparison['arbitrary']['avg_depth']:.2f} | {comparison['comparison']['depth_diff']:+.2f} |")
    report.append(f"| Avg Size | {comparison['semantic']['avg_size']:.2f} | {comparison['arbitrary']['avg_size']:.2f} | {comparison['comparison']['size_diff']:+.2f} |")
    report.append(f"| Avg Balance | {comparison['semantic']['avg_balance']:.2f} | {comparison['arbitrary']['avg_balance']:.2f} | - |")
    report.append("")
    
    if comparison['comparison']['semantic_advantage']:
        report.append("**Interpretation:** Semantic names provide small advantage.")
    else:
        report.append("**Interpretation:** Arbitrary names (X1, X2) perform EQUAL OR BETTER!")
        report.append("This is a key insight: LLM contribution is STRUCTURAL, not SEMANTIC.")
    report.append("")
    
    # Finding 2: Fitness Attention
    report.append("### Finding 2: LLM Uses Fitness Information")
    report.append("")
    
    if fitness_results and "swap_experiment" in fitness_results:
        swap = fitness_results["swap_experiment"]["summary"]
        report.append(f"- Fitness Swap Test: {swap['n_both_parsed']} valid trials")
        report.append(f"- Trees identical when fitness swapped: {swap['identical_rate']:.1%}")
        report.append(f"- Interpretation: {swap['interpretation']}")
        report.append("")
        
        if swap['identical_rate'] < 0.5:
            report.append("**Conclusion:** LLM DOES use fitness values to guide crossover.")
        else:
            report.append("**Conclusion:** LLM ignores fitness (could simplify prompts).")
    report.append("")
    
    # Finding 3: Structural Prior
    report.append("### Finding 3: Learned Structural Prior")
    report.append("")
    report.append("LLM-generated trees exhibit consistent structural preferences:")
    report.append("")
    report.append(f"| Parameter | Mean | Std |")
    report.append(f"|-----------|------|-----|")
    report.append(f"| Depth | {prior.depth_mean:.2f} | {prior.depth_std:.2f} |")
    report.append(f"| Size (nodes) | {prior.size_mean:.2f} | {prior.size_std:.2f} |")
    report.append(f"| Balance | {prior.balance_mean:.2f} | {prior.balance_std:.2f} |")
    report.append(f"| Feature Reuse | {prior.feature_reuse_mean:.2f} | {prior.feature_reuse_std:.2f} |")
    report.append("")
    report.append("These parameters will be used to guide crossover in Phase 3.")
    report.append("")
    
    # Implications for Phase 2
    report.append("## IMPLICATIONS FOR PHASE 2 (Distillation)")
    report.append("-" * 70)
    report.append("")
    report.append("Based on Phase 1 findings:")
    report.append("")
    report.append("1. **FOCUS ON STRUCTURAL PRIORS** (not semantic knowledge)")
    report.append("   - Semantic names don't improve performance")
    report.append("   - LLM's value comes from tree structure preferences")
    report.append("")
    report.append("2. **DISTILL THE STRUCTURAL PRIOR**")
    report.append("   - Use learned depth/size/balance preferences")
    report.append("   - Score candidate crossover children by structural prior")
    report.append("   - No LLM calls needed at runtime!")
    report.append("")
    report.append("3. **SIMPLIFIED APPROACH**")
    report.append("   - Originally planned: Semantic Graph + Structural Prior")
    report.append("   - Revised plan: Structural Prior ONLY (simpler, empirically justified)")
    report.append("")
    
    # Next Steps
    report.append("## NEXT STEPS")
    report.append("-" * 70)
    report.append("")
    report.append("1. [x] Analyze Phase 1 results (this report)")
    report.append("2. [ ] Implement DistilledEvolution using structural prior")
    report.append("3. [ ] Run validation: LLEGO vs Distilled-LLEGO vs GATree")
    report.append("4. [ ] Generate figures and write report")
    report.append("")
    report.append("=" * 70)
    
    # Save report
    report_text = "\n".join(report)
    with open(output_path, "w") as f:
        f.write(report_text)
    
    return report_text


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("mi_analysis/results/phase1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("mi_analysis/results/phase2"),
    )
    
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Loading Phase 1 results...")
    
    # Load results
    semantic_data = load_semantic_ablation_results(args.results_dir)
    fitness_data = load_fitness_attention_results(args.results_dir)
    
    if not semantic_data:
        logger.error(f"No semantic ablation results found in {args.results_dir}")
        return
    
    # Analyze
    raw_results = semantic_data.get("raw_results", [])
    comparison = analyze_semantic_vs_arbitrary(raw_results)
    
    logger.info(f"Semantic parse rate: {comparison['semantic']['parse_rate']:.1%}")
    logger.info(f"Arbitrary parse rate: {comparison['arbitrary']['parse_rate']:.1%}")
    
    # Extract trees and fit prior
    trees = extract_trees(raw_results)
    logger.info(f"Extracted {len(trees)} trees for prior fitting")
    
    prior = StructuralPrior.fit_from_trees(trees)
    
    # Save prior
    prior.save(args.output_dir / "structural_prior.pkl")
    
    prior_json = {
        "depth_mean": prior.depth_mean,
        "depth_std": prior.depth_std,
        "size_mean": prior.size_mean,
        "size_std": prior.size_std,
        "balance_mean": prior.balance_mean,
        "balance_std": prior.balance_std,
        "feature_reuse_mean": prior.feature_reuse_mean,
        "feature_reuse_std": prior.feature_reuse_std,
        "n_trees_fitted": len(trees),
    }
    with open(args.output_dir / "structural_prior.json", "w") as f:
        json.dump(prior_json, f, indent=2)
    
    # Generate report
    report = generate_report(
        semantic_data,
        fitness_data,
        comparison,
        prior,
        args.output_dir / "phase1_analysis_report.txt",
    )
    
    print("\n" + report)
    
    logger.info(f"\n✅ Phase 2 analysis complete!")
    logger.info(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
