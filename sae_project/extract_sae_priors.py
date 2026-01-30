#!/usr/bin/env python
"""
Extract SAE semantic priors for all datasets.

This runs ONCE per dataset to extract the similarity matrix,
then saves it for use in validation experiments.

Usage:
    python extract_sae_priors.py --datasets breast heart diabetes
"""

import sys
import argparse
import logging
from pathlib import Path

# Setup paths
_this_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(_this_dir))

from sae_semantic_prior import SAESemanticPrior, SAEConfig

logger = logging.getLogger(__name__)


# Dataset feature mappings
DATASET_FEATURES = {
    "breast": [
        "mean radius", "mean texture", "mean perimeter", "mean area",
        "mean smoothness", "mean compactness", "mean concavity", 
        "mean concave points", "mean symmetry", "mean fractal dimension",
        "radius error", "texture error", "perimeter error", "area error",
        "smoothness error", "compactness error", "concavity error",
        "concave points error", "symmetry error", "fractal dimension error",
        "worst radius", "worst texture", "worst perimeter", "worst area",
        "worst smoothness", "worst compactness", "worst concavity",
        "worst concave points", "worst symmetry", "worst fractal dimension"
    ],
    "heart": [
        "age", "sex", "chest pain type", "resting blood pressure", 
        "serum cholesterol", "fasting blood sugar", "resting electrocardiographic",
        "maximum heart rate", "exercise induced angina", "oldpeak",
        "slope of peak exercise ST segment", "number of major vessels",
        "thalassemia"
    ],
    "diabetes": [
        "age", "sex", "body mass index", "average blood pressure",
        "serum1", "serum2", "serum3", "serum4", "serum5", "serum6"
    ],
    "liver": [
        "mean corpuscular volume", "alkaline phosphotase", "alanine aminotransferase",
        "aspartate aminotransferase", "gamma-glutamyl transpeptidase", "drinks per day"
    ],
    "credit-g": [
        "checking account status", "duration in months", "credit history", "purpose",
        "credit amount", "savings account", "employment duration", "installment rate",
        "personal status", "other debtors", "residence duration", "property",
        "age", "other installment plans", "housing", "existing credits", "job",
        "number of dependents", "telephone", "foreign worker"
    ],
}


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    parser = argparse.ArgumentParser(description="Extract SAE semantic priors")
    parser.add_argument("--datasets", nargs="+", default=["breast", "heart", "diabetes"])
    parser.add_argument("--output-dir", type=Path, default=Path("sae_project/priors"))
    parser.add_argument("--model", type=str, default="gemma-2-2b")
    parser.add_argument("--sae-release", type=str, default="gemma-scope-2b-pt-res-canonical")
    parser.add_argument("--sae-id", type=str, default="layer_12/width_16k/canonical")
    parser.add_argument("--method", type=str, default="jaccard", 
                       choices=["jaccard", "weighted_jaccard", "cosine", "ensemble"])
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--domain", type=str, default="healthcare")
    
    args = parser.parse_args()
    
    config = SAEConfig(
        model_name=args.model,
        sae_release=args.sae_release,
        sae_id=args.sae_id,
        similarity_method=args.method,
        activation_threshold=args.threshold,
        domain=args.domain,
    )
    
    logger.info("Initializing SAE Semantic Prior extractor...")
    prior = SAESemanticPrior(config)
    
    for dataset in args.datasets:
        if dataset not in DATASET_FEATURES:
            logger.warning(f"Unknown dataset: {dataset}, skipping")
            continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Extracting prior for: {dataset}")
        logger.info(f"{'='*60}")
        
        features = DATASET_FEATURES[dataset]
        output_path = args.output_dir / dataset
        
        try:
            result = prior.extract(features)
            result.save(output_path)
            
            # Print preview
            df = result.to_dataframe()
            logger.info(f"\nSimilarity matrix preview (first 5x5):")
            print(df.iloc[:5, :5].round(3).to_string())
            
        except Exception as e:
            logger.error(f"Failed to extract for {dataset}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    logger.info(f"\n✅ Extraction complete! Priors saved to {args.output_dir}")


if __name__ == "__main__":
    main()
