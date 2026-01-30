"""
SAE Semantic Prior Extraction

Extracts semantic similarity between dataset features using Sparse Autoencoders.
Run ONCE per dataset, then use the resulting matrix to guide evolution without LLM calls.

Key Contribution:
- LLEGO uses LLM prompts to leverage semantic knowledge (expensive, repeated)
- We extract the same knowledge from LLM internals via SAEs (cheap, one-shot)
"""

import os
import re
import json
import pickle
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from pathlib import Path
from typing import List, Optional, Literal, Dict, Tuple
from dataclasses import dataclass, field, asdict
from itertools import combinations
from tqdm import tqdm

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


@dataclass
class SAEConfig:
    """Configuration for SAE extraction."""
    model_name: str = "gemma-2-2b"
    sae_release: str = "gemma-scope-2b-pt-res-canonical"
    sae_id: str = "layer_12/width_16k/canonical"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "bfloat16"
    
    # Extraction settings
    similarity_method: str = "jaccard"  # jaccard, weighted_jaccard, cosine, ensemble
    activation_threshold: float = 0.1   # For jaccard methods
    domain: str = "healthcare"          # For prompt context
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> "SAEConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass  
class ExtractionResult:
    """Results from SAE feature extraction."""
    feature_names: List[str]
    similarity_matrix: np.ndarray
    feature_vectors: Optional[np.ndarray] = None  # Raw SAE activations
    top_features_per_column: Optional[Dict[str, List[int]]] = None  # For interpretability
    sparsity_stats: Optional[Dict] = None
    config: Optional[SAEConfig] = None
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert similarity matrix to labeled DataFrame."""
        return pd.DataFrame(
            self.similarity_matrix,
            index=self.feature_names,
            columns=self.feature_names
        )
    
    def save(self, path: Path):
        """Save extraction results."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save similarity matrix as CSV (human readable)
        self.to_dataframe().to_csv(path / "similarity_matrix.csv")
        
        # Save full results as pickle (for reloading)
        with open(path / "extraction_result.pkl", "wb") as f:
            pickle.dump(self, f)
        
        # Save config as JSON
        if self.config:
            with open(path / "config.json", "w") as f:
                json.dump(self.config.to_dict(), f, indent=2)
        
        print(f"Saved results to {path}")
    
    @classmethod
    def load(cls, path: Path) -> "ExtractionResult":
        """Load extraction results."""
        with open(Path(path) / "extraction_result.pkl", "rb") as f:
            return pickle.load(f)


class SAESemanticPrior:
    """
    Extracts semantic relationships between features using SAEs.
    
    Usage:
        prior = SAESemanticPrior(config)
        result = prior.extract(["Age", "Cholesterol", "BloodPressure"])
        similarity_df = result.to_dataframe()
    """
    
    def __init__(self, config: Optional[SAEConfig] = None):
        self.config = config or SAEConfig()
        self.model = None
        self.sae = None
        self.hook_point = None
        self._loaded = False
    
    def _load_models(self):
        """Lazy-load model and SAE (heavy operation)."""
        if self._loaded:
            return
        
        print(f"Loading model: {self.config.model_name}...")
        from transformer_lens import HookedTransformer
        from sae_lens import SAE
        
        dtype = getattr(torch, self.config.dtype) if isinstance(self.config.dtype, str) else self.config.dtype
        
        self.model = HookedTransformer.from_pretrained(
            self.config.model_name,
            device=self.config.device,
            dtype=dtype,
            center_unembed=False,
        )
        
        print(f"Loading SAE: {self.config.sae_id}...")
        self.sae = SAE.from_pretrained(
            release=self.config.sae_release,
            sae_id=self.config.sae_id,
            device=self.config.device
        )
        
        # Determine hook point (various API versions)
        self.hook_point = (
            getattr(self.sae, "hook_name", None) or
            getattr(self.sae.cfg, "hook_name", None) or
            getattr(self.sae.cfg, "hook_point", None)
        )
        
        if not self.hook_point:
            # Fallback: construct from SAE ID
            layer_match = re.search(r"layer_(\d+)", self.config.sae_id)
            if layer_match:
                layer_num = layer_match.group(1)
                self.hook_point = f"blocks.{layer_num}.hook_resid_post"
            else:
                raise ValueError("Could not determine hook_point")
        
        print(f"Hook point: {self.hook_point}")
        print(f"SAE dimensions: d_in={self.sae.cfg.d_in}, d_sae={self.sae.cfg.d_sae}")
        self._loaded = True
    
    def _build_prompt(self, feature_name: str) -> str:
        """Build contextual prompt for feature extraction."""
        domain_prompts = {
            "healthcare": f"In a clinical dataset, the column '{feature_name}' represents a patient health metric: {feature_name}.",
            "finance": f"In a financial dataset, the column '{feature_name}' represents an economic indicator: {feature_name}.",
            "generic": f"Dataset column: '{feature_name}'. This feature measures: {feature_name}."
        }
        return domain_prompts.get(self.config.domain, domain_prompts["generic"])
    
    def extract_feature_vectors(
        self, 
        feature_names: List[str],
        use_sae: bool = True,
        batch_size: int = 8
    ) -> torch.Tensor:
        """
        Extract SAE (or residual) representations for each feature.
        
        Returns:
            Tensor of shape [n_features, d_sae] or [n_features, d_model]
        """
        self._load_models()
        
        all_vectors = []
        
        # Process in batches for memory efficiency
        for i in range(0, len(feature_names), batch_size):
            batch_names = feature_names[i:i+batch_size]
            prompts = [self._build_prompt(name) for name in batch_names]
            
            with torch.no_grad():
                _, cache = self.model.run_with_cache(prompts, names_filter=[self.hook_point])
                # Last token position for each prompt
                resid_acts = cache[self.hook_point][:, -1, :]  # [batch, d_model]
                
                if use_sae:
                    feature_acts = self.sae.encode(resid_acts)  # [batch, d_sae]
                    all_vectors.append(feature_acts)
                else:
                    all_vectors.append(resid_acts)
        
        return torch.cat(all_vectors, dim=0)
    
    def compute_similarity(
        self,
        vec_a: torch.Tensor,
        vec_b: torch.Tensor,
        method: Optional[str] = None,
        threshold: Optional[float] = None
    ) -> float:
        """Compute similarity between two feature vectors."""
        method = method or self.config.similarity_method
        threshold = threshold if threshold is not None else self.config.activation_threshold
        
        if method == "cosine":
            return F.cosine_similarity(vec_a.unsqueeze(0), vec_b.unsqueeze(0)).item()
        
        elif method == "jaccard":
            # Binary Jaccard
            active_a = (vec_a > threshold).float()
            active_b = (vec_b > threshold).float()
            intersection = torch.sum(torch.min(active_a, active_b))
            union = torch.sum(torch.max(active_a, active_b))
            return (intersection / union).item() if union > 0 else 0.0
        
        elif method == "weighted_jaccard":
            # Weighted by activation magnitude
            active_a = torch.clamp(vec_a, min=0) * (vec_a > threshold)
            active_b = torch.clamp(vec_b, min=0) * (vec_b > threshold)
            intersection = torch.sum(torch.min(active_a, active_b))
            union = torch.sum(torch.max(active_a, active_b))
            return (intersection / union).item() if union > 0 else 0.0
        
        elif method == "ensemble":
            cosine = self.compute_similarity(vec_a, vec_b, "cosine")
            jaccard = self.compute_similarity(vec_a, vec_b, "jaccard", threshold)
            return 0.5 * cosine + 0.5 * jaccard
        
        else:
            raise ValueError(f"Unknown similarity method: {method}")
    
    def _analyze_sparsity(self, vectors: torch.Tensor) -> Dict:
        """Analyze sparsity of extracted vectors."""
        threshold = self.config.activation_threshold
        active_mask = (vectors > threshold).float()
        
        return {
            "mean_active": active_mask.sum(dim=-1).mean().item(),
            "std_active": active_mask.sum(dim=-1).std().item(),
            "density": active_mask.mean().item(),
            "mean_activation": vectors[vectors > threshold].mean().item() if (vectors > threshold).any() else 0.0,
        }
    
    def _get_top_features(self, vectors: torch.Tensor, feature_names: List[str], k: int = 10) -> Dict[str, List[int]]:
        """Get top-k SAE features for each column (for interpretability)."""
        top_features = {}
        for i, name in enumerate(feature_names):
            vec = vectors[i]
            top_k = torch.topk(vec, k=min(k, len(vec)))
            top_features[name] = top_k.indices.tolist()
        return top_features
    
    def extract(
        self, 
        feature_names: List[str],
        compute_interpretability: bool = True
    ) -> ExtractionResult:
        """
        Main extraction method: builds complete similarity matrix.
        
        Args:
            feature_names: List of dataset column names
            compute_interpretability: Whether to compute top SAE features per column
            
        Returns:
            ExtractionResult with similarity matrix and metadata
        """
        print(f"Extracting vectors for {len(feature_names)} features...")
        vectors = self.extract_feature_vectors(feature_names, use_sae=True)
        
        # Analyze sparsity
        sparsity_stats = self._analyze_sparsity(vectors)
        print(f"Sparsity: {sparsity_stats['mean_active']:.1f} active features (density: {sparsity_stats['density']*100:.2f}%)")
        
        # Build similarity matrix
        n = len(feature_names)
        matrix = np.zeros((n, n))
        
        print("Computing similarity matrix...")
        for i in tqdm(range(n)):
            for j in range(n):
                if i == j:
                    matrix[i, j] = 1.0
                else:
                    matrix[i, j] = self.compute_similarity(vectors[i], vectors[j])
        
        # Interpretability: top features per column
        top_features = None
        if compute_interpretability:
            top_features = self._get_top_features(vectors, feature_names)
        
        return ExtractionResult(
            feature_names=feature_names,
            similarity_matrix=matrix,
            feature_vectors=vectors.cpu().numpy(),
            top_features_per_column=top_features,
            sparsity_stats=sparsity_stats,
            config=self.config
        )
    
    def get_shared_features(self, result: ExtractionResult, col_a: str, col_b: str, k: int = 5) -> List[Tuple[int, float, float]]:
        """
        Get SAE features that are active in both columns.
        Returns list of (feature_idx, activation_a, activation_b).
        
        Useful for interpretability: "Cholesterol and BloodPressure share SAE feature 1234"
        """
        if result.feature_vectors is None:
            raise ValueError("feature_vectors not available in result")
        
        idx_a = result.feature_names.index(col_a)
        idx_b = result.feature_names.index(col_b)
        
        vec_a = result.feature_vectors[idx_a]
        vec_b = result.feature_vectors[idx_b]
        
        threshold = self.config.activation_threshold
        active_both = (vec_a > threshold) & (vec_b > threshold)
        
        shared = []
        for feat_idx in np.where(active_both)[0]:
            shared.append((int(feat_idx), float(vec_a[feat_idx]), float(vec_b[feat_idx])))
        
        # Sort by minimum activation (features strongly active in both)
        shared.sort(key=lambda x: min(x[1], x[2]), reverse=True)
        return shared[:k]


# =============================================================================
# Convenience functions
# =============================================================================

def extract_for_dataset(
    feature_names: List[str],
    output_dir: Path,
    config: Optional[SAEConfig] = None,
) -> ExtractionResult:
    """
    Convenience function to extract and save semantic prior for a dataset.
    """
    prior = SAESemanticPrior(config)
    result = prior.extract(feature_names)
    result.save(output_dir)
    return result


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract SAE semantic prior")
    parser.add_argument("--features", nargs="+", default=["Age", "Sex", "Cholesterol", "BloodPressure", "HeartRate"])
    parser.add_argument("--output", type=str, default="./results/sae_prior")
    parser.add_argument("--domain", type=str, default="healthcare")
    parser.add_argument("--method", type=str, default="jaccard", choices=["jaccard", "weighted_jaccard", "cosine", "ensemble"])
    parser.add_argument("--threshold", type=float, default=0.1)
    args = parser.parse_args()
    
    config = SAEConfig(
        domain=args.domain,
        similarity_method=args.method,
        activation_threshold=args.threshold
    )
    
    result = extract_for_dataset(
        feature_names=args.features,
        output_dir=Path(args.output),
        config=config
    )
    
    print("\n=== Similarity Matrix ===")
    print(result.to_dataframe().round(3))
    
    print("\n=== Top SAE Features per Column ===")
    for col, feats in (result.top_features_per_column or {}).items():
        print(f"  {col}: {feats[:5]}")
