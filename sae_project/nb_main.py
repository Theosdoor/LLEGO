#%%
# ==========================================
# 1. SETUP & IMPORTS
# ==========================================
import os
import re
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from typing import List, Optional, Literal
from transformer_lens import HookedTransformer
from sae_lens import SAE

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Configuration
MODEL_NAME = "gemma-2-2b"
# https://huggingface.co/google/gemma-scope-2b-pt-res
SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"
SAE_ID = "layer_12/width_16k/canonical" 

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on {device}...")

RESULTS_DIR = "./results/"
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

#%%
# ==========================================
# 2. LOAD MODELS (The Heavy Lifting)
# ==========================================
# Load the base model (HookedTransformer wraps HuggingFace models for mech interp)
model = HookedTransformer.from_pretrained(
    MODEL_NAME,
    device=device,
    dtype=torch.bfloat16,  # Fixed: use torch.bfloat16 instead of string
    center_unembed=False,
)

# Load the Sparse Autoencoder (SAE)
sae = SAE.from_pretrained(
    release=SAE_RELEASE,
    sae_id=SAE_ID,
    device=device
)

# Try multiple ways to get hook_point (sae_lens API varies by SAE type)
hook_point = (
    getattr(sae, "hook_name", None) or  # gemma-scope SAEs store it on the object
    getattr(sae.cfg, "hook_name", None) or 
    getattr(sae.cfg, "hook_point", None)
)

# Fallback: construct from SAE_ID (e.g., "layer_12/..." -> "blocks.12.hook_resid_post")
if not hook_point:
    layer_match = re.search(r"layer_(\d+)", SAE_ID)
    if layer_match:
        layer_num = layer_match.group(1)
        hook_point = f"blocks.{layer_num}.hook_resid_post"
        print(f"Constructed hook_point from SAE_ID: {hook_point}")
    else:
        raise ValueError("Could not determine hook_point from SAE config or SAE_ID")

print(f"Loaded Model & SAE. SAE Hook Point: {hook_point}")
print(f"SAE dimensions: d_in={sae.cfg.d_in}, d_sae={sae.cfg.d_sae}")

#%%
# ==========================================
# 3. DEFINE EXTRACTION LOGIC (The "White-Box" Core)
# ==========================================

def build_semantic_prompt(feature_name: str, domain: str = "healthcare") -> str:
    """
    Build a contextual prompt that disambiguates feature meaning.
    
    The key insight: LLMs interpret tokens based on context. 
    "FBS" alone might mean anything, but with medical context, 
    it activates the right semantic features.
    """
    domain_contexts = {
        "healthcare": f"In a clinical dataset, the column '{feature_name}' represents: {feature_name}. This is a patient health metric.",
        "finance": f"In a financial dataset, the column '{feature_name}' represents: {feature_name}. This is an economic indicator.",
        "generic": f"Data column: '{feature_name}'. Description: {feature_name}."
    }
    return domain_contexts.get(domain, domain_contexts["generic"])


def get_concept_vectors_batched(
    feature_names: List[str], 
    use_sae: bool = True,
    domain: str = "healthcare"
) -> torch.Tensor:
    """
    Extracts internal representations for multiple features in a batch.
    
    Args:
        feature_names: List of column names (e.g., ["Cholesterol", "Age"])
        use_sae: If True, returns sparse SAE features. If False, returns raw residual stream.
        domain: Context domain for prompting ("healthcare", "finance", "generic")
        
    Returns:
        Tensor of shape [num_features, d_sae] if use_sae else [num_features, d_model]
    """
    # Build contextual prompts for each feature
    prompts = [build_semantic_prompt(feat, domain) for feat in feature_names]
    
    with torch.no_grad():
        # Run model and capture activations at the SAE hook point
        _, cache = model.run_with_cache(prompts, names_filter=[hook_point])
        
        # Extract last token activations for each prompt
        # Shape: [batch_size, seq_len, d_model] -> [batch_size, d_model]
        resid_acts = cache[hook_point][:, -1, :]
        
    if not use_sae:
        return resid_acts
    
    # Encode through SAE
    with torch.no_grad():
        feature_acts = sae.encode(resid_acts)
    
    return feature_acts


def get_concept_vector(
    feature_name: str, 
    use_sae: bool = True,
    domain: str = "healthcare"
) -> torch.Tensor:
    """Single-feature wrapper for get_concept_vectors_batched."""
    return get_concept_vectors_batched([feature_name], use_sae, domain)[0]


def compute_similarity(
    vec_a: torch.Tensor, 
    vec_b: torch.Tensor, 
    method: Literal["cosine", "jaccard", "weighted_jaccard", "ensemble"] = "jaccard", 
    threshold: float = 0.1
) -> float:
    """
    Computes semantic similarity between two concept vectors.
    
    Methods:
        - cosine: Standard cosine similarity (works on any vectors)
        - jaccard: Binary intersection/union of active features (sparse-friendly)
        - weighted_jaccard: Jaccard weighted by activation magnitude (captures intensity)
        - ensemble: Average of cosine and jaccard (robust fallback)
    """
    if method == "cosine":
        return F.cosine_similarity(vec_a.unsqueeze(0), vec_b.unsqueeze(0)).item()
    
    elif method == "jaccard":
        # Binary Jaccard: Intersection/Union of ACTIVE features
        active_a = (vec_a > threshold).float()
        active_b = (vec_b > threshold).float()
        
        intersection = torch.sum(torch.min(active_a, active_b))
        union = torch.sum(torch.max(active_a, active_b))
        
        if union == 0: 
            return 0.0
        return (intersection / union).item()
    
    elif method == "weighted_jaccard":
        # Weighted Jaccard: Uses activation magnitudes, not just binary presence
        # This captures "how strongly" features are shared
        active_a = torch.clamp(vec_a, min=0)  # ReLU to handle any negatives
        active_b = torch.clamp(vec_b, min=0)
        
        # Apply threshold to filter noise
        active_a = active_a * (active_a > threshold)
        active_b = active_b * (active_b > threshold)
        
        intersection = torch.sum(torch.min(active_a, active_b))
        union = torch.sum(torch.max(active_a, active_b))
        
        if union == 0:
            return 0.0
        return (intersection / union).item()
    
    elif method == "ensemble":
        # Robust fallback: average of cosine and jaccard
        cosine_sim = compute_similarity(vec_a, vec_b, method="cosine")
        jaccard_sim = compute_similarity(vec_a, vec_b, method="jaccard", threshold=threshold)
        return 0.5 * cosine_sim + 0.5 * jaccard_sim
    
    else:
        raise ValueError(f"Unknown similarity method: {method}")


def analyze_sparsity(vectors: torch.Tensor, threshold: float = 0.1) -> dict:
    """
    Analyze the sparsity of SAE feature activations.
    Useful for debugging and validating SAE behavior.
    """
    active_mask = (vectors > threshold).float()
    
    return {
        "mean_active_features": active_mask.sum(dim=-1).mean().item(),
        "std_active_features": active_mask.sum(dim=-1).std().item(),
        "max_active_features": active_mask.sum(dim=-1).max().item(),
        "min_active_features": active_mask.sum(dim=-1).min().item(),
        "mean_activation": vectors[vectors > threshold].mean().item() if (vectors > threshold).any() else 0.0,
        "density": active_mask.mean().item(),  # Fraction of active features
    }

#%%
# ==========================================
# 4. EXECUTE EXTRACTION (The "One-Shot" Step)
# ==========================================

# Mock Dataset Columns (Heart Disease Example)
# Note: "CP" is Chest Pain, "FBS" is Fasting Blood Sugar, "ExAng" is Exercise-induced Angina
columns = ["Age", "Sex", "CP", "RestingBP", "Cholesterol", "FBS", "MaxHR", "ExAng"]

print("Extracting Concept Vectors...")
vectors_tensor = get_concept_vectors_batched(columns, use_sae=True, domain="healthcare")

# Sparsity sanity check
sparsity_stats = analyze_sparsity(vectors_tensor, threshold=0.1)
print(f"\n[Sparsity Analysis]")
print(f"  Mean active features: {sparsity_stats['mean_active_features']:.1f} / {sae.cfg.d_sae}")
print(f"  Density: {sparsity_stats['density']*100:.2f}%")
print(f"  Mean activation (>threshold): {sparsity_stats['mean_activation']:.3f}")

# Warn if SAE is too sparse or too dense
if sparsity_stats['mean_active_features'] < 10:
    print("  ⚠️  WARNING: Very few active features. Consider lowering threshold or using raw residuals.")
elif sparsity_stats['density'] > 0.1:
    print("  ⚠️  WARNING: High density. SAE may not be providing good sparsity.")

# Build the Semantic Matrix
print("\nBuilding Semantic Affinity Matrix...")
n_cols = len(columns)
matrix = np.zeros((n_cols, n_cols))

for i in tqdm(range(n_cols), desc="Computing similarities"):
    for j in range(n_cols):
        if i == j:
            matrix[i, j] = 1.0
        else:
            sim = compute_similarity(
                vectors_tensor[i], 
                vectors_tensor[j], 
                method="jaccard",
                threshold=0.1
            )
            matrix[i, j] = sim

# Convert to DataFrame for visualization
semantic_df = pd.DataFrame(matrix, index=columns, columns=columns)
csv_path = os.path.join(RESULTS_DIR, "semantic_affinity_matrix.csv")
semantic_df.to_csv(csv_path)
print(f"Matrix saved to {csv_path}")

#%%
# ==========================================
# 5. VISUALIZATION & SANITY CHECK
# ==========================================

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Heatmap
sns.heatmap(semantic_df, annot=True, cmap="viridis", fmt=".2f", ax=axes[0])
axes[0].set_title(f"SAE-Derived Affinity ({MODEL_NAME})\nMethod: Jaccard (Last Token)")

# Sanity Check: Compare with expected groupings
# In healthcare, we expect bio-metrics to cluster (Cholesterol, RestingBP, MaxHR)
# and demographics to be separate (Age, Sex)
expected_similar = [("Cholesterol", "RestingBP"), ("Cholesterol", "MaxHR"), ("RestingBP", "MaxHR")]
expected_dissimilar = [("Cholesterol", "Sex"), ("Age", "MaxHR")]

sanity_data = []
for pair in expected_similar:
    sanity_data.append({"pair": f"{pair[0]}-{pair[1]}", "similarity": semantic_df.loc[pair[0], pair[1]], "expected": "similar"})
for pair in expected_dissimilar:
    sanity_data.append({"pair": f"{pair[0]}-{pair[1]}", "similarity": semantic_df.loc[pair[0], pair[1]], "expected": "dissimilar"})

sanity_results = pd.DataFrame(sanity_data)
colors = ['green' if x == 'similar' else 'red' for x in sanity_results['expected']]
axes[1].barh(sanity_results['pair'], sanity_results['similarity'], color=colors, alpha=0.7)
axes[1].set_xlabel("Jaccard Similarity")
axes[1].set_title("Sanity Check: Expected Groupings\n(Green=should be similar, Red=should be dissimilar)")
axes[1].axvline(x=semantic_df.values[np.triu_indices(n_cols, k=1)].mean(), color='blue', linestyle='--', label='Mean similarity')
axes[1].legend()

plt.tight_layout()
fig_path = os.path.join(RESULTS_DIR, "semantic_affinity_analysis.png")
plt.savefig(fig_path, dpi=300)
print(f"Analysis saved to {fig_path}")
plt.show()

#%%
# ==========================================
# 6. MOCK OPTIMIZATION (The "Distilled" Logic)
# ==========================================
# This demonstrates how you use the matrix in the Genetic Algorithm (GA)

def get_mutation_candidate(
    current_feature: str, 
    semantic_matrix: pd.DataFrame, 
    all_features: List[str],
    temperature: float = 1.0
) -> str:
    """
    Selects a feature to swap based on semantic affinity.
    
    Args:
        current_feature: The feature to mutate from
        semantic_matrix: DataFrame of pairwise similarities
        all_features: List of all available features
        temperature: Controls exploration (higher = more random)
    """
    # 1. Get row for current feature
    probs = semantic_matrix.loc[current_feature].values.copy()
    
    # 2. Zero out self-loop (don't swap with self)
    current_idx = all_features.index(current_feature)
    probs[current_idx] = 0
    
    # 3. Apply temperature scaling
    if temperature != 1.0 and probs.sum() > 0:
        probs = np.power(probs, 1.0 / temperature)
    
    # 4. Normalize to probability distribution
    if probs.sum() == 0:
        # Fallback to uniform random if no semantic links found
        probs = np.ones(len(probs)) 
        probs[current_idx] = 0
    
    probs = probs / probs.sum()
    
    # 5. Sample
    chosen_feature = np.random.choice(all_features, p=probs)
    return chosen_feature

# Demo
np.random.seed(42)
current = "Cholesterol"
print(f"\n[Mutation Logic Demo]")
print(f"Current Node Split: {current}")
print("Candidate Probabilities from SAE:")
print(semantic_df.loc[current].sort_values(ascending=False).head(3))

next_feature = get_mutation_candidate(current, semantic_df, columns, temperature=1.0)
print(f"-> Mutated to: {next_feature} (guided by SAE priors)")

# Show distribution of mutations over many samples
mutations = [get_mutation_candidate(current, semantic_df, columns) for _ in range(100)]
print(f"\nMutation distribution (100 samples from '{current}'):")
for feat, count in pd.Series(mutations).value_counts().head(5).items():
    print(f"  {feat}: {count}%")