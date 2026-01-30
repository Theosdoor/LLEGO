#%%
# ==========================================
# 1. SETUP & IMPORTS
# ==========================================
import os
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
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
    dtype="bfloat16",
    center_unembed=False,
    )

# Load the Sparse Autoencoder (SAE)
sae = SAE.from_pretrained(
    release=SAE_RELEASE,
    sae_id=SAE_ID,
    device=device
)

hook_point = getattr(sae.cfg, "hook_name", None) or getattr(sae.cfg, "hook_point", None)
if not hook_point:
    raise ValueError("Could not determine hook_point from SAE config")

print(f"Loaded Model & SAE. SAE Hook Point: {hook_point}")

#%%
# ==========================================
# 3. DEFINE EXTRACTION LOGIC (The "White-Box" Core)
# ==========================================

def get_concept_vector(feature_names: str, use_sae=True):
    """
    Extracts the internal representation of a tabular feature.
    
    Args:
        feature_name: The column name (e.g. "Cholesterol")
        use_sae: If True, returns sparse SAE features. If False, returns raw residual stream.
    """
    # 1. Contextual Prompting
    # We wrap the feature in a natural sentence to force the model to "think" about its meaning.
    # Without this, "CP" might be interpreted as "Club Penguin" instead of "Chest Pain".
    prompts = [
        f"Data column: '{feat}'. Description: The patient's {feat}" 
        for feat in feature_names
    ]    
    # 2. Run Model with Cache
    # We only need the activation at the specific hook point the SAE was trained on.
    with torch.no_grad():
        _, cache = model.run_with_cache(prompts, names_filter=[hook_point])
        resid_acts = cache[hook_point][0, -1, :]
        
    if not use_sae:
        return resid_acts # Shape: [batch_size, seq_len, d_model] -> [batch_size, d_model]

    # 3. Encode with SAE
    # Transforms dense vector -> Sparse feature vector (d_sae size, e.g. 16k)
    with torch.no_grad():
        feature_acts = sae.encode(resid_acts)
    
    return feature_acts

def compute_similarity(vec_a, vec_b, method="jaccard", threshold=0.1):
    """
    Computes semantic similarity between two concept vectors.
    """
    if method == "cosine":
        return F.cosine_similarity(vec_a.unsqueeze(0), vec_b.unsqueeze(0)).item()
    
    elif method == "jaccard":
        # Jaccard = Intersection / Union of ACTIVE features
        # Only makes sense for SAE vectors (which are sparse)
        
        # Threshold to consider a feature "active" (filter out noise)
        active_a = (vec_a > threshold).float()
        active_b = (vec_b > threshold).float()
        
        intersection = torch.sum(torch.min(active_a, active_b))
        union = torch.sum(torch.max(active_a, active_b))
        
        if union == 0: return 0.0
        return (intersection / union).item()

#%%
# ==========================================
# 4. EXECUTE EXTRACTION (The "One-Shot" Step)
# ==========================================

# Mock Dataset Columns (Heart Disease Example)
# Note: "CP" is Chest Pain, "FBS" is Fasting Blood Sugar
columns = ["Age", "Sex", "CP", "RestingBP", "Cholesterol", "FBS", "MaxHR", "ExAng"]

print("Extracting Concept Vectors...")
vectors_tensor = get_concept_vectors_batched(columns, use_sae=True)


# Build the Semantic Matrix
n_cols = len(columns)
matrix = np.zeros((n_cols, n_cols))

for i in tqdm(range(n_cols)):
    for j in range(n_cols):
        if i == j:
            matrix[i, j] = 1.0
        else:
            sim = compute_similarity(
                vectors_tensor[i], 
                vectors_tensor[j], 
                method="jaccard",
                threshold=0.1 # Adjust if heatmap is all zeros
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

plt.figure(figsize=(10, 6))
sns.heatmap(semantic_df, annot=True, cmap="viridis", fmt=".2f")
plt.title(f"SAE-Derived Affinity ({MODEL_NAME})\nMethod: Jaccard (Last Token)")
plt.tight_layout()

fig_path = os.path.join(RESULTS_DIR, "semantic_affinity_matrix.png")
plt.savefig(fig_path, dpi=300)
print(f"Heatmap saved to {fig_path}")
plt.close()

# Interpretation Check:
# "Cholesterol" should have higher similarity to "RestingBP" or "MaxHR" (Health/Bio)
# than to "Sex" (Demographic).

#%%
# ==========================================
# 6. MOCK OPTIMIZATION (The "Distilled" Logic)
# ==========================================
# This demonstrates how you use the matrix in the Genetic Algorithm (GA)

def get_mutation_candidate(current_feature, semantic_matrix, all_features):
    """
    Selects a feature to swap based on semantic affinity.
    """
    # 1. Get row for current feature
    probs = semantic_matrix.loc[current_feature].values
    
    # 2. Zero out self-loop (don't swap with self)
    current_idx = all_features.index(current_feature)
    probs[current_idx] = 0
    
    # 3. Normalize to probability distribution
    if probs.sum() == 0:
        # Fallback to random if no semantic links found
        probs = np.ones(len(probs)) 
    
    probs = probs / probs.sum()
    
    # 4. Sample
    chosen_feature = np.random.choice(all_features, p=probs)
    return chosen_feature

# Demo
current = "Cholesterol"
print(f"\n[Mutation Logic Demo]")
print(f"Current Node Split: {current}")
print("Candidate Probabilities from SAE:")
print(semantic_df.loc[current].sort_values(ascending=False).head(3))

next_feature = get_mutation_candidate(current, semantic_df, columns)
print(f"-> Mutated to: {next_feature} (guided by SAE priors)")