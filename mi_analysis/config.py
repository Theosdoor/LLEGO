"""
Configuration for MI analysis experiments.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class MIConfig:
    """Configuration for mechanistic interpretability experiments."""
    
    # Model settings
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    device: str = "cuda"
    torch_dtype: str = "bfloat16"
    
    # Analysis settings
    n_samples: int = 50  # Number of crossover prompts to analyze
    batch_size: int = 4
    max_new_tokens: int = 512
    
    # Paths
    output_dir: Path = field(default_factory=lambda: Path("mi_analysis/results"))
    cache_dir: Path = field(default_factory=lambda: Path("mi_analysis/cache"))
    
    # Experiment flags
    save_activations: bool = True
    save_attention: bool = True
    
    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.cache_dir = Path(self.cache_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)


@dataclass  
class SemanticAblationConfig(MIConfig):
    """Config specific to semantic ablation experiments."""
    
    # Datasets to test
    datasets: list = field(default_factory=lambda: [
        "breast",      # High semantic (medical)
        "heart-statlog",  # High semantic (medical)
        "diabetes",    # High semantic (medical)
        "iris",        # Medium semantic (biology)
    ])
    
    # Number of crossover operations per condition
    n_crossovers_per_condition: int = 20
    
    # Seeds for reproducibility
    seeds: list = field(default_factory=lambda: [0, 1, 2])


@dataclass
class FitnessAttentionConfig(MIConfig):
    """Config specific to fitness attention analysis."""
    
    # Layers to analyze attention
    layers_to_analyze: list = field(default_factory=lambda: [0, 8, 16, 24, 31])
    
    # Whether to run causal intervention (swapping fitness labels)
    run_fitness_swap: bool = True


@dataclass
class StructuralPriorConfig(MIConfig):
    """Config specific to structural prior elicitation."""
    
    # Number of trees to generate for distribution analysis
    n_trees_to_generate: int = 100
    
    # Tree depth range to analyze
    max_depth: int = 6
    
    # Use arbitrary feature names to isolate structural preferences
    use_arbitrary_features: bool = True
