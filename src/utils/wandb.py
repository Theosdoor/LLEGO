from typing import Union

import wandb
from omegaconf import DictConfig

from utils.extraction import flatten_config


def maybe_initialize_wandb(
    cfg: DictConfig, dedicated_exp_name: str, dedicated_exp_dir: str
) -> Union[str, None]:
    """Initialize wandb if necessary."""
    cfg_flat = flatten_config(cfg)
    
    # Keys to completely remove (sensitive or not useful)
    sensitive_keys = ['api_key', 'api-key', 'apikey', 'token', 'password', 'secret']
    keys_to_remove = [
        'log_wandb',  # Always true if we're here
        # Class paths - just implementation details, not useful for analysis
        'crossover', 'filter', 'fitness_eval', 'hof', 'llm_api', 
        'metrics_logger', 'mutation', 'pop_init', 'selection',
    ]
    
    cfg_filtered = {}
    for k, v in cfg_flat.items():
        # Skip sensitive keys
        if any(sens in k.lower() for sens in sensitive_keys):
            continue
        # Skip keys to remove
        if k in keys_to_remove:
            continue
        # Skip class path values (strings containing module paths)
        if isinstance(v, str) and v.startswith('llego.'):
            continue
        cfg_filtered[k] = v
    
    if cfg.log_wandb:
        wandb.init(
            project="LLEGO",
            config=cfg_filtered,
            name=dedicated_exp_name,
            dir=dedicated_exp_dir,
        )
        assert wandb.run is not None
        run_id = wandb.run.id
        assert isinstance(run_id, str)
        return run_id
    else:
        return None
