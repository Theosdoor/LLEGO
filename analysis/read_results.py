# %%
import pickle as pkl
import json
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

# %%
import sys
import os

src_path = os.path.abspath('../src')
parent_path = os.path.dirname(src_path)
sys.path.insert(0, parent_path)

# %%
# unzip results file
!unzip results/results_files.zip -d results/

# %% [markdown]
# # Classification

# %% [markdown]
# ### Baselines

# %%
def read_results_baselines(datasets, depths, seeds, methods_config, base_path="./results", folder_name="baselines_classification"):
    
    
    results_data = []
    
    # Process each method
    for depth in depths:
        for method in methods_config:
            config = methods_config[method]
            metric_key = config["metric_key"]
            ordering = config["ordering"]
            
            for dataset in datasets:
                for seed in seeds:
                    # Construct the file path
                    file_pattern = config["file_pattern"].format(seed=seed)
                    file_path = Path(base_path) / folder_name / depth / dataset / method / file_pattern
                    
                    try:
                        # Read and extract result based on method type
                        if config["result_type"] == "json":
                            with open(file_path, "r") as f:
                                result_json = json.load(f)
                            
                            # Navigate through nested dictionary to get the result
                            result_value = result_json
                            for key in config["result_key"]:
                                result_value = result_value[key]
                            
                            metric = result_value
                            
                        elif config["result_type"] == "pkl":
                            with open(file_path, "rb") as f:
                                result_pkl = pkl.load(f)
                            
                            # Extract best individual from the last iteration
                            last_iter = config["last_iter"]
                            population = result_pkl[last_iter]["population"]
                            
                            # Sort by fitness function
                            sorting_function = lambda x: (x.fitness[f"{metric_key}_train"] + 
                                                         2 * x.fitness[f"{metric_key}_val"])
                            reverse = True if ordering == "max" else False
                            population.sort(key=sorting_function, reverse=reverse)
                            
                            best_ind = population[0]
                            metric = best_ind.fitness[f"{metric_key}_test"]
                        
                        # Store the result
                        results_data.append({
                            "depth": depth,
                            "method": method,
                            "dataset": dataset,
                            "seed": seed,
                            metric_key: metric
                        })
                        
                    except Exception as e:
                        print(f"Error reading {file_path}: {str(e)}")
                        # Add a row with NaN for balanced_accuracy
                        results_data.append({
                            "depth": depth,
                            "method": method,
                            "dataset": dataset,
                            "seed": seed,
                            metric_key: np.nan
                        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results_data)
    return results_df

def read_results_llego(datasets, depths, seeds, methods_config, base_path="/home/nvth2/llego/results", folder_name="llego_classification"):
    
    # Initialize an empty list to collect results
    results_data = []
    metric_key = methods_config["LLEGO"]["metric_key"]
    ordering = methods_config["LLEGO"]["ordering"]
    
    for dataset in datasets:
        for depth in depths:
            for seed in seeds:
                # Construct the file path
                file_path = Path(base_path) / folder_name / dataset / depth / str(seed) / "llego" / "llego_hof.pkl"
                
                try:
                    # Read the hall of fame pickle file
                    with open(file_path, "rb") as f:
                        hof = pkl.load(f)
                    
                    # Calculate average fitness for each individual
                    average_fitness = [(0.5 * ind.fitness[f'{metric_key}_train'] + 
                                        ind.fitness[f'{metric_key}_val']) / 2 
                                       for ind in hof]
                    
                    # Find the index of the best individual
                    best_index = average_fitness.index(max(average_fitness)) if ordering == "max" else average_fitness.index(min(average_fitness))
                    
                    # Get the best individual
                    best_ind = hof[best_index]
                    
                    # Extract the metric
                    metric = best_ind.fitness[f"{metric_key}_test"]
                    
                    # Store the result
                    results_data.append({
                        "depth": depth,
                        "method": "LLEGO",  # Adding method name for consistency
                        "dataset": dataset,
                        "seed": seed,
                        metric_key: metric
                    })
                    
                except Exception as e:
                    print(f"Error reading {file_path}: {str(e)}")
                    results_data.append({
                        "depth": depth,
                        "method": "LLEGO",
                        "dataset": dataset,
                        "seed": seed,
                        metric_key: np.nan
                    })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results_data)
    return results_df

def combine_all_results(datasets, depths, seeds, baselines_config, llego_config, base_path="./results", folder_name_baselines="baselines_classification", folder_name_llego="llego_classification"):
    # Get baseline results
    baseline_results = read_results_baselines(
        datasets=datasets, 
        depths=depths, 
        seeds=seeds, 
        methods_config=baselines_config,
        base_path=base_path, 
        folder_name=folder_name_baselines
    )
    
    # Get LLEGO results
    llego_results = read_results_llego(
        datasets=datasets, 
        depths=depths, 
        seeds=seeds, 
        methods_config=llego_config,
        base_path=base_path, 
        folder_name=folder_name_llego
    )
    
    # Combine the results
    combined_results = pd.concat([baseline_results, llego_results], ignore_index=True)
    
    return combined_results

def compute_summary_statistics(results_df, metric_key):
    
    # Group by depth, method, and dataset, then calculate mean and std
    summary = results_df.groupby(['depth', 'method', 'dataset'])[metric_key].agg(['mean', 'std']).reset_index()
    
    return summary


# Define constants
datasets = ["breast", "compas", "credit-g", "diabetes", "heart-statlog", "liver", "vehicle"]
depths = ["3", "4"]
seeds = list(range(5))  
metric_key = "balanced_accuracy"
ordering = "max"
baselines_config = {
        "CART": {
            "file_pattern": "results_seed_{seed}.json",
            "result_type": "json",
            "result_key": ["test", "balanced_accuracy"],
            "metric_key": metric_key,
            "ordering": ordering
        },
        "DL85": {
            "file_pattern": "results_seed_{seed}.json",
            "result_type": "json",
            "result_key": ["test", "balanced_accuracy"],
            "metric_key": metric_key,
            "ordering": ordering
        },
        "GOSDT": {
            "file_pattern": "results_seed_{seed}.json",
            "result_type": "json", 
            "result_key": ["test", "balanced_accuracy"],
            "metric_key": metric_key,
            "ordering": ordering
        },
        "C45": {
            "file_pattern": "results_seed_{seed}.json",
            "result_type": "json",
            "result_key": ["test", "balanced_accuracy"],
            "metric_key": metric_key,
            "ordering": ordering
        },
        # GATREE method (pickle results)
        "GATREE": {
            "file_pattern": "gatree_search_populations_seed_{seed}.pkl",
            "result_type": "pkl",
            "last_iter": 25,
            "metric_key": metric_key,
            "ordering": ordering
        },
        
    }

llego_config = {
    "LLEGO": {
        "metric_key": metric_key,
        "ordering": ordering
    }
}

# Read all results
print("Reading results...")
results_df_clas = combine_all_results(datasets, depths, seeds, baselines_config, llego_config, base_path="./results", folder_name_baselines="baselines_classification", folder_name_llego="llego_classification")
# Save the raw results
results_df_clas.to_csv("artifacts/all_classification_results.csv", index=False)
print(f"Raw results saved to all_classification_results.csv")

# Compute summary statistics
summary_df_clas = compute_summary_statistics(results_df_clas, metric_key)
summary_df_clas.to_csv("artifacts/summary_classification_results.csv", index=False)
print(f"Summary results saved to summary_classification_results.csv")

# Print some basic information
print("\nResults overview:")
print(f"Total results: {len(results_df_clas)}")
print(f"Missing values: {results_df_clas['balanced_accuracy'].isna().sum()}")

# Print final joint summary
print("\nJoint summary by depth-method-dataset:")
print(summary_df_clas.head())

# %% [markdown]
# ### Compute ranks

# %%
def compute_method_ranks(summary_df, depth):
    # Filter by the specified depth
    depth_df = summary_df[summary_df['depth'] == depth].copy()
    
    # Create a list to store rank results
    rank_results = []
    
    # Process each dataset
    for dataset in depth_df['dataset'].unique():
        # Get data for this dataset
        dataset_df = depth_df[depth_df['dataset'] == dataset]
        
        # Skip datasets with too few methods (need at least 2 for ranking)
        if len(dataset_df) < 2:
            continue
            
        ranks = stats.rankdata(-dataset_df['mean'].values)  # Negative for descending order
        
        # Add ranks to the results
        for i, (_, row) in enumerate(dataset_df.iterrows()):
            rank_results.append({
                'method': row['method'],
                'dataset': dataset,
                'rank': ranks[i]
            })
    
    rank_df = pd.DataFrame(rank_results)
    
    method_ranks = rank_df.groupby('method')['rank'].agg(['mean', 'std', 'count']).reset_index()
    
    method_ranks = method_ranks.sort_values('mean')
    
    return method_ranks

# Compute ranks for each depth
depth_ranks = {}
for depth in depths:
    depth_ranks[depth] = compute_method_ranks(summary_df_clas, depth)
    
# Print the ranks
for depth, ranks in depth_ranks.items():
    print(f"\nRanks for depth {depth}:")
    print(ranks)
    print("\n")

# %% [markdown]
# ### Regression results
# 

# %%
# Define constants
datasets = ["abalone", "cars", "cholesterol", "wage", "wine"]
depths = ["3", "4"]
seeds = list(range(5))  # [0, 1, 2, 3, 4]
metric_key = "mse"
ordering = "min"
baselines_config = {
        # Standard methods (JSON results)
        "CART": {
            "file_pattern": "results_seed_{seed}.json",
            "result_type": "json",
            "result_key": ["test", metric_key],
            "metric_key": metric_key,
            "ordering": ordering
        },
        "GATREE": {
            "file_pattern": "gatree_search_populations_seed_{seed}.pkl",
            "result_type": "pkl",
            "last_iter": 25,
            "metric_key": metric_key,
            "ordering": ordering
        },
        
    }

llego_config = {
    "LLEGO": {
        "metric_key": metric_key,
        "ordering": ordering
    }
}

# Read all results
print("Reading results...")
results_df_reg = combine_all_results(datasets, depths, seeds, baselines_config, llego_config, base_path="./results", folder_name_baselines="baselines_regression", folder_name_llego="llego_regression")
# Save the raw results
results_df_reg.to_csv("artifacts/all_regression_results.csv", index=False)


# Compute summary statistics
summary_df_reg = compute_summary_statistics(results_df_reg, metric_key)
summary_df_reg.to_csv("artifacts/summary_regression_results.csv", index=False)

# Print some basic information
print("\nResults overview:")
print(f"Total results: {len(results_df_reg)}")
print(f"Missing values: {results_df_reg[metric_key].isna().sum()}")

# Print final joint summary
print("\nJoint summary by depth-method-dataset:")
print(summary_df_reg.head())


