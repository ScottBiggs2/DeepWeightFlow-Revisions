import sys
import torch
import numpy as np
from collections import defaultdict
from typing import NamedTuple
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import copy
import logging

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
import copy
from collections import defaultdict
import os
import traceback

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from scipy.stats import entropy, wasserstein_distance
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import os

# Import flowmatching utilities for ResNet handling and canonicalization (git rebasin)
from flowmatching.train_and_generate import load_config, get_data_loader, convert_models_to_weight_space
from flowmatching.canonicalization import get_permuted_models_data
from flowmatching.flow_matching import FlowMatching, WeightSpaceFlowModel
from flowmatching.utils import (
    WeightSpaceObjectResnet,
    count_parameters,
    evaluate_model,
    print_stats as fm_print_stats,
    load_cifar10,
    recalibrate_bn_stats,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(stream=sys.stdout, format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO, datefmt='%I:%M:%S')

@torch.no_grad()
def compute_wrong_indices_batch(models, test_loader, device='cuda'):
    """
    Efficiently compute wrong indices for all models at once.
    Returns a matrix where each row is the wrong prediction mask for a model.
    """
    n_models = len(models)
    all_wrong_masks = []
    
    # Move all models to device and set to eval
    models = [model.to(device).eval() for model in models]
    
    # Collect all predictions
    all_targets = []
    all_predictions = [[] for _ in range(n_models)]
    
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        all_targets.append(target.cpu())
        
        # Get predictions from all models in one pass
        for i, model in enumerate(models):
            output = model(data)
            pred = torch.max(output, 1)[1]
            all_predictions[i].append(pred.cpu())
    
    # Concatenate all targets and predictions
    all_targets = torch.cat(all_targets).numpy()
    
    # Compute wrong masks for all models
    wrong_masks = []
    for i in range(n_models):
        all_preds = torch.cat(all_predictions[i]).numpy()
        wrong_mask = (all_preds != all_targets)
        wrong_masks.append(wrong_mask)
    
    return np.array(wrong_masks)

def compute_wrong_iou_vectorized(mask_a, mask_b):
    """
    Compute IoU between two boolean masks efficiently.
    """
    intersection = np.logical_and(mask_a, mask_b)
    union = np.logical_or(mask_a, mask_b)
    union_sum = np.sum(union)
    
    if union_sum == 0:
        return 1.0  # Both models are perfect
    
    return np.sum(intersection) / union_sum

def compute_max_similarity_vs_originals(models, original_models, test_loader, device='cuda'):
    """
    Compute maximum IoU similarity for each model against all original models.
    This provides a consistent baseline comparison.
    """
    # Get wrong prediction masks for both sets
    all_masks = compute_wrong_indices_batch(models + original_models, test_loader, device)
    model_masks = all_masks[:len(models)]
    original_masks = all_masks[len(models):]
    
    max_similarities = []

    # Detect if 'models' and 'original_models' are the same sequence (identity or elementwise)
    same_lists = False
    try:
        if len(models) == len(original_models) and all(models[k] is original_models[k] for k in range(len(models))):
            same_lists = True
    except Exception:
        same_lists = False

    for i in tqdm(range(len(models)), desc="Computing max similarities vs originals"):
        similarities = []
        for j in range(len(original_models)):
            # When comparing originals vs originals, skip self-comparison to avoid trivial IoU==1.0
            if same_lists and i == j:
                continue
            iou = compute_wrong_iou_vectorized(model_masks[i], original_masks[j])
            similarities.append(iou)

        max_similarities.append(max(similarities) if similarities else 0.0)
    
    return np.array(max_similarities)

def add_noise_to_models(models, noise_std=0.01):
    """Add Gaussian noise to model parameters."""
    noisy_models = []
    for model in models:
        noisy_model = copy.deepcopy(model)
        with torch.no_grad():
            for param in noisy_model.parameters():
                noise = torch.randn_like(param) * noise_std
                param.add_(noise)
        noisy_models.append(noisy_model)
    return noisy_models

def jensen_shannon_distance(p, q):
    """Compute Jensen-Shannon distance between two distributions."""
    # Ensure distributions are normalized
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    # Compute M = (P + Q) / 2
    m = (p + q) / 2
    
    # Compute JS divergence
    js_div = 0.5 * entropy(p, m) + 0.5 * entropy(q, m)
    
    # JS distance is sqrt of JS divergence
    return np.sqrt(js_div)

def nearest_neighbor_analysis(original_models, generated_models, device='cuda', num_bins=100):
    """
    Enhanced nearest neighbor analysis with Wasserstein and Jensen-Shannon distances.
    """
    def flatten_models(models):
        flats = []
        for m in models:
            # If model has a `flatten` method (weight-space object), use it
            if hasattr(m, 'flatten'):
                flat = m.flatten()
            else:
                # Generic fallback: concatenate all parameters
                parts = []
                for p in m.parameters():
                    parts.append(p.detach().cpu().flatten())
                flat = torch.cat(parts)
            flats.append(flat.cpu().numpy())
        return np.stack(flats)

    orig_flat = flatten_models(original_models)
    gen_flat = flatten_models(generated_models)

    print(f"Num Models - Original: {len(orig_flat)}, Generated: {len(gen_flat)}")
    
    # Compute all pairwise distances
    orig_to_orig_distances = cdist(orig_flat, orig_flat, metric='euclidean')
    gen_to_gen_distances = cdist(gen_flat, gen_flat, metric='euclidean')
    orig_to_gen_distances = cdist(orig_flat, gen_flat, metric='euclidean')
    gen_to_orig_distances = orig_to_gen_distances.T
    
    # Extract nearest neighbor distances
    np.fill_diagonal(orig_to_orig_distances, np.inf)
    np.fill_diagonal(gen_to_gen_distances, np.inf)
    
    nn_orig_to_orig = orig_to_orig_distances.min(axis=1)
    nn_gen_to_gen = gen_to_gen_distances.min(axis=1)
    nn_orig_to_gen = orig_to_gen_distances.min(axis=1)
    nn_gen_to_orig = gen_to_orig_distances.min(axis=1)
    
    # Print statistics
    print("\n=== Nearest Neighbor Distance Statistics ===")
    print(f"Original → Original (mean ± std): {nn_orig_to_orig.mean():.4f} ± {nn_orig_to_orig.std():.4f}")
    print(f"Generated → Generated (mean ± std): {nn_gen_to_gen.mean():.4f} ± {nn_gen_to_gen.std():.4f}")
    print(f"Original → Generated (mean ± std): {nn_orig_to_gen.mean():.4f} ± {nn_orig_to_gen.std():.4f}")
    print(f"Generated → Original (mean ± std): {nn_gen_to_orig.mean():.4f} ± {nn_gen_to_orig.std():.4f}")
    
    # Compute histograms
    all_distances = np.concatenate([nn_orig_to_orig, nn_gen_to_gen, nn_orig_to_gen, nn_gen_to_orig])
    min_d, max_d = all_distances.min(), all_distances.max()
    bins = np.linspace(min_d, max_d, num_bins + 1)
    
    hist_orig_orig, _ = np.histogram(nn_orig_to_orig, bins=bins, density=True)
    hist_gen_gen, _ = np.histogram(nn_gen_to_gen, bins=bins, density=True)
    hist_orig_gen, _ = np.histogram(nn_orig_to_gen, bins=bins, density=True)
    hist_gen_orig, _ = np.histogram(nn_gen_to_orig, bins=bins, density=True)
    
    # Enhanced distance metrics
    print("\n=== Enhanced Distance Metrics ===")
    
    # L2 Distance (mean Euclidean distance between distributions)
    try:
        l2_orig_gen = np.mean(orig_to_gen_distances)
        l2_gen_gen = np.mean(gen_to_gen_distances[np.triu_indices_from(gen_to_gen_distances, k=1)])
        l2_orig_orig = np.mean(orig_to_orig_distances[np.triu_indices_from(orig_to_orig_distances, k=1)])
        
        print(f"L2 Distance (Orig→Orig): {l2_orig_orig:.4f}")
        print(f"L2 Distance (Gen→Gen): {l2_gen_gen:.4f}")
        print(f"L2 Distance (Orig→Gen): {l2_orig_gen:.4f}")
    except Exception as e:
        print(f"Error computing L2 distances: {e}")
    
    # Cosine Similarity (mean cosine similarity between distributions)
    try:
        # Compute mean vectors for each group
        orig_mean = np.mean(orig_flat, axis=0)
        gen_mean = np.mean(gen_flat, axis=0)
        
        # Cosine similarity between mean vectors
        cos_sim_means = np.dot(orig_mean, gen_mean) / (np.linalg.norm(orig_mean) * np.linalg.norm(gen_mean))
        
        # Pairwise cosine similarities
        orig_to_gen_cosine = cdist(orig_flat, gen_flat, metric='cosine')
        gen_to_gen_cosine = cdist(gen_flat, gen_flat, metric='cosine')
        orig_to_orig_cosine = cdist(orig_flat, orig_flat, metric='cosine')
        
        # Mean cosine similarities (converting distance to similarity: 1 - distance)
        mean_cos_orig_orig = 1 - np.mean(orig_to_orig_cosine[np.triu_indices_from(orig_to_orig_cosine, k=1)])
        mean_cos_gen_gen = 1 - np.mean(gen_to_gen_cosine[np.triu_indices_from(gen_to_gen_cosine, k=1)])
        mean_cos_orig_gen = 1 - np.mean(orig_to_gen_cosine)
        
        print(f"Cosine Similarity (Orig Mean vs Gen Mean): {cos_sim_means:.4f}")
        print(f"Mean Cosine Similarity (Orig→Orig): {mean_cos_orig_orig:.4f}")
        print(f"Mean Cosine Similarity (Gen→Gen): {mean_cos_gen_gen:.4f}")
        print(f"Mean Cosine Similarity (Orig→Gen): {mean_cos_orig_gen:.4f}")
    except Exception as e:
        print(f"Error computing Cosine similarities: {e}")
    
    # Wasserstein distances
    try:
        wasserstein_gen_gen = wasserstein_distance(nn_orig_to_orig, nn_gen_to_gen)
        wasserstein_orig_gen = wasserstein_distance(nn_orig_to_orig, nn_orig_to_gen)
        wasserstein_gen_orig = wasserstein_distance(nn_orig_to_orig, nn_gen_to_orig)
        
        print(f"Wasserstein(Orig→Orig, Gen→Gen): {wasserstein_gen_gen:.4f}")
        print(f"Wasserstein(Orig→Orig, Orig→Gen): {wasserstein_orig_gen:.4f}")
        print(f"Wasserstein(Orig→Orig, Gen→Orig): {wasserstein_gen_orig:.4f}")
    except Exception as e:
        print(f"Error computing Wasserstein distances: {e}")
    
    # Jensen-Shannon distances
    try:
        js_gen_gen = jensen_shannon_distance(hist_orig_orig + 1e-12, hist_gen_gen + 1e-12)
        js_orig_gen = jensen_shannon_distance(hist_orig_orig + 1e-12, hist_orig_gen + 1e-12)
        js_gen_orig = jensen_shannon_distance(hist_orig_orig + 1e-12, hist_gen_orig + 1e-12)
        
        print(f"Jensen-Shannon(Orig→Orig, Gen→Gen): {js_gen_gen:.4f}")
        print(f"Jensen-Shannon(Orig→Orig, Orig→Gen): {js_orig_gen:.4f}")
        print(f"Jensen-Shannon(Orig→Orig, Gen→Orig): {js_gen_orig:.4f}")
    except Exception as e:
        print(f"Error computing Jensen-Shannon distances: {e}")
    
    return {
        "nn_orig_to_orig": nn_orig_to_orig,
        "nn_gen_to_gen": nn_gen_to_gen,
        "nn_orig_to_gen": nn_orig_to_gen,
        "nn_gen_to_orig": nn_gen_to_orig,
        "hist_orig_orig": hist_orig_orig,
        "hist_gen_gen": hist_gen_gen,
        "hist_orig_gen": hist_orig_gen,
        "hist_gen_orig": hist_gen_orig,
        "bins": bins
    }

def plot_similarity_vs_accuracy(original_models, generated_models, test_loader, model_type, device='cuda'):
    """
    Create scatter plots showing max similarity vs test accuracy for different model types.
    Now compares all models against original models as baseline.
    """
    # Test accuracies
    orig_accuracies = [test_mlp(model, test_loader) for model in original_models]
    gen_accuracies = [test_mlp(model, test_loader) for model in generated_models]
    
    # Max similarities vs original models (as baseline)
    orig_max_sim = compute_max_similarity_vs_originals(original_models, original_models, test_loader, device)
    gen_max_sim = compute_max_similarity_vs_originals(generated_models, original_models, test_loader, device)
    
    # Generate noisy versions of ORIGINAL models (not generated)
    orig_noise_001 = add_noise_to_models(original_models, 0.001)
    orig_noise_01 = add_noise_to_models(original_models, 0.01)
    
    orig_noise_001_acc = [test_mlp(model, test_loader) for model in orig_noise_001]
    orig_noise_01_acc = [test_mlp(model, test_loader) for model in orig_noise_01]
    
    orig_noise_001_sim = compute_max_similarity_vs_originals(orig_noise_001, original_models, test_loader, device)
    orig_noise_01_sim = compute_max_similarity_vs_originals(orig_noise_01, original_models, test_loader, device)
    
    # Prepare data for CSV export
    plot_data = []
    
    # Add original models data
    for i, (acc, sim) in enumerate(zip(orig_accuracies, orig_max_sim)):
        plot_data.append({
            'model_type': 'Original',
            'model_id': i,
            'accuracy': acc,
            'max_iou_similarity': sim
        })
    
    # Add generated models data
    for i, (acc, sim) in enumerate(zip(gen_accuracies, gen_max_sim)):
        plot_data.append({
            'model_type': 'Generated',
            'model_id': i,
            'accuracy': acc,
            'max_iou_similarity': sim
        })
    
    # Add original + noise data
    for i, (acc, sim) in enumerate(zip(orig_noise_001_acc, orig_noise_001_sim)):
        plot_data.append({
            'model_type': 'Original + N(0, 0.001)',
            'model_id': i,
            'accuracy': acc,
            'max_iou_similarity': sim
        })
    
    for i, (acc, sim) in enumerate(zip(orig_noise_01_acc, orig_noise_01_sim)):
        plot_data.append({
            'model_type': 'Original + N(0, 0.01)',
            'model_id': i,
            'accuracy': acc,
            'max_iou_similarity': sim
        })
    
    # Save to CSV
    import pandas as pd
    df = pd.DataFrame(plot_data)
    df.to_csv(f'similarity_vs_accuracy_data_{model_type}.csv', index=False)
    logging.info(f"Saved scatter plot data to similarity_vs_accuracy_data_{model_type}.csv")
    
    # Create scatter plot

    plt.figure(figsize=(12, 8)) 
    plt.rcParams.update({'font.size': 16}) # hardcode to make them look pretty
   
    plt.scatter(orig_max_sim, orig_accuracies, alpha=0.6, label='Original', color='blue', s=50)
    plt.scatter(gen_max_sim, gen_accuracies, alpha=0.6, label='Generated', color='red', s=50)
    plt.scatter(orig_noise_001_sim, orig_noise_001_acc, alpha=0.6, label='Original + N(0, 0.001)', color='orange', s=50)
    plt.scatter(orig_noise_01_sim, orig_noise_01_acc, alpha=0.6, label='Original + N(0, 0.01)', color='green', s=50)
    
    plt.xlabel('Maximum IoU Similarity vs Original Models')
    plt.ylabel('Test Accuracy (%)')
    # plt.xlim(0.6, 1.0) # hardcode to make them look pretty
    # plt.ylim(0.8, 1.0) # hardcode to make them look pretty
    # plt.title(f'Max Similarity vs Test Accuracy - {model_type}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'similarity_vs_accuracy_{model_type}.png', dpi=150, bbox_inches='tight')
    plt.show()
    # plt.close()
    
    return {
        'orig_acc': orig_accuracies, 'gen_acc': gen_accuracies,
        'orig_noise_001_acc': orig_noise_001_acc, 'orig_noise_01_acc': orig_noise_01_acc,
        'orig_sim': orig_max_sim, 'gen_sim': gen_max_sim,
        'orig_noise_001_sim': orig_noise_001_sim, 'orig_noise_01_sim': orig_noise_01_sim,
        'plot_data_df': df
    }

def plot_similarity_vs_accuracy_violin(original_models, generated_models, test_loader, model_type, device='cuda'):
    """
    Create violin plots showing test accuracy distributions for different noise levels,
    arranged by maximum similarity. Now uses original models with noise.
    """
    # Generate different noise levels for ORIGINAL models
    orig_noise_0001 = add_noise_to_models(original_models, 0.0001)
    orig_noise_001 = add_noise_to_models(original_models, 0.001)
    orig_noise_01 = add_noise_to_models(original_models, 0.01)
    orig_noise_1 = add_noise_to_models(original_models, 0.1)
    
    # Compute accuracies
    gen_acc = [test_mlp(model, test_loader) for model in generated_models]
    orig_noise_0001_acc = [test_mlp(model, test_loader) for model in orig_noise_0001]
    orig_noise_001_acc = [test_mlp(model, test_loader) for model in orig_noise_001]
    orig_noise_01_acc = [test_mlp(model, test_loader) for model in orig_noise_01]
    orig_noise_1_acc = [test_mlp(model, test_loader) for model in orig_noise_1]
    
    # Compute max similarities vs original models
    gen_sim = compute_max_similarity_vs_originals(generated_models, original_models, test_loader, device)
    orig_noise_0001_sim = compute_max_similarity_vs_originals(orig_noise_0001, original_models, test_loader, device)
    orig_noise_001_sim = compute_max_similarity_vs_originals(orig_noise_001, original_models, test_loader, device)
    orig_noise_01_sim = compute_max_similarity_vs_originals(orig_noise_01, original_models, test_loader, device)
    orig_noise_1_sim = compute_max_similarity_vs_originals(orig_noise_1, original_models, test_loader, device)
    
    # Prepare data for violin plot and CSV export
    data_for_plot = []
    categories = ['Generated', 'Orig + N(0,0.0001)', 'Orig + N(0,0.001)', 'Orig + N(0,0.01)', 'Orig + N(0,0.1)']
    accuracies = [gen_acc, orig_noise_0001_acc, orig_noise_001_acc, orig_noise_01_acc, orig_noise_1_acc]
    similarities = [gen_sim, orig_noise_0001_sim, orig_noise_001_sim, orig_noise_01_sim, orig_noise_1_sim]
    
    violin_data = []
    for i, (cat, acc, sim) in enumerate(zip(categories, accuracies, similarities)):
        for j, (a, s) in enumerate(zip(acc, sim)):
            data_point = {
                'Category': cat,
                'model_id': j,
                'Accuracy': a,
                'Max_Similarity': s,
                'Position': i
            }
            data_for_plot.append(data_point)
            violin_data.append(data_point)
    
    # Save violin plot data to CSV
    import pandas as pd
    violin_df = pd.DataFrame(violin_data)
    violin_df.to_csv(f'violin_plot_data_{model_type}.csv', index=False)
    logging.info(f"Saved violin plot data to violin_plot_data_{model_type}.csv")
    
    df = pd.DataFrame(data_for_plot)
    
    # Create the plot
    plt.rcParams.update({'font.size': 16}) # hardcode to make them look pretty

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Top plot: Violin plot of accuracies
    sns.violinplot(data=df, x='Category', y='Accuracy', ax=ax1)
    ax1.set_title(f'Test Accuracy Distributions - {model_type}')
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Bottom plot: Violin plot of max similarities
    sns.violinplot(data=df, x='Category', y='Max_Similarity', ax=ax2)
    # ax2.set_title(f'Max IoU Similarity vs Original Models - {model_type}')
    ax2.set_ylabel('Maximum IoU Similarity')
    ax2.set_xlabel('Model Type')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'violin_plots_{model_type}.png', dpi=150, bbox_inches='tight')
    plt.show()
    # plt.close()
    
    # Log aggregate IoUs
    print(f"\n=== Aggregate IoU Statistics vs Original Models for {model_type} ===")
    for cat, sim in zip(categories, similarities):
        print(f"{cat} - Mean Max IoU: {np.mean(sim):.4f} ± {np.std(sim):.4f}")
    
    return violin_df

# [Previous classes remain the same: MLP, PermutationSpec, WeightSpaceObject, etc.]

class MLP(nn.Module):
    def __init__(self, init_type='xavier', seed=None):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(784, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 10)
        
        if seed is not None:
            torch.manual_seed(seed)

        self.init_weights(init_type)

    def init_weights(self, init_type):
        if init_type == 'xavier':
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.xavier_uniform_(self.fc2.weight)
            nn.init.xavier_uniform_(self.fc3.weight)
        elif init_type == 'he':
            nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
            nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
            nn.init.kaiming_uniform_(self.fc3.weight, nonlinearity='relu')
        else:
            nn.init.normal_(self.fc1.weight)
            nn.init.normal_(self.fc2.weight)
            nn.init.normal_(self.fc3.weight)
        
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class PermutationSpec(NamedTuple):
    perm_to_axes: dict
    axes_to_perm: dict

def permutation_spec_from_axes_to_perm(axes_to_perm: dict) -> PermutationSpec:
    perm_to_axes = defaultdict(list)
    for wk, axis_perms in axes_to_perm.items():
        for axis, perm in enumerate(axis_perms):
            if perm is not None:
                perm_to_axes[perm].append((wk, axis))
    return PermutationSpec(dict(perm_to_axes), axes_to_perm)

def mlp_permutation_spec() -> PermutationSpec:
    """Permutation spec for 3-layer MLP"""
    return permutation_spec_from_axes_to_perm({
        "fc1.weight": (None, "P_0"),
        "fc1.bias": ("P_0",),
        "fc2.weight": ("P_0", "P_1"),
        "fc2.bias": ("P_1",),
        "fc3.weight": ("P_1", None),
        "fc3.bias": (None,),
    })

def get_permuted_param(ps: PermutationSpec, perm, k: str, params, except_axis=None):
    w = params[k]
    for axis, p in enumerate(ps.axes_to_perm[k]):
        if axis == except_axis:
            continue
        if p is not None:
            w = torch.index_select(w, axis, torch.tensor(perm[p], device=w.device))
    return w

def apply_permutation(ps: PermutationSpec, perm, params):
    return {k: get_permuted_param(ps, perm, k, params) for k in params.keys()}

def update_model_weights(model, aligned_params):
    model.fc1.weight.data = aligned_params["fc1.weight"].T
    model.fc1.bias.data = aligned_params["fc1.bias"]
    model.fc2.weight.data = aligned_params["fc2.weight"].T
    model.fc2.bias.data = aligned_params["fc2.bias"]
    model.fc3.weight.data = aligned_params["fc3.weight"].T
    model.fc3.bias.data = aligned_params["fc3.bias"]

def weight_matching(ps: PermutationSpec, params_a, params_b, max_iter=100, init_perm=None, silent=True, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    params_a = {k: v.to(device) for k, v in params_a.items()}
    params_b = {k: v.to(device) for k, v in params_b.items()}
    
    perm_sizes = {p: params_a[axes[0][0]].shape[axes[0][1]] for p, axes in ps.perm_to_axes.items()}
    
    perm = {p: torch.arange(n, device=device) for p, n in perm_sizes.items()} if init_perm is None else {p: v.to(device) for p, v in init_perm.items()}
    
    perm_names = list(perm.keys())
    rng = np.random.RandomState(42)

    for _ in range(max_iter):
        progress = False
        for p_ix in rng.permutation(len(perm_names)):
            p = perm_names[p_ix]
            n = perm_sizes[p]
            A = torch.zeros((n, n), device=device)
            for wk, axis in ps.perm_to_axes[p]:
                w_a = params_a[wk]
                w_b = get_permuted_param(ps, perm, wk, params_b, except_axis=axis)
                w_a = w_a.moveaxis(axis, 0).reshape((n, -1))
                w_b = w_b.moveaxis(axis, 0).reshape((n, -1))
                A += w_a @ w_b.T
            ri, ci = linear_sum_assignment(A.detach().cpu().numpy(), maximize=True)
            eye_old = torch.eye(n, device=device)[perm[p]]
            eye_new = torch.eye(n, device=device)[ci]
            oldL = torch.tensordot(A, eye_old, dims=([0,1],[0,1]))
            newL = torch.tensordot(A, eye_new, dims=([0,1],[0,1]))
            progress = progress or newL > oldL + 1e-12
            perm[p] = torch.tensor(ci, device=device)
        if not progress:
            break
    return perm

# Using `get_permuted_models_data` imported from `flowmatching.canonicalization`
# for canonicalization / git rebasin logic. Local MLP-specific utilities were
# removed to avoid shadowing the canonical implementation.


class Bunch:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class WeightSpaceObject:
    def __init__(self, weights, biases):
        self.weights = tuple(weights)
        self.biases = tuple(biases)
        
    def flatten(self, device=None):
        flat = torch.cat([w.flatten() for w in self.weights] + 
                        [b.flatten() for b in self.biases])
        return flat.to(device) if device else flat
    
    @classmethod
    def from_flat(cls, flat, layers, device=None):
        """Create WeightSpaceObject from flattened vector and layer sizes"""
        sizes = []
        # Calculate sizes for weight matrices
        for i in range(len(layers) - 1):
            sizes.append(layers[i] * layers[i+1])  # Weight matrix
        # Calculate sizes for bias vectors
        for i in range(1, len(layers)):
            sizes.append(layers[i])  # Bias vector
            
        # Split flat tensor into parts
        parts = []
        start = 0
        for size in sizes:
            parts.append(flat[start:start+size])
            start += size
            
        # Reshape into weight matrices and bias vectors
        weights = []
        biases = []
        for i in range(len(layers) - 1):
            weights.append(parts[i].reshape(layers[i+1], layers[i]))
            biases.append(parts[i + len(layers) - 1])
            
        return cls(weights, biases)

# Use `FlowMatching` and `WeightSpaceFlowModel` from `flowmatching.flow_matching`
# (imported at the top of this file). Local/duplicate implementations were
# removed to ensure we use the library's canonical implementations.

def test_mlp(model, test_loader):
    # Generic evaluator wrapper that delegates to flowmatching.utils.evaluate_model
    return evaluate_model(model, test_loader, device)

def print_stats(models, test_loader):
    # Delegate to flowmatching's print_stats which handles different model types
    try:
        return fm_print_stats(models, test_loader, device)
    except Exception:
        # Fallback: compute accuracies manually
        accuracies = [evaluate_model(m, test_loader, device) for m in models]
        accuracies = np.array(accuracies)
        mean = accuracies.mean()
        std = accuracies.std()
        logging.info("\n=== Summary (fallback) ===")
        logging.info(f"Average Accuracy: {mean:.2f}% ± {std:.2f}%")
        return mean, std


def plot_source_distribution_comparison(original_models, cfm, flat_dim, layer_layout, test_loader, model_type, device='cuda', n_samples=100):
    """
    Create scatter plot comparing generated models from different source distributions.
    """
    source_stds = [0.001, 0.005, 0.01]
    all_generated_data = []
    
    logging.info("Adding original models for comparison context")
    orig_accuracies = [test_mlp(model, test_loader) for model in original_models]
    orig_similarities = compute_max_similarity_vs_originals(original_models, original_models, test_loader, device)
    
    for i, (acc, sim) in enumerate(zip(orig_accuracies, orig_similarities)):
        all_generated_data.append({
            'source_std': 'original',
            'model_id': i,
            'accuracy': acc,
            'max_iou_similarity': sim,
            'model_type': 'Original'
        })
    
    for source_std in source_stds:
        logging.info(f"Generating models with source std: {source_std}")
        
        # Generate models with this source distribution
        random_flat = torch.randn(n_samples, flat_dim, device=device) * source_std
        new_weights_flat = cfm.map(random_flat, n_steps=100, method="rk4")
        
        generated_models = []
        for i in range(n_samples):
            new_wso = WeightSpaceObject.from_flat(
                new_weights_flat[i],
                layers=np.array(layer_layout),
                device=device
            )
            
            model = MLP()
            model.fc1.weight.data = new_wso.weights[0].clone()
            model.fc1.bias.data = new_wso.biases[0].clone()
            model.fc2.weight.data = new_wso.weights[1].clone()
            model.fc2.bias.data = new_wso.biases[1].clone()
            model.fc3.weight.data = new_wso.weights[2].clone()
            model.fc3.bias.data = new_wso.biases[2].clone()
            
            generated_models.append(model)
        
        # Compute accuracies and similarities
        accuracies = [test_mlp(model, test_loader) for model in generated_models]
        similarities = compute_max_similarity_vs_originals(generated_models, original_models, test_loader, device)
        
        # Store data
        for i, (acc, sim) in enumerate(zip(accuracies, similarities)):
            all_generated_data.append({
                'source_std': source_std,
                'model_id': i,
                'accuracy': acc,
                'max_iou_similarity': sim,
                'model_type': f'Generated (σ={source_std})'
            })
        
        logging.info(f"Source std {source_std} - Mean accuracy: {np.mean(accuracies):.2f}% ± {np.std(accuracies):.2f}%")
        logging.info(f"Source std {source_std} - Mean IoU vs orig: {np.mean(similarities):.4f} ± {np.std(similarities):.4f}")
    
    # Save source distribution data to CSV
    import pandas as pd
    source_df = pd.DataFrame(all_generated_data)
    source_df.to_csv(f'source_distribution_comparison_{model_type}.csv', index=False)
    logging.info(f"Saved source distribution data to source_distribution_comparison_{model_type}.csv")
    
    # Create scatter plot
    plt.figure(figsize=(12, 8))
    plt.rcParams.update({'font.size': 16})
    
    # Plot original models first (blue dots) - THIS WAS MISSING!
    orig_subset = source_df[source_df['source_std'] == 'original']
    plt.scatter(orig_subset['max_iou_similarity'], orig_subset['accuracy'], 
               alpha=0.6, label='Original', color='blue', s=50)
    
    # Plot generated models from different source distributions
    colors = ['red', 'orange', 'green']
    for i, source_std in enumerate(source_stds):
        subset = source_df[source_df['source_std'] == source_std]
        plt.scatter(subset['max_iou_similarity'], subset['accuracy'],
                   alpha=0.6, label=f'Generated (σ={source_std})',
                   color=colors[i], s=50)
    
    plt.xlabel('Maximum IoU Similarity vs Original Models')
    plt.ylabel('Test Accuracy (%)')
    # plt.title(f'Source Distribution Comparison - {model_type}')
    plt.legend()
    # plt.xlim(0.6, 1.05)
    # plt.ylim(0.6, 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'source_distribution_comparison_{model_type}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return source_df


def plot_source_distribution_comparison_resnet(original_models, cfm, flat_dim, weight_shapes, bias_shapes, test_loader, model_type, device='cuda', n_samples=100):
    """
    Source distribution comparison for ResNet weight-space generation.
    Generates models from several source stds, reconstructs ResNets, computes
    accuracies and max-IoU similarities vs original models, saves CSV and plots.
    """
    from flowmatching.models import get_resnet18

    source_stds = [0.001, 0.005, 0.01]
    all_generated_data = []

    logging.info("Adding original models for comparison context (ResNet)")
    orig_accuracies = [test_mlp(model, test_loader) for model in original_models]
    orig_similarities = compute_max_similarity_vs_originals(original_models, original_models, test_loader, device)

    for i, (acc, sim) in enumerate(zip(orig_accuracies, orig_similarities)):
        all_generated_data.append({
            'source_std': 'original',
            'model_id': i,
            'accuracy': acc,
            'max_iou_similarity': sim,
            'model_type': 'Original'
        })

    for source_std in source_stds:
        logging.info(f"Generating ResNet models with source std: {source_std}")

        random_flat = torch.randn(n_samples, flat_dim, device=device) * source_std
        generated_flat = cfm.map(random_flat, n_steps=100, method="rk4")

        generated_models = []
        for i in range(n_samples):
            flat_vec = generated_flat[i]
            new_wso = WeightSpaceObjectResnet.from_flat(flat_vec, weight_shapes, bias_shapes, device=device)

            model = get_resnet18(num_classes=10)
            param_dict = {}
            w_idx = 0
            b_idx = 0
            for name, param in model.named_parameters():
                if 'weight' in name:
                    param_dict[name] = new_wso.weights[w_idx]
                    w_idx += 1
                elif 'bias' in name:
                    param_dict[name] = new_wso.biases[b_idx]
                    b_idx += 1

            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in param_dict:
                        param.copy_(param_dict[name])

            try:
                model = recalibrate_bn_stats(model, device=device)
            except Exception:
                pass

            generated_models.append(model.to(device))

        # Compute accuracies and similarities
        accuracies = [test_mlp(model, test_loader) for model in generated_models]
        similarities = compute_max_similarity_vs_originals(generated_models, original_models, test_loader, device)

        for i, (acc, sim) in enumerate(zip(accuracies, similarities)):
            all_generated_data.append({
                'source_std': source_std,
                'model_id': i,
                'accuracy': acc,
                'max_iou_similarity': sim,
                'model_type': f'Generated (σ={source_std})'
            })

        logging.info(f"Source std {source_std} - Mean accuracy: {np.mean(accuracies):.2f}% ± {np.std(accuracies):.2f}%")
        logging.info(f"Source std {source_std} - Mean IoU vs orig: {np.mean(similarities):.4f} ± {np.std(similarities):.4f}")

    # Save to CSV
    import pandas as pd
    source_df = pd.DataFrame(all_generated_data)
    fname = f'source_distribution_comparison_{model_type}.csv'
    source_df.to_csv(fname, index=False)
    logging.info(f"Saved source distribution data to {fname}")

    # Plot
    plt.figure(figsize=(12, 8))
    plt.rcParams.update({'font.size': 16})

    orig_subset = source_df[source_df['source_std'] == 'original']
    plt.scatter(orig_subset['max_iou_similarity'], orig_subset['accuracy'], alpha=0.6, label='Original', color='blue', s=50)

    colors = ['red', 'orange', 'green']
    for i, source_std in enumerate(source_stds):
        subset = source_df[source_df['source_std'] == source_std]
        plt.scatter(subset['max_iou_similarity'], subset['accuracy'], alpha=0.6, label=f'Generated (σ={source_std})', color=colors[i], s=50)

    plt.xlabel('Maximum IoU Similarity vs Original Models')
    plt.ylabel('Test Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'source_distribution_comparison_{model_type}.png', dpi=150, bbox_inches='tight')
    plt.show()

    return source_df

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    # Load config and set up for resnet18 CIFAR10 pipeline
    cfg_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'flowmatching', 'constants.json'))
    config = load_config(cfg_path) if os.path.exists(cfg_path) else load_config('flowmatching/constants.json')

    model_key = 'resnet18_cifar10'
    model_config = config['models'][model_key]
    model_dir = config['directories'][model_key]

    logging.info(f"Using model config: {model_key}")

    # Data loader (CIFAR10)
    test_loader = get_data_loader(model_config['dataset'], batch_size=128, train=False)

    logging.info("Creating permuted model dataset using rebasin (canonicalization)...")
    ref_point = 0
    num_models = 50

    ref_model, org_models, permuted_models = get_permuted_models_data(
        model_name=model_key,
        model_dir=model_dir,
        pretrained_model_name=model_config['pretrained_model_name'],
        num_models=num_models,
        ref_point=ref_point,
        device=device,
        model_config=model_config,
    )

    logging.info("Original models stats:")
    fm_print_stats(org_models, test_loader, device)
    logging.info("Permuted (rebasined) models stats:")
    fm_print_stats(permuted_models, test_loader, device)

    # Choose training mode
    for training_mode in model_config.get('training_modes', ['with_gitrebasin', 'without_rebasin']):
        models_to_use = permuted_models if training_mode == 'with_gitrebasin' else org_models

        logging.info(f"Converting {len(models_to_use)} models to weight-space objects ({training_mode})...")
        weight_space_objects = convert_models_to_weight_space(models_to_use, model_config)

        flat_target_weights = torch.stack([wso.flatten(device) for wso in weight_space_objects]).to(device)
        flat_dim = flat_target_weights.shape[1]
        logging.info(f"Flat weight dimension: {flat_dim}")

        # PCA compression if configured
        ipca = None
        if model_config.get('use_pca') and model_config.get('pca_components'):
            from sklearn.decomposition import IncrementalPCA
            ipca = IncrementalPCA(n_components=model_config['pca_components'], batch_size=10)
            flat_latent = ipca.fit_transform(flat_target_weights.cpu().numpy())
            target_tensor = torch.tensor(flat_latent, dtype=torch.float32, device=device)
            actual_dim = model_config['pca_components']
        else:
            target_tensor = flat_target_weights
            actual_dim = flat_dim

        source_std = model_config['source_std']
        source_tensor = torch.randn_like(target_tensor) * source_std

        sourceloader = DataLoader(TensorDataset(source_tensor), batch_size=model_config['batch_size'], shuffle=True, drop_last=True)
        targetloader = DataLoader(TensorDataset(target_tensor), batch_size=model_config['batch_size'], shuffle=True, drop_last=True)

        # Build flow model
        hidden_dim = model_config['flow_hidden_dims'][0]
        flow_model = WeightSpaceFlowModel(actual_dim, hidden_dim, time_embed_dim=model_config['time_embed_dim'], dropout=model_config['dropout']).to(device)
        logging.info(f"Flow model parameters: {count_parameters(flow_model):,}")

        cfm = FlowMatching(
            sourceloader=sourceloader,
            targetloader=targetloader,
            model=flow_model,
            mode='velocity',
            t_dist=config.get('t_dist', 'uniform'),
            device=device
        )

        optimizer = torch.optim.AdamW(flow_model.parameters(), lr=model_config['lr'], weight_decay=model_config['weight_decay'], betas=(0.9, 0.95))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=model_config['n_iters'], eta_min=1e-6)

        train_kwargs = {
            'n_iters': model_config['n_iters'],
            'optimizer': optimizer,
            'scheduler': scheduler,
            'sigma': model_config['sigma'],
            'patience': model_config['patience'],
            'log_freq': 50
        }
        if model_config.get('gradient_accumulation_steps'):
            train_kwargs['accum_steps'] = model_config['gradient_accumulation_steps']

        logging.info("Starting flow training...")
        cfm.train(**train_kwargs)

        # Generation
        n_samples = model_config['n_samples']
        random_flat = torch.randn(n_samples, actual_dim, device=device) * source_std

        generated_models = []

        logging.info("Generating new ResNet weights from flow...")
        generated_flat = cfm.map(random_flat, n_steps=model_config['integration_steps'], method=model_config['integration_method'])
        if ipca is not None:
            generated_flat = ipca.inverse_transform(generated_flat.cpu().numpy())
            generated_flat = torch.tensor(generated_flat, dtype=torch.float32, device=device)

        # Get reference shapes
        ref_wso = weight_space_objects[0]
        weight_shapes = getattr(ref_wso, 'weight_shapes', None)
        bias_shapes = getattr(ref_wso, 'bias_shapes', None)

        from flowmatching.models import get_resnet18

        for i in tqdm(range(n_samples), desc='Reconstructing generated ResNets'):
            flat_vec = generated_flat[i]
            new_wso = WeightSpaceObjectResnet.from_flat(flat_vec, weight_shapes, bias_shapes, device=device)

            model = get_resnet18(num_classes=10)
            # Map weights/biases back into model parameters in order
            param_dict = {}
            w_idx = 0
            b_idx = 0
            for name, param in model.named_parameters():
                if 'weight' in name:
                    param_dict[name] = new_wso.weights[w_idx]
                    w_idx += 1
                elif 'bias' in name:
                    param_dict[name] = new_wso.biases[b_idx]
                    b_idx += 1

            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in param_dict:
                        param.copy_(param_dict[name])

            # Recalibrate BN statistics if requested
            if model_config.get('recalibrate_bn', False):
                model = recalibrate_bn_stats(model, device=device)

            generated_models.append(model.to(device))

        logging.info("Evaluation of generated models")
        mean_acc, std_acc = fm_print_stats(generated_models, test_loader, device)
        logging.info(f"Generated models mean accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")

        # Produce and save max-IoU vs accuracy CSV/plots (matching original script outputs)
        try:
            logging.info("Saving Max-IoU vs Accuracy CSV and plots...")
            plot_results = plot_similarity_vs_accuracy(models_to_use, generated_models, test_loader, model_type=f"{model_key}_{training_mode}", device=device)
            violin_results = plot_similarity_vs_accuracy_violin(models_to_use, generated_models, test_loader, model_type=f"{model_key}_{training_mode}", device=device)
            logging.info(f"Saved similarity/violin CSVs for {model_key}_{training_mode}")
        except Exception as e:
            logging.info(f"Failed to save Max-IoU CSVs/plots: {e}")

        # Also produce source-distribution comparison plots/CSVs for ResNet
        try:
            logging.info("Saving source-distribution comparison (ResNet)...")
            source_comp = plot_source_distribution_comparison_resnet(models_to_use, cfm, flat_dim, weight_shapes, bias_shapes, test_loader, model_type=f"{model_key}_{training_mode}", device=device, n_samples=n_samples)
            logging.info(f"Saved source-distribution CSVs for {model_key}_{training_mode}")
        except Exception as e:
            logging.info(f"Failed to save source-distribution CSVs/plots: {e}")

        # Nearest neighbor analysis (in weight-space)
        nn_distances = nearest_neighbor_analysis(models_to_use, generated_models, device=device)

        # Save distance metrics and histograms to CSV for later analysis
        try:
            import pandas as pd

            distances_df = pd.DataFrame({
                'nn_orig_to_orig': nn_distances['nn_orig_to_orig'],
                'nn_gen_to_gen': nn_distances['nn_gen_to_gen'],
                'nn_orig_to_gen': nn_distances['nn_orig_to_gen'],
                'nn_gen_to_orig': nn_distances['nn_gen_to_orig'],
            })
            distances_fname = f"nn_distances_{model_key}_{training_mode}.csv"
            distances_df.to_csv(distances_fname, index=False)
            logging.info(f"Saved nearest-neighbor distances to {distances_fname}")

            # Histogram data: convert bin edges to centers
            bins = nn_distances.get('bins')
            if bins is not None:
                centers = (bins[:-1] + bins[1:]) / 2
            else:
                centers = np.arange(len(nn_distances['hist_orig_orig']))

            hist_df = pd.DataFrame({
                'bin_center': centers,
                'hist_orig_orig': nn_distances['hist_orig_orig'],
                'hist_gen_gen': nn_distances['hist_gen_gen'],
                'hist_orig_gen': nn_distances['hist_orig_gen'],
                'hist_gen_orig': nn_distances['hist_gen_orig'],
            })
            hist_fname = f"distance_hist_{model_key}_{training_mode}.csv"
            hist_df.to_csv(hist_fname, index=False)
            logging.info(f"Saved histogram data to {hist_fname}")
        except Exception as e:
            logging.info(f"Failed to save distance CSVs: {e}")

        # Optional: save or plot results
        logging.info("Finished generation and analysis for training mode: %s", training_mode)

if __name__ == "__main__":
    logging.info("MLP - MNIST embed 512")
    main()
