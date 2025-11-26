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
    
    for i in tqdm(range(len(models)), desc="Computing max similarities vs originals"):
        similarities = []
        for j in range(len(original_models)):
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
            weights = (
                m.fc1.weight.data.clone(),
                m.fc2.weight.data.clone(),
                m.fc3.weight.data.clone()
            )
            biases = (
                m.fc1.bias.data.clone(),
                m.fc2.bias.data.clone(),
                m.fc3.bias.data.clone()
            )
            flat = torch.cat([w.flatten() for w in weights] + [b.flatten() for b in biases])
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

def get_permuted_models_data(ref_point=0, model_dir="mnist_models", num_models=100, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Reference model
    ref_model = MLP()
    ref_path = f"{model_dir}/mlp_seed{ref_point}.pt"
    ref_model.load_state_dict(torch.load(ref_path, map_location=device))
    ref_model = ref_model.to(device)
    
    ps = mlp_permutation_spec()
    params_a = {k: v.T if "weight" in k else v for k, v in ref_model.state_dict().items() if k in ps.axes_to_perm}
    
    permuted_models, org_models = [ref_model], [ref_model]
    
    for i in tqdm(range(num_models), desc="Processing models"):
        if i == ref_point:
            continue
        path = f"{model_dir}/mlp_seed{i}.pt"
        if not os.path.exists(path):
            logging.info(f"Skipping model {i} - file not found")
            continue
        
        model = MLP()
        model.load_state_dict(torch.load(path, map_location=device))
        model = model.to(device)
        org_models.append(model)
        
        params_b = {k: v.T if "weight" in k else v for k, v in model.state_dict().items() if k in ps.axes_to_perm}
        perm = weight_matching(ps, params_a, params_b, device=device)
        aligned_params = apply_permutation(ps, perm, params_b)
        
        reconstructed = copy.deepcopy(model)
        update_model_weights(reconstructed, aligned_params)
        permuted_models.append(reconstructed.to(device))
    
    return ref_model, org_models, permuted_models

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

class FlowMatching:
    def __init__(
        self,
        sourceloader,
        targetloader,
        model,
        mode="velocity",
        t_dist="uniform",
        device=None,
        normalize_pred=False,
        geometric=False,
    ):
        # Device
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sourceloader = sourceloader
        self.targetloader = targetloader
        self.model = model.to(self.device)
        self.mode = mode
        self.t_dist = t_dist
        self.sigma = 0.001
        self.normalize_pred = normalize_pred
        self.geometric = geometric

        # Metrics tracking
        self.metrics = {"train_loss": [], "time": [], "grad_norm": [], "flow_norm": [], "true_norm": []}
        self.best_loss = float('inf')
        self.best_model_state = None

        self.input_dim = None

    def sample_from_loader(self, loader):
        """Sample a batch from dataloader"""
        try:
            if not hasattr(loader, '_iterator') or loader._iterator is None:
                loader._iterator = iter(loader)
            try:
                batch = next(loader._iterator)
            except StopIteration:
                loader._iterator = iter(loader)
                batch = next(loader._iterator)
            return batch[0].to(self.device)
        except Exception as e:
            logging.info(f"Error sampling from loader: {str(e)}")
            if hasattr(loader.dataset, '__getitem__'):
                dummy = loader.dataset[0][0]
                return torch.zeros(loader.batch_size, *dummy.shape, device=self.device)
            return torch.zeros(loader.batch_size, 1, device=self.device)

    def sample_time_and_flow(self):
        """Sample time t and flow (for velocity or target mode)"""
        x0 = self.sample_from_loader(self.sourceloader)
        x1 = self.sample_from_loader(self.targetloader)
        batch_size = min(x0.size(0), x1.size(0))
        x0, x1 = x0[:batch_size], x1[:batch_size]

        if self.t_dist == "beta":
            alpha, beta_param = 2.0, 5.0
            t = torch.distributions.Beta(alpha, beta_param).sample((batch_size,)).to(self.device)
        else:
            t = torch.rand(batch_size, device=self.device)

        t_pad = t.view(-1, *([1] * (x0.dim() - 1)))
        mu_t = (1 - t_pad) * x0 + t_pad * x1
        epsilon = torch.randn_like(x0) * self.sigma
        xt = mu_t + epsilon
        ut = x1 - x0

        return Bunch(t=t.unsqueeze(-1), x0=x0, xt=xt, x1=x1, ut=ut, eps=epsilon, batch_size=batch_size)

    def forward(self, flow):
        flow_pred = self.model(flow.xt, flow.t)
        return None, flow_pred

    def loss_fn(self, flow_pred, flow):
        if self.mode == "target":
            l_flow = torch.mean((flow_pred.squeeze() - flow.x1) ** 2)
        else:
            l_flow = torch.mean((flow_pred.squeeze() - flow.ut) ** 2)
        return None, l_flow

    def vector_field(self, xt, t):
        """Compute vector field at point xt and time t"""
        _, pred = self.forward(Bunch(xt=xt, t=t, batch_size=xt.size(0)))
        return pred if self.mode == "velocity" else pred - xt

    def train(self, n_iters=10, optimizer=None, scheduler=None, sigma=0.001, patience=1e99, log_freq=5):
        self.sigma = sigma
        last_loss = 1e99
        patience_count = 0
        pbar = tqdm(range(n_iters), desc="Training steps")
        for i in pbar:
            try:
                optimizer.zero_grad()
                flow = self.sample_time_and_flow()
                _, flow_pred = self.forward(flow)
                _, loss = self.loss_fn(flow_pred, flow)

                if not torch.isnan(loss) and not torch.isinf(loss):
                    loss.backward()
                    optimizer.step()
                    if scheduler: scheduler.step()

                    # Save best model
                    if loss.item() < self.best_loss:
                        self.best_loss = loss.item()
                        self.best_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                else:
                    logging.info(f"Skipping step {i} due to invalid loss: {loss.item()}")
                    continue

                # Early stopping
                if loss.item() > last_loss:
                    patience_count += 1
                    if patience_count >= patience:
                        logging.info(f"Early stopping at iteration {i}")
                        break
                else:
                    patience_count = 0

                last_loss = loss.item()

                if i % log_freq == 0:
                    true_tensor = flow.ut if self.mode == "velocity" else flow.x1
                    grad_norm = self.get_grad_norm()
                    self.metrics["train_loss"].append(loss.item())
                    self.metrics["flow_norm"].append(flow_pred.norm(p=2, dim=1).mean().item())
                    self.metrics["time"].append(flow.t.mean().item())
                    self.metrics["true_norm"].append(true_tensor.norm(p=2, dim=1).mean().item())
                    self.metrics["grad_norm"].append(grad_norm)
                    pbar.set_description(f"Iters [loss {loss.item():.6f}, ∇ norm {grad_norm:.6f}]")
            except Exception as e:
                logging.info(f"Error during training iteration {i}: {str(e)}")
                traceback.print_exc()
                continue

    def get_grad_norm(self):
        total = 0
        for p in self.model.parameters():
            if p.grad is not None:
                total += p.grad.detach().norm(2).item() ** 2
        return total ** 0.5

    def map(self, x0, n_steps=50, return_traj=False, method="euler"):
        if self.best_model_state is not None:
            current_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            self.model.load_state_dict(self.best_model_state)

        self.model.eval()
        batch_size, flat_dim = x0.size()
        traj = [x0.detach().clone()] if return_traj else None
        xt = x0.clone()
        times = torch.linspace(0, 1, n_steps, device=self.device)
        dt = times[1] - times[0]

        for i, t in enumerate(times[:-1]):
            with torch.no_grad():
                t_tensor = torch.ones(batch_size, 1, device=self.device) * t
                pred = self.model(xt, t_tensor)
                if pred.dim() > 2: pred = pred.squeeze(-1)
                vt = pred if self.mode == "velocity" else pred - xt
                if method == "euler":
                    xt = xt + vt * dt
                elif method == "rk4":
                    # RK4 steps
                    k1 = vt
                    k2 = self.model(xt + 0.5 * dt * k1, t_tensor + 0.5 * dt)
                    if k2.dim() > 2: k2 = k2.squeeze(-1)
                    k2 = k2 if self.mode == "velocity" else k2 - (xt + 0.5 * dt * k1)
                    k3 = self.model(xt + 0.5 * dt * k2, t_tensor + 0.5 * dt)
                    if k3.dim() > 2: k3 = k3.squeeze(-1)
                    k3 = k3 if self.mode == "velocity" else k3 - (xt + 0.5 * dt * k2)
                    k4 = self.model(xt + dt * k3, t_tensor + dt)
                    if k4.dim() > 2: k4 = k4.squeeze(-1)
                    k4 = k4 if self.mode == "velocity" else k4 - (xt + dt * k3)
                    xt = xt + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

                if return_traj:
                    traj.append(xt.detach().clone())

        if self.best_model_state is not None:
            self.model.load_state_dict(current_state)
        self.model.train()
        return traj if return_traj else xt

    def generate_weights(self, n_samples=10, source_noise_std=0.001, **map_kwargs):
        assert self.input_dim is not None, "Set `self.input_dim` before generating weights."
        source_samples = torch.randn(n_samples, self.input_dim, device=self.device) * source_noise_std
        return self.map(source_samples, **map_kwargs)

    def plot_metrics(self):
        labels = list(self.metrics.keys())
        lists = list(self.metrics.values())
        n = len(lists)
        fig, axs = plt.subplots(1, n, figsize=(3 * n, 3))
        for i, (label, lst) in enumerate(zip(labels, lists)):
            axs[i].plot(lst)
            axs[i].grid()
            axs[i].title.set_text(label)
            if label == "train_loss":
                axs[i].set_yscale("log")
        plt.tight_layout()
        # plt.close()
        plt.show()


class WeightSpaceFlowModel(nn.Module):
    def __init__(self, input_dim, time_embed_dim=64, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.time_embed_dim = time_embed_dim
        self.dropout = dropout
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.GELU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        hidden_dim = min(512, input_dim // 4)
        logging.info(f"hidden_dim:{hidden_dim}")
        
        self.net = nn.Sequential(
            nn.Linear(input_dim + time_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2), 
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            
            nn.Linear(hidden_dim, input_dim)
        )
        
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
    
    def forward(self, x, t):
        t_embed = self.time_embed(t)
        combined = torch.cat([x, t_embed], dim=-1)
        return self.net(combined)

def test_mlp(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    model = model.to(device)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 100 * correct / total

def print_stats(models, test_loader):
    accuracies = []
    for i, model in enumerate(models):
        acc = test_mlp(model, test_loader)
        accuracies.append(acc)

    accuracies = np.array(accuracies)
    mean = accuracies.mean()
    std = accuracies.std()
    min_acc = accuracies.min()
    max_acc = accuracies.max()

    logging.info("\n=== Summary ===")
    logging.info(f"Average Accuracy: {mean:.2f}% ± {std:.2f}%")
    logging.info(f"Min Accuracy: {min_acc:.2f}%")
    logging.info(f"Max Accuracy: {max_acc:.2f}%")


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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    layer_layout = [784, 32, 32, 10]
    batch_size = 8
    
    transform = transforms.Compose([transforms.ToTensor()])
    test_data = datasets.MNIST(root="data", train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)
  
    logging.info("Creating permuted model dataset using rebasin...")
    ref_point = 12

    ref_model, org_models, permuted_models = get_permuted_models_data(ref_point=ref_point)
    logging.info("Original Models")
    print_stats(org_models, test_loader)
    logging.info("Permuted Models")
    print_stats(permuted_models, test_loader)

    for init_type in ["gaussian_0.01"]:
        for model_type in ["with_gitrebasin", "without_rebasin"]:
            if model_type == "with_gitrebasin":
                models_to_use = permuted_models
            else:
               models_to_use = org_models 
    
            logging.info("Converting models to WeightSpaceObjects...")
            weights_list = []
            for model in tqdm(models_to_use):
                weights = (
                    model.fc1.weight.data.clone(),
                    model.fc2.weight.data.clone(),
                    model.fc3.weight.data.clone()
                )
                
                biases = (
                    model.fc1.bias.data.clone(),
                    model.fc2.bias.data.clone(), 
                    model.fc3.bias.data.clone()
                )
                
                wso = WeightSpaceObject(weights, biases)
                weights_list.append(wso)
            
            logging.info(f"Created {len(weights_list)} permuted weight configurations")
            
            logging.info("Converting to flat tensors...")
            flat_target_weights = torch.stack([wso.flatten(device) for wso in weights_list])
            flat_dim = flat_target_weights.shape[1]
            
            n_samples = 100
            
            if "gaussian" in init_type:
                if init_type == "gaussian_0.01":
                    source_std = 0.01
                else:
                    source_std = 0.001
            
                flat_source_weights = torch.randn(len(weights_list), flat_dim, device=device) * source_std
                random_flat = torch.randn(n_samples, flat_dim, device=device) * source_std
            
            elif "kaimings" in init_type:
                flat_source_weights = []
                for wso in weights_list:
                    kaiming_weights = []
                    for w in wso.weights:
                        fan_in = w.shape[1]
                        std = np.sqrt(2.0 / fan_in)
                        kaiming_w = torch.randn_like(w) * std
                        kaiming_weights.append(kaiming_w)
                    
                    biases = [b.clone() for b in wso.biases]
                    
                    flat_source_weights.append(WeightSpaceObject(kaiming_weights, biases).flatten(device))
                
                flat_source_weights = torch.stack(flat_source_weights)
                
                fan_in_global = weights_list[0].weights[0].shape[1]
                kaiming_std = np.sqrt(2.0 / fan_in_global)
                random_flat = torch.randn(n_samples, flat_dim, device=device) * kaiming_std
    
            else:
                raise ValueError(f"Unknown init_type: {init_type}")
                
            source_dataset = TensorDataset(flat_source_weights)
            target_dataset = TensorDataset(flat_target_weights)
            
            sourceloader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
            targetloader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        
            flow_model = WeightSpaceFlowModel(flat_dim).to(device)
            flow_model.train()
            
            t_dist = "beta"
            logging.info(f"t_dist type:{t_dist}")
            cfm = FlowMatching(
                sourceloader=sourceloader,
                targetloader=targetloader,
                model=flow_model,
                mode="velocity",
                t_dist=t_dist, 
                device=device
            )
            
            n_params_base = sum(p.numel() for p in MLP().parameters())
            n_params_flow = count_parameters(flow_model)
            logging.info(f"MLP params:{n_params_base}")
            logging.info(f"Flow model params:{n_params_flow}")
        
            optimizer = torch.optim.AdamW(
                flow_model.parameters(), 
                lr=5e-4, # changed from 1e-4
                weight_decay=1e-5,
                betas=(0.9, 0.95)
            )
            
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=30000, eta_min=1e-6
            )
            
            cfm.train(
                n_iters=30000,
                optimizer=optimizer,
                scheduler=scheduler,
                sigma=0.001,
                patience=100,
                log_freq=10
            )
            
            logging.info("Generating new MLP weights...")
                        
            for gen_method in ["rk4"]:
                new_weights_flat = cfm.map(random_flat, n_steps=100, method=gen_method)
                generated_models = []        
    
                for i in range(n_samples):
                    new_wso = WeightSpaceObject.from_flat(
                        new_weights_flat[i], 
                        layers=np.array(layer_layout), 
                        device=device
                    )
            
                    expected_weight_shapes = [(32, 784), (32, 32), (10, 32)]
                    expected_bias_shapes = [(32,), (32,), (10,)]
            
                    assert len(new_wso.weights) == 3, f"Expected 3 weight matrices, got {len(new_wso.weights)}"
                    assert len(new_wso.biases) == 3, f"Expected 3 bias vectors, got {len(new_wso.biases)}"
                    
                    for j, (w, expected_shape) in enumerate(zip(new_wso.weights, expected_weight_shapes)):
                        assert w.shape == expected_shape, f"Weight {j} has shape {w.shape}, expected {expected_shape}"
                    
                    for j, (b, expected_shape) in enumerate(zip(new_wso.biases, expected_bias_shapes)):
                        assert b.shape == expected_shape, f"Bias {j} has shape {b.shape}, expected {expected_shape}"
            
                    model = MLP()
                    model.fc1.weight.data = new_wso.weights[0].clone()
                    model.fc1.bias.data = new_wso.biases[0].clone()
                    model.fc2.weight.data = new_wso.weights[1].clone()
                    model.fc2.bias.data = new_wso.biases[1].clone()
                    model.fc3.weight.data = new_wso.weights[2].clone()
                    model.fc3.bias.data = new_wso.biases[2].clone()
    
                    generated_models.append(model)
                    
                logging.info(f"Init Type: {init_type}, Model Type: {model_type}, Generation Method: {gen_method}")
                print_stats(generated_models, test_loader)

                # Averaging generated models
                if generated_models:
                    logging.info(f"Averaging {len(generated_models)} generated models...")
                    
                    # Create a template model to accumulate weights
                    avg_model = MLP().to(device)
                    
                    # Zero out the parameters of the avg_model
                    for param in avg_model.parameters():
                        param.data.zero_()
                    
                    # Sum the parameters of all generated models
                    for model in generated_models:
                        for avg_param, param in zip(avg_model.parameters(), model.parameters()):
                            avg_param.data.add_(param.data)
                            
                    # Divide by the number of models to get the average
                    num_models = len(generated_models)
                    for param in avg_model.parameters():
                        param.data.div_(num_models)
                        
                    # Evaluate the averaged model
                    avg_accuracy = test_mlp(avg_model, test_loader)
                    logging.info(f"Averaged Model Accuracy: {avg_accuracy:.2f}%")

                logging.info(f" --- {len(generated_models)} Generated Models ---")
                
                # Enhanced nearest neighbor analysis
                nn_distances = nearest_neighbor_analysis(models_to_use, generated_models, device=device)

                # IoU similarity analysis with plots (using original models as baseline)
                logging.info("=== IoU Similarity Analysis (vs Original Models) ===")
                similarity_data = plot_similarity_vs_accuracy(models_to_use, generated_models, test_loader, model_type, device=device)
                violin_data = plot_similarity_vs_accuracy_violin(models_to_use, generated_models, test_loader, model_type, device=device)
                
                # Source distribution comparison
                logging.info("=== Source Distribution Comparison ===")
                source_comparison_data = plot_source_distribution_comparison(
                    models_to_use, cfm, flat_dim, layer_layout, test_loader, model_type, device=device, n_samples=n_samples
                )

if __name__ == "__main__":
    logging.info("MLP - MNIST embed 512")
    main()
