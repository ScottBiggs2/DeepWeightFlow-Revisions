#!/usr/bin/env python3
"""
Purpose:
  - Compute PCA on very-high-dimensional flattened model weight vectors
    (e.g., DistilBERT / BERT parameters) using a single-pass random-projection
    sketch + randomized SVD on the sketch. Designed to use minimal RAM by:
      * streaming each sample's weights chunk-by-chunk
      * generating random-projection blocks on-the-fly (no R_full stored)
      * keeping only an r x r sketch in memory

High-level algorithm (Path A):
  For each sample (model checkpoint) do:
    for each parameter tensor in checkpoint, for each block of that tensor:
      - dequantize / read block (float32)
      - generate R_block (r x block_len) from RNG (reproducible)
      - z = R_block @ x_block               # r-dim
      - C += z @ z.T                        # accumulate r x r sketch
  After all samples:
    - Run randomized_svd(C) -> top-k eigenvectors U_r (r x k), eigenvalues s
    - Reconstruct full-space PCs by streaming blocks again:
        for each block: W_block = R_block.T @ U_r  # block_len x k
      Concatenate W_block across all blocks -> W_full (D x k)

Key choices & tradeoffs accepted in this implementation:
  * Sketch dimension r: must be >= k and preferably r = oversample * k
    Tradeoff: larger r -> better approximation but more memory and compute.
  * Block size: controls memory per-block; smaller => less RAM but more RNG/ops.
  * Random matrix distribution: Gaussian N(0,1) scaled by 1/sqrt(r).
    Alternative: structured/projection (SRHT) could be faster, but more complex.
  * We accumulate C = sum z z.T (r x r). This approximates global covariance in
    the randomized projected space and is small (r x r).
  * Reconstruction step requires forming W_full (D x k). This can be big but
    is usually manageable for modest k (e.g., k <= 512). If D*k is too large,
    store blocks to disk or use W as an on-disk memory-mapped array.
  * This sketch assumes you have multiple samples (models or snapshots).
    If you only have one sample, PCA is meaningless; for demos we simulate
    multiple samples by adding small noise to a single checkpoint.
  * We use FP32 everywhere (no float64 upcasting). If numerical stability is
    critical, accumulate C in FP64 (costs memory) or increase r/k.

Dependencies:
  - torch
  - transformers
  - numpy
  - scikit-learn (for randomized_svd)
  - tqdm (optional, progress bars)

Run:
  python streaming_random_projection_pca.py --model distilbert-base-uncased \
       --n_samples 16 --k 128 --r_factor 2 --block_size 200_000

Author: (sketch)
"""

import argparse
import math
import os
import tempfile
from functools import partial

import numpy as np
import torch
from sklearn.utils.extmath import randomized_svd
from tqdm import tqdm
from transformers import AutoModel

# -----------------------------
# Helper utilities
# -----------------------------
def iter_model_param_blocks(model_or_state_dict, block_size):
    """
    Yields (param_name, block_index, block_array) where block_array is a 1D numpy float32 array.
    Accepts either a torch.nn.Module or a state_dict-like dict mapping names->tensors.
    The order is deterministic (sorted by key).
    """
    if isinstance(model_or_state_dict, dict):
        items = sorted(model_or_state_dict.items())
    else:
        # model (torch.nn.Module)
        items = sorted(model_or_state_dict.state_dict().items())

    for name, tensor in items:
        arr = tensor.detach().cpu().numpy().ravel().astype(np.float32)
        n = arr.size
        num_blocks = math.ceil(n / block_size)
        for b in range(num_blocks):
            start = b * block_size
            end = min(n, (b + 1) * block_size)
            yield name, b, arr[start:end]


def rng_for_block(seed, global_block_idx):
    """
    Returns a numpy RandomState seeded deterministically from (seed, global_block_idx).
    We'll use this RNG to generate the block's projection matrix.
    """
    return np.random.RandomState(seed ^ (global_block_idx * 0x9e3779b1 & 0xFFFFFFFF))


def make_R_block(rng, r, block_len):
    """
    Draw Gaussian projection matrix (r x block_len) scaled appropriately.
    We intentionally do NOT store it; it's small per block: r * block_len floats.
    For speed in real code, consider structured transforms (SRHT) or sparse R.
    """
    # Normal(0,1/sqrt(r)) -> so E[||R x||^2] ~ ||x||^2
    return rng.randn(r, block_len).astype(np.float32) / math.sqrt(r)


# -----------------------------
# Core sketching + reconstruction
# -----------------------------
def build_sketch_from_samples(sample_sources, seed, k, r_factor, block_size):
    """
    sample_sources: iterable of sample provider objects. Each provider is either:
      - a path to a checkpoint compatible with transformers.AutoModel
      - a dict-like state_dict (name -> torch.tensor)
      - a (callable) function that returns a state_dict / model when called
    For demo we accept a single base model and simulate multiple samples by adding noise.
    Returns:
      - C: r x r covariance sketch (numpy float32)
      - meta: dict with sketch params (r, k, D, block_info list)
    """
    # First pass: compute total flattened dimension D and number of global blocks
    # We'll walk ONE representative sample to count blocks/tensor sizes.
    # For safety, load first sample now to determine layout
    first_provider = sample_sources[0]
    if callable(first_provider):
        model_or_sd = first_provider()
    elif isinstance(first_provider, str):
        # load using AutoModel, returns torch model
        model_or_sd = AutoModel.from_pretrained(first_provider)
    else:
        model_or_sd = first_provider  # assume dict or model

    # Build list of (name, size) and a global block index map to reproducibly generate R per block
    block_layout = []  # list of tuples: (param_name, block_len)
    total_D = 0
    for name, _, block in iter_model_param_blocks(model_or_sd, block_size):
        block_layout.append((name, block.size))
        total_D += block.size
    num_global_blocks = len(block_layout)
    print(f"total flattened dim D={total_D}, num_global_blocks={num_global_blocks}")

    # sketch dimension
    r = max(int(r_factor * k), k + 4)

    # initialize sketch accumulator (r x r)
    C = np.zeros((r, r), dtype=np.float64)  # accumulate in float64 for stability; cast later if desired
    global_block_idx = 0

    # Iterate samples (models) one-by-one; for each, stream param blocks
    for sample_idx, provider in enumerate(tqdm(sample_sources, desc="samples")):
        # obtain a model/state_dict for this sample
        if callable(provider):
            model_or_sd = provider()
        elif isinstance(provider, str):
            model_or_sd = AutoModel.from_pretrained(provider)
        else:
            model_or_sd = provider

        # iterate blocks in consistent order
        for name, bidx, block in iter_model_param_blocks(model_or_sd, block_size):
            x = block.astype(np.float32)  # 1D float32
            block_len = x.size
            # RNG seeded deterministically by (seed, global_block_idx) so same R_block on reconstruction pass
            rng = rng_for_block(seed, global_block_idx)
            R = make_R_block(rng, r, block_len)  # r x block_len
            # z = R @ x  -> shape (r,)
            # Use float64 accumulation for C to reduce numeric issues across many samples
            z = R.dot(x).astype(np.float64)
            # accumulate outer product
            C += np.outer(z, z)
            global_block_idx += 1

        # free model if it's a transformers model
        if hasattr(model_or_sd, "cpu") and hasattr(model_or_sd, "state_dict"):
            # try to free memory
            del model_or_sd

    meta = dict(D=total_D, r=r, k=k, block_layout=block_layout, num_global_blocks=num_global_blocks)
    return C, meta


def randomized_svd_on_sketch(C, k, n_iter=2, random_state=0):
    """
    Runs randomized_svd on the sketch matrix C (r x r).
    Returns top-k left singular vectors U_r (r x k) and singular values s.
    """
    # Convert to float32 for downstream operations if desired; sklearn supports float64 too
    U, s, Vt = randomized_svd(C, n_components=k, n_iter=n_iter, random_state=random_state)
    # U is r x k
    return U, s


def reconstruct_full_pcs(U_r, meta, seed, block_size, sample_provider_for_layout=None, dtype=np.float32, out_path=None):
    """
    Reconstructs full-space principal directions W_full (D x k) by streaming over the same block layout
    and generating R_block on-the-fly.

    If out_path is provided, write a memory-mapped np.memmap of shape (D, k) to disk to avoid large RAM.
    Returns W_full (numpy array or memmap).
    """
    D = meta["D"]
    k = U_r.shape[1]
    block_layout = meta["block_layout"]

    if out_path:
        W = np.memmap(out_path, dtype=dtype, mode="w+", shape=(D, k))
    else:
        W = np.zeros((D, k), dtype=dtype)

    pos = 0
    global_block_idx = 0
    for (name, block_len) in block_layout:
        rng = rng_for_block(seed, global_block_idx)
        R = make_R_block(rng, meta["r"], block_len)  # r x block_len
        # W_block = R.T @ U_r  -> shape (block_len x k)
        W_block = R.T.dot(U_r).astype(dtype)
        # place block into W
        W[pos:pos + block_len, :] = W_block
        pos += block_len
        global_block_idx += 1

    assert pos == D, "reconstruction dimension mismatch"
    return W


# -----------------------------
# Demo / CLI
# -----------------------------
def make_noisy_provider_from_base(model_name_or_path, noise_scale=1e-3):
    """
    Returns a callable that when called loads the model and returns state_dict with tiny noise added.
    Used to simulate multiple samples without having multiple checkpoints on disk.
    """
    base_model = AutoModel.from_pretrained(model_name_or_path)
    sd = base_model.state_dict()

    def provider():
        # add small gaussian noise to each tensor to simulate distinct samples
        noisy = {}
        for k, v in sd.items():
            arr = v.detach().cpu().numpy().astype(np.float32)
            noisy_arr = arr + np.random.randn(*arr.shape).astype(np.float32) * noise_scale
            noisy[k] = torch.tensor(noisy_arr)
        return noisy

    return provider


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="distilbert-base-uncased", help="HF model name (example uses DistilBERT).")
    p.add_argument("--n_samples", type=int, default=8, help="Number of samples (checkpoints) to process. Demo simulates them.")
    p.add_argument("--k", type=int, default=128, help="Target number of principal components")
    p.add_argument("--r_factor", type=float, default=2.0, help="Oversampling factor: r = r_factor * k")
    p.add_argument("--block_size", type=int, default=200_000, help="Block size (number of floats) to stream at once")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="out_pca")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # For demo: create providers = list of callables that return state_dict per sample
    # In production, replace with provider paths: e.g., list of checkpoint folders or functions that load and return state_dicts.
    providers = [make_noisy_provider_from_base(args.model, noise_scale=1e-3) for _ in range(args.n_samples)]

    print("Building sketch (single-pass over samples)...")
    C, meta = build_sketch_from_samples(providers, seed=args.seed, k=args.k, r_factor=args.r_factor, block_size=args.block_size)
    print(f"Built sketch C with shape {C.shape}")

    print("Running randomized SVD on sketch...")
    U_r, s = randomized_svd_on_sketch(C, k=args.k, n_iter=2, random_state=args.seed)
    print(f"Top singular values (sketch): {s[:10]}")

    # Reconstruction may be large: write to memmap file if D*k big
    D = meta["D"]
    approx_size_bytes = D * args.k * 4
    memmap_path = None
    if approx_size_bytes > 2_000_000_000:  # > ~2GB
        memmap_path = os.path.join(args.out_dir, "W_full.memmap")
        print(f"Writing full principal directions to memmap at {memmap_path} (size {approx_size_bytes / 1e9:.2f} GB)")

    print("Reconstructing full-space principal directions (streaming second pass)...")
    W_full = reconstruct_full_pcs(U_r, meta, seed=args.seed, block_size=args.block_size, out_path=memmap_path)
    print("Reconstruction complete.")
    # Save top-k singular values and optionally W_full
    np.save(os.path.join(args.out_dir, "sketch_singular_values.npy"), s)
    if memmap_path:
        print(f"W_full saved as memmap at {memmap_path}")
    else:
        np.save(os.path.join(args.out_dir, "W_full.npy"), W_full)
        print("W_full saved to .npy")

    print("Done. Note: W_full columns are approximate principal directions in parameter space.")


if __name__ == "__main__":
    main()

# python BERT/stream_pca_distilbert.py --model distilbert-base-uncased --k 64 --block_size 100000