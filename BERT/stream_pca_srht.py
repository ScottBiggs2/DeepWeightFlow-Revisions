#!/usr/bin/env python3
"""
stream_pca_srht.py

Single-file streaming PCA for huge model weight vectors (DistilBERT/BERT) using:
  - Per-block SRHT projection (Subsampled Randomized Hadamard Transform)
  - r = 4 * k (default) oversampling for accuracy
  - Accumulation of small r x r covariance sketch C
  - Randomized SVD on C to obtain top-k sketch eigenvectors U_r
  - Streaming second pass to reconstruct W_full = (R^T @ U_r) in original space (memmapped)
  - Computes per-sample L2 / relative L2 / explained-variance metrics

Outputs:
  out_dir/
    - sketch_C.npy         : (r x r) float64 covariance sketch
    - U_r.npy, s_r.npy     : (r x k), (k,) left singular vectors and singular values
    - W_full.memmap        : memmapped (D x k) float32 full principal directions
    - meta.json            : metadata (D, r, k, block_layout, seed, etc)
    - errors.json          : per-sample reconstruction errors (L2, rel, explained_variance)

Mathematical background (short):
  - We want the top-k principal directions of flattened model weight vectors x ∈ ℝ^D,
    but D is huge (tens/hundreds of millions). Forming full covariance Σ = Σ_i x_i x_i^T
    is impossible in RAM.
  - SRHT projects each block of x to r ≪ D dimensions with high probability preserving
    the dominant subspace (Johnson-Lindenstrauss + Halko/Tropp theory).
    For block x (length n), SRHT does:
        z = sqrt(m/r) * (S * H_norm * D * x_pad)
      where:
        - m = next_power_of_two(n) (pad x to length m)
        - D = diag(random ±1)
        - H_norm = normalized Hadamard (so H_norm H_norm^T = I)
        - S = subsampling operator picking r rows
    We accumulate C += z z^T over all blocks and all samples. Then randomized SVD(C)
    gives top-k sketch-space eigenvectors U_r (r x k).
  - To recover full-space principal directions, for each block we compute:
        W_block = R_block.T @ U_r
    and concatenate blocks -> W_full (D x k). For SRHT we compute R_block implicitly
    using the same D, S indices and FWHT operator (no need to store R_full).

Tradeoffs / choices:
  - r = 4*k is the default; increase for higher accuracy (r=8k recommended if budget allows).
  - SRHT requires padding each block to power-of-two length (minor overhead).
  - C is accumulated in float64 for numerical stability; final SVD uses sklearn.randomized_svd.
  - W_full (D x k) is memmapped if large; keep k modest (<= 1024 recommended).
  - This script streams one model at a time; you can pass multiple checkpoint providers.
  - By default, the script simulates multiple samples by adding small noise to a base
    model — replace with your list of checkpoint dirs for real data.

Dependencies:
  - numpy, torch, transformers, tqdm, scikit-learn
  - Tested in env with CPU; better performance on machines with large RAM for memmap.
"""

import argparse
import math
import os
import json
import tempfile
from functools import partial

import numpy as np
import torch
from tqdm import tqdm
from sklearn.utils.extmath import randomized_svd

# -----------------------
# Utilities: block iteration
# -----------------------
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
    Deterministic RNG for a block using XOR-hash with golden ratio constant.
    Returns a numpy RandomState.
    """
    # combine seed and block idx into a 32-bit-ish integer
    mix = (seed ^ (global_block_idx * 0x9e3779b1)) & 0xFFFFFFFF
    return np.random.RandomState(int(mix))


# -----------------------
# SRHT helpers (FWHT)
# -----------------------
def next_pow2(x):
    return 1 << (x - 1).bit_length()

def fwht_inplace(a):
    """
    In-place unnormalized FWHT on 1D numpy array. Length must be power of two.
    """
    h = 1
    n = a.shape[0]
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                x = a[j]
                y = a[j + h]
                a[j] = x + y
                a[j + h] = x - y
        h *= 2
    return a

def fwht_normalized_vector(x):
    """
    Return H_norm * x where H_norm = H / sqrt(m).
    """
    m = x.shape[0]
    y = x.astype(np.float64).copy()
    fwht_inplace(y)
    return (y / math.sqrt(float(m)))

def fwht_normalized_matrix_cols(M):
    """
    Apply normalized FWHT to each column (in-place) of 2D numpy array M (shape m x k).
    m must be power of two.
    """
    # operate columnwise
    m, k = M.shape
    for col in range(k):
        colv = M[:, col]
        # use float64 for stability
        colv64 = colv.astype(np.float64)
        fwht_inplace(colv64)
        M[:, col] = (colv64 / math.sqrt(float(m))).astype(M.dtype)
    return M

def srht_project_block(x, rng, r):
    """
    Project 1D numpy array x (length n) to z (length r) via SRHT:
      z = sqrt(m/r) * (S * H_norm * D * x_pad)
    where:
      - m = next_pow2(n)
      - D = random +-1 diagonal drawn from rng
      - H_norm = normalized Hadamard (m x m)
      - S selects r rows (without replacement) with indices drawn from rng
    Returns z as float64 for stable accumulation.
    """
    n = x.size
    m = next_pow2(n)
    # pad to m
    if m != n:
        x_pad = np.zeros(m, dtype=np.float32)
        x_pad[:n] = x
    else:
        x_pad = x.astype(np.float32).copy()

    # D: random signs
    signs = rng.choice([-1.0, 1.0], size=m).astype(np.float32)
    y = signs * x_pad                # D * x_pad
    # H_norm * (D x) -> use fwht normalized
    y = fwht_normalized_vector(y)    # length m, float64
    # subsample r indices deterministically
    idx = rng.choice(m, size=r, replace=False)
    z = y[idx] * math.sqrt(float(m) / float(r))
    # return z and metadata needed for reconstruction (signs, idx, m)
    return z.astype(np.float64), signs.astype(np.float32), idx.astype(np.int64), m

def reconstruct_block_srht_from_U(U_r, signs, idx, m, n):
    """
    Given U_r (r x k) for a block (same U_r for all blocks),
    and the block-specific SRHT metadata (signs, idx, m, n),
    compute W_block = R_block.T @ U_r  which results in shape (n, k),
    using the identity:
      R = sqrt(m/r) * S * H_norm * D
      => R.T @ U = sqrt(m/r) * D * H_norm * (S^T @ U)
    Here:
      - S^T @ U produces an (m x k) matrix with rows idx populated from U and zeros elsewhere.
      - Apply H_norm to each column, multiply by D (signs), then take first n rows.
    Returns W_block (n x k) as float32 (suitable for memmap storage).
    """
    r, k = U_r.shape
    # build M = S^T @ U => m x k (float64)
    M = np.zeros((m, k), dtype=np.float64)
    # place U rows into M at positions idx
    M[idx, :] = U_r.astype(np.float64)
    # apply H_norm to columns
    fwht_normalized_matrix_cols(M)   # in-place (float64)
    # multiply by signs (D) elementwise across rows
    M = (M * signs[:, None])
    # scale by sqrt(m / r)
    M *= math.sqrt(float(m) / float(r))
    # take first n rows (original block length)
    W_block = M[:n, :].astype(np.float32)
    return W_block


# -----------------------
# Core sketch-building and reconstruction
# -----------------------
def build_srht_sketch_from_samples(sample_sources, seed, k, r_factor, block_size):
    """
    Build SRHT sketch C by streaming over sample_sources.
    Returns:
      - C (r x r) float64 sketch
      - meta dict with layout and SRHT parameters
    sample_sources: list of providers (callables returning state_dict, or path strings, or dicts)
    """
    # load first provider to build block layout
    first_provider = sample_sources[0]
    if callable(first_provider):
        model_or_sd = first_provider()
    elif isinstance(first_provider, str):
        from transformers import AutoModel
        model_or_sd = AutoModel.from_pretrained(first_provider)
    else:
        model_or_sd = first_provider

    block_layout = []  # list of (param_name, block_len)
    total_D = 0
    for name, _, block in iter_model_param_blocks(model_or_sd, block_size):
        block_layout.append((name, int(block.size)))
        total_D += int(block.size)
    num_global_blocks = len(block_layout)
    print(f"total flattened dim D={total_D}, num_global_blocks={num_global_blocks}")

    # r dimension
    r = max(int(r_factor * k), k + 4)
    # initialize sketch accumulator
    C = np.zeros((r, r), dtype=np.float64)
    global_block_idx = 0

    # we will record per-block SRHT metadata needed later for reconstruction (signs, idx, m)
    # To avoid storing huge metadata, we only store idx arrays and signs as small arrays per block.
    block_meta = []

    # iterate over samples
    for sample_idx, provider in enumerate(tqdm(sample_sources, desc="samples")):
        # get model/state_dict
        if callable(provider):
            model_or_sd = provider()
        elif isinstance(provider, str):
            from transformers import AutoModel
            model_or_sd = AutoModel.from_pretrained(provider)
        else:
            model_or_sd = provider

        for name, bidx, block in iter_model_param_blocks(model_or_sd, block_size):
            x = block.astype(np.float32)
            block_len = x.size
            rng = rng_for_block(seed, global_block_idx)
            z, signs, idx, m = srht_project_block(x, rng, r)
            # accumulate C
            C += np.outer(z, z)
            # Save block meta for later reconstruction (just once per global block, independent of sample)
            # But careful: block_meta should be consistent across samples (we assume same layout)
            if sample_idx == 0:
                # store minimal metadata per block: signs (m,), idx (r,), m, n
                block_meta.append(dict(signs=signs.tolist(), idx=idx.tolist(), m=int(m), n=int(block_len)))
            global_block_idx += 1

        # free model if HF model
        if hasattr(model_or_sd, "cpu") and hasattr(model_or_sd, "state_dict"):
            del model_or_sd

    meta = dict(D=total_D, r=r, k=k, block_layout=block_layout, num_global_blocks=num_global_blocks, block_meta=block_meta)
    return C, meta


def randomized_svd_on_sketch(C, k, n_iter=2, random_state=0):
    """
    Run randomized SVD on small sketch C (r x r). Returns U_r (r x k), s (k,)
    """
    U, s, Vt = randomized_svd(C, n_components=k, n_iter=n_iter, random_state=random_state)
    return U.astype(np.float32), s.astype(np.float32)


def reconstruct_full_pcs_srht(U_r, meta, out_path=None):
    """
    Reconstruct W_full = concatenation of block W_blocks, using SRHT block metadata recorded in meta['block_meta'].
    Writes to memmap out_path if provided, else returns numpy array (may be very large).
    """
    D = meta["D"]
    k = U_r.shape[1]
    block_layout = meta["block_layout"]
    block_meta = meta["block_meta"]
    assert len(block_layout) == len(block_meta), "block layout and meta mismatch"

    if out_path:
        W = np.memmap(out_path, dtype=np.float32, mode="w+", shape=(D, k))
    else:
        W = np.zeros((D, k), dtype=np.float32)

    pos = 0
    for j, ((name, block_len), bm) in enumerate(zip(block_layout, block_meta)):
        signs = np.array(bm["signs"], dtype=np.float32)
        idx = np.array(bm["idx"], dtype=np.int64)
        m = int(bm["m"])
        n = int(bm["n"])
        # reconstruct W_block from U_r and block meta
        W_block = reconstruct_block_srht_from_U(U_r.astype(np.float64), signs, idx, m, n)  # returns float32
        # place
        W[pos:pos + block_len, :] = W_block
        pos += block_len

    assert pos == D, f"reconstruction length mismatch {pos} != {D}"
    return W


# -----------------------
# Error computation (streaming)
# -----------------------
def compute_reconstruction_errors(sample_providers, W_full_path_or_array, meta, block_size, seed=42):
    """
    For each provider, compute:
        - L2 error e = ||x - W W^T x||_2
        - relative error e_rel = e / ||x||_2
        - explained variance evr = 1 - (e^2 / ||x||^2)
    Streaming, no full vectors in RAM.
    """
    D = meta["D"]
    k = meta["k"]

    # load W_full either as memmap or array
    if isinstance(W_full_path_or_array, str):
        W = np.memmap(W_full_path_or_array, dtype=np.float32, mode="r", shape=(D, k))
    else:
        W = W_full_path_or_array

    results = []
    for provider in tqdm(sample_providers, desc="computing_errors"):
        if callable(provider):
            model_sd = provider()
        elif isinstance(provider, str):
            from transformers import AutoModel
            model_sd = AutoModel.from_pretrained(provider).state_dict()
        else:
            model_sd = provider

        # Step A: compute scores = W^T x (stream)
        scores = np.zeros((k,), dtype=np.float64)
        pos = 0
        for name, bidx, block in iter_model_param_blocks(model_sd, block_size):
            x_block = block.astype(np.float32).ravel()
            W_block = W[pos:pos + x_block.size, :]      # (block_len, k)
            # accumulate
            scores += W_block.T.dot(x_block.astype(np.float64))
            pos += x_block.size

        # Step B: compute reconstruction residuals streaming
        pos = 0
        squared_resid = 0.0
        squared_norm = 0.0
        for name, bidx, block in iter_model_param_blocks(model_sd, block_size):
            x_block = block.astype(np.float32).ravel()
            W_block = W[pos:pos + x_block.size, :]
            x_rec_block = W_block.dot(scores).astype(np.float64)
            diff = x_block.astype(np.float64) - x_rec_block
            squared_resid += float((diff ** 2).sum())
            squared_norm += float((x_block.astype(np.float64) ** 2).sum())
            pos += x_block.size

        e = math.sqrt(squared_resid)
        e_rel = e / (math.sqrt(squared_norm) + 1e-30)
        evr = 1.0 - (squared_resid / (squared_norm + 1e-30))
        results.append(dict(L2=float(e), rel=float(e_rel), explained_variance=float(evr)))

    # close memmap if used
    if isinstance(W_full_path_or_array, str):
        del W

    return results


# -----------------------
# Demo helpers: noisy provider to simulate many checkpoints
# -----------------------
def make_noisy_provider_from_base(model_name_or_path, noise_scale=1e-3):
    """
    Load a transformer model once and return a callable that when called returns a state_dict
    with small gaussian noise added to each parameter (useful to simulate multiple samples).
    """
    from transformers import AutoModel
    base_model = AutoModel.from_pretrained(model_name_or_path)
    sd = base_model.state_dict()
    # convert to numpy once
    sd_np = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in sd.items()}

    def provider():
        noisy = {}
        for k, arr in sd_np.items():
            noisy_arr = arr + np.random.randn(*arr.shape).astype(np.float32) * noise_scale
            noisy[k] = torch.tensor(noisy_arr)
        return noisy

    return provider


# -----------------------
# CLI + main
# -----------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="distilbert-base-uncased", help="HF model name for demo.")
    p.add_argument("--n_samples", type=int, default=8, help="Number of samples (simulated if using noisy provider).")
    p.add_argument("--k", type=int, default=64, help="Target number of principal components")
    p.add_argument("--r_factor", type=float, default=4.0, help="r = r_factor * k (default 4*k recommended).")
    p.add_argument("--block_size", type=int, default=200_000, help="Block size (number of floats) to stream at once")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="out_pca")
    p.add_argument("--noise_scale", type=float, default=1e-3, help="If >0, use noisy provider to simulate multiple samples")
    p.add_argument("--n_iter_svd", type=int, default=2, help="randomized_svd iterations")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Providers: for demo, simulate n_samples noisy variants of base model.
    # Replace providers with list of checkpoint folder paths or provider functions to use real checkpoints.
    providers = [make_noisy_provider_from_base(args.model, noise_scale=args.noise_scale) for _ in range(args.n_samples)]

    # Build sketch
    print("Building SRHT sketch (single-pass over samples)...")
    C, meta = build_srht_sketch_from_samples(providers, seed=args.seed, k=args.k, r_factor=args.r_factor, block_size=args.block_size)
    print(f"Built sketch C shape: {C.shape}")

    # Save sketch and meta
    sketch_path = os.path.join(args.out_dir, "sketch_C.npy")
    np.save(sketch_path, C)
    meta_to_save = dict(meta)
    meta_to_save.update(dict(seed=args.seed, r_factor=args.r_factor, block_size=args.block_size))
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta_to_save, f, indent=2)

    # Randomized SVD on sketch
    print("Running randomized SVD on sketch...")
    U_r, s_r = randomized_svd_on_sketch(C, k=args.k, n_iter=args.n_iter_svd, )
    np.save(os.path.join(args.out_dir, "U_r.npy"), U_r)
    np.save(os.path.join(args.out_dir, "s_r.npy"), s_r)
    print(f"Top singular values (sketch): {s_r[:min(10, s_r.size)].tolist()}")

    # Reconstruct full PCs (memmap if large)
    D = meta["D"]
    approx_size_bytes = D * args.k * 4
    memmap_path = os.path.join(args.out_dir, "W_full.memmap") if approx_size_bytes > 2_000_000_000 else None
    if memmap_path:
        print(f"Will write W_full memmap at {memmap_path} (size {approx_size_bytes / 1e9:.2f} GB)")

    print("Reconstructing full-space principal directions (SRHT second pass)...")
    W_full = reconstruct_full_pcs_srht(U_r, meta, out_path=memmap_path)
    if memmap_path:
        print(f"W_full saved as memmap at {memmap_path}")
    else:
        np.save(os.path.join(args.out_dir, "W_full.npy"), W_full)
        print("W_full saved to .npy")

    # Compute reconstruction errors per sample
    print("Computing L2 / relative errors per sample...")
    W_ref = memmap_path if memmap_path else os.path.join(args.out_dir, "W_full.npy")
    errors = compute_reconstruction_errors(providers, W_ref, meta, block_size=args.block_size, seed=args.seed)
    with open(os.path.join(args.out_dir, "errors.json"), "w") as f:
        json.dump(errors, f, indent=2)

    # Summarize
    rels = [e["rel"] for e in errors]
    evrs = [e["explained_variance"] for e in errors]
    print("Per-sample relative L2 error: mean %.6f median %.6f" % (float(np.mean(rels)), float(np.median(rels))))
    print("Per-sample explained variance: mean %.6f median %.6f" % (float(np.mean(evrs)), float(np.median(evrs))))

    print("All outputs written to:", args.out_dir)
    print("Files: sketch_C.npy, U_r.npy, s_r.npy, W_full.memmap (or W_full.npy), meta.json, errors.json")

if __name__ == "__main__":
    main()
