#!/usr/bin/env python3
"""
stream_pca_srht_fast_torch.py

PyTorch-accelerated SRHT streaming PCA for huge model weight vectors.
Designed for M1 Mac (PyTorch with MPS/Metal) but falls back to CPU if MPS not available.

Features:
 - Blockwise SRHT projection implemented in PyTorch with vectorized FWHT (fast on MPS/Metal)
 - r = r_factor * k (default r_factor = 4)
 - Accumulate small r x r covariance sketch C (float64) for stability
 - Randomized SVD on C (scikit-learn)
 - Streaming reconstruction pass that computes full-space PCs W_full (memmap if large)
 - Per-sample L2/relative/explained-variance reporting
 - Progress bars and intermediate logging

Save outputs (out_dir):
  - sketch_C.npy        : r x r float64 sketch
  - U_r.npy, s_r.npy    : r x k left singular vectors and singular values
  - W_full.memmap       : memmap (D x k) float32
  - meta.json           : metadata
  - errors.json         : per-sample error metrics

Usage example (recommended for M1 Mac):
  python BERT/stream_pca_srht_fast_torch.py \
    --model distilbert-base-uncased --n_samples 4 --k 128 \
    --r_factor 4 --block_size 200000 --out_dir out_pca

Notes:
 - For real multi-checkpoint runs, replace the simulated noisy providers in main()
   with a list of checkpoint folder paths or loader callables.
 - Ensure your PyTorch supports MPS (Apple Metal) for best performance on M1.
"""

import argparse
import math
import os
import json
import time
from functools import partial

import numpy as np
import torch
from tqdm import tqdm
from sklearn.utils.extmath import randomized_svd
from transformers import AutoModel

# ----------------------------
# Device selection: prefer MPS (Apple Metal) on M1/M2, fallback to CPU
# ----------------------------
def get_torch_device():
    if hasattr(torch, "has_mps") and torch.has_mps:
        try:
            dev = torch.device("mps")
            # quick sanity check
            _ = torch.tensor([1.0], device=dev)
            print("Using device: MPS (Apple Metal)")
            return dev
        except Exception:
            pass
    # fallback
    dev = torch.device("cpu")
    print("Using device: CPU")
    return dev

DEVICE = get_torch_device()

# ----------------------------
# Utilities: iterate blocks
# ----------------------------
def iter_model_param_blocks(model_or_state_dict, block_size):
    """
    Yields (param_name, block_index, block_array) where block_array is 1D numpy float32.
    Order deterministic (sorted by key).
    """
    if isinstance(model_or_state_dict, dict):
        items = sorted(model_or_state_dict.items())
    else:
        items = sorted(model_or_state_dict.state_dict().items())

    for name, tensor in items:
        arr = tensor.detach().cpu().numpy().ravel().astype(np.float32)
        n = arr.size
        num_blocks = math.ceil(n / block_size)
        for b in range(num_blocks):
            s = b * block_size
            e = min(n, (b + 1) * block_size)
            yield name, b, arr[s:e]


def rng_for_block(seed, global_block_idx):
    """
    Deterministic numpy RandomState for per-block metadata generation.
    """
    mix = (seed ^ (global_block_idx * 0x9e3779b1)) & 0xFFFFFFFF
    return np.random.RandomState(int(mix))

# ----------------------------
# Fast FWHT (Hadamard) in PyTorch
# Vectorized: accepts a 2D tensor (batch x m) and performs FWHT on each row.
# m must be a power of two.
# ----------------------------
def fwht_torch_batch(x: torch.Tensor):
    """
    In-place-like FWHT on a batch matrix x (batch, m) on the selected DEVICE.
    Uses iterative butterfly operations with reshape, works in O(m log m).
    Returns transformed tensor (same shape).
    """
    # We operate along last dim
    assert x.ndim == 2
    batch, m = x.shape
    h = 1
    out = x
    while h < m:
        # reshape to (batch, m//(2h), 2h)
        out = out.reshape(batch, -1, 2 * h)
        left = out[:, :, :h]
        right = out[:, :, h:2 * h]
        # butterfly
        a = left + right
        b = left - right
        out = torch.cat([a, b], dim=2)
        out = out.reshape(batch, m)
        h *= 2
    return out

# Apply normalized Hadamard: H / sqrt(m)
def fwht_normalized_torch_batch(x: torch.Tensor):
    out = fwht_torch_batch(x)
    m = out.shape[1]
    return out / math.sqrt(float(m))

# ----------------------------
# SRHT projection (torch version) — project a block into r dims
# Returns:
#   z: torch tensor (r,) float64 for stable accumulation
#   signs (numpy float32), idx (numpy int64), m (int) metadata for reconstruction
# Implementation:
#   - pad to m = next_pow2(n)
#   - D: random +/-1 (generated as numpy, sent to torch)
#   - y = D * x_pad (torch)
#   - compute H_norm * y using fwht_normalized_torch_batch (batch=1)
#   - select r indices via numpy RNG idx (deterministic per-block)
#   - return z = sqrt(m/r) * y[idx]
# ----------------------------
def next_pow2(x):
    return 1 << (x - 1).bit_length()

def srht_project_block_torch(x_np: np.ndarray, rng: np.random.RandomState, r: int, device):
    """
    x_np: 1D numpy float32 array
    returns: z (np.float64 1D array length r), metadata signs (np.float32 m,),
             idx (np.int64 r,), m (int), n (int)
    """
    n = x_np.size
    m = next_pow2(n)
    # pad
    if m != n:
        x_pad = np.zeros(m, dtype=np.float32)
        x_pad[:n] = x_np
    else:
        x_pad = x_np.astype(np.float32).copy()

    # signs D: +/-1
    signs = rng.choice([-1.0, 1.0], size=m).astype(np.float32)
    # use torch tensor on device for FWHT
    # y = D * x_pad
    y = torch.from_numpy(x_pad).to(device=device, dtype=torch.float32) * torch.from_numpy(signs).to(device=device, dtype=torch.float32)
    # apply normalized hadamard (batched)
    y = y.unsqueeze(0)  # shape (1, m)
    y = fwht_normalized_torch_batch(y)  # (1, m) on device
    y = y.squeeze(0)  # (m,)
    # pick idx
    idx = rng.choice(m, size=r, replace=False)
    z = y[idx].cpu().numpy().astype(np.float32) * math.sqrt(float(m) / float(r))
    return z, signs, idx.astype(np.int64), int(m), int(n)

# ----------------------------
# Reconstruct W_block from U_r and SRHT metadata (torch-accelerated)
# Given:
#   U_r (r x k) as numpy or torch
#   signs (m,), idx (r,), m, n
# Process:
#   M = zeros(m, k); M[idx, :] = U_r
#   apply H_norm to columns via torch FWHT (batched)
#   multiply by signs (D)
#   scale by sqrt(m/r)
#   take first n rows -> W_block (n x k) float32
# ----------------------------
def reconstruct_block_srht_from_U_torch(U_r_np: np.ndarray, signs_np: np.ndarray, idx_np: np.ndarray, m: int, n: int, device):
    """
    U_r_np: numpy (r x k) float32/float64
    signs_np: numpy (m,) float32
    idx_np: numpy (r,) int64
    returns: W_block numpy (n x k) float32
    """
    r, k = U_r_np.shape
    # Build M (m x k), place U rows
    # We'll build in torch for speed
    M = torch.zeros((m, k), dtype=torch.float32, device=device)  # use float64 on device if supported, else float32
    # Put U_r into M at rows idx
    U_torch = torch.from_numpy(U_r_np.astype(np.float32)).to(device=device)
    M_idx = torch.from_numpy(idx_np).to(device=device)
    M[M_idx, :] = U_torch
    # apply H_norm to columns: we need shape (k, m) to use our batched fwht (batch over columns)
    M_t = M.transpose(0, 1)  # (k, m)
    # FWHT batch expects (batch, m) where batch=k
    M_t = fwht_normalized_torch_batch(M_t)  # (k, m) float64 on device
    # transpose back
    M = M_t.transpose(0, 1)  # (m, k)
    # multiply by signs D
    signs_t = torch.from_numpy(signs_np.astype(np.float32)).to(device=device)[:, None]
    M = M * signs_t
    # scale
    M = M * math.sqrt(float(m) / float(r))
    # take first n rows
    W_block = M[:n, :].to(dtype=torch.float32, device='cpu').numpy()  # move to CPU numpy
    return W_block

# ----------------------------
# Core sketch build (single-pass)
# ----------------------------
def build_srht_sketch_from_samples_torch(sample_sources, seed, k, r_factor, block_size, device):
    """
    sample_sources: list of providers (callables returning state_dict or path strings or dicts)
    returns:
      C: numpy float64 (r x r)
      meta: dict with block layout and SRHT metadata (per global block)
    """
    # load first provider to build block layout
    first = sample_sources[0]
    if callable(first):
        model_or_sd = first()
    elif isinstance(first, str):
        model_or_sd = AutoModel.from_pretrained(first)
    else:
        model_or_sd = first

    block_layout = []
    total_D = 0
    for name, _, block in iter_model_param_blocks(model_or_sd, block_size):
        block_layout.append((name, int(block.size)))
        total_D += int(block.size)
    num_global_blocks = len(block_layout)
    print(f"total flattened dim D={total_D}, num_global_blocks={num_global_blocks}")

    r = max(int(r_factor * k), k + 4)
    C = np.zeros((r, r), dtype=np.float32)

    # block_meta recorded from the first sample only (consistent layout)
    block_meta = []
    global_block_idx = 0

    for sample_idx, provider in enumerate(tqdm(sample_sources, desc="samples")):
        if callable(provider):
            model_or_sd = provider()
        elif isinstance(provider, str):
            model_or_sd = AutoModel.from_pretrained(provider)
        else:
            model_or_sd = provider

        for name, bidx, block in iter_model_param_blocks(model_or_sd, block_size):
            x = block.astype(np.float32)
            
            norm = np.linalg.norm(x) + 1e-12 # quickly normalize blocks to prevent scale issues
            x = x / norm
            rng = rng_for_block(seed, global_block_idx)
            z, signs, idx, m, n = srht_project_block_torch(x, rng, r, device)
            # accumulate
            C += np.outer(z, z)
            if sample_idx == 0:
                # store minimal metadata
                block_meta.append(dict(signs=signs.tolist(), idx=idx.tolist(), m=int(m), n=int(n)))
            global_block_idx += 1

        # free HF model if loaded
        if hasattr(model_or_sd, "cpu") and hasattr(model_or_sd, "state_dict"):
            del model_or_sd

    meta = dict(D=total_D, r=r, k=k, block_layout=block_layout, num_global_blocks=num_global_blocks, block_meta=block_meta)
    return C, meta

# ----------------------------
# randomized SVD on sketch
# ----------------------------
def randomized_svd_on_sketch(C, k, n_iter=2, random_state=0):
    U, s, Vt = randomized_svd(C, n_components=k, n_iter=n_iter, random_state=random_state)
    return U.astype(np.float32), s.astype(np.float32)

# ----------------------------
# reconstruct full PCs (second pass, accelerated)
# ----------------------------
def reconstruct_full_pcs_srht_torch(U_r, meta, out_path=None, device=torch.device("cpu")):
    """
    Reconstruct W_full using block_meta stored in meta.
    Writes memmap to out_path if provided; else returns numpy array.
    Uses torch for FWHT-heavy inner loops.
    """
    D = meta["D"]
    k = U_r.shape[1]
    block_layout = meta["block_layout"]
    block_meta = meta["block_meta"]
    assert len(block_layout) == len(block_meta)

    if out_path:
        W = np.memmap(out_path, dtype=np.float32, mode="w+", shape=(D, k))
    else:
        W = np.zeros((D, k), dtype=np.float32)

    pos = 0
    total_blocks = len(block_layout)
    start = time.time()
    for j, ((name, block_len), bm) in enumerate(tqdm(list(zip(block_layout, block_meta)), desc="reconstruct_blocks", total=total_blocks)):
        signs = np.array(bm["signs"], dtype=np.float32)
        idx = np.array(bm["idx"], dtype=np.int64)
        m = int(bm["m"])
        n = int(bm["n"])
        # reconstruct W_block using torch-accelerated function
        W_block = reconstruct_block_srht_from_U_torch(U_r, signs, idx, m, n, device=device)
        W[pos:pos + block_len, :] = W_block
        pos += block_len
        # occasionally flush memmap to disk (if using memmap)
        if out_path and (j % 50 == 0):
            W.flush()
    end = time.time()
    if out_path:
        W.flush()
    assert pos == D, f"reconstructed length {pos} != D {D}"
    return W

# ----------------------------
# compute reconstruction errors streaming
# ----------------------------
def compute_reconstruction_errors(sample_providers, W_full_path_or_array, meta, block_size, device):
    D = meta["D"]
    k = meta["k"]
    # load W
    if isinstance(W_full_path_or_array, str):
        W = np.memmap(W_full_path_or_array, dtype=np.float32, mode="r", shape=(D, k))
    else:
        W = W_full_path_or_array

    results = []
    for provider in tqdm(sample_providers, desc="computing_errors"):
        if callable(provider):
            model_sd = provider()
        elif isinstance(provider, str):
            model_sd = AutoModel.from_pretrained(provider).state_dict()
        else:
            model_sd = provider

        # compute scores streaming
        scores = np.zeros((k,), dtype=np.float32)
        pos = 0
        for name, bidx, block in iter_model_param_blocks(model_sd, block_size):
            x_block = block.astype(np.float32).ravel()
            W_block = W[pos:pos + x_block.size, :]
            scores += W_block.T.dot(x_block.astype(np.float32))
            pos += x_block.size

        # reconstruct residual streaming
        pos = 0
        sq_resid = 0.0
        sq_norm = 0.0
        for name, bidx, block in iter_model_param_blocks(model_sd, block_size):
            x_block = block.astype(np.float32).ravel()
            W_block = W[pos:pos + x_block.size, :]
            x_rec_block = W_block.dot(scores).astype(np.float32)
            diff = x_block.astype(np.float32) - x_rec_block
            sq_resid += float((diff ** 2).sum())
            sq_norm += float((x_block.astype(np.float32) ** 2).sum())
            pos += x_block.size

        e = math.sqrt(sq_resid)
        e_rel = e / (math.sqrt(sq_norm) + 1e-30)
        evr = 1.0 - (sq_resid / (sq_norm + 1e-30))
        results.append(dict(L2=float(e), rel=float(e_rel), explained_variance=float(evr)))

    if isinstance(W_full_path_or_array, str):
        del W
    return results

# ----------------------------
# demo noisy provider (simulate multiple checkpoints)
# ----------------------------
def make_noisy_provider_from_base(model_name_or_path, noise_scale=1e-3):
    base_model = AutoModel.from_pretrained(model_name_or_path)
    sd = base_model.state_dict()
    sd_np = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in sd.items()}

    def provider():
        noisy = {}
        for kname, arr in sd_np.items():
            noisy_arr = arr + np.random.randn(*arr.shape).astype(np.float32) * noise_scale
            noisy[kname] = torch.tensor(noisy_arr)
        return noisy
    return provider

# ----------------------------
# CLI + main
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="distilbert-base-uncased")
    p.add_argument("--n_samples", type=int, default=4)
    p.add_argument("--k", type=int, default=128)
    p.add_argument("--r_factor", type=float, default=4.0)
    p.add_argument("--block_size", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="out_pca")
    p.add_argument("--noise_scale", type=float, default=1e-3)
    p.add_argument("--n_iter_svd", type=int, default=2)
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # device for heavy ops (FWHT)
    device = DEVICE

    # Build providers: replace with real checkpoint loaders if you have them
    providers = [make_noisy_provider_from_base(args.model, noise_scale=args.noise_scale) for _ in range(args.n_samples)]

    print("Building SRHT sketch (single-pass) — this will stream model blocks and run FWHT on device...")
    C, meta = build_srht_sketch_from_samples_torch(providers, seed=args.seed, k=args.k, r_factor=args.r_factor, block_size=args.block_size, device=device)
    print(f"Built sketch C shape: {C.shape}")

    # save sketch & meta
    np.save(os.path.join(args.out_dir, "sketch_C.npy"), C)
    meta_save = dict(meta)
    meta_save.update(dict(seed=args.seed, r_factor=args.r_factor, block_size=args.block_size))
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta_save, f, indent=2)

    # randomized SVD
    print("Running randomized SVD on sketch (tiny matrix)...")
    U_r, s_r = randomized_svd_on_sketch(C, k=args.k, n_iter=args.n_iter_svd)
    np.save(os.path.join(args.out_dir, "U_r.npy"), U_r)
    np.save(os.path.join(args.out_dir, "s_r.npy"), s_r)
    print("Top sketch singular values:", s_r[:10].tolist())

    # reconstruct W_full (memmap if large)
    D = meta["D"]
    approx_bytes = D * args.k * 4
    memmap_path = os.path.join(args.out_dir, "W_full.memmap") if approx_bytes > 2_000_000_000 else None
    if memmap_path:
        print(f"Writing W_full memmap at {memmap_path} (size {approx_bytes/1e9:.2f} GB)")

    print("Reconstructing full-space principal directions (SRHT second pass) — heavy compute, progress shown per block...")
    W_full = reconstruct_full_pcs_srht_torch(U_r, meta, out_path=memmap_path, device=device)
    if memmap_path:
        print(f"W_full memmap written to {memmap_path}")
    else:
        np.save(os.path.join(args.out_dir, "W_full.npy"), W_full)
        print("W_full saved to .npy")

    # compute errors
    print("Computing per-sample reconstruction errors...")
    W_ref = memmap_path if memmap_path else os.path.join(args.out_dir, "W_full.npy")
    errors = compute_reconstruction_errors(providers, W_ref, meta, block_size=args.block_size, device=device)
    with open(os.path.join(args.out_dir, "errors.json"), "w") as f:
        json.dump(errors, f, indent=2)
    rels = [e["rel"] for e in errors]
    evrs = [e["explained_variance"] for e in errors]
    print("Per-sample relative L2 error: mean %.6f median %.6f" % (float(np.mean(rels)), float(np.median(rels))))
    print("Per-sample explained variance: mean %.6f median %.6f" % (float(np.mean(evrs)), float(np.median(evrs))))

    print("All done. Outputs in:", args.out_dir)
    print("Files: sketch_C.npy, U_r.npy, s_r.npy, W_full.memmap (or W_full.npy), meta.json, errors.json")

if __name__ == "__main__":
    main()
