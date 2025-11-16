#!/usr/bin/env python3
"""
stream_pca_srht_h200.py

H200-optimized SRHT streaming PCA for huge parameter vectors.

Key points:
 - CUDA-only (expects an NVIDIA GPU like H100/H200)
 - Uses FP16 for FWHT/SRHT heavy work, keeps sketch accumulation in float64 for stability
 - Uses pinned memory for fast host->device transfers
 - Uses torch.linalg.eigh on GPU for the symmetric sketch (faster & stable)
 - Streams blocks and writes W_full as memmap (D x k)
 - Saves: sketch_C.npy, U_r.npy, s_r.npy, W_full.memmap, meta.json, errors.json
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
from transformers import AutoModel

# ----------------------------
# Device (CUDA) and global config
# ----------------------------
def get_device():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this H200-optimized script.")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(dev)
    print(f"Using device: {props.name} ({props.total_memory/1e9:.2f} GB)")
    return dev

DEVICE = get_device()

# ----------------------------
# Utilities: iterate blocks
# ----------------------------
def iter_model_param_blocks(model_or_state_dict, block_size):
    """
    Yields (param_name, block_index, block_array) where block_array is 1D numpy float32.
    Deterministic order (sorted keys).
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
    mix = (seed ^ (global_block_idx * 0x9e3779b1)) & 0xFFFFFFFF
    return np.random.RandomState(int(mix))

# ----------------------------
# Fast FWHT (batched) on GPU
# ----------------------------
@torch.inference_mode()
def fwht_batch(x: torch.Tensor):
    """
    Batched FWHT (Hadamard) in-place style on GPU.
    x: (batch, m) tensor, m must be power of two.
    We assume x on DEVICE.
    """
    assert x.dim() == 2
    batch, m = x.shape
    h = 1
    out = x
    while h < m:
        out = out.view(batch, -1, 2 * h)
        left = out[:, :, :h]
        right = out[:, :, h:2 * h]
        a = left + right
        b_ = left - right
        out = torch.cat([a, b_], dim=2)
        out = out.view(batch, m)
        h *= 2
    return out

def fwht_normalized(x: torch.Tensor):
    return fwht_batch(x) * (1.0 / math.sqrt(x.shape[1]))

# ----------------------------
# SRHT projection: GPU-backed
# ----------------------------
def next_pow2_int(n: int) -> int:
    return 1 << (n - 1).bit_length()

def srht_project_block_gpu(x_np: np.ndarray, rng: np.random.RandomState, r: int, device):
    """
    Project a 1D numpy block x_np -> z (r,)
    Returns: z (np.float64), signs (np.float32 list), idx (np.int64 list), m, n
    """
    n = x_np.size
    m = next_pow2_int(n)
    # pad
    if m != n:
        x_pad = np.zeros(m, dtype=np.float32)
        x_pad[:n] = x_np
    else:
        x_pad = x_np

    # D diag signs (numpy)
    signs = rng.choice([-1.0, 1.0], size=m).astype(np.float32)
    # sample indices
    idx = rng.choice(m, size=r, replace=False).astype(np.int64)

    # pinned transfer to device
    x_t = torch.from_numpy(x_pad).pin_memory().to(device=device, non_blocking=True)
    d_t = torch.from_numpy(signs).pin_memory().to(device=device, non_blocking=True)

    # do SRHT on gpu under autocast fp16
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
        y = x_t * d_t             # D * x
        y = y.unsqueeze(0)        # (1, m)
        y = fwht_normalized(y)    # (1, m) fp16
        y = y.squeeze(0)          # (m,)

        idx_t = torch.from_numpy(idx).to(device=device, non_blocking=True)
        z_t = torch.index_select(y, 0, idx_t)
        z_t = z_t * math.sqrt(float(m) / float(r))

    z = z_t.float().cpu().numpy().astype(np.float64)
    return z, signs, idx, int(m), int(n)

# ----------------------------
# Reconstruct block W_block from U_r using GPU (accelerated)
# ----------------------------
def reconstruct_block_from_U_gpu(U_r: np.ndarray, signs: np.ndarray, idx: np.ndarray, m: int, n: int, device):
    """
    Given U_r (r x k) numpy float32/float64, reconstruct W_block (n x k) as numpy float32.
    Uses the identity: R = sqrt(m/r) * S * H_norm * D
    => R.T @ U = sqrt(m/r) * D * H_norm * (S^T @ U)
    Where S^T @ U is m x k with zeros except rows idx filled with U.
    """
    r, k = U_r.shape
    # Build M (m x k) on GPU
    # dtype: use float32 for memory but computations done under autocast
    M = torch.zeros((m, k), dtype=torch.float32, device=device)
    U_t = torch.from_numpy(U_r.astype(np.float32)).to(device=device)
    idx_t = torch.from_numpy(idx.astype(np.int64)).to(device=device)
    M.index_copy_(0, idx_t, U_t)  # place U rows into M at idx

    # FWHT on columns: we want to apply H_norm to each column -> do batch over columns
    M_t = M.t().contiguous()  # (k, m)
    # perform FWHT normalized in fp16 for speed
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
        M_t = fwht_normalized(M_t)
    M = M_t.t()  # (m, k)

    # multiply by signs D
    signs_t = torch.from_numpy(signs.astype(np.float32)).to(device=device)[:, None]
    M = M * signs_t

    # scale
    M = M * math.sqrt(float(m) / float(r))

    # take first n rows and move to CPU (float32)
    W_block = M[:n, :].to(dtype=torch.float32, device='cpu').numpy()
    return W_block

# ----------------------------
# Build SRHT sketch (single pass)
# ----------------------------
def build_srht_sketch(samples, seed, k, r_factor, block_size, device):
    """
    samples: list of providers (callables or state_dicts or checkpoint paths)
    returns:
      C_cpu: numpy float64 (r x r) -- accumulated sketch (on CPU)
      meta: metadata dict including block_meta (signs, idx, m, n for each global block)
    """
    # examine first sample to get layout
    first = samples[0]
    if callable(first):
        sd = first()
    elif isinstance(first, str):
        sd = AutoModel.from_pretrained(first)
    else:
        sd = first

    block_layout = []
    total_D = 0
    for name, _, block in iter_model_param_blocks(sd, block_size):
        block_layout.append((name, int(block.size)))
        total_D += int(block.size)
    num_blocks = len(block_layout)
    print(f"Total D={total_D}, num_global_blocks={num_blocks}")

    r = max(int(r_factor * k), k + 4)
    # accumulate C on GPU in float64 for stability, then move to CPU
    C_dev = torch.zeros((r, r), dtype=torch.float64, device=device)

    block_meta = []
    global_idx = 0

    for sample_idx, provider in enumerate(tqdm(samples, desc="samples")):
        if callable(provider):
            sd = provider()
        elif isinstance(provider, str):
            sd = AutoModel.from_pretrained(provider)
        else:
            sd = provider

        for name, bidx, block in iter_model_param_blocks(sd, block_size):
            x = block.astype(np.float32)
            rng = rng_for_block(seed, global_idx)

            z, signs, idx, m, n = srht_project_block_gpu(x, rng, r, device)
            # accumulate into C_dev: need to move z to device float64
            z_t = torch.from_numpy(z).to(device=device, dtype=torch.float64)
            # outer product
            C_dev += torch.outer(z_t, z_t)

            if sample_idx == 0:
                block_meta.append(dict(signs=signs.tolist(), idx=idx.tolist(), m=int(m), n=int(n)))
            global_idx += 1

        # attempt to free HF model
        if hasattr(sd, "cpu") and hasattr(sd, "state_dict"):
            del sd

    # move C to CPU numpy float64
    C_cpu = C_dev.cpu().numpy().astype(np.float64)
    meta = dict(D=total_D, r=r, k=k, block_layout=block_layout, num_global_blocks=num_blocks, block_meta=block_meta)
    return C_cpu, meta

# ----------------------------
# Eigendecompose sketch on GPU (symmetric)
# ----------------------------
def eig_on_gpu(C_cpu: np.ndarray, k: int, device):
    """
    Move small symmetric C to GPU and compute top-k eigenvectors (largest).
    Returns U_r (r x k) float32, s_r (k,) float32.
    """
    C_t = torch.from_numpy(C_cpu).to(device=device, dtype=torch.float64)
    # use eigh since C is symmetric PSD
    vals, vecs = torch.linalg.eigh(C_t)  # ascending eigenvalues
    # take top-k
    vals = vals.to(torch.float32)
    vecs = vecs.to(torch.float32)
    top_vals = vals[-k:][::-1].cpu().numpy()
    top_vecs = vecs[:, -k:].cpu().numpy()[:, ::-1]  # r x k
    return top_vecs.astype(np.float32), top_vals.astype(np.float32)

# ----------------------------
# Reconstruct full PCs (second pass)
# ----------------------------
def reconstruct_full_W(U_r: np.ndarray, meta: dict, out_path: str, device):
    """
    Reconstruct W_full by streaming over block_meta and reconstructing W_block for each block.
    Writes to memmap out_path (D x k).
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
    for j, ((name, block_len), bm) in enumerate(tqdm(list(zip(block_layout, block_meta)), desc="reconstruct_blocks", total=len(block_layout))):
        signs = np.array(bm["signs"], dtype=np.float32)
        idx = np.array(bm["idx"], dtype=np.int64)
        m = int(bm["m"])
        n = int(bm["n"])
        W_block = reconstruct_block_from_U_gpu(U_r, signs, idx, m, n, device)
        W[pos:pos + block_len, :] = W_block
        pos += block_len
        if out_path and (j % 32 == 0):
            W.flush()
    if out_path:
        W.flush()
    assert pos == D
    return W

# ----------------------------
# Compute reconstruction errors streaming (uses memmap W)
# ----------------------------
def compute_errors(samples, W_ref_path_or_array, meta, block_size, device):
    D = meta["D"]
    k = meta["k"]

    if isinstance(W_ref_path_or_array, str):
        W = np.memmap(W_ref_path_or_array, dtype=np.float32, mode="r", shape=(D, k))
    else:
        W = W_ref_path_or_array

    results = []
    for provider in tqdm(samples, desc="compute_errors"):
        if callable(provider):
            sd = provider()
        elif isinstance(provider, str):
            sd = AutoModel.from_pretrained(provider)
        else:
            sd = provider

        # compute scores streaming (double precision accumulation)
        scores = np.zeros((k,), dtype=np.float64)
        pos = 0
        for name, bidx, block in iter_model_param_blocks(sd, block_size):
            x = block.astype(np.float32).ravel()
            W_block = W[pos:pos + x.size, :]
            scores += W_block.T.dot(x.astype(np.float64))
            pos += x.size

        # residuals
        pos = 0
        sq_resid = 0.0
        sq_norm = 0.0
        for name, bidx, block in iter_model_param_blocks(sd, block_size):
            x = block.astype(np.float32).ravel()
            W_block = W[pos:pos + x.size, :]
            x_rec = W_block.dot(scores).astype(np.float64)
            diff = x.astype(np.float64) - x_rec
            sq_resid += float((diff ** 2).sum())
            sq_norm += float((x.astype(np.float64) ** 2).sum())
            pos += x.size

        e = math.sqrt(sq_resid)
        e_rel = e / (math.sqrt(sq_norm) + 1e-30)
        evr = 1.0 - (sq_resid / (sq_norm + 1e-30))
        results.append(dict(L2=float(e), rel=float(e_rel), explained_variance=float(evr)))

    if isinstance(W_ref_path_or_array, str):
        del W
    return results

# ----------------------------
# Noisy provider helper (simulate many checkpoints)
# ----------------------------
def make_noisy_provider_from_base(model_name_or_path, noise_scale=1e-6):
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
# CLI & main
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="distilbert-base-uncased")
    p.add_argument("--n_samples", type=int, default=64)
    p.add_argument("--k", type=int, default=256)
    p.add_argument("--r_factor", type=float, default=4.0)
    p.add_argument("--block_size", type=int, default=500_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="out_pca")
    p.add_argument("--noise_scale", type=float, default=1e-6)
    p.add_argument("--n_iter_svd", type=int, default=2)
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = DEVICE

    # providers: replace with real checkpoint paths or provider functions for production
    providers = [make_noisy_provider_from_base(args.model, noise_scale=args.noise_scale) for _ in range(args.n_samples)]

    print("Building SRHT sketch (single pass) ...")
    C_cpu, meta = build_srht_sketch(providers, seed=args.seed, k=args.k, r_factor=args.r_factor, block_size=args.block_size, device=device)
    print("Sketch built:", C_cpu.shape)

    # save sketch and meta
    np.save(os.path.join(args.out_dir, "sketch_C.npy"), C_cpu)
    meta_save = dict(meta)
    meta_save.update(dict(seed=args.seed, r_factor=args.r_factor, block_size=args.block_size))
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta_save, f, indent=2)

    # eigendecompose sketch on GPU
    print("Eigendecomposing sketch on GPU ...")
    U_r, s_r = eig_on_gpu(C_cpu, k=args.k, device=device)
    np.save(os.path.join(args.out_dir, "U_r.npy"), U_r)
    np.save(os.path.join(args.out_dir, "s_r.npy"), s_r)
    print("Top sketch vals:", s_r[:10].tolist())

    # reconstruct full W (memmap)
    D = meta["D"]
    approx_size_bytes = D * args.k * 4
    memmap_path = os.path.join(args.out_dir, "W_full.memmap") if approx_size_bytes > 2_000_000_000 else None
    if memmap_path:
        print(f"Writing W_full memmap at {memmap_path} (size {approx_size_bytes/1e9:.2f} GB)")

    print("Reconstructing full-space principal directions (GPU-accelerated) ...")
    W_full = reconstruct_full_W(U_r, meta, out_path=memmap_path, device=device)
    if memmap_path:
        print("W_full memmap written to", memmap_path)
    else:
        np.save(os.path.join(args.out_dir, "W_full.npy"), W_full)
        print("W_full saved to .npy")

    # compute errors
    print("Computing reconstruction errors ...")
    W_ref = memmap_path if memmap_path else os.path.join(args.out_dir, "W_full.npy")
    errors = compute_errors(providers, W_ref, meta, block_size=args.block_size, device=device)
    with open(os.path.join(args.out_dir, "errors.json"), "w") as f:
        json.dump(errors, f, indent=2)

    rels = [e["rel"] for e in errors]
    evrs = [e["explained_variance"] for e in errors]
    print("Per-sample relative L2 error: mean %.6f median %.6f" % (float(np.mean(rels)), float(np.median(rels))))
    print("Per-sample explained variance: mean %.6f median %.6f" % (float(np.mean(evrs)), float(np.median(evrs))))

    print("Done. Outputs in:", args.out_dir)

if __name__ == "__main__":
    main()
