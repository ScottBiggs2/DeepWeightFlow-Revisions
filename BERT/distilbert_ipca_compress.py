#!/usr/bin/env python3
"""
distilbert_ipca_blockwise_reconstruct_v2.py

Blockwise per-parameter IncrementalPCA compression and reconstruction for transformer blocks.
- Skips PCA for 1D parameters (biases, LayerNorm)
- Fits IPCA for 2D weight matrices only
- Produces per-parameter, per-block, and global reconstruction metrics
- Computes end-to-end reconstruction MSE for full model
- Pings original and reconstructed models
- Optional save of reconstructed model (default: False)
"""

import argparse
import copy
from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import IncrementalPCA
from transformers import AutoModel, AutoTokenizer

# -----------------------
# Metrics utilities
# -----------------------
def compute_metrics_np(orig: np.ndarray, recon: np.ndarray) -> Dict[str, float]:
    diff = recon - orig
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    max_abs = float(np.max(np.abs(diff)))
    orig_l2 = float(np.linalg.norm(orig))
    recon_l2 = float(np.linalg.norm(recon))
    l2_err = float(np.linalg.norm(diff))
    rel_l2 = l2_err / (orig_l2 + 1e-12)
    orig_var = float(np.var(orig))
    rel_mse = mse / (orig_var + 1e-12)
    denom = (orig_l2 * recon_l2 + 1e-12)
    cosine = float(np.dot(orig, recon) / denom)
    return {
        "mse": mse,
        "rmse": rmse,
        "max_abs": max_abs,
        "orig_l2": orig_l2,
        "recon_l2": recon_l2,
        "l2_error": l2_err,
        "relative_l2_error": rel_l2,
        "relative_mse": rel_mse,
        "cosine_similarity": cosine,
        "n_params": orig.size,
    }

def compute_global_mse(orig_model: nn.Module, rec_model: nn.Module) -> float:
    orig_flat, rec_flat = [], []
    for o_param, r_param in zip(orig_model.parameters(), rec_model.parameters()):
        orig_flat.append(o_param.detach().cpu().float().numpy().ravel())
        rec_flat.append(r_param.detach().cpu().float().numpy().ravel())
    orig_flat = np.concatenate(orig_flat)
    rec_flat = np.concatenate(rec_flat)
    mse = float(np.mean((rec_flat - orig_flat) ** 2))
    return mse

# -----------------------
# Model ping utility
# -----------------------
def ping_model(model: nn.Module, device="cpu"):
    """Run a dummy forward pass to verify model is functional."""
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    sample_text = ["Hello world!", "Testing model ping."]
    inputs = tokenizer(sample_text, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    # only print tensor outputs
    out_shapes = {k: v.shape for k, v in outputs.items() if isinstance(v, torch.Tensor)}
    print("Model ping successful. Output shapes:", out_shapes)

# -----------------------
# Utilities to handle parameters
# -----------------------
def param_to_samples(param_tensor: torch.Tensor) -> np.ndarray:
    arr = param_tensor.detach().cpu().float().numpy()
    if arr.ndim == 2:
        return arr
    elif arr.ndim == 1:
        return arr.reshape(-1, 1)
    else:
        return arr.reshape(arr.shape[0], -1)

def samples_to_param(samples: np.ndarray, orig_shape) -> np.ndarray:
    return samples.reshape(-1)[: int(np.prod(orig_shape))]

# -----------------------
# Compression
# -----------------------
def fit_ipca_for_param(samples: np.ndarray, n_components_target: int, row_batch: int):
    n_rows, n_features = samples.shape
    k = min(n_components_target, n_rows, n_features)
    if k < 1:
        return None, samples.copy()
    ipca = IncrementalPCA(n_components=k)
    first_batch_size = min(n_rows, k)
    ipca.partial_fit(samples[:first_batch_size])
    start = first_batch_size
    while start < n_rows:
        end = min(n_rows, start + row_batch)
        ipca.partial_fit(samples[start:end])
        start = end
    Z = ipca.transform(samples).astype(np.float32)
    return ipca, Z

def ipca_inverse(ipca: IncrementalPCA, Z: np.ndarray) -> np.ndarray:
    return ipca.inverse_transform(Z).astype(np.float32)

def compress_model_blockwise(model: nn.Module,
                             components_per_matrix: int = 128,
                             row_batch: int = 256,
                             verbose: bool = True):
    if hasattr(model, "transformer") and hasattr(model.transformer, "layer"):
        blocks = list(model.transformer.layer)
    elif hasattr(model, "bert") and hasattr(model.bert.encoder, "layer"):
        blocks = list(model.bert.encoder.layer)
    else:
        blocks = [model]

    all_blocks_meta: List[Dict[str, Any]] = []

    for bi, block in enumerate(blocks):
        if verbose:
            print(f"\n=== Block {bi} ===")
        params = []
        for name, p in block.named_parameters(recurse=True):
            params.append((name, p.detach().cpu(), p.shape))
        block_meta = {"block_index": bi, "params": []}

        for pname, ptensor, orig_shape in params:
            samples = param_to_samples(ptensor)
            n_rows, n_features = samples.shape if samples.size else (0, 0)

            if n_rows == 0 or n_features == 0:
                block_meta["params"].append({
                    "name": pname,
                    "orig_shape": orig_shape,
                    "n_rows": n_rows,
                    "n_features": n_features,
                    "k": 0,
                    "ipca": None,
                    "latent": np.zeros(0, dtype=np.float32)
                })
                continue

            # Skip PCA for vectors
            if n_features == 1:
                latent = samples.copy().ravel().astype(np.float32)
                ipca = None
                k = 1
                if verbose:
                    print(f"  param {pname}: vector, skipping PCA, storing as-is")
            else:
                ipca, Z = fit_ipca_for_param(samples, components_per_matrix, row_batch)
                latent = Z.ravel().astype(np.float32)
                k = Z.shape[1]
                if verbose:
                    print(f"  param {pname}: rows={n_rows}, feats={n_features}, k={k}")

            block_meta["params"].append({
                "name": pname,
                "orig_shape": orig_shape,
                "n_rows": n_rows,
                "n_features": n_features,
                "k": k,
                "ipca": ipca,
                "latent": latent
            })

        block_latents = [e["latent"] for e in block_meta["params"] if e["latent"].size > 0]
        block_meta["block_latent"] = np.concatenate(block_latents).astype(np.float32) if block_latents else np.zeros(0, dtype=np.float32)
        all_blocks_meta.append(block_meta)

    return all_blocks_meta

# -----------------------
# Reconstruction
# -----------------------
def reconstruct_model_from_meta(orig_model: nn.Module, meta, device: str = "cpu", verbose: bool = True):
    rec_model = copy.deepcopy(orig_model).to(device)
    if hasattr(rec_model, "transformer") and hasattr(rec_model.transformer, "layer"):
        blocks = list(rec_model.transformer.layer)
    elif hasattr(rec_model, "bert") and hasattr(rec_model.bert.encoder, "layer"):
        blocks = list(rec_model.bert.encoder.layer)
    else:
        blocks = [rec_model]

    per_param_metrics, per_block_metrics = [], []

    for bmeta, block in zip(meta, blocks):
        if verbose:
            print(f"\n--- Reconstruct Block {bmeta['block_index']} ---")

        orig_flat_parts, recon_flat_parts = [], []

        for pentry in bmeta["params"]:
            pname = pentry["name"]
            orig_shape = pentry["orig_shape"]
            n_rows, n_features, k = pentry["n_rows"], pentry["n_features"], pentry["k"]

            orig_param = getattr_in_named_module(orig_model, bmeta['block_index'], pname)
            orig_flat = orig_param.detach().cpu().float().numpy().ravel()

            if pentry["ipca"] is None:
                recon_flat = pentry["latent"].copy()
            else:
                Z = pentry["latent"].reshape((n_rows, k)).astype(np.float32)
                recon_flat = samples_to_param(ipca_inverse(pentry["ipca"], Z), orig_shape)

            assign_param_from_flat(block, pname, recon_flat)

            per_param_metrics.append({
                "block": bmeta['block_index'],
                "name": pname,
                **compute_metrics_np(orig_flat, recon_flat)
            })
            orig_flat_parts.append(orig_flat)
            recon_flat_parts.append(recon_flat)

            if verbose:
                print(f"  reconstructed {pname}: params={orig_flat.size}, k={k}")

        if len(orig_flat_parts) > 0:
            orig_block_flat = np.concatenate(orig_flat_parts).astype(np.float32)
            recon_block_flat = np.concatenate(recon_flat_parts).astype(np.float32)
            block_metrics = compute_metrics_np(orig_block_flat, recon_block_flat)
        else:
            block_metrics = compute_metrics_np(np.zeros(0), np.zeros(0))

        block_metrics["block_index"] = bmeta["block_index"]
        per_block_metrics.append(block_metrics)

        if verbose:
            print(f"  Block {bmeta['block_index']} MSE={block_metrics['mse']:.4e} cosine={block_metrics['cosine_similarity']:.6f}")

    return rec_model, per_param_metrics, per_block_metrics

# -----------------------
# Helpers
# -----------------------
def getattr_in_named_module(model: nn.Module, block_index: int, pname: str):
    if hasattr(model, "transformer") and hasattr(model.transformer, "layer"):
        target_block = list(model.transformer.layer)[block_index]
    elif hasattr(model, "bert") and hasattr(model.bert.encoder, "layer"):
        target_block = list(model.bert.encoder.layer)[block_index]
    else:
        target_block = model

    for full_name, p in target_block.named_parameters(recurse=True):
        if full_name == pname or full_name.endswith(pname):
            return p
    raise KeyError(f"Param {pname} not found in block {block_index}")

def assign_param_from_flat(block: nn.Module, pname: str, flat_array: np.ndarray):
    for full_name, p in block.named_parameters(recurse=True):
        if full_name == pname or full_name.endswith(pname):
            tensor = torch.from_numpy(flat_array.reshape(p.shape)).to(p.device).to(p.dtype)
            p.data.copy_(tensor)
            return
    raise KeyError(f"Cannot assign param {pname} in block; name lookup failed.")

# -----------------------
# Reporting
# -----------------------
def summarize(per_param_metrics: List[Dict[str, Any]], per_block_metrics: List[Dict[str, Any]], global_mse: float = None):
    print("\n================ GLOBAL SUMMARY ================\n")
    total_params = sum(p["n_params"] for p in per_param_metrics)
    weighted_mse = sum(p["mse"] * p["n_params"] for p in per_param_metrics) / (total_params + 1e-12)
    print(f"Total params considered: {total_params}")
    print(f"Weighted MSE: {weighted_mse:.6e}")
    if global_mse is not None:
        print(f"Global reconstruction MSE (all parameters): {global_mse:.6e}")

    worst_params = sorted(per_param_metrics, key=lambda x: x["mse"], reverse=True)[:10]
    print("\nTop 10 worst parameters by MSE:")
    for p in worst_params:
        print(f" Block {p['block']}, {p['name']}: mse={p['mse']:.4e}, cos={p['cosine_similarity']:.6f}, params={p['n_params']}")

    print("\nPer-block summary:")
    for b in per_block_metrics:
        print(f" Block {b['block_index']}: mse={b['mse']:.4e}, cos={b['cosine_similarity']:.6f}, params={b['n_params']}")

    avg_cos = float(np.mean([p["cosine_similarity"] for p in per_param_metrics]))
    avg_rel_l2 = float(np.mean([p["relative_l2_error"] for p in per_param_metrics]))
    print(f"\nAverage cosine across params: {avg_cos:.6f}")
    print(f"Average relative L2 error across params: {avg_rel_l2:.6e}")

# -----------------------
# CLI + Main
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="Blockwise per-parameter IPCA compression with full reconstruction")
    parser.add_argument("--model", type=str, default="distilbert-base-uncased")
    parser.add_argument("--components-per-matrix", type=int, default=128)
    parser.add_argument("--row-batch", type=int, default=256)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--no-reconstruct", action="store_true")
    parser.add_argument("--save-reconstruction", action="store_true",
                        help="Save reconstructed model to disk (default False)")
    args = parser.parse_args()

    print(f"Loading model {args.model} ...")
    model = AutoModel.from_pretrained(args.model)
    model.eval()

    # Ping original
    print("\nPinging original model...")
    ping_model(model, device=args.device)

    # Compress
    print("\nCompressing blocks (per-parameter IPCA)...")
    meta = compress_model_blockwise(model, components_per_matrix=args.components_per_matrix,
                                    row_batch=args.row_batch, verbose=True)

    if args.no_reconstruct:
        print("Skipping reconstruction as requested. Exiting.")
        return

    # Reconstruct
    print("\nReconstructing model from compressed meta and computing metrics...")
    rec_model, per_param_metrics, per_block_metrics = reconstruct_model_from_meta(model, meta,
                                                                                   device=args.device, verbose=True)

    # Ping reconstructed
    print("\nPinging reconstructed model...")
    ping_model(rec_model, device=args.device)

    # Compute global MSE and summarize
    global_mse = compute_global_mse(model, rec_model)
    summarize(per_param_metrics, per_block_metrics, global_mse=global_mse)

    # Optionally save
    if args.save_reconstruction:
        torch.save(rec_model.state_dict(), "reconstructed_model.pt")
        print("Reconstructed model saved to reconstructed_model.pt")

    print("\nDone.")

if __name__ == "__main__":
    main()

# python BERT/distilbert_ipca_compress.py \
#     --model distilbert-base-uncased \
#     --components-per-matrix 256 \
#     --row-batch 256 \
#     --device cpu