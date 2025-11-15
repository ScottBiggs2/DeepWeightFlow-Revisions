# Mixed notes here to avoid cluttering readme.md

To compress/reconstruct: 
`scores = W_full.T @ x` then `x_rec = W_full @ scores`


Reconstruction demo (same idea as before): 
```python 
import numpy as np
W = np.memmap("out_pca/W_full.memmap", dtype=np.float32, mode="r", shape=(D,k))
# flatten model into x (D,)
scores = W.T.dot(x)          # (k,)
x_recon = W.dot(scores)      # (D,)
```

This method is: 
- fast (faster than the Gaussian approach anyways)
- accurate (within reason)
- memory friendly (I can run it for Distil. BERT on my 2020 M1 MacBook)

To increase fidelity, increase k/r ratio and both k and r. We can really push on this with GPUs and high vRAM. 

Additional notes: 

``` bash
Verify reconstruction error
The script writes errors.json with per-sample L2 and relative errors. Lower relative error => better PC capture.

Tuning knobs

Increase r_factor (=> r) for higher accuracy; r=4*k is a good start.

Increase k to capture more variance.

Reduce block_size to lower peak memory at cost of more CPU loops.

SRHT details

SRHT requires padding to nearest power-of-two per-block. Padding is implemented and generally negligible for large blocks.

This SRHT implementation is simple and CPU-based (FWHT in Python). For very large models / speed, consider a C/C++ or NumPy vectorized FWHT, or using PyTorch FFT tricks / Triton kernel.

Interpreting the sketch_C and U_r

sketch_C = Σ (R x_i)(R x_i)^T — covariance in sketch space.

U_r are the sketch-space eigenvectors; W_full = R^T U_r are the lifted full-space approximate PCs.
```

# 📚 Sources:

Randomized PCA / Randomized SVD

Halko, Martinsson & Tropp (2011)
“Finding Structure with Randomness: Probabilistic Algorithms for Constructing Approximate Matrix Decompositions.” (Foundational randomized SVD paper.)
https://arxiv.org/abs/0909.4061

Liberty et al. (2007)
Randomized algorithms for the low-rank approximation of matrices.
https://www.pnas.org/doi/10.1073/pnas.0709640104 

Rokhlin, Szlam & Tygert (2009)
A Randomized Algorithm for Principal Component Analysis.
https://arxiv.org/abs/0809.2274 

SRHT (Subsampled Randomized Hadamard Transform)

Tropp (2011)
Improved analysis of the subsampled randomized Hadamard transform.
https://arxiv.org/abs/1011.1595 

Ailon & Chazelle (2009)
The Fast Johnson–Lindenstrauss Transform and Approximate Nearest Neighbors.
https://www.cs.princeton.edu/~chazelle/pubs/FJLT-sicomp09.pdf 

Ailon & Liberty (2013)
Fast Dimension Reduction Using Rademacher Series on Dual BCH Codes.
https://www.cs.yale.edu/homes/el327/papers/SODA_FJLT_revisited.pdf 

Streaming / Incremental PCA

Boutsidis, Drineas & Magdon-Ismail (there are too many dates on this one)
Near-Optimal Coresets for PCA via Incremental SVD.
https://arxiv.org/abs/1202.3505 

Warmuth & Kuzmin (2008)
Randomized Online PCA Algorithms with Regret Bounds.
https://jmlr.csail.mit.edu/papers/volume9/warmuth08a/warmuth08a.pdf 

Ross et al. (2008)
Incremental Learning for Robust Visual Tracking.
https://www.cs.toronto.edu/~dross/ivt/RossLimLinYang_ijcv.pdf

scikit-learn: IncrementalPCA
https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html 

# 🔩 Engineering References

Facebook Research: FALCONN (Hadamard + cross-polytope LSH)
https://github.com/falconn-lib/falconn 

JAX’s fast Walsh–Hadamard

Faster Walsh-Hadamard Transform and Matrix Multiplication over Finite Fields using Lookup Tables, Alman 2022
https://arxiv.org/abs/2211.04643 

# 🧠 Related Work in (Linear) Model Compression / Weight PCA 

ALBERT (Lan et al. 2019)
https://arxiv.org/abs/1909.11942 

"Low-Rank Adaptation" (LoRA)
https://arxiv.org/abs/2106.09685 

Achlioptas (2003)
Database-friendly random projections.
https://www.sciencedirect.com/science/article/pii/S0022000003000254

# Referenced/Related Libraries 

scikit-learn random projection module
https://scikit-learn.org/stable/modules/random_projection.html 

fbpca (Facebook)
https://github.com/facebookarchive/fbpca 

# 🧱 HuggingFace References (BERT/DistilBERT stuff)

HuggingFace Transformers Weights & Architecture
https://github.com/huggingface/transformers
