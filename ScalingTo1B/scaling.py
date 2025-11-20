import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import IncrementalPCA
from torch.utils.data import Dataset, DataLoader
import logging
from tqdm import tqdm
from typing import List, Tuple, Optional
import gc

logging.basicConfig(level=logging.INFO)

# ============================================================================
# 1. DATA GENERATION: Create billion-scale test data from known series
# ============================================================================

def generate_pi_digits_tensor(n_features: int, n_samples: int, 
                               seed: int = 42) -> np.ndarray:
    """
    Generate test data using digits of pi with variations.
    Creates n_samples variations of a pi-digit-based sequence.
    """
    np.random.seed(seed)
    
    # Generate base pi sequence (using approximation for simplicity)
    # In practice, you'd use mpmath or similar for real pi digits
    base = np.random.RandomState(31415926).rand(n_features) 
    
    # Create variations by adding small noise
    samples = []
    for i in range(n_samples):
        noise_scale = 0.01 * (i / n_samples)  # Increasing noise
        variation = base + np.random.randn(n_features) * noise_scale
        samples.append(variation)
    
    return np.array(samples, dtype=np.float32)


def generate_sine_wave_tensor(n_features: int, n_samples: int,
                               seed: int = 42) -> np.ndarray:
    """
    Generate test data using sine waves with different frequencies.
    """
    np.random.seed(seed)
    
    x = np.linspace(0, 100 * np.pi, n_features)
    samples = []
    
    for i in range(n_samples):
        freq = 1.0 + i * 0.1  # Varying frequency
        phase = np.random.rand() * 2 * np.pi
        amplitude = 1.0 + np.random.rand() * 0.5
        
        wave = amplitude * np.sin(freq * x + phase)
        samples.append(wave)
    
    return np.array(samples, dtype=np.float32)


# ============================================================================
# 2. CHUNKED IPCA COMPRESSION
# ============================================================================

class ChunkedIPCA:
    """
    Compress billion-scale features by:
    1. Splitting into manageable chunks
    2. Applying IPCA to each chunk independently
    3. Concatenating compressed representations
    """
    
    def __init__(self, 
                 n_features: int,
                 chunk_size: int = 10_000_000,  # 10M features per chunk
                 total_latent_dim: int = 10_000,  # Target total compressed dimension
                 batch_size: int = 10):
        
        self.n_features = n_features
        self.chunk_size = chunk_size
        self.total_latent_dim = total_latent_dim
        self.batch_size = batch_size
        
        # Calculate chunking
        self.n_chunks = int(np.ceil(n_features / chunk_size))
        self.chunk_boundaries = []
        
        for i in range(self.n_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, n_features)
            self.chunk_boundaries.append((start, end))
        
        # Distribute components across chunks proportionally to their size
        self.components_per_chunk = []
        self.ipca_models = []
        
        total_chunk_features = sum(end - start for start, end in self.chunk_boundaries)
        
        for start, end in self.chunk_boundaries:
            chunk_features = end - start
            # Allocate components proportional to chunk size
            n_components = max(1, int(total_latent_dim * (chunk_features / total_chunk_features)))
            self.components_per_chunk.append(n_components)
        
        # Adjust to hit exact target (handle rounding)
        current_total = sum(self.components_per_chunk)
        if current_total != total_latent_dim:
            diff = total_latent_dim - current_total
            # Add/subtract from largest chunk
            largest_idx = np.argmax(self.components_per_chunk)
            self.components_per_chunk[largest_idx] = max(1, self.components_per_chunk[largest_idx] + diff)
        
        self.actual_latent_dim = sum(self.components_per_chunk)
        
        logging.info(f"ChunkedIPCA Configuration:")
        logging.info(f"  Total features: {n_features:,}")
        logging.info(f"  Chunk size: {chunk_size:,}")
        logging.info(f"  Number of chunks: {self.n_chunks}")
        logging.info(f"  Target latent dim: {total_latent_dim:,}")
        logging.info(f"  Actual latent dim: {self.actual_latent_dim:,}")
        logging.info(f"  Components per chunk: {self.components_per_chunk}")
        logging.info(f"  Compression ratio: {self.actual_latent_dim/n_features:.6f} ({self.actual_latent_dim/n_features*100:.4f}%)")
    
    def fit(self, data: np.ndarray) -> 'ChunkedIPCA':
        """
        Fit IPCA models to each chunk.
        
        Args:
            data: (n_samples, n_features) array
        """
        n_samples = data.shape[0]
        logging.info(f"Fitting ChunkedIPCA on {n_samples} samples...")
        
        for chunk_idx, (start, end) in enumerate(tqdm(self.chunk_boundaries, 
                                                       desc="Fitting chunks")):
            chunk_data = data[:, start:end]
            n_components = self.components_per_chunk[chunk_idx]
            
            # Fit IPCA on this chunk
            ipca = IncrementalPCA(n_components=n_components, batch_size=self.batch_size)
            ipca.fit(chunk_data)
            
            self.ipca_models.append(ipca)
            
            # Log variance explained
            var_explained = np.sum(ipca.explained_variance_ratio_)
            logging.info(f"  Chunk {chunk_idx}: {var_explained:.2%} variance, "
                        f"{n_components} components")
            
            del chunk_data
            gc.collect()
        
        return self
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform data through chunked IPCA.
        
        Returns:
            (n_samples, total_latent_dim) compressed array
        """
        n_samples = data.shape[0]
        compressed_chunks = []
        
        for chunk_idx, (start, end) in enumerate(self.chunk_boundaries):
            chunk_data = data[:, start:end]
            compressed = self.ipca_models[chunk_idx].transform(chunk_data)
            compressed_chunks.append(compressed)
            
            del chunk_data
            gc.collect()
        
        # Concatenate all compressed chunks
        return np.hstack(compressed_chunks)
    
    def inverse_transform(self, latent: np.ndarray) -> np.ndarray:
        """
        Reconstruct original features from compressed representation.
        
        Args:
            latent: (n_samples, total_latent_dim) array
            
        Returns:
            (n_samples, n_features) reconstructed array
        """
        n_samples = latent.shape[0]
        reconstructed_chunks = []
        
        offset = 0
        for chunk_idx, n_components in enumerate(self.components_per_chunk):
            # Extract chunk's latent representation
            chunk_latent = latent[:, offset:offset + n_components]
            
            # Inverse transform
            reconstructed = self.ipca_models[chunk_idx].inverse_transform(chunk_latent)
            reconstructed_chunks.append(reconstructed)
            
            offset += n_components
            del chunk_latent
            gc.collect()
        
        # Concatenate reconstructed chunks
        return np.hstack(reconstructed_chunks)


# ============================================================================
# 3. MEMORY-MAPPED DATASETS
# ============================================================================

class MemoryMappedDataset(Dataset):
    """Dataset that loads from memory-mapped file."""
    
    def __init__(self, data: np.ndarray, device: str = 'cpu'):
        self.data = data
        self.device = device
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]).to(self.device)


class StreamingNoiseDataset(Dataset):
    """Generate Gaussian noise on-the-fly."""
    
    def __init__(self, n_samples: int, latent_dim: int, 
                 std: float = 0.001, device: str = 'cpu'):
        self.n_samples = n_samples
        self.latent_dim = latent_dim
        self.std = std
        self.device = device
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return torch.randn(self.latent_dim, device=self.device) * self.std


# ============================================================================
# 4. FLOW MATCHING MODEL (unchanged from your original)
# ============================================================================

class FlowModel(nn.Module):
    """Lightweight flow model for compressed latent space."""
    
    def __init__(self, latent_dim: int, hidden_dim: int = 1024, 
                 time_embed_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.time_embed_dim = time_embed_dim
        
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.GELU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        self.net = nn.Sequential(
            nn.Linear(latent_dim + time_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            
            nn.Linear(hidden_dim, latent_dim)
        )
        
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
    
    def forward(self, x, t):
        t_embed = self.time_embed(t)
        combined = torch.cat([x, t_embed], dim=-1)
        return self.net(combined)


class Bunch:
    """Simple container for flow data."""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FlowMatching:
    """Flow matching trainer (simplified from your version)."""
    
    def __init__(self, sourceloader, targetloader, model, device):
        self.sourceloader = sourceloader
        self.targetloader = targetloader
        self.model = model.to(device)
        self.device = device
        self.sigma = 0.001
        self.best_loss = float('inf')
        self.best_model_state = None
    
    def sample_from_loader(self, loader):
        try:
            batch = next(iter(loader))
            return batch.to(self.device)
        except:
            return torch.zeros(loader.batch_size, 
                             self.model.latent_dim, 
                             device=self.device)
    
    def sample_time_and_flow(self):
        x0 = self.sample_from_loader(self.sourceloader)
        x1 = self.sample_from_loader(self.targetloader)
        
        batch_size = min(x0.size(0), x1.size(0))
        x0, x1 = x0[:batch_size], x1[:batch_size]
        
        t = torch.rand(batch_size, device=self.device)
        t_pad = t.view(-1, 1)
        
        mu_t = (1 - t_pad) * x0 + t_pad * x1
        epsilon = torch.randn_like(x0) * self.sigma
        xt = mu_t + epsilon
        ut = x1 - x0
        
        return Bunch(t=t.unsqueeze(-1), x0=x0, xt=xt, x1=x1, ut=ut)
    
    def train(self, n_iters: int, optimizer, scheduler=None, log_freq: int = 100):
        pbar = tqdm(range(n_iters), desc="Training")
        
        for i in pbar:
            flow = self.sample_time_and_flow()
            
            flow_pred = self.model(flow.xt, flow.t)
            loss = torch.mean((flow_pred - flow.ut) ** 2)
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if scheduler:
                scheduler.step()
            
            if loss.item() < self.best_loss:
                self.best_loss = loss.item()
                self.best_model_state = {k: v.clone() 
                                        for k, v in self.model.state_dict().items()}
            
            if i % log_freq == 0:
                pbar.set_description(f"Loss: {loss.item():.6f}")
    
    def generate(self, n_samples: int, source_std: float = 0.001, 
                n_steps: int = 50, method: str = "euler") -> torch.Tensor:
        """
        Generate samples using numerical integration.
        
        Args:
            n_samples: Number of samples to generate
            source_std: Standard deviation of source noise
            n_steps: Number of integration steps
            method: Integration method ("euler" or "rk4")
        """
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
        
        self.model.eval()
        
        x = torch.randn(n_samples, self.model.latent_dim, 
                       device=self.device) * source_std
        
        times = torch.linspace(0, 1, n_steps, device=self.device)
        dt = times[1] - times[0]
        
        with torch.no_grad():
            if method == "euler":
                for t in times[:-1]:
                    t_batch = torch.ones(n_samples, 1, device=self.device) * t
                    vt = self.model(x, t_batch)
                    x = x + vt * dt
                    
            elif method == "rk4":
                for i, t in enumerate(times[:-1]):
                    t_batch = torch.ones(n_samples, 1, device=self.device) * t
                    
                    # RK4 integration
                    k1 = self.model(x, t_batch)
                    
                    k2 = self.model(x + 0.5 * dt * k1, t_batch + 0.5 * dt)
                    
                    k3 = self.model(x + 0.5 * dt * k2, t_batch + 0.5 * dt)
                    
                    k4 = self.model(x + dt * k3, t_batch + dt)
                    
                    x = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            else:
                raise ValueError(f"Unknown method: {method}")
        
        self.model.train()
        return x


# ============================================================================
# 5. VALIDATION METRICS
# ============================================================================

class ValidationMetrics:
    """Compute comprehensive validation metrics for flow matching."""
    
    @staticmethod
    def compute_reconstruction_error(original: np.ndarray, 
                                    reconstructed: np.ndarray) -> dict:
        """
        Compute reconstruction quality metrics.
        
        Args:
            original: (n_samples, n_features) ground truth
            reconstructed: (n_samples, n_features) reconstructed
        """
        # Mean Squared Error
        mse = np.mean((original - reconstructed) ** 2)
        
        # Root Mean Squared Error
        rmse = np.sqrt(mse)
        
        # Mean Absolute Error
        mae = np.mean(np.abs(original - reconstructed))
        
        # Relative Error
        relative_error = np.mean(np.abs(original - reconstructed) / 
                                (np.abs(original) + 1e-8))
        
        # Per-sample MSE
        per_sample_mse = np.mean((original - reconstructed) ** 2, axis=1)
        
        # Correlation (per sample)
        correlations = []
        for i in range(len(original)):
            corr = np.corrcoef(original[i], reconstructed[i])[0, 1]
            correlations.append(corr)
        
        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'relative_error': float(relative_error),
            'per_sample_mse_mean': float(np.mean(per_sample_mse)),
            'per_sample_mse_std': float(np.std(per_sample_mse)),
            'correlation_mean': float(np.mean(correlations)),
            'correlation_std': float(np.std(correlations)),
            'correlation_min': float(np.min(correlations)),
        }
    
    @staticmethod
    def compute_distribution_metrics(real: np.ndarray, 
                                     generated: np.ndarray) -> dict:
        """
        Compare statistical properties of real vs generated distributions.
        
        Args:
            real: (n_samples, n_features) real data
            generated: (n_samples, n_features) generated data
        """
        # Mean comparison
        real_mean = np.mean(real, axis=0)
        gen_mean = np.mean(generated, axis=0)
        mean_diff = np.mean(np.abs(real_mean - gen_mean))
        
        # Std comparison
        real_std = np.std(real, axis=0)
        gen_std = np.std(generated, axis=0)
        std_diff = np.mean(np.abs(real_std - gen_std))
        
        # Global statistics
        metrics = {
            'mean_difference': float(mean_diff),
            'std_difference': float(std_diff),
            'real_mean': float(np.mean(real)),
            'gen_mean': float(np.mean(generated)),
            'real_std': float(np.std(real)),
            'gen_std': float(np.std(generated)),
            'real_min': float(np.min(real)),
            'real_max': float(np.max(real)),
            'gen_min': float(np.min(generated)),
            'gen_max': float(np.max(generated)),
        }
        
        return metrics
    
    @staticmethod
    def compute_latent_space_metrics(compressor: ChunkedIPCA,
                                     original_data: np.ndarray,
                                     generated_latent: np.ndarray) -> dict:
        """
        Evaluate quality of latent space representations.
        
        Args:
            compressor: Fitted ChunkedIPCA object
            original_data: Original high-dimensional data
            generated_latent: Generated latent codes
        """
        # Compress original data
        original_latent = compressor.transform(original_data)
        
        # Compare latent distributions
        latent_metrics = ValidationMetrics.compute_distribution_metrics(
            original_latent, generated_latent
        )
        
        # Latent space coverage (how much of original space is covered)
        # Use min/max ranges per dimension
        coverage_per_dim = []
        for i in range(original_latent.shape[1]):
            orig_range = np.ptp(original_latent[:, i])  # peak-to-peak
            gen_range = np.ptp(generated_latent[:, i])
            if orig_range > 0:
                coverage = min(gen_range / orig_range, 1.0)
                coverage_per_dim.append(coverage)
        
        latent_metrics['latent_coverage_mean'] = float(np.mean(coverage_per_dim))
        latent_metrics['latent_coverage_std'] = float(np.std(coverage_per_dim))
        
        return latent_metrics
    
    @staticmethod
    def compute_compression_metrics(compressor: ChunkedIPCA,
                                    original_data: np.ndarray) -> dict:
        """
        Evaluate compression quality per chunk.
        
        Args:
            compressor: Fitted ChunkedIPCA object
            original_data: Original data to compress
        """
        metrics = {
            'n_chunks': compressor.n_chunks,
            'total_features': compressor.n_features,
            'total_latent_dim': compressor.actual_latent_dim,
            'compression_ratio': compressor.actual_latent_dim / compressor.n_features,
            'chunk_metrics': []
        }
        
        # Per-chunk variance explained
        for chunk_idx, ipca in enumerate(compressor.ipca_models):
            start, end = compressor.chunk_boundaries[chunk_idx]
            chunk_data = original_data[:, start:end]
            
            # Compress and reconstruct
            compressed = ipca.transform(chunk_data)
            reconstructed = ipca.inverse_transform(compressed)
            
            # Compute reconstruction error
            chunk_mse = np.mean((chunk_data - reconstructed) ** 2)
            
            chunk_metrics = {
                'chunk_idx': chunk_idx,
                'start': start,
                'end': end,
                'n_features': end - start,
                'n_components': compressor.components_per_chunk[chunk_idx],
                'variance_explained': float(np.sum(ipca.explained_variance_ratio_)),
                'reconstruction_mse': float(chunk_mse),
            }
            
            metrics['chunk_metrics'].append(chunk_metrics)
            
            del chunk_data, compressed, reconstructed
            gc.collect()
        
        return metrics
    
    @staticmethod
    def print_validation_report(metrics: dict):
        """Print formatted validation report."""
        print("\n" + "="*80)
        print("VALIDATION REPORT")
        print("="*80)
        
        if 'reconstruction' in metrics:
            print("\n📊 Reconstruction Quality:")
            r = metrics['reconstruction']
            print(f"  MSE:              {r['mse']:.6e}")
            print(f"  RMSE:             {r['rmse']:.6e}")
            print(f"  MAE:              {r['mae']:.6e}")
            print(f"  Relative Error:   {r['relative_error']:.6f}")
            print(f"  Correlation:      {r['correlation_mean']:.4f} ± {r['correlation_std']:.4f}")
            print(f"  Min Correlation:  {r['correlation_min']:.4f}")
        
        if 'distribution' in metrics:
            print("\n📈 Distribution Comparison:")
            d = metrics['distribution']
            print(f"  Mean Difference:  {d['mean_difference']:.6e}")
            print(f"  Std Difference:   {d['std_difference']:.6e}")
            print(f"  Real:  μ={d['real_mean']:.4f}, σ={d['real_std']:.4f}")
            print(f"  Gen:   μ={d['gen_mean']:.4f}, σ={d['gen_std']:.4f}")
        
        if 'latent' in metrics:
            print("\n🎯 Latent Space Quality:")
            l = metrics['latent']
            print(f"  Coverage:         {l['latent_coverage_mean']:.2%} ± {l['latent_coverage_std']:.2%}")
            print(f"  Mean Difference:  {l['mean_difference']:.6e}")
            print(f"  Std Difference:   {l['std_difference']:.6e}")
        
        if 'compression' in metrics:
            print("\n🗜️  Compression Performance:")
            c = metrics['compression']
            print(f"  Total Features:   {c['total_features']:,}")
            print(f"  Latent Dimension: {c['total_latent_dim']:,}")
            print(f"  Compression:      {c['compression_ratio']:.4%}")
            print(f"  Number of Chunks: {c['n_chunks']}")
            
            print("\n  Per-Chunk Performance:")
            for cm in c['chunk_metrics']:
                print(f"    Chunk {cm['chunk_idx']:2d}: "
                      f"{cm['n_features']:,} → {cm['n_components']:,} features, "
                      f"var={cm['variance_explained']:.2%}, "
                      f"mse={cm['reconstruction_mse']:.6e}")
        
        if 'flow_quality' in metrics:
            print("\n🌊 Flow Quality:")
            f = metrics['flow_quality']
            print(f"  Integration Method: {f['method']}")
            print(f"  Steps:            {f['n_steps']}")
            print(f"  Best Loss:        {f['best_loss']:.6e}")
        
        print("\n" + "="*80)


# ============================================================================
# 6. MAIN PIPELINE (WITH VALIDATION)
# ============================================================================

def run_billion_scale_experiment(
    n_features: int = 1_000_000_000,  # 1 billion
    n_samples: int = 100,
    chunk_size: int = 50_000_000,  # 50M per chunk
    total_latent_dim: int = 10_000,  # Target compressed dimension
    data_type: str = "sine",
    integration_method: str = "rk4",  # "euler" or "rk4"
    integration_steps: int = 50,
    n_generated: int = 10,  # Number of samples to generate for validation
):
    """
    Complete pipeline for billion-scale flow matching with validation.
    
    Args:
        n_features: Total number of features (e.g., 1B)
        n_samples: Number of training samples
        chunk_size: Features per chunk for IPCA
        total_latent_dim: Target total latent dimension after compression
        data_type: "sine" or "pi"
        integration_method: "euler" or "rk4"
        integration_steps: Number of ODE integration steps
        n_generated: Number of samples to generate for validation
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logging.info("="*80)
    logging.info("BILLION-SCALE FLOW MATCHING EXPERIMENT")
    logging.info("="*80)
    logging.info(f"Features: {n_features:,}")
    logging.info(f"Samples: {n_samples}")
    logging.info(f"Chunk size: {chunk_size:,}")
    logging.info(f"Target latent dim: {total_latent_dim:,}")
    logging.info(f"Integration method: {integration_method}")
    logging.info(f"Integration steps: {integration_steps}")
    
    # Step 1: Generate data
    logging.info("\n[1/6] Generating data...")
    if data_type == "sine":
        data = generate_sine_wave_tensor(n_features, n_samples)
    else:
        data = generate_pi_digits_tensor(n_features, n_samples)
    
    logging.info(f"  Data shape: {data.shape}")
    logging.info(f"  Memory: {data.nbytes / 1e9:.2f} GB")
    
    # Step 2: Fit chunked IPCA
    logging.info("\n[2/6] Fitting Chunked IPCA...")
    compressor = ChunkedIPCA(
        n_features=n_features,
        chunk_size=chunk_size,
        total_latent_dim=total_latent_dim,
        batch_size=10
    )
    compressor.fit(data)
    
    # Step 3: Transform to latent space
    logging.info("\n[3/6] Compressing to latent space...")
    latent_data = compressor.transform(data)
    logging.info(f"  Compressed shape: {latent_data.shape}")
    logging.info(f"  Compression: {n_features:,} → {latent_data.shape[1]:,} "
                f"({latent_data.shape[1]/n_features*100:.3f}%)")
    
    # Step 4: Train flow matching
    logging.info("\n[4/6] Training flow matching...")
    
    source_dataset = StreamingNoiseDataset(n_samples, latent_data.shape[1], 
                                          std=0.001, device='cpu')
    target_dataset = MemoryMappedDataset(latent_data, device='cpu')
    
    sourceloader = DataLoader(source_dataset, batch_size=8, shuffle=True)
    targetloader = DataLoader(target_dataset, batch_size=8, shuffle=True)
    
    flow_model = FlowModel(latent_dim=latent_data.shape[1], hidden_dim=512)
    cfm = FlowMatching(sourceloader, targetloader, flow_model, device)
    
    optimizer = torch.optim.AdamW(flow_model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5000)
    
    cfm.train(n_iters=5000, optimizer=optimizer, scheduler=scheduler, log_freq=100)
    
    # Step 5: Generate and reconstruct
    logging.info(f"\n[5/6] Generating {n_generated} new samples...")
    generated_latent = cfm.generate(
        n_samples=n_generated, 
        n_steps=integration_steps,
        method=integration_method
    )
    generated_latent_np = generated_latent.cpu().numpy()
    
    logging.info("  Reconstructing to original space...")
    reconstructed = compressor.inverse_transform(generated_latent_np)
    
    logging.info(f"  Reconstructed shape: {reconstructed.shape}")
    
    # Step 6: Comprehensive validation
    logging.info("\n[6/6] Computing validation metrics...")
    
    validation_metrics = {}
    
    # 6a. Compression quality
    logging.info("  Computing compression metrics...")
    validation_metrics['compression'] = ValidationMetrics.compute_compression_metrics(
        compressor, data
    )
    
    # 6b. Reconstruction quality (compare generated to real samples)
    logging.info("  Computing reconstruction metrics...")
    # Use first n_generated samples from original data for comparison
    validation_metrics['reconstruction'] = ValidationMetrics.compute_reconstruction_error(
        data[:n_generated], reconstructed
    )
    
    # 6c. Distribution comparison
    logging.info("  Computing distribution metrics...")
    validation_metrics['distribution'] = ValidationMetrics.compute_distribution_metrics(
        data[:n_generated], reconstructed
    )
    
    # 6d. Latent space quality
    logging.info("  Computing latent space metrics...")
    validation_metrics['latent'] = ValidationMetrics.compute_latent_space_metrics(
        compressor, data[:n_generated], generated_latent_np
    )
    
    # 6e. Flow quality
    validation_metrics['flow_quality'] = {
        'method': integration_method,
        'n_steps': integration_steps,
        'best_loss': cfm.best_loss,
    }
    
    # Print comprehensive report
    ValidationMetrics.print_validation_report(validation_metrics)
    
    logging.info("\n" + "="*80)
    logging.info("EXPERIMENT COMPLETE")
    logging.info("="*80)
    
    return {
        'compressor': compressor,
        'flow_model': cfm,
        'generated_latent': generated_latent_np,
        'reconstructed': reconstructed,
        'validation_metrics': validation_metrics,
        'original_data': data,
    }


# ============================================================================
# 7. QUICK TEST
# ============================================================================

if __name__ == "__main__":
    # Test with both integration methods
    logging.info("Running comparative test (Euler vs RK4)...")
    
    # Small-scale test with Euler
    logging.info("\n" + "🔹"*40)
    logging.info("TEST 1: Euler Integration")
    logging.info("🔹"*40)
    results_euler = run_billion_scale_experiment(
        n_features=10_000_000,  # 10M for testing
        n_samples=50,
        chunk_size=2_000_000,  # 2M per chunk (5 chunks)
        total_latent_dim=100_000,  # 100K latent (1% of 10M)
        data_type="sine",
        integration_method="euler",
        integration_steps=50,
        n_generated=10
    )
    
    # Small-scale test with RK4
    logging.info("\n" + "🔹"*40)
    logging.info("TEST 2: RK4 Integration")
    logging.info("🔹"*40)
    results_rk4 = run_billion_scale_experiment(
        n_features=10_000_000,  # 10M for testing
        n_samples=50,
        chunk_size=2_000_000,  # 2M per chunk (5 chunks)
        total_latent_dim=100_000,  # 100K latent (1% of 10M)
        data_type="sine",
        integration_method="rk4",
        integration_steps=50,
        n_generated=10
    )
    
    # Compare methods
    logging.info("\n" + "="*80)
    logging.info("METHOD COMPARISON")
    logging.info("="*80)
    
    euler_mse = results_euler['validation_metrics']['reconstruction']['mse']
    rk4_mse = results_rk4['validation_metrics']['reconstruction']['mse']
    
    euler_corr = results_euler['validation_metrics']['reconstruction']['correlation_mean']
    rk4_corr = results_rk4['validation_metrics']['reconstruction']['correlation_mean']
    
    logging.info(f"\nReconstruction MSE:")
    logging.info(f"  Euler: {euler_mse:.6e}")
    logging.info(f"  RK4:   {rk4_mse:.6e}")
    logging.info(f"  Winner: {'RK4' if rk4_mse < euler_mse else 'Euler'} "
                f"({abs(euler_mse - rk4_mse)/min(euler_mse, rk4_mse)*100:.1f}% better)")
    
    logging.info(f"\nCorrelation:")
    logging.info(f"  Euler: {euler_corr:.4f}")
    logging.info(f"  RK4:   {rk4_corr:.4f}")
    logging.info(f"  Winner: {'RK4' if rk4_corr > euler_corr else 'Euler'}")
    
    logging.info("\n" + "="*80)
    logging.info("\nTo run billion-scale:")
    logging.info("  results = run_billion_scale_experiment(")
    logging.info("      n_features=1_000_000_000,  # 1 billion")
    logging.info("      n_samples=100,")
    logging.info("      chunk_size=50_000_000,     # 50M per chunk (20 chunks)")
    logging.info("      total_latent_dim=1_000_000, # 1M latent (0.1% compression)")
    logging.info("      integration_method='rk4',")
    logging.info("      integration_steps=100,")
    logging.info("      n_generated=20")
    logging.info("  )")