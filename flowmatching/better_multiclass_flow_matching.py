import torch
import torch.nn as nn
import logging
from tqdm import tqdm
from utils import Bunch

class MultiClassFlowMatching:
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
        cfg_dropout_prob=0.1,  # NEW: Probability of dropping class conditioning
        label_noise_std=0.05,  # NEW: Std dev of label noise for smoothing
        unconditional_token=-1.0,  # NEW: Token to indicate unconditional generation
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sourceloader = sourceloader
        self.targetloader = targetloader
        self.model = model.to(self.device)
        self.mode = mode
        self.t_dist = t_dist
        self.sigma = 0.001
        self.normalize_pred = normalize_pred
        self.geometric = geometric
        
        # CFG and label smoothing parameters
        self.cfg_dropout_prob = cfg_dropout_prob
        self.label_noise_std = label_noise_std
        self.unconditional_token = unconditional_token

        self.best_loss = float('inf')
        self.best_model_state = None
        self.input_dim = None

    def sample_from_loader(self, loader):
        """Sample a batch from dataloader, returns (data, labels)"""
        try:
            if not hasattr(loader, '_iterator') or loader._iterator is None:
                loader._iterator = iter(loader)
            try:
                batch = next(loader._iterator)
            except StopIteration:
                loader._iterator = iter(loader)
                batch = next(loader._iterator)
            
            # Handle both (data,) and (data, labels) returns
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                return batch[0].to(self.device), batch[1].to(self.device)
            else:
                data = batch[0].to(self.device) if isinstance(batch, (list, tuple)) else batch.to(self.device)
                return data, None
                
        except Exception as e:
            logging.info(f"Error sampling from loader: {str(e)}")
            if hasattr(loader.dataset, '__getitem__'):
                dummy = loader.dataset[0][0]
                return torch.zeros(loader.batch_size, *dummy.shape, device=self.device), None
            return torch.zeros(loader.batch_size, 1, device=self.device), None

    def sample_time_and_flow(self):
        """Sample time t and flow with class conditioning + CFG + label noise"""
        x0, _ = self.sample_from_loader(self.sourceloader)  # Source is just noise, no class
        x1, c1 = self.sample_from_loader(self.targetloader)  # Target has class labels
        
        batch_size = min(x0.size(0), x1.size(0))
        x0, x1 = x0[:batch_size], x1[:batch_size]
        
        # Handle class labels - ensure they're the right shape [batch_size, 1]
        if c1 is not None:
            c1 = c1[:batch_size]
            if c1.dim() == 1:
                c1 = c1.unsqueeze(-1)
        else:
            c1 = torch.zeros(batch_size, 1, device=self.device)
        
        # ========================================================================
        # LABEL NOISE: Add slight noise to class labels for smoother interpolation
        # ========================================================================
        if self.label_noise_std > 0 and self.training:
            noise = torch.randn_like(c1) * self.label_noise_std
            c1 = c1 + noise
        
        # ========================================================================
        # CLASSIFIER-FREE GUIDANCE: Randomly drop class conditioning during training
        # ========================================================================
        if self.cfg_dropout_prob > 0 and self.training:
            # Create mask for which samples should be unconditional
            dropout_mask = torch.rand(batch_size, 1, device=self.device) < self.cfg_dropout_prob
            # Replace masked labels with unconditional token
            c1 = torch.where(dropout_mask, 
                           torch.full_like(c1, self.unconditional_token), 
                           c1)

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

        return Bunch(t=t.unsqueeze(-1), x0=x0, xt=xt, x1=x1, ut=ut, 
                    eps=epsilon, batch_size=batch_size, c=c1)

    def forward(self, flow):
        """Forward pass with class conditioning"""
        flow_pred = self.model(flow.xt, flow.t, flow.c)
        return None, flow_pred

    def loss_fn(self, flow_pred, flow):
        if self.mode == "target":
            l_flow = torch.mean((flow_pred.squeeze() - flow.x1) ** 2)
        else:
            l_flow = torch.mean((flow_pred.squeeze() - flow.ut) ** 2)
        return None, l_flow

    def vector_field(self, xt, t, c):
        """Compute vector field at point xt, time t, and class c"""
        _, pred = self.forward(Bunch(xt=xt, t=t, c=c, batch_size=xt.size(0)))
        return pred if self.mode == "velocity" else pred - xt

    def train(self, n_iters=10, optimizer=None, scheduler=None, sigma=0.001, patience=1e99, 
              log_freq=5, accum_steps=None):
        """Train the flow model with optional gradient accumulation"""
        self.sigma = sigma
        self.training = True  # NEW: Flag for training mode
        last_loss = 1e99
        patience_count = 0
        pbar = tqdm(range(n_iters), desc="Training steps")
        
        accum_count = 0
        accumulated_loss = 0
        
        use_grad_accum = accum_steps is not None and accum_steps > 1
        effective_accum_steps = accum_steps if use_grad_accum else 1
        
        for i in pbar:
            flow = self.sample_time_and_flow()
            _, flow_pred = self.forward(flow)
            _, loss = self.loss_fn(flow_pred, flow)
            
            if torch.isnan(loss) or torch.isinf(loss):
                logging.info(f"Skipping step {i} due to invalid loss: {loss.item()}")
                continue
            
            loss_scaled = loss / effective_accum_steps if use_grad_accum else loss
            loss_scaled.backward()
            accum_count += 1
            accumulated_loss += loss.item()
            
            should_update = (not use_grad_accum) or (accum_count == effective_accum_steps) or (i == n_iters - 1)
            
            if should_update:
                optimizer.step()
                optimizer.zero_grad()
                
                if scheduler:
                    scheduler.step()
                
                avg_loss = accumulated_loss / accum_count
                
                if avg_loss < self.best_loss:
                    self.best_loss = avg_loss
                    self.best_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                
                if avg_loss > last_loss:
                    patience_count += 1
                    if patience_count >= patience:
                        logging.info(f"Early stopping at iteration {i}")
                        break
                else:
                    patience_count = 0
                
                last_loss = avg_loss
                accum_count = 0
                accumulated_loss = 0
            
            if i % log_freq == 0:
                desc = f"Iters [loss {loss.item():.6f}"
                if use_grad_accum:
                    desc += f", accum {accum_count}/{effective_accum_steps}"
                desc += "]"
                pbar.set_description(desc)
        
        if use_grad_accum and accum_count > 0:
            optimizer.step()
            optimizer.zero_grad()
        
        self.training = False  # NEW: Reset training flag

    def map(self, x0, class_label, n_steps=50, return_traj=False, method="euler", 
            guidance_scale=3.0):  # NEW: CFG guidance scale
        """
        Map from source to target for a specific class.
        
        Supports class interpolation: non-integer class_label values will 
        interpolate between classes (e.g., class_label=0.5 interpolates 
        between class 0 and class 1).
        
        Args:
            x0: Initial points [batch_size, dim]
            class_label: Target class - can be:
                - int: specific class (e.g., 0, 1, 2)
                - float: interpolation between classes (e.g., 0.5, 1.3)
                - tensor: per-sample class labels [batch_size, 1]
            n_steps: Number of integration steps
            return_traj: Whether to return full trajectory
            method: Integration method ('euler' or 'rk4')
            guidance_scale: CFG strength (1.0 = no guidance, 2.0-3.0 = strong guidance)
        
        Returns:
            Generated weights at target class (or trajectory if return_traj=True)
        """
        if self.best_model_state is not None:
            current_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            self.model.load_state_dict(self.best_model_state)

        self.model.eval()
        self.training = False  # Ensure training flag is off
        batch_size = x0.size(0)
        
        # Handle class label - supports scalars, floats, and tensors
        if isinstance(class_label, (int, float)):
            c = torch.full((batch_size, 1), float(class_label), device=self.device, dtype=torch.float32)
        else:
            c = class_label.to(self.device).float()
            if c.dim() == 1:
                c = c.unsqueeze(-1)
        
        # ========================================================================
        # CFG: Prepare unconditional labels if guidance is enabled
        # ========================================================================
        use_cfg = guidance_scale != 1.0 and self.cfg_dropout_prob > 0
        if use_cfg:
            c_uncond = torch.full_like(c, self.unconditional_token)
        else:
            c_uncond = None
        
        traj = [x0.detach().clone()] if return_traj else None
        xt = x0.clone()
        times = torch.linspace(0, 1, n_steps, device=self.device)
        dt = times[1] - times[0]

        for i, t in enumerate(times[:-1]):
            with torch.no_grad():
                t_tensor = torch.ones(batch_size, 1, device=self.device) * t
                
                # ================================================================
                # CFG: Compute both conditional and unconditional predictions
                # ================================================================
                if use_cfg:
                    # Conditional prediction
                    pred_cond = self.model(xt, t_tensor, c)
                    if pred_cond.dim() > 2:
                        pred_cond = pred_cond.squeeze(-1)
                    
                    # Unconditional prediction
                    pred_uncond = self.model(xt, t_tensor, c_uncond)
                    if pred_uncond.dim() > 2:
                        pred_uncond = pred_uncond.squeeze(-1)
                    
                    # CFG formula: pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
                    pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
                else:
                    # Standard prediction without CFG
                    pred = self.model(xt, t_tensor, c)
                    if pred.dim() > 2:
                        pred = pred.squeeze(-1)
                
                vt = pred if self.mode == "velocity" else pred - xt
                
                if method == "euler":
                    xt = xt + vt * dt
                elif method == "rk4":
                    # Note: For RK4 with CFG, we need to recompute at each sub-step
                    # This is more expensive but more accurate
                    k1 = vt
                    
                    # k2 computation
                    xt_k2 = xt + 0.5 * dt * k1
                    t_k2 = t_tensor + 0.5 * dt
                    if use_cfg:
                        pred_k2_cond = self.model(xt_k2, t_k2, c)
                        if pred_k2_cond.dim() > 2:
                            pred_k2_cond = pred_k2_cond.squeeze(-1)
                        pred_k2_uncond = self.model(xt_k2, t_k2, c_uncond)
                        if pred_k2_uncond.dim() > 2:
                            pred_k2_uncond = pred_k2_uncond.squeeze(-1)
                        k2 = pred_k2_uncond + guidance_scale * (pred_k2_cond - pred_k2_uncond)
                    else:
                        k2 = self.model(xt_k2, t_k2, c)
                        if k2.dim() > 2:
                            k2 = k2.squeeze(-1)
                    k2 = k2 if self.mode == "velocity" else k2 - xt_k2
                    
                    # k3 computation
                    xt_k3 = xt + 0.5 * dt * k2
                    if use_cfg:
                        pred_k3_cond = self.model(xt_k3, t_k2, c)
                        if pred_k3_cond.dim() > 2:
                            pred_k3_cond = pred_k3_cond.squeeze(-1)
                        pred_k3_uncond = self.model(xt_k3, t_k2, c_uncond)
                        if pred_k3_uncond.dim() > 2:
                            pred_k3_uncond = pred_k3_uncond.squeeze(-1)
                        k3 = pred_k3_uncond + guidance_scale * (pred_k3_cond - pred_k3_uncond)
                    else:
                        k3 = self.model(xt_k3, t_k2, c)
                        if k3.dim() > 2:
                            k3 = k3.squeeze(-1)
                    k3 = k3 if self.mode == "velocity" else k3 - xt_k3
                    
                    # k4 computation
                    xt_k4 = xt + dt * k3
                    t_k4 = t_tensor + dt
                    if use_cfg:
                        pred_k4_cond = self.model(xt_k4, t_k4, c)
                        if pred_k4_cond.dim() > 2:
                            pred_k4_cond = pred_k4_cond.squeeze(-1)
                        pred_k4_uncond = self.model(xt_k4, t_k4, c_uncond)
                        if pred_k4_uncond.dim() > 2:
                            pred_k4_uncond = pred_k4_uncond.squeeze(-1)
                        k4 = pred_k4_uncond + guidance_scale * (pred_k4_cond - pred_k4_uncond)
                    else:
                        k4 = self.model(xt_k4, t_k4, c)
                        if k4.dim() > 2:
                            k4 = k4.squeeze(-1)
                    k4 = k4 if self.mode == "velocity" else k4 - xt_k4
                    
                    xt = xt + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

                if return_traj:
                    traj.append(xt.detach().clone())

        if self.best_model_state is not None:
            self.model.load_state_dict(current_state)
        self.model.train()
        return traj if return_traj else xt

    def generate_weights(self, n_samples=10, class_label=0, source_noise_std=0.001, **map_kwargs):
        """Generate weights for a specific class"""
        assert self.input_dim is not None, "Set `self.input_dim` before generating weights."
        source_samples = torch.randn(n_samples, self.input_dim, device=self.device) * source_noise_std
        return self.map(source_samples, class_label=class_label, **map_kwargs)


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.residual = (in_dim == out_dim)

    def forward(self, x):
        out = self.norm(self.activation(self.linear(x)))
        if self.residual:
            return out + x
        return self.dropout(out)


class MultiClassWeightSpaceFlowModel(nn.Module):
    """
    Enhanced multiclass flow model with larger embeddings for better class separation
    """
    def __init__(self, input_dim, hidden_dim, dropout=0.1, time_embed_dim=64, class_embed_dim=128):  # Larger default!
        super().__init__()
        self.input_dim = input_dim
        self.time_embed_dim = time_embed_dim
        self.class_embed_dim = class_embed_dim 
        self.hidden_dims = [hidden_dim]
        
        # ========================================================================
        # ENHANCED CLASS EMBEDDING: Deeper and more expressive for better separation
        # ========================================================================
        self.class_embed = nn.Sequential(
            nn.Linear(1, class_embed_dim),
            nn.LayerNorm(class_embed_dim),  # Add normalization
            nn.GELU(),
            nn.Dropout(dropout * 0.5),  # Light dropout
            nn.Linear(class_embed_dim, class_embed_dim),
            nn.LayerNorm(class_embed_dim),
            nn.GELU(),
            nn.Linear(class_embed_dim, class_embed_dim)  # Third layer for more capacity
        )

        self.time_embed = nn.Sequential( 
            nn.Linear(1, time_embed_dim),
            nn.GELU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        ) 

        # self.input_project = nn.Sequential(
        #     nn.Linear(input_dim, hidden_dim),
        #     nn.LayerNorm(hidden_dim),
        #     nn.GELU(),
        #     nn.Dropout(dropout * 0.5),  # Light dropout
        #     nn.Linear(hidden_dim, hidden_dim),
        # )

        # # Main MLP blocks with conditioning re-injection
        self.block_1 = ResidualBlock(input_dim + time_embed_dim + class_embed_dim, hidden_dim, dropout=dropout)
        self.block_2 = ResidualBlock(hidden_dim + time_embed_dim + class_embed_dim, hidden_dim, dropout=dropout)
        self.block_3 = ResidualBlock(hidden_dim + time_embed_dim + class_embed_dim, hidden_dim, dropout=dropout)
        self.block_4 = ResidualBlock(hidden_dim, hidden_dim, dropout=0)
        self.block_5 = ResidualBlock(hidden_dim, hidden_dim, dropout=0)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

        # Main MLP blocks with conditioning re-injection
        # self.block_1 = ResidualBlock(hidden_dim + time_embed_dim + class_embed_dim, hidden_dim, dropout=dropout)
        # self.block_2 = ResidualBlock(hidden_dim, hidden_dim, dropout=dropout)
        # self.block_3 = ResidualBlock(hidden_dim, hidden_dim//2, dropout=dropout)
        # self.block_4 = ResidualBlock(hidden_dim//2, hidden_dim, dropout=0)
        # self.block_5 = ResidualBlock(hidden_dim, hidden_dim, dropout=0)
        # self.output_layer = nn.Linear(hidden_dim, input_dim)

        # Initialize output layer to near-zero for stability
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x, t, c): 
        t_embed = self.time_embed(t)
        c_embed = self.class_embed(c)
        # x_embed = self.input_project(x)

        combined_input = torch.cat([x, t_embed, c_embed], dim=-1)
        h1 = self.block_1(combined_input)

        h1 = torch.cat([h1, t_embed, c_embed], dim=-1)
        h2 = self.block_2(h1)

        h2 = torch.cat([h2, t_embed, c_embed], dim=-1)
        h3 = self.block_3(h2)

        h4 = self.block_4(h3)

        h5 = self.block_5(h4)

        output = self.output_layer(h5)

        return output