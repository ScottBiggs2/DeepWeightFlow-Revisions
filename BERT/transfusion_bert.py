"""
Complete BERT Transfusion Implementation
Corrected and organized version
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import copy
from typing import List, Tuple, Optional, Dict, Any
from scipy.optimize import linear_sum_assignment
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from datasets import load_dataset
from sklearn.metrics import r2_score
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== BERT MODEL ARCHITECTURE ====================

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = FeedForward(embed_dim, int(embed_dim * mlp_ratio), dropout)
        
    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x


class BERTTransformer(nn.Module):
    def __init__(self, vocab_size=30522, max_seq_len=128, embed_dim=512, 
                 num_heads=8, num_layers=8, num_classes=1, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self._init_weights()
        
    def _init_weights(self):
        nn.init.normal_(self.token_embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
        
    def forward(self, input_ids, mask=None):
        B, N = input_ids.shape
        x = self.token_embed(input_ids)
        x = x + self.pos_embed[:, :N, :]
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x, mask)
        x = self.norm(x)
        x = x[:, 0]
        x = self.head(x)
        return x


def create_bert_50m(num_classes=1):
    return BERTTransformer(
        vocab_size=30522, max_seq_len=128, embed_dim=768,
        num_heads=12, num_layers=8, num_classes=num_classes, dropout=0.05
    )

# ==================== DATASET ====================

class YelpReviewDataset(Dataset):
    def __init__(self, split='train', max_length=128, subset_size=10000):
        raw_data = load_dataset("yelp_review_full", split=split)
        self.data = raw_data.select(range(min(len(raw_data), subset_size)))
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item['text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze(0)
        label = torch.tensor(item['label'] / 4.0, dtype=torch.float32)
        return input_ids, label


# ==================== BERT WEIGHT SPACE OBJECTS ====================

class BERTAttentionWeights:
    """Weight space object for BERT attention layers"""
    def __init__(self, qkv_weight, qkv_bias, proj_weight, proj_bias, num_heads):
        self.qkv_weight = qkv_weight
        self.qkv_bias = qkv_bias
        self.proj_weight = proj_weight
        self.proj_bias = proj_bias
        self.num_heads = num_heads
        
    def split_heads(self):
        """Split QKV weights into per-head Q, K, V matrices"""
        embed_dim = self.qkv_weight.shape[1]
        head_dim = embed_dim // self.num_heads
        
        # QKV weight is [3*embed_dim, embed_dim]
        qkv = self.qkv_weight.reshape(3, embed_dim, embed_dim)
        q_weight, k_weight, v_weight = qkv[0], qkv[1], qkv[2]
        
        # Split into heads: [num_heads, head_dim, embed_dim]
        q_heads = q_weight.reshape(self.num_heads, head_dim, embed_dim)
        k_heads = k_weight.reshape(self.num_heads, head_dim, embed_dim)
        v_heads = v_weight.reshape(self.num_heads, head_dim, embed_dim)
        
        return q_heads, k_heads, v_heads


class BERTTransformerBlockWeights:
    """
    Weight space object for a BERT transformer block
    NOW USES TUPLES like VisionTransformerWeightSpace!
    """
    def __init__(self, attention, norm1_weight, norm1_bias, 
                 mlp_weights, mlp_biases,
                 norm2_weight, norm2_bias):
        self.attention = attention
        self.norm1_weight = norm1_weight
        self.norm1_bias = norm1_bias
        self.mlp_weights = mlp_weights
        self.mlp_biases = mlp_biases
        self.norm2_weight = norm2_weight
        self.norm2_bias = norm2_bias


class BERTWeightSpace:
    """Weight space object for BERT Transformers"""
    
    def __init__(self, 
                 token_embed: torch.Tensor,
                 pos_embed: torch.Tensor,
                 blocks: List[BERTTransformerBlockWeights],
                 norm_weight: torch.Tensor,
                 norm_bias: torch.Tensor,
                 head_weight: torch.Tensor,
                 head_bias: torch.Tensor):
        
        self.token_embed = token_embed
        self.pos_embed = pos_embed
        self.blocks = blocks
        self.norm_weight = norm_weight
        self.norm_bias = norm_bias
        self.head_weight = head_weight
        self.head_bias = head_bias
        
    @classmethod
    def from_bert_model(cls, model: nn.Module):
        """Extract weights from a BERT model"""
        blocks = []
        
        for block in model.blocks:
            attn = block.attn
            attention_weights = BERTAttentionWeights(
                qkv_weight=attn.qkv.weight.data.clone(),
                qkv_bias=attn.qkv.bias.data.clone() if attn.qkv.bias is not None else None,
                proj_weight=attn.proj.weight.data.clone(),
                proj_bias=attn.proj.bias.data.clone() if attn.proj.bias is not None else None,
                num_heads=attn.num_heads
            )
            
            mlp_weights = []
            mlp_biases = []
            for name, layer in block.mlp.named_children():
                if hasattr(layer, 'weight'):
                    mlp_weights.append(layer.weight.data.clone())
                    if hasattr(layer, 'bias') and layer.bias is not None:
                        mlp_biases.append(layer.bias.data.clone())
            mlp_weights = tuple(mlp_weights)
            mlp_biases = tuple(mlp_biases)
            
            block_weights = BERTTransformerBlockWeights(
                attention=attention_weights,
                norm1_weight=block.norm1.weight.data.clone(),
                norm1_bias=block.norm1.bias.data.clone(),
                mlp_weights=mlp_weights,
                mlp_biases=mlp_biases,
                norm2_weight=block.norm2.weight.data.clone(),
                norm2_bias=block.norm2.bias.data.clone()
            )
            blocks.append(block_weights)
        
        return cls(
            token_embed=model.token_embed.weight.data.clone(),
            pos_embed=model.pos_embed.data.clone(),
            blocks=blocks,
            norm_weight=model.norm.weight.data.clone(),
            norm_bias=model.norm.bias.data.clone(),
            head_weight=model.head.weight.data.clone(),
            head_bias=model.head.bias.data.clone()
        )
    
    def apply_to_model(self, model: nn.Module):
        """Apply weights to a BERT model"""
        with torch.no_grad():
            model.token_embed.weight.data.copy_(self.token_embed)
            model.pos_embed.data.copy_(self.pos_embed)
            
            for block, block_weights in zip(model.blocks, self.blocks):
                attn = block.attn
                attn.qkv.weight.data.copy_(block_weights.attention.qkv_weight)
                if block_weights.attention.qkv_bias is not None:
                    attn.qkv.bias.data.copy_(block_weights.attention.qkv_bias)
                attn.proj.weight.data.copy_(block_weights.attention.proj_weight)
                if block_weights.attention.proj_bias is not None:
                    attn.proj.bias.data.copy_(block_weights.attention.proj_bias)
                
                block.norm1.weight.data.copy_(block_weights.norm1_weight)
                block.norm1.bias.data.copy_(block_weights.norm1_bias)
                block.norm2.weight.data.copy_(block_weights.norm2_weight)
                block.norm2.bias.data.copy_(block_weights.norm2_bias)
                
                mlp_layers = [layer for name, layer in block.mlp.named_children() 
                             if hasattr(layer, 'weight')]
                for layer, weight in zip(mlp_layers, block_weights.mlp_weights):
                    layer.weight.data.copy_(weight)
                    
                mlp_bias_idx = 0
                for name, layer in block.mlp.named_children():
                    if hasattr(layer, 'bias') and layer.bias is not None:
                        if mlp_bias_idx < len(block_weights.mlp_biases):
                            layer.bias.data.copy_(block_weights.mlp_biases[mlp_bias_idx])
                            mlp_bias_idx += 1
            
            model.norm.weight.data.copy_(self.norm_weight)
            model.norm.bias.data.copy_(self.norm_bias)
            model.head.weight.data.copy_(self.head_weight)
            model.head.bias.data.copy_(self.head_bias)
    
    def flatten(self, device=None) -> torch.Tensor:
        """Flatten all weights into a single vector"""
        all_params = []
        
        all_params.append(self.token_embed.flatten())
        all_params.append(self.pos_embed.flatten())
        
        for block in self.blocks:
            all_params.append(block.attention.qkv_weight.flatten())
            if block.attention.qkv_bias is not None:
                all_params.append(block.attention.qkv_bias.flatten())
            all_params.append(block.attention.proj_weight.flatten())
            if block.attention.proj_bias is not None:
                all_params.append(block.attention.proj_bias.flatten())
            
            all_params.append(block.norm1_weight.flatten())
            all_params.append(block.norm1_bias.flatten())
            all_params.append(block.norm2_weight.flatten())
            all_params.append(block.norm2_bias.flatten())
            
            for w in block.mlp_weights:
                all_params.append(w.flatten())
            for b in block.mlp_biases:
                all_params.append(b.flatten())
        
        all_params.append(self.norm_weight.flatten())
        all_params.append(self.norm_bias.flatten())
        all_params.append(self.head_weight.flatten())
        all_params.append(self.head_bias.flatten())
        
        flat = torch.cat(all_params)
        if device:
            flat = flat.to(device)
        return flat
    
    @classmethod
    def from_flat(cls, flat_tensor, reference_ws, device=None):
        """Reconstruct BERTWeightSpace from flattened weights"""
        if device is None:
            device = flat_tensor.device
            
        param_shapes = []
        param_types = []
        
        param_shapes.append(reference_ws.token_embed.shape)
        param_types.append('token_embed')
        
        param_shapes.append(reference_ws.pos_embed.shape)
        param_types.append('pos_embed')
        
        for block_idx, block in enumerate(reference_ws.blocks):
            param_shapes.append(block.attention.qkv_weight.shape)
            param_types.append(f'block_{block_idx}_attn_qkv_weight')
            
            if block.attention.qkv_bias is not None:
                param_shapes.append(block.attention.qkv_bias.shape)
                param_types.append(f'block_{block_idx}_attn_qkv_bias')
            
            param_shapes.append(block.attention.proj_weight.shape)
            param_types.append(f'block_{block_idx}_attn_proj_weight')
            
            if block.attention.proj_bias is not None:
                param_shapes.append(block.attention.proj_bias.shape)
                param_types.append(f'block_{block_idx}_attn_proj_bias')
            
            param_shapes.append(block.norm1_weight.shape)
            param_types.append(f'block_{block_idx}_norm1_weight')
            
            param_shapes.append(block.norm1_bias.shape)
            param_types.append(f'block_{block_idx}_norm1_bias')
            
            param_shapes.append(block.norm2_weight.shape)
            param_types.append(f'block_{block_idx}_norm2_weight')
            
            param_shapes.append(block.norm2_bias.shape)
            param_types.append(f'block_{block_idx}_norm2_bias')
            
            for mlp_idx, mlp_weight in enumerate(block.mlp_weights):
                param_shapes.append(mlp_weight.shape)
                param_types.append(f'block_{block_idx}_mlp_weight_{mlp_idx}')
            
            for mlp_idx, mlp_bias in enumerate(block.mlp_biases):
                param_shapes.append(mlp_bias.shape)
                param_types.append(f'block_{block_idx}_mlp_bias_{mlp_idx}')
        
        param_shapes.append(reference_ws.norm_weight.shape)
        param_types.append('norm_weight')
        
        param_shapes.append(reference_ws.norm_bias.shape)
        param_types.append('norm_bias')
        
        param_shapes.append(reference_ws.head_weight.shape)
        param_types.append('head_weight')
        
        param_shapes.append(reference_ws.head_bias.shape)
        param_types.append('head_bias')
        
        sizes = [np.prod(shape) for shape in param_shapes]
        parts = []
        start = 0
        
        for size in sizes:
            parts.append(flat_tensor[start:start+size])
            start += size
        
        reconstructed_params = {}
        for i, (shape, param_type) in enumerate(zip(param_shapes, param_types)):
            reconstructed_params[param_type] = parts[i].reshape(shape).to(device)
        
        reconstructed_blocks = []
        num_blocks = len(reference_ws.blocks)
        
        for block_idx in range(num_blocks):
            qkv_weight = reconstructed_params[f'block_{block_idx}_attn_qkv_weight']
            qkv_bias = reconstructed_params.get(f'block_{block_idx}_attn_qkv_bias', None)
            proj_weight = reconstructed_params[f'block_{block_idx}_attn_proj_weight']  
            proj_bias = reconstructed_params.get(f'block_{block_idx}_attn_proj_bias', None)
            
            attention = BERTAttentionWeights(
                qkv_weight=qkv_weight,
                qkv_bias=qkv_bias,
                proj_weight=proj_weight,
                proj_bias=proj_bias,
                num_heads=reference_ws.blocks[block_idx].attention.num_heads
            )
            
            mlp_weights = []
            mlp_biases = []
            
            mlp_weight_idx = 0
            while f'block_{block_idx}_mlp_weight_{mlp_weight_idx}' in reconstructed_params:
                mlp_weights.append(reconstructed_params[f'block_{block_idx}_mlp_weight_{mlp_weight_idx}'])
                mlp_weight_idx += 1
            
            mlp_bias_idx = 0
            while f'block_{block_idx}_mlp_bias_{mlp_bias_idx}' in reconstructed_params:
                mlp_biases.append(reconstructed_params[f'block_{block_idx}_mlp_bias_{mlp_bias_idx}'])
                mlp_bias_idx += 1
            
            block = BERTTransformerBlockWeights(
                attention=attention,
                norm1_weight=reconstructed_params[f'block_{block_idx}_norm1_weight'],
                norm1_bias=reconstructed_params[f'block_{block_idx}_norm1_bias'],
                mlp_weights=tuple(mlp_weights),
                mlp_biases=tuple(mlp_biases),
                norm2_weight=reconstructed_params[f'block_{block_idx}_norm2_weight'],
                norm2_bias=reconstructed_params[f'block_{block_idx}_norm2_bias']
            )
            
            reconstructed_blocks.append(block)
        
        return cls(
            token_embed=reconstructed_params['token_embed'],
            pos_embed=reconstructed_params['pos_embed'],
            blocks=reconstructed_blocks,
            norm_weight=reconstructed_params['norm_weight'], 
            norm_bias=reconstructed_params['norm_bias'],
            head_weight=reconstructed_params['head_weight'],
            head_bias=reconstructed_params['head_bias']
        )


# ==================== BERT PERMUTATION SPEC ====================

class BERTPermutationSpec:
    """Specification for BERT permutations"""
    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        self.block_perms = []
        for _ in range(num_blocks):
            self.block_perms.append({
                'attention_out': None,
                'mlp1': None,
            })
    
    def set_block_perm(self, block_idx: int, perm_type: str, perm: torch.Tensor):
        """Set a specific permutation for a block"""
        self.block_perms[block_idx][perm_type] = perm


# ==================== BERT TRANSFUSION MATCHER ====================

class TransFusionMatcher:
    """
    Weight matching using TransFusion approach for Vision Transformers
    """
    
    def __init__(self, num_iterations: int = 5, epsilon: float = 1e-8):
        self.num_iterations = num_iterations
        self.epsilon = epsilon
        
    def compute_spectral_distance(self, weight1: torch.Tensor, weight2: torch.Tensor) -> float:
        """Compute permutation-invariant distance using singular values"""
        try:
            _, s1, _ = torch.svd(weight1)
            _, s2, _ = torch.svd(weight2)
        except:
            _, s1, _ = np.linalg.svd(weight1.cpu().numpy())
            _, s2, _ = np.linalg.svd(weight2.cpu().numpy())
            s1 = torch.tensor(s1)
            s2 = torch.tensor(s2)
        
        max_len = max(len(s1), len(s2))
        if len(s1) < max_len:
            s1 = torch.cat([s1, torch.zeros(max_len - len(s1))])
        if len(s2) < max_len:
            s2 = torch.cat([s2, torch.zeros(max_len - len(s2))])
        
        return torch.norm(s1 - s2).item()
    
    def compose_attention_permutation(self, inter_head_perm: torch.Tensor,
                                     intra_head_perms: List[torch.Tensor],
                                     d_model: int, num_heads: int) -> torch.Tensor:
        """Compose inter and intra head permutations into a single block diagonal matrix"""
        head_dim = d_model // num_heads
        P_attn = torch.zeros(d_model, d_model)
        
        for i in range(num_heads):
            j = torch.argmax(inter_head_perm[:, i]).item()
            P_intra = intra_head_perms[j] if j < len(intra_head_perms) else torch.eye(head_dim)
            
            start_i = i * head_dim
            end_i = (i + 1) * head_dim
            start_j = j * head_dim
            end_j = (j + 1) * head_dim
            
            P_attn[start_i:end_i, start_j:end_j] = P_intra
        
        return P_attn
    
    def match_attention_heads(self, attn1, attn2) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """Two-level matching for attention heads"""
        q1, k1, v1 = attn1.split_heads()
        q2, k2, v2 = attn2.split_heads()
        
        num_heads = attn1.num_heads
        d_model = attn1.qkv_weight.shape[1]
        head_dim = d_model // num_heads
        
        distance_matrix = torch.zeros(num_heads, num_heads)
        
        for i in range(num_heads):
            for j in range(num_heads):
                dist_q = self.compute_spectral_distance(q1[i], q2[j])
                dist_k = self.compute_spectral_distance(k1[i], k2[j])
                dist_v = self.compute_spectral_distance(v1[i], v2[j])
                distance_matrix[i, j] = dist_q + dist_k + dist_v
        
        row_ind, col_ind = linear_sum_assignment(distance_matrix.detach().cpu().numpy())
        
        inter_head_perm = torch.zeros(num_heads, num_heads)
        inter_head_perm[row_ind, col_ind] = 1.0
        
        intra_head_perms = []
        for i, j in zip(row_ind, col_ind):
            cost_q = -torch.mm(q2[j], q1[i].t())
            cost_k = -torch.mm(k2[j], k1[i].t())
            cost_v = -torch.mm(v2[j], v1[i].t())
            cost_matrix = (cost_q + cost_k + cost_v) / 3.0
            
            row_ind_intra, col_ind_intra = linear_sum_assignment(cost_matrix.detach().cpu().numpy())
            
            perm = torch.zeros(head_dim, head_dim)
            perm[row_ind_intra, col_ind_intra] = 1.0
            intra_head_perms.append(perm)
        
        composed_perm = self.compose_attention_permutation(
            inter_head_perm, intra_head_perms, d_model, num_heads
        )
        
        return inter_head_perm, intra_head_perms, composed_perm
    
    def match_mlp_layer(self, weight1: torch.Tensor, weight2: torch.Tensor,
                       prev_perm: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Match MLP layers using Hungarian algorithm"""
        device = weight1.device
        
        if prev_perm is not None:
            prev_perm = prev_perm.to(device)
            if prev_perm.shape[0] == weight1.shape[1]:
                weight1_permuted = torch.mm(weight1, prev_perm.t())
            else:
                weight1_permuted = weight1
        else:
            weight1_permuted = weight1
        
        cost_matrix = -torch.mm(weight2, weight1_permuted.t())
        row_ind, col_ind = linear_sum_assignment(cost_matrix.detach().cpu().numpy())
        
        n = weight1.shape[0]
        perm = torch.zeros(n, n, device=device)
        perm[row_ind, col_ind] = 1.0
        
        return perm
    
    def apply_permutation_to_weights(self, weights, perm_spec) -> Any:
        """Apply computed permutations to weight space"""
        result = copy.deepcopy(weights)
        
        prev_output_perm = None
        
        for block_idx, (source_block, result_block) in enumerate(zip(weights.blocks, result.blocks)):
            block_perm = perm_spec.block_perms[block_idx]
            d_model = source_block.attention.qkv_weight.shape[1]
            device = result_block.attention.qkv_weight.device
            
            if block_perm['attention_out'] is not None:
                P_attn = block_perm['attention_out'].to(device)
                P_attn_expanded = torch.block_diag(P_attn, P_attn, P_attn).to(device)
                
                if prev_output_perm is not None:
                    result_block.attention.qkv_weight = torch.mm(
                        torch.mm(P_attn_expanded, result_block.attention.qkv_weight),
                        prev_output_perm.t()
                    )
                else:
                    result_block.attention.qkv_weight = torch.mm(
                        P_attn_expanded, result_block.attention.qkv_weight
                    )
                
                if result_block.attention.qkv_bias is not None:
                    result_block.attention.qkv_bias = torch.mv(
                        P_attn_expanded, result_block.attention.qkv_bias
                    )
                
                result_block.attention.proj_weight = torch.mm(
                    result_block.attention.proj_weight, P_attn.t()
                )
            
            if len(result_block.mlp_weights) >= 2 and block_perm['mlp1'] is not None:
                P_mlp1 = block_perm['mlp1'].to(device)
                
                result_block.mlp_weights = list(result_block.mlp_weights)
                if prev_output_perm is not None:
                    result_block.mlp_weights[0] = torch.mm(
                        torch.mm(P_mlp1, result_block.mlp_weights[0]),
                        prev_output_perm.t()
                    )
                else:
                    result_block.mlp_weights[0] = torch.mm(P_mlp1, result_block.mlp_weights[0])
                
                if len(result_block.mlp_biases) > 0:
                    result_block.mlp_biases = list(result_block.mlp_biases)
                    result_block.mlp_biases[0] = torch.mv(P_mlp1, result_block.mlp_biases[0])
                
                result_block.mlp_weights[1] = torch.mm(result_block.mlp_weights[1], P_mlp1.t())
                
                result_block.mlp_weights = tuple(result_block.mlp_weights)
                result_block.mlp_biases = tuple(result_block.mlp_biases)
                
                prev_output_perm = torch.eye(d_model, device=device)
            else:
                prev_output_perm = torch.eye(d_model, device=device)
        
        return result
    
    def canonicalize_model(self, models: List[Any], reference_idx: int = 0) -> List[Any]:
        """Canonicalize multiple models using one as reference"""
        reference = models[reference_idx]
        canonicalized = []
        
        for i, model in enumerate(models):
            if i == reference_idx:
                canonicalized.append(reference)
            else:
                perm_spec = BERTPermutationSpec(len(model.blocks))
                current_model = copy.deepcopy(model)
                
                for iteration in range(self.num_iterations):
                    current_dim_perm = None
                    
                    for block_idx in range(len(current_model.blocks)):
                        current_block = current_model.blocks[block_idx]
                        reference_block = reference.blocks[block_idx]
                        
                        d_model = current_block.attention.qkv_weight.shape[1]
                        
                        inter_perm, intra_perms, composed_perm = self.match_attention_heads(
                            current_block.attention, reference_block.attention
                        )
                        perm_spec.set_block_perm(block_idx, 'attention_out', composed_perm)
                        
                        if len(current_block.mlp_weights) >= 1:
                            mlp1_perm = self.match_mlp_layer(
                                current_block.mlp_weights[0],
                                reference_block.mlp_weights[0],
                                prev_perm=current_dim_perm
                            )
                            perm_spec.set_block_perm(block_idx, 'mlp1', mlp1_perm)
                            current_dim_perm = torch.eye(d_model)
                    
                    current_model = self.apply_permutation_to_weights(current_model, perm_spec)

                
                canonicalized.append(current_model)
        
        return canonicalized


# ==================== UTILITY FUNCTIONS ====================

def evaluate_model(model, loader, device):
    """Evaluate model and return R² score"""
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for input_ids, y in loader:
            input_ids, y = input_ids.to(device), y.to(device)
            out = model(input_ids).squeeze(-1)
            preds.extend(out.cpu().numpy())
            labels.extend(y.cpu().numpy())
    r2 = r2_score(labels, preds)
    return r2


# ==================== MAIN TRANSFUSION FUNCTION ====================

def apply_transfusion(model_dir, num_models, pretrained_model_name, ref_point=0, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ref_model = create_bert_50m(num_classes=1).to(device)
    ref_model_path = f"{model_dir}/{pretrained_model_name}{ref_point}_best.pt"
    
    try:
        ref_model.load_state_dict(torch.load(ref_model_path, map_location=device))
        print(f"Loaded reference model from {ref_model_path}")
    except Exception as e:
        print(f"Failed to load reference model: {e}")
        raise e
    
    ref_ws = BERTWeightSpace.from_bert_model(ref_model)
    weight_space_objects = [ref_ws]
    
    matcher = TransFusionMatcher(num_iterations=2)
    
    for i in tqdm(range(num_models), desc="Processing Bert models"):
        if i == ref_point:
            continue
        
        model_path = f"{model_dir}/{pretrained_model_name}{i}_best.pt"
        if not os.path.exists(model_path):
            print(f"Skipping model {i} - file not found")
            continue
        
        model = create_bert_50m(num_classes=1).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        ws =BERTWeightSpace.from_bert_model(model)
        
        canonicalized_list = matcher.canonicalize_model([ref_ws, ws], reference_idx=0)
        aligned_ws = canonicalized_list[1]
        
        weight_space_objects.append(aligned_ws)
    
    print(f"Successfully processed {len(weight_space_objects)} Bert models")    
    return ref_model, weight_space_objects


def main():
    print("=" * 70)
    print("BERT TRANSFUSION - Weight Matching Demo")
    print("=" * 70)
    
    # Configuration
    model_dir = "yelp_bert_regression_models"
    num_models = 4
    ref_point = 0
    
    # Load test dataset
    print("\nLoading test dataset...")
    test_dataset = YelpReviewDataset('test', subset_size=10000)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    print(f"Test set size: {len(test_dataset)}")
    
    # Load and align models
    print("\n" + "=" * 70)
    print("Loading models using TransFusion...")
    print("=" * 70)
    
    ref_model, aligned_weight_spaces = apply_transfusion(
        model_dir=model_dir,
        num_models=num_models,
        pretrained_model_name="bert_",
        ref_point=ref_point,
        device=device
    )
    
    print(f"\nLoaded {len(aligned_weight_spaces)} models into weight space")
    
    r2_ref = evaluate_model(ref_model, test_loader, device)
    print(f"Reference model (Model {ref_point}) R²: {r2_ref:.4f}")

    for i in range (1,num_models):
        generated_ws = aligned_weight_spaces[i]
        flat = aligned_weight_spaces[i].flatten()
        generated_ws = BERTWeightSpace.from_flat(flat, aligned_weight_spaces[0], device)
        new_model = create_bert_50m().to(device)
        generated_ws.apply_to_model(new_model)
        new_model.eval()
        
        r2_aligned = evaluate_model(new_model, test_loader, device)
    
        print(f"Aligned model R²: {r2_aligned:.4f}")
        
if __name__ == "__main__":
    main()