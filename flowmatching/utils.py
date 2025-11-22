import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np
from typing import Optional, List
from torchvision import datasets, transforms
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from models import AttentionWeights, TransformerBlockWeights


class VisionTransformerWeightSpace:
    """Weight space object for Vision Transformers"""
    
    def __init__(self, 
                 patch_embed_weight: torch.Tensor,
                 patch_embed_bias: Optional[torch.Tensor],
                 cls_token: torch.Tensor,
                 pos_embed: torch.Tensor,
                 blocks: List[TransformerBlockWeights],
                 norm_weight: torch.Tensor,
                 norm_bias: torch.Tensor,
                 head_weight: torch.Tensor,
                 head_bias: torch.Tensor):
        
        self.patch_embed_weight = patch_embed_weight
        self.patch_embed_bias = patch_embed_bias
        self.cls_token = cls_token
        self.pos_embed = pos_embed
        self.blocks = blocks
        self.norm_weight = norm_weight
        self.norm_bias = norm_bias
        self.head_weight = head_weight
        self.head_bias = head_bias
        
    @classmethod
    def from_vit_model(cls, model: nn.Module):
        """Extract weights from a ViT model"""
        blocks = []
        
        for block in model.blocks:
            attn = block.attn
            attention_weights = AttentionWeights(
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
            
            block_weights = TransformerBlockWeights(
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
            patch_embed_weight=model.patch_embed.proj.weight.data.clone(),
            patch_embed_bias=model.patch_embed.proj.bias.data.clone() 
                            if model.patch_embed.proj.bias is not None else None,
            cls_token=model.cls_token.data.clone(),
            pos_embed=model.pos_embed.data.clone(),
            blocks=blocks,
            norm_weight=model.norm.weight.data.clone(),
            norm_bias=model.norm.bias.data.clone(),
            head_weight=model.head.weight.data.clone(),
            head_bias=model.head.bias.data.clone()
        )
    
    def apply_to_model(self, model: nn.Module):
        """Apply weights to a ViT model"""
        with torch.no_grad():
            model.patch_embed.proj.weight.data.copy_(self.patch_embed_weight)
            if self.patch_embed_bias is not None:
                model.patch_embed.proj.bias.data.copy_(self.patch_embed_bias)
            
            model.cls_token.data.copy_(self.cls_token)
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
        
        all_params.append(self.patch_embed_weight.flatten())
        if self.patch_embed_bias is not None:
            all_params.append(self.patch_embed_bias.flatten())
        
        all_params.append(self.cls_token.flatten())
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
        """Reconstruct VisionTransformerWeightSpace from flattened weights"""
        if device is None:
            device = flat_tensor.device
            
        param_shapes = []
        param_types = []
        
        param_shapes.append(reference_ws.patch_embed_weight.shape)
        param_types.append('patch_embed_weight')
        
        if reference_ws.patch_embed_bias is not None:
            param_shapes.append(reference_ws.patch_embed_bias.shape)
            param_types.append('patch_embed_bias')
        
        param_shapes.append(reference_ws.cls_token.shape)
        param_types.append('cls_token')
        
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
            
            attention = AttentionWeights(
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
            
            block = TransformerBlockWeights(
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
            patch_embed_weight=reconstructed_params['patch_embed_weight'],
            patch_embed_bias=reconstructed_params.get('patch_embed_bias', None),
            cls_token=reconstructed_params['cls_token'],
            pos_embed=reconstructed_params['pos_embed'],
            blocks=reconstructed_blocks,
            norm_weight=reconstructed_params['norm_weight'], 
            norm_bias=reconstructed_params['norm_bias'],
            head_weight=reconstructed_params['head_weight'],
            head_bias=reconstructed_params['head_bias']
        )

class Bunch:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def safe_deflatten(flat, batch_size, starts, ends):
    """Safely deflatten a tensor without index errors"""
    parts = []
    actual_batch_size = flat.size(0)
    
    safe_batch_size = min(actual_batch_size, batch_size)
    
    for i in range(safe_batch_size):
        batch_parts = []
        for si, ei in zip(starts, ends):
            if si < ei: 
                batch_parts.append(flat[i][si:ei])
        parts.append(batch_parts)
    
    return parts

def count_parameters(model):
    """Count trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_cifar10(batch_size=128):

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return test_loader

def load_mnist(batch_size=32):
    transform = transforms.Compose([transforms.ToTensor()])
    test_data = datasets.MNIST(root="data", train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return test_loader

def load_fashion_mnist(batch_size=32):
    transform = transforms.Compose([transforms.ToTensor()])
    test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return test_loader

def get_california_loaders(batch_size=64, test_size=0.2, random_state=None):
    # Load dataset
    california = fetch_california_housing()
    X, y = california.data, california.target

    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.FloatTensor(y_train).view(-1, 1)
    y_test = torch.FloatTensor(y_test).view(-1, 1)

    # Create DataLoader
    # train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)
    # train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return test_loader

def load_iris_dataset(batch_size=32):
    iris = load_iris()
    X, y = iris.data, iris.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)
    
    test_ds = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    return test_loader

class WeightSpaceObjectMLP:
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

class WeightSpaceObjectResnet:
    def __init__(self, weights, biases):
        self.weights = tuple(weights)
        self.biases = tuple(biases)
        
    def flatten(self, device=None):
        flat = torch.cat([w.flatten() for w in self.weights] + 
                        [b.flatten() for b in self.biases])
        return flat.to(device) if device else flat
    
    @classmethod
    def from_flat(cls, flat, weight_shapes, bias_shapes, device=None):
        sizes = ([np.prod(s) for s in weight_shapes + bias_shapes])
        parts = []
        start = 0
        for size in sizes:
            parts.append(flat[start:start+size])
            start += size
            
        n_weights = len(weight_shapes)
        n_biases = len(bias_shapes)
        
        weights = [parts[i].reshape(weight_shapes[i]) for i in range(n_weights)]
        biases = [parts[n_weights + i].reshape(bias_shapes[i]) for i in range(n_biases)]        
        return cls(weights, biases)

def evaluate_model(model, test_loader, device):
    """Evaluate model accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    return 100 * correct / total

def test_ensemble(models, test_loader, device="cuda"):
    """Test ensemble of models"""
    for m in models:
        m.eval()
        m.to(device)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            logits_sum = None
            for model in models:
                output = model(data)
                if logits_sum is None:
                    logits_sum = output
                else:
                    logits_sum += output
            
            avg_logits = logits_sum / len(models)
            _, predicted = torch.max(avg_logits, 1)
            
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    return 100 * correct / total

def print_stats(models, test_loader, device):
    """Print statistics for generated models"""
    accuracies = []
    for model in models:
        acc = evaluate_model(model, test_loader, device)
        accuracies.append(acc)
    
    accuracies = np.array(accuracies)
    mean = accuracies.mean()
    std = accuracies.std()
    min_acc = accuracies.min()
    max_acc = accuracies.max()
    
    print("\n=== Summary ===")
    print(f"Average Accuracy: {mean:.2f}% ± {std:.2f}%")
    print(f"Min Accuracy: {min_acc:.2f}%")
    print(f"Max Accuracy: {max_acc:.2f}%")
    
    return mean, std

def print_regression_stats(models, test_loader, device):
    criterion = nn.MSELoss()  # Regression: MSE loss
    for model in models: 
        # Evaluate
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                out = model(xb)
                preds.append(out)
                targets.append(yb)
        preds = torch.cat(preds, dim=0)
        targets = torch.cat(targets, dim=0)

        # Calculate metrics
        mse = nn.MSELoss()(preds, targets).item()
        mae = nn.L1Loss()(preds, targets).item()
        print(f"Model {model}: Test MSE={mse:.4f}, MAE={mae:.4f}")
        return mse, mae

def recalibrate_bn_stats(model, device='cuda', print_stats=False):
    """Recalculate BatchNorm statistics for generated weights"""
    model.train()
    model.to(device)
    test_loader = load_cifar10(batch_size=128)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None
    
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            _ = model(inputs)
    
    if print_stats:
        for name, m in model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                print(f"{name}: mean={m.running_mean.mean().item():.4f}, "
                      f"var={m.running_var.mean().item():.4f}, "
                      f"num_batches_tracked={m.num_batches_tracked.item()}")
    

    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = 0.1
    
    return model

def get_fewshot_loaders(dataset_name='STL10', batch_size=32, num_samples_per_class=5, num_classes=10, few_shot=False):
    """
    Load datasets (STL-10 or SVHN). If few_shot=True, returns few-shot training set.
    """
    if dataset_name.upper() == 'STL10':
        transform_train = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4467, 0.4398, 0.4066),
                                 (0.2241, 0.2215, 0.2239))
        ])
        transform_test = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4467, 0.4398, 0.4066),
                                 (0.2241, 0.2215, 0.2239))
        ])
        train_dataset = datasets.STL10(root='./data', split='train', download=True, transform=transform_train)
        test_dataset = datasets.STL10(root='./data', split='test', download=True, transform=transform_test)

    elif dataset_name.upper() == 'SVHN':
        transform_train = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728),
                                 (0.1980, 0.2010, 0.1970))
        ])
        transform_test = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728),
                                 (0.1980, 0.2010, 0.1970))
        ])
        train_dataset = datasets.SVHN(root='./data', split='train', download=True, transform=transform_train)
        test_dataset = datasets.SVHN(root='./data', split='test', download=True, transform=transform_test)

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if few_shot:
        # Build few-shot subset
        class_indices = defaultdict(list)
        for idx in range(len(train_dataset)):
            _, label = train_dataset[idx]
            label = label % 10 if dataset_name.upper() == 'SVHN' else label  # fix SVHN 10->0
            class_indices[label].append(idx)

        few_shot_indices = []
        for class_id in range(num_classes):
            if class_id in class_indices:
                few_shot_indices.extend(class_indices[class_id][:num_samples_per_class])

        train_dataset = Subset(train_dataset, few_shot_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"{dataset_name} loaded: {len(train_loader.dataset)} training samples, {len(test_loader.dataset)} test samples")
    return train_loader, test_loader

def get_cifar10_loaders(batch_size=128):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    
    train_ds = datasets.CIFAR10(root="data", train=True, download=True, transform=transform_train)
    test_ds = datasets.CIFAR10(root="data", train=False, download=True, transform=transform_test)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, test_loader