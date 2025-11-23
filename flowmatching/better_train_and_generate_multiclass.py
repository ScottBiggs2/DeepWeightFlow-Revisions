import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import numpy as np
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from sklearn.decomposition import IncrementalPCA

from utils import WeightSpaceObjectMLP, WeightSpaceObjectResnet, VisionTransformerWeightSpace, count_parameters, recalibrate_bn_stats
from models import MC_MLP_MNIST, MC_MLP_Fashion_MNIST, MLP_Iris, ResNet20, get_resnet18, create_vit_small
from better_multiclass_flow_matching import MultiClassFlowMatching, MultiClassWeightSpaceFlowModel
from canonicalization import get_permuted_models_data

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(config_file='constants.json'):
    """Load config file, checking both current dir and script dir"""
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            return json.load(f)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, config_file)
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    
    raise FileNotFoundError(
        f"Could not find '{config_file}' in:\n"
        f"  - Current directory: {os.getcwd()}\n"
        f"  - Script directory: {script_dir}\n"
        f"Please either:\n"
        f"  1. Run from the directory containing {config_file}\n"
        f"  2. Use --config with full path\n"
        f"  3. Place {config_file} in current directory"
    )


def generate_at_class_label(cfm, class_label, n_samples, actual_dim, source_std, 
                           reference_config, ipca=None):
    """
    Generate models at a specific class label (supports interpolation).
    
    Args:
        cfm: MultiClassFlowMatching instance
        class_label: Target class (int or float)
        n_samples: Number of models to generate
        actual_dim: Dimension of the weight space
        source_std: Standard deviation of source noise
        reference_config: Config dict with integration settings
        ipca: PCA transform to invert (or None)
    
    Returns:
        torch.Tensor: Generated weights [n_samples, actual_dim] (after inverse PCA if applicable)
    """
    random_flat = torch.randn(n_samples, actual_dim, device=device) * source_std
    new_weights_flat = cfm.map(
        random_flat,
        class_label=class_label,
        n_steps=reference_config['integration_steps'],
        method=reference_config['integration_method'],
        guidance_scale=reference_config.get('guidance_scale', 1.0)
    )
    
    # Inverse PCA if used
    if ipca is not None:
        new_weights_flat = ipca.inverse_transform(new_weights_flat.cpu().numpy())
        new_weights_flat = torch.tensor(new_weights_flat, dtype=torch.float32, device=device)
    
    return new_weights_flat


def reconstruct_mlp_models(weights_flat, model_config, dataset):
    """Reconstruct MLP models from flattened weights."""
    generated_models = []
    n_samples = weights_flat.shape[0]
    
    for i in range(n_samples):
        new_wso = WeightSpaceObjectMLP.from_flat(
            weights_flat[i],
            layers=np.array(model_config['layer_layout']),
            device=device
        )
        
        # Create appropriate model type based on dataset
        if 'fashion' in dataset:
            model = MC_MLP_Fashion_MNIST()
        elif 'mnist' in dataset:
            model = MC_MLP_MNIST()
        elif "iris" in dataset: 
            model = MLP_Iris()
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        
        # Load weights into model
        for idx in range(len(new_wso.weights)):
            getattr(model, f'fc{idx+1}').weight.data = new_wso.weights[idx].clone()
            getattr(model, f'fc{idx+1}').bias.data = new_wso.biases[idx].clone()
        
        generated_models.append(model.to(device))
    
    return generated_models


def reconstruct_resnet_models(weights_flat, model_config, dataset):
    """Reconstruct ResNet models from flattened weights."""
    generated_models = []
    n_samples = weights_flat.shape[0]
    
    # Create template model to get shapes
    if "resnet20" in model_config['architecture']:
        template_model = ResNet20()
    elif "resnet18" in model_config['architecture']:
        template_model = get_resnet18()
    else:
        raise ValueError(f"Unknown ResNet architecture: {model_config['architecture']}")
    
    # Extract weight and bias shapes
    weight_shapes, bias_shapes = [], []
    for name, param in template_model.named_parameters():
        if "weight" in name:
            weight_shapes.append(tuple(param.shape))
        elif "bias" in name:
            bias_shapes.append(tuple(param.shape))
    
    # Reconstruct each model
    for i in range(n_samples):
        new_wso = WeightSpaceObjectResnet.from_flat(
            weights_flat[i],
            weight_shapes,
            bias_shapes,
            device=device
        )
        
        # Create fresh model
        if "resnet20" in model_config['architecture']:
            model = ResNet20()
        elif "resnet18" in model_config['architecture']:
            model = get_resnet18()
        
        # Load weights from weight space object
        param_dict = {}
        weight_idx, bias_idx = 0, 0
        for name, param in model.named_parameters():
            if "weight" in name:
                param_dict[name] = new_wso.weights[weight_idx]
                weight_idx += 1
            elif "bias" in name:
                param_dict[name] = new_wso.biases[bias_idx]
                bias_idx += 1
        
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in param_dict:
                    param.copy_(param_dict[name])
        
        # Recalibrate BatchNorm if configured
        if model_config.get('recalibrate_bn', False):
            model = recalibrate_bn_stats(model, device)
        
        generated_models.append(model.to(device))
    
    return generated_models


def reconstruct_vit_models(weights_flat, model_config, reference_ws, ipca=None):
    """Reconstruct ViT models from flattened weights."""
    generated_models = []
    n_samples = weights_flat.shape[0]
    
    for i in range(n_samples):
        generated_ws = VisionTransformerWeightSpace.from_flat(
            weights_flat[i], reference_ws, device
        )
        
        new_model = create_vit_small().to(device)
        generated_ws.apply_to_model(new_model)
        generated_models.append(new_model)
    
    return generated_models


def reconstruct_models(weights_flat, model_config, dataset, reference_ws=None, ipca=None):
    """
    Route to appropriate reconstruction function based on architecture.
    
    Args:
        weights_flat: Flattened weight tensors [n_models, flat_dim]
        model_config: Config dict
        dataset: Dataset name
        reference_ws: Reference weight space object (for ViT)
        ipca: PCA object (for ViT)
    
    Returns:
        List of reconstructed models
    """
    architecture = model_config.get('architecture', 'mlp').lower()
    
    if 'vit' in architecture:
        return reconstruct_vit_models(weights_flat, model_config, reference_ws, ipca)
    elif 'resnet' in architecture:
        return reconstruct_resnet_models(weights_flat, model_config, dataset)
    elif 'mlp' in architecture or architecture == 'mlp':
        return reconstruct_mlp_models(weights_flat, model_config, dataset)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def convert_models_to_weight_space(models_to_use, model_config):
    """Convert list of models to weight space objects."""
    weight_space_objects = []
    architecture = model_config.get('architecture', 'mlp').lower()
    
    if 'vit' in architecture:
        for model in models_to_use:
            if hasattr(model, 'flatten'):
                weight_space_objects.append(model)
            else:
                ws_obj = VisionTransformerWeightSpace.from_vit_model(model.to(device))
                weight_space_objects.append(ws_obj)
    
    elif 'resnet' in architecture:
        for model in tqdm(models_to_use, desc="Converting ResNet to weight space"):
            weights, biases, weight_shapes, bias_shapes = [], [], [], []
            for name, param in model.named_parameters():
                param = param.detach().to(device)
                if torch.isnan(param).any() or torch.isinf(param).any():
                    logging.warning(f"NaN/Inf detected in {name}, replacing with zeros")
                    param = torch.zeros_like(param)

                if "weight" in name:
                    weights.append(param.clone())
                    weight_shapes.append(param.shape)
                elif "bias" in name:
                    biases.append(param.clone())
                    bias_shapes.append(param.shape)
            
            wso = WeightSpaceObjectResnet(weights, biases)
            wso.weight_shapes, wso.bias_shapes = weight_shapes, bias_shapes
            weight_space_objects.append(wso)
    
    else:  # MLP
        for model in tqdm(models_to_use, desc="Converting MLP to weight space"):
            weights, biases = [], []
            for name, param in model.named_parameters():
                param = param.detach().to(device)
                if torch.isnan(param).any() or torch.isinf(param).any():
                    logging.warning(f"NaN/Inf detected in {name}, replacing with zeros")
                    param = torch.zeros_like(param)
                if "weight" in name:
                    weights.append(param.clone())
                elif "bias" in name:
                    biases.append(param.clone())
            wso = WeightSpaceObjectMLP(weights, biases)
            weight_space_objects.append(wso)
    
    return weight_space_objects


def prepare_class_data(models, model_config, class_label):
    """
    Prepare weight space data for a single class.
    
    Returns:
        tuple: (target_tensor, labels, ipca, flat_dim, actual_dim, reference_ws)
    """
    # Convert to weight space
    weight_space_objects = convert_models_to_weight_space(models, model_config)
    flat_target_weights = torch.stack([wso.flatten(device) for wso in weight_space_objects]).to(device)
    
    flat_dim = flat_target_weights.shape[1]
    n_samples = flat_target_weights.shape[0]
    print(f"Class {class_label} weight space dimension: {flat_dim:,}")
    print(f"Class {class_label} number of samples: {n_samples}")
    
    # Apply PCA if configured
    ipca = None
    if model_config.get('use_pca', False) and model_config.get('pca_components'):
        n_components = model_config['pca_components']
        
        # Ensure we don't request more components than possible
        max_possible_components = min(n_samples - 1, flat_dim)
        if n_components > max_possible_components:
            raise ValueError(
                f"Requested {n_components} PCA components but only {max_possible_components} possible "
                f"(n_samples={n_samples}, n_features={flat_dim}). "
                f"Either reduce pca_components in config or use more models (--num_models)."
            )
        
        print(f"Applying PCA with {n_components} components")
        
        # Use IncrementalPCA with small batch_size for memory efficiency
        # The batch_size here is just for PCA computation, not flow matching
        ipca = IncrementalPCA(n_components=n_components, batch_size=min(10, n_samples))
        flat_latent = ipca.fit_transform(flat_target_weights.cpu().numpy())
        
        # Verify we got what we asked for
        if flat_latent.shape[1] != n_components:
            raise ValueError(
                f"PCA returned {flat_latent.shape[1]} components instead of requested {n_components}. "
                f"This should not happen. n_samples={n_samples}, flat_dim={flat_dim}"
            )
        
        target_tensor = torch.tensor(flat_latent, dtype=torch.float32)
        actual_dim = n_components
        print(f"  Variance explained: {ipca.explained_variance_ratio_.sum():.4f}")
        print(f"  Output shape: {flat_latent.shape}")
    else:
        target_tensor = flat_target_weights
        actual_dim = flat_dim
    
    print(f"  Final actual_dim = {actual_dim}")
    
    # Create labels
    labels = torch.full([target_tensor.shape[0], 1], class_label, dtype=torch.float32)
    
    # Store reference weight space object for ViT
    reference_ws = weight_space_objects[0] if 'vit' in model_config.get('architecture', '').lower() else None
    
    return target_tensor, labels, ipca, flat_dim, actual_dim, reference_ws


def train_and_generate(args):
    """Main training and generation function for multiclass models."""
    
    config = load_config(args.config)
    
    # Define classes - customize this list for your use case
    model_classes = [
        ('mc_mlp_mnist_compressed', 0),
        ('mc_mlp_fashion_mnist_compressed', 1),
        ('resnet20_cifar10_compressed', 2),
    ]
    
    print(f"Training multiclass flow matching for {len(model_classes)} classes")
    print("Using CLIP-style conditioning: N(0,1) + class_embed -> target weights")
    
    for training_mode in config['training_modes']:
        if args.mode and training_mode != args.mode:
            continue
        
        print(f"\n{'='*80}")
        print(f"TRAINING MODE: {training_mode}")
        print(f"{'='*80}")
        
        mode_class_data = []
        
        for model_name, class_label in model_classes:
            print(f"\n{'-'*60}")
            print(f"Preparing class {class_label}: {model_name}")
            print(f"{'-'*60}")
            
            model_config = config['models'][model_name]
            model_dir_raw = config['directories'][model_name]
            
            # Resolve path
            if not os.path.isabs(model_dir_raw):
                model_dir = model_dir_raw
                if not os.path.exists(model_dir):
                    config_dir = os.path.dirname(os.path.abspath(args.config))
                    model_dir = os.path.join(config_dir, model_dir_raw)
                    if not os.path.exists(model_dir):
                        parent_dir = os.path.dirname(config_dir)
                        model_dir = os.path.normpath(os.path.join(parent_dir, model_dir_raw))
            else:
                model_dir = model_dir_raw
            
            if not os.path.exists(model_dir):
                raise FileNotFoundError(f"Could not find model directory: {model_dir_raw}")
            
            print(f"Using model directory: {model_dir}")
            pretrained_model_name = model_config.get('pretrained_model_name', 'mlp_seed')
            
            # Get permuted/original models
            ref_model, org_models, permuted_models = get_permuted_models_data(
                model_name=model_name,
                model_dir=model_dir,
                pretrained_model_name=pretrained_model_name,
                num_models=args.num_models,
                ref_point=args.ref_point,
                device=device,
                model_config=model_config
            )
            
            # Choose which set of models to use
            models_to_use = permuted_models if training_mode == "with_gitrebasin" else org_models
            
            # Prepare weight space data
            target_tensor, labels, ipca, flat_dim, actual_dim, reference_ws = prepare_class_data(
                models_to_use, model_config, class_label
            )
            
            mode_class_data.append({
                'model_name': model_name,
                'class_label': class_label,
                'target_tensor': target_tensor,
                'labels': labels,
                'ipca': ipca,
                'flat_dim': flat_dim,
                'actual_dim': actual_dim,
                'model_config': model_config,
                'training_mode': training_mode,
                'dataset': model_config['dataset'],
                'model_dir': model_dir,
                'pretrained_model_name': pretrained_model_name,
                'org_models': org_models,
                'permuted_models': permuted_models,
                'reference_ws': reference_ws  # For ViT
            })
        
        # Verify all classes have same dimensionality
        dims = [d['actual_dim'] for d in mode_class_data]
        flat_dims = [d['flat_dim'] for d in mode_class_data]
        
        if not all(d == dims[0] for d in dims):
            print("\n" + "!"*80)
            print("ERROR: Dimension mismatch across classes!")
            print("\nDimensions found:")
            for class_data in mode_class_data:
                print(f"  Class {class_data['class_label']} ({class_data['model_name']}): "
                      f"flat={class_data['flat_dim']:,}, actual={class_data['actual_dim']:,}")
            print("!"*80)
            raise ValueError(f"Dimension mismatch: {dims}")
        
        actual_dim = dims[0]
        print(f"\n✓ All {len(mode_class_data)} classes aligned at dimension: {actual_dim:,}")
        if mode_class_data[0]['ipca'] is not None:
            print(f"  (Original dimension: {flat_dims[0]:,}, reduced via PCA)")
        
        # Use first class config as reference
        reference_config = mode_class_data[0]['model_config']
        
        # Train for each hidden dimension configuration
        for hidden_dim in reference_config['flow_hidden_dims']:
            if args.hidden_dim and hidden_dim != args.hidden_dim:
                continue
            
            print(f"\n{'='*80}")
            print(f"Training Flow Model: hidden_dim={hidden_dim}, mode={training_mode}")
            print(f"{'='*80}")
            
            # Create combined dataset
            target_datasets = [
                TensorDataset(d['target_tensor'], d['labels']) 
                for d in mode_class_data
            ]
            combined_target_dataset = ConcatDataset(target_datasets)
            
            # Create source (noise) dataset
            source_std = reference_config['source_std']
            total_samples = sum(d['target_tensor'].shape[0] for d in mode_class_data)
            source_tensor = torch.randn(total_samples, actual_dim) * source_std
            source_labels = torch.zeros(total_samples, 1)
            source_dataset = TensorDataset(source_tensor, source_labels)
            
            print(f"Source distribution: N(0, {source_std}²)")
            print(f"Total training samples: {total_samples:,}")
            
            # Create dataloaders
            def collate_fn(batch):
                flats, labs = zip(*batch)
                return torch.stack(flats), torch.stack(labs)
            
            sourceloader = DataLoader(
                source_dataset, 
                batch_size=reference_config['batch_size'], 
                shuffle=True, 
                drop_last=True
            )
            
            targetloader = DataLoader(
                combined_target_dataset,
                batch_size=reference_config['batch_size'],
                shuffle=True,
                drop_last=True,
                collate_fn=collate_fn
            )
            
            # Create flow model
            flow_model = MultiClassWeightSpaceFlowModel(
                actual_dim,
                hidden_dim,
                time_embed_dim=reference_config['time_embed_dim'],
                class_embed_dim=reference_config.get('class_embed_dim', 128),
                dropout=reference_config['dropout']
            ).to(device)
            
            print(f"Flow model parameters: {count_parameters(flow_model):,}")
            
            # Create flow matcher with CFG
            cfm = MultiClassFlowMatching(
                sourceloader=sourceloader,
                targetloader=targetloader,
                model=flow_model,
                mode="velocity",
                t_dist=config['t_dist'],
                device=device,
                cfg_dropout_prob=reference_config.get('cfg_dropout_prob', 0.1),
                label_noise_std=reference_config.get('label_noise_std', 0.05),
            )
            cfm.input_dim = actual_dim
            
            # Setup optimizer and scheduler
            optimizer = torch.optim.AdamW(
                flow_model.parameters(),
                lr=reference_config['lr'],
                weight_decay=reference_config['weight_decay'],
                betas=(0.9, 0.95)
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=reference_config['n_iters'],
                eta_min=1e-6
            )
            
            # Train
            train_kwargs = {
                'n_iters': reference_config['n_iters'],
                'optimizer': optimizer,
                'scheduler': scheduler,
                'sigma': reference_config['sigma'],
                'patience': reference_config['patience'],
                'log_freq': 10
            }
            grad_accum_steps = reference_config.get('gradient_accumulation_steps')
            if grad_accum_steps is not None:
                train_kwargs['accum_steps'] = grad_accum_steps
            
            print("\nStarting training...")
            cfm.train(**train_kwargs)
            print(f"✓ Training complete! Best loss: {cfm.best_loss:.6f}")
            
            # Save checkpoint if requested
            if args.save_models:
                checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(
                    checkpoint_dir,
                    f'flow_model_{training_mode}_hidden{hidden_dim}.pt'
                )
                torch.save({
                    'model_state_dict': cfm.best_model_state,
                    'flow_model_config': {
                        'input_dim': actual_dim,
                        'hidden_dim': hidden_dim,
                        'time_embed_dim': reference_config['time_embed_dim'],
                        'class_embed_dim': reference_config.get('class_embed_dim', 128),
                        'dropout': reference_config['dropout']
                    },
                    'training_config': reference_config,
                    'best_loss': cfm.best_loss,
                    'num_classes': len(mode_class_data),
                    'class_info': [(d['model_name'], d['class_label']) for d in mode_class_data]
                }, checkpoint_path)
                print(f"✓ Saved checkpoint to {checkpoint_path}")
            
            # Generate and evaluate models for each class
            n_samples = reference_config['n_samples']
            evaluation_results = []
            
            for class_data in mode_class_data:
                class_label = class_data['class_label']
                model_name = class_data['model_name']
                dataset = class_data['dataset']
                ipca = class_data['ipca']
                model_config = class_data['model_config']
                reference_ws = class_data['reference_ws']
                
                print(f"\n{'='*80}")
                print(f"Generating {n_samples} models for class {class_label} ({model_name})")
                print(f"{'='*80}")
                
                # Generate weights
                new_weights_flat = generate_at_class_label(
                    cfm, class_label, n_samples, actual_dim, source_std,
                    reference_config, ipca=ipca
                )
                
                # Reconstruct models
                generated_models = reconstruct_models(
                    new_weights_flat, model_config, dataset, 
                    reference_ws=reference_ws, ipca=ipca
                )
                
                # Evaluate
                test_loader = get_data_loader(dataset, batch_size=32, train=False)
                original_models = class_data['permuted_models'] if training_mode == "with_gitrebasin" else class_data['org_models']
                original_models_subset = original_models[:n_samples]
                
                eval_stats = evaluate_and_compare_simple(
                    generated_models,
                    original_models_subset,
                    test_loader,
                    device,
                    class_label,
                    model_name
                )
                eval_stats['class_label'] = class_label
                eval_stats['model_name'] = model_name
                evaluation_results.append(eval_stats)
                
                # Save if requested
                if args.save_models:
                    save_dir = os.path.join(
                        args.output_dir,
                        f'{training_mode}_hidden{hidden_dim}',
                        f'class_{class_label}_{model_name}'
                    )
                    os.makedirs(save_dir, exist_ok=True)
                    
                    for idx, model in enumerate(generated_models):
                        model_path = os.path.join(save_dir, f'generated_model_{idx}.pt')
                        torch.save(model.state_dict(), model_path)
                    
                    stats_path = os.path.join(save_dir, 'generation_stats.json')
                    with open(stats_path, 'w') as f:
                        json.dump({
                            'class_label': int(class_label),
                            'model_name': model_name,
                            'n_samples': n_samples,
                            'original_mean_accuracy': float(eval_stats['original_mean']),
                            'original_std_accuracy': float(eval_stats['original_std']),
                            'generated_mean_accuracy': float(eval_stats['generated_mean']),
                            'generated_std_accuracy': float(eval_stats['generated_std']),
                            'accuracy_difference': float(eval_stats['difference']),
                            'relative_diff_pct': float(eval_stats['relative_diff_pct']),
                            'training_mode': training_mode,
                            'hidden_dim': hidden_dim,
                            'integration_steps': reference_config['integration_steps'],
                            'integration_method': reference_config['integration_method'],
                            'used_pca': ipca is not None,
                            'pca_components': actual_dim if ipca is not None else None
                        }, f, indent=2)
                    
                    print(f"✓ Saved {n_samples} models to {save_dir}")
                
                del generated_models
            
            # Print summary
            print(f"\n{'='*80}")
            print(f"FINAL SUMMARY - {training_mode}, hidden_dim={hidden_dim}")
            print(f"{'='*80}")
            for result in evaluation_results:
                print(f"\nClass {result['class_label']} ({result['model_name']}):")
                print(f"  Original:  {result['original_mean']:.4f} ± {result['original_std']:.4f}")
                print(f"  Generated: {result['generated_mean']:.4f} ± {result['generated_std']:.4f}")
                print(f"  Δ Accuracy: {result['difference']:.4f} ({result['relative_diff_pct']:.2f}%)")
            print(f"{'='*80}")
            
            del flow_model, cfm
            torch.cuda.empty_cache()


def evaluate_and_compare_simple(generated_models, original_models, test_loader, device, class_label, model_name):
    """Evaluate and compare generated vs original models."""
    from utils import print_stats
    
    print(f"\n{'='*80}")
    print(f"EVALUATION - Class {class_label} ({model_name})")
    print(f"{'='*80}")
    
    print(f"\n[Original Models - {len(original_models)} models]")
    orig_mean, orig_std = print_stats(original_models, test_loader, device)
    
    print(f"\n[Generated Models - {len(generated_models)} models]")
    gen_mean, gen_std = print_stats(generated_models, test_loader, device)
    
    print(f"\n{'─'*80}")
    print("COMPARISON:")
    print(f"{'─'*80}")
    print(f"Original: {orig_mean:.4f} ± {orig_std:.4f}")
    print(f"Generated: {gen_mean:.4f} ± {gen_std:.4f}")
    print(f"Difference: {abs(gen_mean - orig_mean):.4f} ({'+' if gen_mean > orig_mean else '-'}{abs((gen_mean - orig_mean)/orig_mean * 100):.2f}%)")
    print(f"{'─'*80}")
    
    return {
        'original_mean': orig_mean,
        'original_std': orig_std,
        'generated_mean': gen_mean,
        'generated_std': gen_std,
        'difference': abs(gen_mean - orig_mean),
        'relative_diff_pct': abs((gen_mean - orig_mean)/orig_mean * 100)
    }


def get_data_loader(dataset_name, batch_size=32, train=False):
    """Helper to get data loader."""
    from utils import load_mnist, load_fashion_mnist, load_iris_dataset, load_cifar10
    
    dataset_name = dataset_name.lower()
    if dataset_name == "mnist":
        return load_mnist(batch_size=batch_size)
    elif dataset_name == "fashion_mnist":
        return load_fashion_mnist(batch_size=batch_size)
    elif dataset_name == "iris":
        return load_iris_dataset(batch_size=batch_size)
    elif dataset_name == "cifar10":
        return load_cifar10(batch_size=batch_size)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def main():
    parser = argparse.ArgumentParser(
        description='Train multiclass flow matching for neural network weight generation'
    )
    parser.add_argument('--config', type=str, default='flowmatching/constants.json',
                       help='Configuration file path')
    parser.add_argument('--num_models', type=int, default=100,
                       help='Number of pretrained models to use PER CLASS')
    parser.add_argument('--ref_point', type=int, default=0,
                       help='Reference model index for canonicalization')
    parser.add_argument('--hidden_dim', type=int, default=None,
                       help='Specific hidden dimension to test')
    parser.add_argument('--mode', type=str, default=None,
                       choices=['with_gitrebasin', 'without_rebasin'],
                       help='Training mode')
    parser.add_argument('--save_models', action='store_true',
                       help='Save generated models and checkpoints')
    parser.add_argument('--output_dir', type=str, default='./generated_models',
                       help='Directory to save outputs')
    
    args = parser.parse_args()
    
    print("="*80)
    print("MULTICLASS FLOW MATCHING - Neural Network Weight Generation")
    print("="*80)
    print(f"Config: {args.config}")
    print(f"Models per class: {args.num_models}")
    print(f"Save models: {args.save_models}")
    print("="*80)
    
    train_and_generate(args)
    
    print("\n" + "="*80)
    print("✓ Training and generation complete!")
    print("="*80)


if __name__ == "__main__":
    main()