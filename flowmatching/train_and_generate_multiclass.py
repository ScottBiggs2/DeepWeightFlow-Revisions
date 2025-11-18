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

from utils import WeightSpaceObjectMLP, count_parameters
from models import MLP_MNIST, MLP_Fashion_MNIST
from multiclass_flow_matching import MultiClassFlowMatching, MultiClassWeightSpaceFlowModel
from canonicalization import get_permuted_models_data

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(config_file='constants.json'):
    with open(config_file, 'r') as f:
        return json.load(f)


def generate_at_class_label(cfm, class_label, n_samples, actual_dim, source_std, 
                           reference_config, ipca=None):
    """
    Generate models at a specific class label (supports interpolation).
    
    This function works for any class label value:
    - Integer labels (0, 1, 2, ...): Generate from that specific class
    - Float labels (0.5, 1.3, ...): Interpolate between classes
    
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
        method=reference_config['integration_method']
    )
    
    # Inverse PCA if used
    if ipca is not None:
        new_weights_flat = ipca.inverse_transform(new_weights_flat.cpu().numpy())
        new_weights_flat = torch.tensor(new_weights_flat, dtype=torch.float32, device=device)
    
    return new_weights_flat


def reconstruct_mlp_models(weights_flat, model_config, dataset):
    """
    Reconstruct MLP models from flattened weights.
    
    Args:
        weights_flat: Flattened weight tensors [n_models, flat_dim]
        model_config: Config dict with layer_layout
        dataset: Dataset name to determine model type
    
    Returns:
        List of reconstructed models
    """
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
            model = MLP_Fashion_MNIST()
        elif 'mnist' in dataset:
            model = MLP_MNIST()
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        
        # Load weights into model
        for idx in range(len(new_wso.weights)):
            getattr(model, f'fc{idx+1}').weight.data = new_wso.weights[idx].clone()
            getattr(model, f'fc{idx+1}').bias.data = new_wso.biases[idx].clone()
        
        generated_models.append(model.to(device))
    
    return generated_models


def convert_models_to_weight_space(models_to_use, model_config):
    """Convert list of models to weight space objects"""
    weight_space_objects = []
    
    for model in tqdm(models_to_use, desc="Converting to weight space"):
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
    
    Supports PCA dimensionality reduction if configured. When PCA is enabled,
    each class gets its own PCA transform, but all classes must project to
    the same number of components for the multiclass flow to work.
    
    Args:
        models: List of trained models for this class
        model_config: Configuration dict for this model class
        class_label: Integer or float label for this class
    
    Returns:
        tuple: (target_tensor, labels, ipca, flat_dim, actual_dim)
            - target_tensor: Weight vectors [n_models, actual_dim]
            - labels: Class labels [n_models, 1]
            - ipca: IncrementalPCA object (or None if PCA not used)
            - flat_dim: Original flattened weight dimension
            - actual_dim: Dimension after PCA (or same as flat_dim if no PCA)
    """
    # Convert to weight space
    weight_space_objects = convert_models_to_weight_space(models, model_config)
    flat_target_weights = torch.stack([wso.flatten(device) for wso in weight_space_objects]).to(device)
    
    flat_dim = flat_target_weights.shape[1]
    print(f"Class {class_label} weight space dimension: {flat_dim:,}")
    
    # Apply PCA if configured
    # NOTE: Each class can have its own PCA transform, but all must project
    # to the same number of components (pca_components) for multiclass flow
    ipca = None
    if model_config.get('use_pca', False) and model_config.get('pca_components'):
        print(f"Applying PCA with {model_config['pca_components']} components")
        ipca = IncrementalPCA(n_components=model_config['pca_components'], batch_size=10)
        flat_latent = ipca.fit_transform(flat_target_weights.cpu().numpy())
        target_tensor = torch.tensor(flat_latent, dtype=torch.float32)
        actual_dim = model_config['pca_components']
        print(f"  Variance explained: {ipca.explained_variance_ratio_.sum():.4f}")
    else:
        target_tensor = flat_target_weights
        actual_dim = flat_dim
    
    # Create labels
    labels = torch.full([target_tensor.shape[0], 1], class_label, dtype=torch.float32)
    
    return target_tensor, labels, ipca, flat_dim, actual_dim


def train_and_generate(args):
    """
    Main training and generation function for multiclass models.
    
    This function is designed to scale to N classes (N >= 2):
    - Each class is trained on permuted/original models from its dataset
    - All classes must have identical weight space dimensions (enforced via checks)
    - PCA is supported per-class (each class can have its own PCA, but must
      project to the same number of components)
    - Class interpolation works for any N >= 2 (e.g., class 0.5 interpolates
      between class 0 and 1, class 1.5 between class 1 and 2)
    
    To add more classes:
    1. Add configs to constants.json with matching layer_layout
    2. Add tuple (model_name, class_label) to model_classes list below
    3. Ensure all have same architecture or same PCA components
    """
    
    config = load_config(args.config)
    
    # ========================================================================
    # DEFINE CLASSES HERE - Add more tuples to scale beyond 2 classes
    # ========================================================================
    # Each tuple is (model_name_in_config, class_label)
    # Class labels should be consecutive integers starting from 0
    model_classes = [
        ('mc_mlp_mnist', 0),
        ('mc_mlp_fashion_mnist', 1),
        # ('mc_mlp_cifar10', 2),  # Example: add more classes here
        # ('mc_mlp_custom', 3),
    ]
    # ========================================================================
    
    print(f"Training multiclass flow matching for {len(model_classes)} classes")
    print("Using CLIP-style conditioning: N(0,1) + class_embed -> target weights")
    
    # Load and prepare data for each class
    all_class_data = []
    
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
            
            # Determine directory based on dataset
            dataset = model_config['dataset']
            if dataset == 'mnist':
                model_dir = config['directories']['mlp_mnist']
            elif dataset == 'fashion_mnist':
                model_dir = config['directories']['mlp_fashion_mnist']
            else:
                raise ValueError(f"Unknown dataset for multiclass: {dataset}")
            
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
            
            # Choose which set of models to use based on training mode
            models_to_use = permuted_models if training_mode == "with_gitrebasin" else org_models
            
            # Prepare weight space data
            target_tensor, labels, ipca, flat_dim, actual_dim = prepare_class_data(
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
                'dataset': dataset
            })
        
        # Verify all classes have same dimensionality
        dims = [d['actual_dim'] for d in mode_class_data]
        flat_dims = [d['flat_dim'] for d in mode_class_data]
        
        if not all(d == dims[0] for d in dims):
            print("\n" + "!"*80)
            print("ERROR: Dimension mismatch across classes!")
            print("This typically happens when:")
            print("  1. Model architectures differ (check layer_layout in config)")
            print("  2. PCA components differ across classes")
            print("  3. Models have different total parameter counts")
            print("\nDimensions found:")
            for i, class_data in enumerate(mode_class_data):
                print(f"  Class {class_data['class_label']} ({class_data['model_name']}): "
                      f"flat={class_data['flat_dim']:,}, actual={class_data['actual_dim']:,}")
            print("!"*80)
            raise ValueError(f"Dimension mismatch: {dims}")
        
        actual_dim = dims[0]
        print(f"\n✓ All {len(mode_class_data)} classes aligned at dimension: {actual_dim:,}")
        if mode_class_data[0]['ipca'] is not None:
            print(f"  (Original dimension: {flat_dims[0]:,}, reduced via PCA)")
        
        # Use first class config as reference for flow model hyperparameters
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
            
            # Create source (noise) dataset - N(0, sigma)
            source_std = reference_config['source_std']
            total_samples = sum(d['target_tensor'].shape[0] for d in mode_class_data)
            source_tensor = torch.randn(total_samples, actual_dim) * source_std
            source_dataset = TensorDataset(source_tensor)
            
            print(f"Source distribution: N(0, {source_std}²)")
            print(f"Total training samples: {total_samples:,}")
            
            # Create dataloaders
            def collate_fn(batch):
                flats, labs = zip(*batch)
                flats = torch.stack(flats)
                labs = torch.stack(labs)
                return flats, labs
            
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
            
            # Create flow model with class conditioning
            flow_model = MultiClassWeightSpaceFlowModel(
                actual_dim,
                hidden_dim,
                time_embed_dim=reference_config['time_embed_dim'],
                class_embed_dim=reference_config.get('class_embed_dim', 64),
                dropout=reference_config['dropout']
            ).to(device)
            
            print(f"Flow model parameters: {count_parameters(flow_model):,}")
            print(f"  - Input dim: {actual_dim}")
            print(f"  - Hidden dim: {hidden_dim}")
            print(f"  - Time embed dim: {reference_config['time_embed_dim']}")
            print(f"  - Class embed dim: {reference_config.get('class_embed_dim', 64)}")
            
            # Create flow matcher
            cfm = MultiClassFlowMatching(
                sourceloader=sourceloader,
                targetloader=targetloader,
                model=flow_model,
                mode="velocity",
                t_dist=config['t_dist'],
                device=device
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
            
            # Save flow model checkpoint if requested
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
                        'class_embed_dim': reference_config.get('class_embed_dim', 64),
                        'dropout': reference_config['dropout']
                    },
                    'training_config': reference_config,
                    'best_loss': cfm.best_loss,
                    'num_classes': len(mode_class_data),
                    'class_info': [(d['model_name'], d['class_label']) for d in mode_class_data]
                }, checkpoint_path)
                print(f"✓ Saved flow model checkpoint to {checkpoint_path}")
            
            # Generate and evaluate models for each class
            n_samples = reference_config['n_samples']
            
            # Also test interpolation between classes
            test_interpolation = len(mode_class_data) == 2  # Only for 2 classes for now
            
            for class_data in mode_class_data:
                class_label = class_data['class_label']
                model_name = class_data['model_name']
                dataset = class_data['dataset']
                ipca = class_data['ipca']
                model_config = class_data['model_config']
                
                print(f"\n{'='*80}")
                print(f"Generating {n_samples} models for class {class_label} ({model_name})")
                print(f"{'='*80}")
                
                # Generate weights for this class
                new_weights_flat = generate_at_class_label(
                    cfm, class_label, n_samples, actual_dim, source_std,
                    reference_config, ipca=ipca
                )
                
                # Reconstruct models
                generated_models = reconstruct_mlp_models(
                    new_weights_flat, model_config, dataset
                )
                
                # Evaluate
                from utils import print_stats
                test_loader = get_data_loader(dataset, batch_size=32, train=False)
                
                mean_acc, std_acc = print_stats(generated_models, test_loader, device)
                
                print(f"\n{'='*80}")
                print(f"RESULTS - Class {class_label} ({model_name})")
                print(f"Mode: {training_mode}, Hidden dim: {hidden_dim}")
                print(f"Mean accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
                print(f"{'='*80}")
                
                # Save generated models if requested
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
                    
                    # Save summary stats
                    stats_path = os.path.join(save_dir, 'generation_stats.json')
                    with open(stats_path, 'w') as f:
                        json.dump({
                            'class_label': int(class_label),
                            'model_name': model_name,
                            'n_samples': n_samples,
                            'mean_accuracy': float(mean_acc),
                            'std_accuracy': float(std_acc),
                            'training_mode': training_mode,
                            'hidden_dim': hidden_dim,
                            'integration_steps': reference_config['integration_steps'],
                            'integration_method': reference_config['integration_method']
                        }, f, indent=2)
                    
                    print(f"✓ Saved {n_samples} generated models to {save_dir}")
                
                del generated_models
            
            # Test class interpolation - works for N classes
            if args.save_models and len(mode_class_data) >= 2:
                print(f"\n{'='*80}")
                print(f"Testing Class Interpolation ({len(mode_class_data)} classes)")
                print(f"{'='*80}")
                
                # Generate interpolation points based on number of classes
                num_classes = len(mode_class_data)
                
                # For 2 classes: test 0.25, 0.5, 0.75
                # For 3 classes: test 0.5, 1.0, 1.5, 2.0, 2.5
                # For N classes: test midpoints and a few beyond
                if num_classes == 2:
                    interp_labels = [0.25, 0.5, 0.75]
                else:
                    # Test midpoints between each adjacent pair plus some extrapolation
                    interp_labels = []
                    for i in range(num_classes - 1):
                        interp_labels.append(i + 0.5)
                    # Add some interpolation/extrapolation points
                    interp_labels.extend([0.25, num_classes - 0.25])
                
                n_interp_samples = 20
                
                for interp_label in interp_labels:
                    print(f"\nGenerating {n_interp_samples} models at class={interp_label:.2f}")
                    
                    # Generate weights at interpolated class
                    interp_weights_flat = generate_at_class_label(
                        cfm, interp_label, n_interp_samples, actual_dim, source_std,
                        reference_config, ipca=mode_class_data[0]['ipca']
                    )
                    
                    # Save interpolated models
                    save_dir = os.path.join(
                        args.output_dir,
                        f'{training_mode}_hidden{hidden_dim}',
                        f'interpolation_class_{interp_label:.2f}'
                    )
                    os.makedirs(save_dir, exist_ok=True)
                    
                    # Reconstruct and save using first class as template
                    interp_models = reconstruct_mlp_models(
                        interp_weights_flat,
                        mode_class_data[0]['model_config'],
                        mode_class_data[0]['dataset']
                    )
                    
                    for i, model in enumerate(interp_models):
                        model_path = os.path.join(save_dir, f'interpolated_model_{i}.pt')
                        torch.save(model.state_dict(), model_path)
                    
                    # Save metadata
                    meta_path = os.path.join(save_dir, 'interpolation_info.json')
                    with open(meta_path, 'w') as f:
                        json.dump({
                            'class_label': float(interp_label),
                            'n_samples': n_interp_samples,
                            'num_classes': num_classes,
                            'class_names': [d['model_name'] for d in mode_class_data],
                            'training_mode': training_mode,
                            'hidden_dim': hidden_dim
                        }, f, indent=2)
                    
                    print(f"✓ Saved {n_interp_samples} interpolated models to {save_dir}")
                    del interp_models
            
            del flow_model, cfm
            torch.cuda.empty_cache()


def get_data_loader(dataset_name, batch_size=32, train=False):
    """Helper to get data loader"""
    from utils import load_mnist, load_fashion_mnist
    
    dataset_name = dataset_name.lower()
    if dataset_name == "mnist":
        return load_mnist(batch_size=batch_size)
    elif dataset_name == "fashion_mnist":
        return load_fashion_mnist(batch_size=batch_size)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def main():
    parser = argparse.ArgumentParser(
        description='Train multiclass flow matching for neural network weight generation'
    )
    parser.add_argument('--config', type=str, default='constants.json',
                       help='Configuration file path')
    parser.add_argument('--num_models', type=int, default=100,
                       help='Number of pretrained models to use per class')
    parser.add_argument('--ref_point', type=int, default=0,
                       help='Reference model index for canonicalization')
    parser.add_argument('--hidden_dim', type=int, default=None,
                       help='Specific hidden dimension to test (tests all if None)')
    parser.add_argument('--mode', type=str, default=None,
                       choices=['with_gitrebasin', 'without_rebasin'],
                       help='Training mode (tests all if None)')
    parser.add_argument('--save_models', action='store_true',
                       help='Save generated models and flow model checkpoints')
    parser.add_argument('--output_dir', type=str, default='./generated_models',
                       help='Directory to save generated models and checkpoints')
    
    args = parser.parse_args()
    
    print("="*80)
    print("MULTICLASS FLOW MATCHING - Neural Network Weight Generation")
    print("="*80)
    print(f"Config file: {args.config}")
    print(f"Models per class: {args.num_models}")
    print(f"Reference point: {args.ref_point}")
    print(f"Architecture: mc_mlp_mnist (class 0) + mc_mlp_fashion_mnist (class 1)")
    print(f"Layer layout: [784, 64, 64, 10]")
    if args.hidden_dim:
        print(f"Hidden dimension: {args.hidden_dim}")
    else:
        print(f"Hidden dimensions: [64] (from config)")
    if args.mode:
        print(f"Training mode: {args.mode}")
    else:
        print(f"Training modes: with_gitrebasin, without_rebasin")
    print(f"Save models: {args.save_models}")
    if args.save_models:
        print(f"Output directory: {args.output_dir}")
    print("="*80)
    
    train_and_generate(args)
    
    print("\n" + "="*80)
    print("✓ Training and generation complete!")
    print("="*80)


if __name__ == "__main__":
    main()