import sys
import os
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import traceback
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from collections import defaultdict
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import torchvision
from utils import *
from models import get_resnet18
import pandas as pd
from tabulate import tabulate

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def finetune_model(model, train_loader, test_loader, epochs=10, lr=1e-4, 
                   detach_ratio=0.4, device='cuda'):
    model = model.to(device)

    all_layers = [
        model.conv1, model.bn1, model.layer1, model.layer2, model.layer3, model.layer4
    ]
    num_to_freeze = int(len(all_layers) * detach_ratio)

    for layer in all_layers[:num_to_freeze]:
        for param in layer.parameters():
            param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_acc = 100.0 * correct / total
        train_loss /= total
        scheduler.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        test_acc = 100.0 * correct / total
        best_acc = max(best_acc, test_acc)

        print(f"  Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Test Acc: {test_acc:.2f}% | Best Acc: {best_acc:.2f}%")

    return model, best_acc


def load_pretrained_models(model_dir, num_models=5, device='cuda'):
    """Load pretrained ResNet18 models from directory"""
    print(f"\n=== Loading {num_models} Pretrained Models from {model_dir} ===")
    
    models = []
    model_files = sorted([f for f in os.listdir(model_dir) if f.endswith('.pt')])[:num_models]
    
    if len(model_files) < num_models:
        raise ValueError(f"Only found {len(model_files)} models in {model_dir}, need {num_models}")
    
    for model_file in tqdm(model_files, desc="Loading models"):
        model_path = os.path.join(model_dir, model_file)
        model = get_resnet18(num_classes=10)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        models.append(model)
        print(f"  Loaded: {model_file}")
    
    return models


def evaluate_adaptability(pretrained_models, epochs_list=[0, 1, 5], device='cuda'):
    """Evaluate transfer learning performance of pretrained models"""
    
    results = {
        'Epoch': [],
        'STL-10-Few-Shot': [],
        'SVHN': [],
    }
    
    print("\n=== Loading Transfer Learning Datasets ===")
    tiny_train, tiny_test = get_fewshot_loaders(dataset_name='STL10')
    svhn_train, svhn_test = get_fewshot_loaders(dataset_name='SVHN')
    
    num_models = len(pretrained_models)
    
    for epochs in epochs_list:
        print(f"\n=== Evaluating at {epochs} epochs ===")
        
        stl_accs = []
        svhn_accs = []
        
        for i, model in enumerate(pretrained_models):
            print(f"\nModel {i+1}/{num_models}:")
            
            if epochs > 0:
                print(f"  Fine-tuning on STL-10 for {epochs} epochs...")
                _, stl_acc = finetune_model(
                    copy.deepcopy(model), 
                    tiny_train, 
                    tiny_test, 
                    epochs=epochs, 
                    lr=1e-4, 
                    detach_ratio=0.0, 
                    device=device
                )
                stl_accs.append(stl_acc)
                
                print(f"  Fine-tuning on SVHN for {epochs} epochs...")
                _, svhn_acc = finetune_model(
                    copy.deepcopy(model), 
                    svhn_train, 
                    svhn_test, 
                    epochs=epochs, 
                    lr=1e-4, 
                    detach_ratio=0.0, 
                    device=device
                )
                svhn_accs.append(svhn_acc)
            else:
                # Zero-shot evaluation
                stl_acc = evaluate_model(copy.deepcopy(model), tiny_test, device)
                stl_accs.append(stl_acc)
                print(f"  STL-10 zero-shot: {stl_acc:.2f}%")
                
                svhn_acc = evaluate_model(copy.deepcopy(model), svhn_test, device)
                svhn_accs.append(svhn_acc)
                print(f"  SVHN zero-shot: {svhn_acc:.2f}%")
        
        # Store results
        results['Epoch'].append(epochs)
        results['STL-10-Few-Shot'].append(f"{np.mean(stl_accs):.2f} ± {np.std(stl_accs):.2f}")
        results['SVHN'].append(f"{np.mean(svhn_accs):.2f} ± {np.std(svhn_accs):.2f}")
        
        print(f"\n--- Summary for {epochs} epochs ---")
        print(f"STL-10: {np.mean(stl_accs):.2f} ± {np.std(stl_accs):.2f}%")
        print(f"SVHN: {np.mean(svhn_accs):.2f} ± {np.std(svhn_accs):.2f}%")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate pretrained models on transfer learning tasks')
    parser.add_argument('--model_dir', type=str, default='resnet18_transfer_learning',
                       help='Directory containing pretrained ResNet18 models')
    parser.add_argument('--num_models', type=int, default=5,
                       help='Number of pretrained models to evaluate')
    parser.add_argument('--output', type=str, default='pretrained_transfer_results.csv',
                       help='Output CSV file for results')
    
    args = parser.parse_args()
    
    print("="*80)
    print("ResNet18 Transfer Learning Evaluation: CIFAR-10 → STL-10/SVHN")
    print(f"Using device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print("="*80)
    
    try:
        # Load pretrained models
        pretrained_models = load_pretrained_models(
            args.model_dir, 
            num_models=args.num_models, 
            device=device
        )
        
        # Evaluate transfer learning
        results = evaluate_adaptability(
            pretrained_models,
            epochs_list=[0, 1, 5],
            device=device
        )
        
        # Create and display results table
        df = pd.DataFrame(results)
        
        print("\n" + "="*80)
        print("FINAL RESULTS")
        print("="*80)
        print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))
        
        # Save results
        df.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")
        
    except Exception as e:
        print(f"Error in execution: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()