# Neural Network Weight Generation via Flow Matching

This project implements a flow matching approach to generate new neural network weights by learning from distributions of pretrained models. It supports multiple architectures including MLPs, ResNets, and Vision Transformers, with optional weight alignment using Git Re-Basin.

## Overview

The system learns to generate functional neural network weights by:
1. Loading collections of pretrained models
2. Optionally aligning them using Git Re-Basin permutation matching
3. Learning a flow from noise to the weight distribution
4. Generating new weight samples that achieve competitive performance

## Supported Models

| Model | Dataset | Architecture |
|-------|---------|--------------|
| `mlp_mnist` | MNIST | 3-layer MLP (784-32-32-10) |
| `mlp_fashion_mnist` | Fashion-MNIST | 3-layer MLP (784-128-128-10) |
| `mlp_iris` | Iris | 2-layer MLP (4-16-3) ||
| `resnet20_cifar10` | CIFAR-10 | ResNet-20 |
| `resnet18_cifar10` | CIFAR-10 | ResNet-18 |
| `vit_cifar10` | CIFAR-10 | ViT-Small  |

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/anonymousacademicc/DeepWeightFlow-ICLR.git
cd DeepWeightFlow-ICLR
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate 
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Generate datasets
generateModelsCode directory has scripts to generate data for different models , run the scripts like mentioned below by substituting `filename` with correct script name.

```bash
python generateModelsCode/{file_name}.py
```

## Directory Structure

### Project File Structure
```
.
├── flowmatching
│   ├── canonicalization.py
│   ├── constants.json
│   ├── flow_matching.py
│   ├── gitrebasin.py
│   ├── models.py
│   ├── permutation_specs.py
│   ├── train_and_generate.py
│   ├── transfer_learning_Resnet18.py
│   └── utils.py
├── generateModelsCode
│   ├── generate_fashion_mnist_weights.py
│   ├── generate_iris_weights.py
│   ├── generate_mnist_weights.py
│   ├── generateResnet18.py
│   ├── generateVitData.py
│   └── getImageNetWeightsResnet20.py
├── LICENSE
├── README.md
└── requirements.txt
```

## Usage

### Basic Training

Run training for a specific model:

```bash
python train_and_generate.py --model <model_name> [options]
```

### Examples

#### 1. Train MLP on MNIST
```bash
# Test all hidden dimensions with both modes
python train_and_generate.py --model mlp_mnist

# Test specific hidden dimension with Git Re-Basin
python train_and_generate.py --model mlp_mnist --hidden_dim 256 --mode with_gitrebasin
```

#### 2. Train MLP on Fashion-MNIST
```bash
python train_and_generate.py --model mlp_fashion_mnist

# Without Git Re-Basin alignment
python train_and_generate.py --model mlp_fashion_mnist --mode without_rebasin
```

#### 3. Train MLP on Iris
```bash
# This will process all 5 initialization types
python train_and_generate.py --model mlp_iris --hidden_dim 128
```

#### 4. Train ResNet-20 on CIFAR-10
```bash
python train_and_generate.py --model resnet20_cifar10

# Test specific configuration
python train_and_generate.py --model resnet20_cifar10 --hidden_dim 256 --num_models 50
```

#### 5. Train ResNet-18 on CIFAR-10 (with PCA)
```bash
# ResNet-18 uses PCA for dimensionality reduction
python train_and_generate.py --model resnet18_cifar10
```

#### 6. Train Vision Transformer on CIFAR-10
```bash
python train_and_generate.py --model vit_cifar10

# ViT uses smaller batch size and different alignment
python train_and_generate.py --model vit_cifar10 --num_models 50
```

#### 6. Transfer Learning Resnet18 on CIFAR-10
```bash
python transfer_learning_Resnet18.py --hidden_dim 512
```
### Command-Line Arguments

| Argument | Description | Default | Options |
|----------|-------------|---------|---------|
| `--model` | Model to train | Required | `mlp_mnist`, `mlp_fashion_mnist`, `mlp_iris`, `resnet20_cifar10`, `resnet18_cifar10`, `vit_cifar10` |
| `--config` | Configuration file path | `constants.json` | - |
| `--num_models` | Number of pretrained models to use | 100 | Integer |
| `--ref_point` | Reference model index for alignment | 0 | Integer |
| `--hidden_dim` | Specific hidden dimension to test | None (tests all) | Integer from config |
| `--mode` | Training mode | None (tests both) | `with_gitrebasin`, `without_rebasin` |

## Configuration

The `constants.json` file contains all model-specific configurations:

### Key Configuration Parameters

- **`flow_hidden_dims`**: List of hidden dimensions to test for the flow model
- **`n_iters`**: Number of training iterations (default: 30000)
- **`n_samples`**: Number of new models to generate
- **`use_pca`**: Whether to use PCA for dimensionality reduction (ResNet-18 only)
- **`recalibrate_bn`**: Whether to recalibrate BatchNorm statistics (ResNets only)
- **`integration_steps`**: Number of ODE integration steps for generation
- **`integration_method`**: ODE solver (`euler` or `rk4`)

### Model-Specific Settings

#### MLPs
- Simple architectures without BatchNorm
- Iris model tests 5 initialization types (default, he, xavier, uniform, normal)
- Fashion-MNIST uses larger hidden dimension (128) than MNIST (32)

#### ResNets
- Require BatchNorm recalibration after weight generation
- ResNet-18 uses PCA to reduce dimensionality (99 components)
- ResNet-20 processes full weight space without PCA

#### Vision Transformer
- Uses TransFusion algorithm for alignment
- Smaller batch size with graduient accumulation due to memory requirements
- Special handling for attention head permutations

## Troubleshooting

### Common Issues

1. **CUDA out of memory**
   - Reduce `batch_size` in constants.json
   - Reduce `n_samples` for generation
   - Use CPU by setting `device = torch.device("cpu")` in code

2. **Missing pretrained models**
   - Ensure all model files exist in the correct directories
   - Check file naming conventions match expected patterns

3. **Poor generation quality**
   - Increase `n_iters` for longer training
   - Try different `hidden_dim` values
   - Ensure Git Re-Basin alignment is working (`with_gitrebasin` mode)

4. **Slow training**
   - Use GPU if available
   - Reduce `integration_steps` for faster (but less accurate) generation
   - Use `euler` instead of `rk4` for integration

### Batch Processing

Run experiments for all models:

```bash
#!/bin/bash
for model in mlp_mnist mlp_fashion_mnist mlp_iris resnet20_cifar10 resnet18_cifar10 vit_cifar10; do
    echo "Training $model..."
    python train.py --model $model
done
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.