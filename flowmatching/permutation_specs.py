from typing import NamedTuple
from collections import defaultdict
import torch

class PermutationSpec(NamedTuple):
    perm_to_axes: dict
    axes_to_perm: dict
    
def permutation_spec_from_axes_to_perm(axes_to_perm: dict) -> PermutationSpec:
    perm_to_axes = defaultdict(list)
    for wk, axis_perms in axes_to_perm.items():
        for axis, perm in enumerate(axis_perms):
            if perm is not None:
                perm_to_axes[perm].append((wk, axis))
    return PermutationSpec(dict(perm_to_axes), axes_to_perm)

    
# --------------------------------------- MLP Fashion MNIST Spec ---------------------------------------
def fashion_mnist_mlp_permutation_spec() -> PermutationSpec:
    """Permutation spec for 3-layer MLP (no .T, natural weight shapes)"""
    return permutation_spec_from_axes_to_perm({
        "fc1.weight": ("P_0", None), 
        "fc1.bias": ("P_0",),
        "fc2.weight": ("P_1", "P_0"), 
        "fc2.bias": ("P_1",),
        "fc3.weight": (None, "P_1"),
        "fc3.bias": (None,),
    })



# --------------------------------------- MLP Iris Spec ---------------------------------------
def iris_mlp_permutation_spec_mlp() -> PermutationSpec:
    """Permutation spec for Iris MLP (no .T, natural weight shapes)"""
    return permutation_spec_from_axes_to_perm({
        "fc1.weight": ("P_0", None),
        "fc1.bias": ("P_0",),
        "fc2.weight": (None, "P_0"),
        "fc2.bias": (None,),
    })

# --------------------------------------- MLP MNIST Spec ---------------------------------------
def mnist_mlp_permutation_spec_mlp() -> PermutationSpec:
    """Permutation spec for 3-layer MNIST MLP (no .T, natural weight shapes)"""
    return permutation_spec_from_axes_to_perm({
        "fc1.weight": ("P_0", None),
        "fc1.bias": ("P_0",),
        "fc2.weight": ("P_1", "P_0"),
        "fc2.bias": ("P_1",),
        "fc3.weight": (None, "P_1"),
        "fc3.bias": (None,),
    })

# --------------------------------------- Resnet20 Spec ---------------------------------------
 
def resnet20_permutation_spec() -> PermutationSpec:
    conv = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in, None, None)}

    norm = lambda name, p: {
        f"{name}.weight": (p,),
        f"{name}.bias": (p,),
        f"{name}.running_mean": (p,),
        f"{name}.running_var": (p,),
    }

    dense = lambda name, p_in, p_out: {
        f"{name}.weight": (p_out, p_in),
        f"{name}.bias": (p_out,)
    }

    easyblock = lambda name, p: {
        **conv(f"{name}.conv1", p, f"P_{name}_inner"),
        **norm(f"{name}.bn1", f"P_{name}_inner"),
        **conv(f"{name}.conv2", f"P_{name}_inner", p),
        **norm(f"{name}.bn2", p),
    }

    shortcutblock = lambda name, p_in, p_out: {
        **conv(f"{name}.conv1", p_in, f"P_{name}_inner"),
        **norm(f"{name}.bn1", f"P_{name}_inner"),
        **conv(f"{name}.conv2", f"P_{name}_inner", p_out),
        **norm(f"{name}.bn2", p_out),
        **conv(f"{name}.shortcut.0", p_in, p_out),
        **norm(f"{name}.shortcut.1", p_out),
    }

    return permutation_spec_from_axes_to_perm({
        **conv("conv1", None, "P_bg0"),
        **norm("bn1", "P_bg0"),

        **easyblock("layer1.0", "P_bg0"),
        **easyblock("layer1.1", "P_bg0"),
        **easyblock("layer1.2", "P_bg0"),

        **shortcutblock("layer2.0", "P_bg0", "P_bg1"),
        **easyblock("layer2.1", "P_bg1"),
        **easyblock("layer2.2", "P_bg1"),

        **shortcutblock("layer3.0", "P_bg1", "P_bg2"),
        **easyblock("layer3.1", "P_bg2"),
        **easyblock("layer3.2", "P_bg2"),

        **dense("linear", "P_bg2", None),
    })

# --------------------------------------- Resnet18 Spec ---------------------------------------

def resnet18_permutation_spec() -> PermutationSpec:
    conv = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in, None, None)}

    norm = lambda name, p: {
        f"{name}.weight": (p,),
        f"{name}.bias": (p,),
        f"{name}.running_mean": (p,),
        f"{name}.running_var": (p,),
    }

    dense = lambda name, p_in, p_out: {
        f"{name}.weight": (p_out, p_in),
        f"{name}.bias": (p_out,)
    }
    
    # Move basicblock here, inside resnet18_permutation_spec
    def basicblock(name, p_in, p_out, has_shortcut=False):
        spec = {
            **conv(f"{name}.conv1", p_in, f"P_{name}_inner"),
            **norm(f"{name}.bn1", f"P_{name}_inner"),
            **conv(f"{name}.conv2", f"P_{name}_inner", p_out),
            **norm(f"{name}.bn2", p_out),
        }
        
        if has_shortcut:
            spec.update({
                **conv(f"{name}.downsample.0", p_in, p_out),
                **norm(f"{name}.downsample.1", p_out),
            })
        
        return spec
    
    return permutation_spec_from_axes_to_perm({
        **conv("conv1", None, "P_64"),
        **norm("bn1", "P_64"),
        
        **basicblock("layer1.0", "P_64", "P_64"),
        **basicblock("layer1.1", "P_64", "P_64"),
        
        **basicblock("layer2.0", "P_64", "P_128", has_shortcut=True),
        **basicblock("layer2.1", "P_128", "P_128"),
        
        **basicblock("layer3.0", "P_128", "P_256", has_shortcut=True),
        **basicblock("layer3.1", "P_256", "P_256"),
        
        **basicblock("layer4.0", "P_256", "P_512", has_shortcut=True),
        **basicblock("layer4.1", "P_512", "P_512"),
        
        **dense("fc", "P_512", None),
    })
    
# --------------------------------------- Resnet18 Spec ---------------------------------------

class ViTPermutationSpec:
    """Specification for permutations applied throughout the network"""
    
    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        # Store permutations for each layer
        self.patch_embed_perm = None
        self.block_perms = []  # List of dicts with attention and mlp permutations
        for _ in range(num_blocks):
            self.block_perms.append({
                'attention_in': None,  # Input permutation to attention
                'attention_out': None,  # Output permutation from attention
                'mlp1': None, 
                'mlp2': None,  # Second MLP layer permutation  
            })
        self.head_perm = None  # Final classification head
        
    def set_block_perm(self, block_idx: int, perm_type: str, perm: torch.Tensor):
        """Set a specific permutation for a block"""
        self.block_perms[block_idx][perm_type] = perm


import os
import torch
import copy
from tqdm import tqdm
from utils import recalibrate_bn_stats

def get_permuted_models_data(
    model_name: str,
    model_dir: str,
    pretrained_model_name: str,
    num_models: int,
    ref_point: int = 0,
    device=None,
    model_config=None
):
    """Apply weight matching to align models with a reference model"""
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Handle ViT models separately
    if "vit" in model_name:
        return apply_transfusion(model_dir, num_models, pretrained_model_name, ref_point, device)
    
    # Get model and permutation spec
    ref_model, ps = get_model_and_spec(model_name, device)
    
    # For MLP models with different initializations, handle seed variations
    if "mlp" in model_name:
        init_types = ["default", "he", "xavier", "uniform", "normal"]
        all_permuted_models = []
        all_org_models = []
        
        for init_type in init_types:
            model_prefix = f"mlp_{init_type}_seed"
            
            # Load reference model
            ref_model_path = f"{model_dir}/{model_prefix}{ref_point}.pt"
            if not os.path.exists(ref_model_path):
                print(f"Skipping init type {init_type} - reference model not found")
                continue
                
            try:
                ref_model_init, _ = get_model_and_spec(model_name, device)
                ref_model_init.load_state_dict(torch.load(ref_model_path, map_location=device))
                ref_model_init = ref_model_init.to(device)
                print(f"Loaded reference model from {ref_model_path}")
            except Exception as e:
                print(f"Failed to load reference model for {init_type}: {e}")
                continue
            
            params_a = {k: v.clone().detach() for k, v in ref_model_init.state_dict().items()
                        if k in ps.axes_to_perm}
            
            permuted_models = [ref_model_init]
            org_models = [ref_model_init]
            
            # Process models for this initialization type
            for i in range(min(num_models, 21)):  # Seeds 0-20
                if i == ref_point:
                    continue
                    
                model_path = f"{model_dir}/{model_prefix}{i}.pt"
                if not os.path.exists(model_path):
                    continue
                    
                try:
                    model_b, _ = get_model_and_spec(model_name, device)
                    model_b.load_state_dict(torch.load(model_path, map_location=device))
                    model_b = model_b.to(device)
                    org_models.append(model_b)
                    
                    params_b = {k: v.clone().detach() for k, v in model_b.state_dict().items()
                                if k in ps.axes_to_perm}
                    
                    perm = weight_matching(ps, params_a, params_b, device=device)
                    permuted_params_b = apply_permutation(ps, perm, params_b)
                    
                    reconstructed_model = copy.deepcopy(model_b)
                    state_dict = reconstructed_model.state_dict()
                    for k in permuted_params_b:
                        state_dict[k] = permuted_params_b[k]
                    reconstructed_model.load_state_dict(state_dict)
                    reconstructed_model = reconstructed_model.to(device)
                    
                    permuted_models.append(reconstructed_model)
                except Exception as e:
                    print(f"Error processing model {model_prefix}{i}: {e}")
                    continue
            
            all_permuted_models.extend(permuted_models)
            all_org_models.extend(org_models)
        
        return ref_model, all_org_models, all_permuted_models
    
    # For ResNet models
    else:
        # Load reference model
        ref_model_path = f"{model_dir}/{pretrained_model_name}{ref_point}.pt"
        try:
            ref_model.load_state_dict(torch.load(ref_model_path, map_location=device))
            ref_model = ref_model.to(device)
            print(f"Loaded reference model from {ref_model_path}")
        except Exception as e:
            print(f"Failed to load reference model: {e}")
            raise e
        
        params_a = {k: v.clone().detach() for k, v in ref_model.state_dict().items()
                    if k in ps.axes_to_perm}
        
        permuted_models = [ref_model]
        org_models = [ref_model]
        
        for i in tqdm(range(num_models), desc="Processing models"):
            if i == ref_point:
                continue
                
            model_path = f"{model_dir}/{pretrained_model_name}{i}.pt"
            if not os.path.exists(model_path):
                print(f"Skipping model {i} - file not found")
                continue
                
            try:
                model_b, _ = get_model_and_spec(model_name, device)
                model_b.load_state_dict(torch.load(model_path, map_location=device))
                model_b = model_b.to(device)
                
                # Store original model
                org_models.append(model_b)
                
                params_b = {k: v.clone().detach() for k, v in model_b.state_dict().items()
                            if k in ps.axes_to_perm}
                
                # Apply weight matching
                perm = weight_matching(ps, params_a, params_b, device=device)
                permuted_params_b = apply_permutation(ps, perm, params_b)
                
                # Create reconstructed model
                reconstructed_model = copy.deepcopy(model_b)
                state_dict = reconstructed_model.state_dict()
                for k in permuted_params_b:
                    state_dict[k] = permuted_params_b[k]
                reconstructed_model.load_state_dict(state_dict)
                reconstructed_model = reconstructed_model.to(device)
                
                # Recalibrate BatchNorm statistics for ResNet models
                if model_config and model_config.get('recalibrate_bn', False):
                    if "resnet" in model_name.lower():
                        reconstructed_model = recalibrate_bn_stats(
                            reconstructed_model, 
                            device=device, 
                            print_stats=False
                        )
                
                permuted_models.append(reconstructed_model)
                
            except Exception as e:
                print(f"Error processing model {i}: {e}")
                continue
                
            torch.cuda.empty_cache()
        
        return ref_model, org_models, permuted_models


def get_model_and_spec(model_name: str, device):
    """Helper function to get model and its permutation spec"""
    if "mlp_fashion_mnist" in model_name:
        from models import MLP_Fashion_MNIST
        from permutation_specs import fashion_mnist_mlp_permutation_spec
        model = MLP_Fashion_MNIST()
        ps = fashion_mnist_mlp_permutation_spec()
    elif "mlp_mnist" in model_name:
        from models import MLP_MNIST
        from permutation_specs import mnist_mlp_permutation_spec_mlp
        model = MLP_MNIST()
        ps = mnist_mlp_permutation_spec_mlp()
    elif "mlp_iris" in model_name:
        from models import MLP_Iris
        from permutation_specs import iris_mlp_permutation_spec_mlp
        model = MLP_Iris()
        ps = iris_mlp_permutation_spec_mlp()
    elif "resnet20" in model_name:
        from models import ResNet20
        from permutation_specs import resnet20_permutation_spec
        model = ResNet20()
        ps = resnet20_permutation_spec()
    elif "resnet18" in model_name:
        from models import get_resnet18
        from permutation_specs import resnet18_permutation_spec
        model = get_resnet18()
        ps = resnet18_permutation_spec()
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    return model.to(device), ps