import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mutual_info_score
import os
import argparse
from typing import Tuple, List, Dict

# --------------------------- DINOv2 Feature Extraction ---------------------------

def image_from_minus_one_to_one(x: torch.Tensor) -> torch.Tensor:
    """
    Scales image tensor from [-1, 1] to [0, 1].
    """
    return (x + 1.0) / 2.0

def preprocess_raw_image(x: torch.Tensor) -> torch.Tensor:
    """
    Preprocesses raw images:
    - Scales from [-1, 1] to [0, 1]
    - Normalizes using ImageNet mean and std
    - Resizes to the required resolution
    """
    resolution = x.shape[-1]
    x = image_from_minus_one_to_one(x)
    x = transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean
                             std=[0.229, 0.224, 0.225])(x)
    # Resize to 224x224 if not already
    if resolution != 224:
        x = torch.nn.functional.interpolate(x, size=224, mode='bicubic', align_corners=False)
    return x

def get_dino_v2_model(model_name: str = 'dinov2_vitb14') -> nn.Module:
    """
    Loads the DINOv2 model from torch.hub, modifies it for feature extraction.
    
    Args:
        model_name (str): The name of the DINOv2 model variant.

    Returns:
        nn.Module: The modified DINOv2 model.
    """
    print(f"[Info] Loading DINOv2 model: {model_name}")
    encoder = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=True)
    encoder.head = nn.Identity()  # Remove the classification head
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    return encoder

@torch.no_grad()
def get_dino_v2_representation(raw_images: torch.Tensor, model: nn.Module) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Extracts DINOv2 representations from raw images, including all hidden layers.
    
    Args:
        raw_images (torch.Tensor): Batch of raw images.
        model (nn.Module): The DINOv2 model.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]: Patch tokens, class tokens, and all hidden layers.
    """
    # Preprocess images
    preprocessed = preprocess_raw_image(raw_images)
    
    # Initialize a dictionary to store hidden states
    hidden_states = {}
    
    # Define a hook function to capture outputs
    def get_hook(name):
        def hook(module, input, output):
            hidden_states[name] = output.detach()
        return hook
    
    # Register hooks on each Transformer block
    hooks = []
    for idx, block in enumerate(model.blocks):
        hook = block.register_forward_hook(get_hook(f'layer_{idx}'))
        hooks.append(hook)
    
    # Forward pass
    features = model.forward_features(preprocessed)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Extract final patch and class tokens
    cls_token = features['x_norm_clstoken']  # Shape: [B, 1, D]
    patch_tokens = features['x_norm_patchtokens']  # Shape: [B, N, D]
    
    return patch_tokens, cls_token, hidden_states

# --------------------------- Mutual Information Functions ---------------------------

def compute_mutual_information(x: np.ndarray, y: np.ndarray, bins: int = 32) -> float:
    """
    Computes mutual information between two 1D discrete variables.
    
    Args:
        x (np.ndarray): First discrete variable.
        y (np.ndarray): Second discrete variable.
        bins (int): Number of bins for discretization.

    Returns:
        float: Mutual information score.
    """
    contingency = np.histogram2d(x, y, bins=bins)[0]
    mi = mutual_info_score(None, None, contingency=contingency)
    return mi

def tensor_to_1d_discrete(tensor_in: torch.Tensor, max_points: int = 10000, bins: int = 64) -> np.ndarray:
    """
    Converts a high-dimensional tensor to a 1D discrete numpy array for mutual information computation.
    
    Args:
        tensor_in (torch.Tensor): Input tensor.
        max_points (int): Maximum number of points to sample.
        bins (int): Number of bins for discretization.

    Returns:
        np.ndarray: Discretized 1D array.
    """
    data = tensor_in.detach().cpu().numpy().reshape(-1)
    if data.shape[0] > max_points:
        idx = np.random.choice(data.shape[0], size=max_points, replace=False)
        data = data[idx]
    dmin, dmax = data.min(), data.max()
    if abs(dmax - dmin) < 1e-7:
        return np.zeros_like(data, dtype=np.int64)
    data_scaled = (data - dmin) / (dmax - dmin + 1e-9) * (bins - 1)
    data_int = data_scaled.astype(np.int64)
    return data_int

# --------------------------- Main Pipeline ---------------------------

def main():
    parser = argparse.ArgumentParser(description="DINOv2 Feature Extraction Pipeline with All Hidden Layers")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a checkpoint file (optional). If provided, the model weights will be loaded from this file.")
    parser.add_argument("--image-size", type=int, default=224, help="Input image resolution (default: 224)")
    parser.add_argument("--in-channels", type=int, default=3, help="Number of input channels (default: 3 for RGB)")
    parser.add_argument("--num-classes", type=int, default=1000, help="Number of classes (default: 1000 for ImageNet)")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for DataLoader (default: 256)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on (default: cuda)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--data-path", type=str, required=True, help="Path to the image dataset (e.g., ImageNet)")
    parser.add_argument("--save-features", type=str, default=None, help="Path to save extracted features (optional)")
    parser.add_argument("--max-images", type=int, default=10000, help="Maximum number of images to process (default: 10000)")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of DataLoader workers (default: 4)")
    args = parser.parse_args()

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Check device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device}")

    # Load DINOv2 model
    model = get_dino_v2_model('dinov2_vitg14').to(device)
    if args.checkpoint:
        print(f"[Info] Loading checkpoint from: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        state_dict = checkpoint.get('ema', checkpoint.get('model', checkpoint))
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[Warning] Missing keys: {missing}")
        if unexpected:
            print(f"[Warning] Unexpected keys: {unexpected}")

    # Prepare Image Dataset and DataLoader
    print("[Info] Preparing Image Dataset and DataLoader...")
    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        # Normalization is handled in preprocess_raw_image
    ])
    dataset = datasets.ImageFolder(root=args.data_path, transform=transform)
    if args.max_images:
        if args.max_images > len(dataset):
            print(f"[Warning] max_images {args.max_images} is greater than dataset size {len(dataset)}. Using full dataset.")
        else:
            dataset = torch.utils.data.Subset(dataset, indices=list(range(args.max_images)))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.num_workers, pin_memory=True)

    # Prepare to store features
    all_patch_tokens = []
    all_cls_tokens = []
    all_hidden_layers = {f"layer_{i}": [] for i in range(8)}  # Assuming 8 layers as in SiT_L2R
    all_labels = []

    # Extract Features
    print("[Info] Starting Feature Extraction...")
    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        patch_tokens, cls_token, hidden_states = get_dino_v2_representation(images, model)  # Shapes: [B, N, D], [B, 1, D], {layer_i: [B, N, D]}
        all_patch_tokens.append(patch_tokens.cpu())
        all_cls_tokens.append(cls_token.cpu())
        all_labels.append(labels.cpu())
        
        # Collect hidden states for each layer
        for layer_key in all_hidden_layers.keys():
            if layer_key in hidden_states:
                all_hidden_layers[layer_key].append(hidden_states[layer_key].cpu())
            else:
                print(f"[Warning] {layer_key} not found in hidden_states.")
        
        processed_images = (batch_idx + 1) * args.batch_size
        if processed_images > args.max_images:
            break
        
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(dataloader):
            print(f"[Info] Processed {min(processed_images, args.max_images)} / {args.max_images} images")

    # Concatenate all features
    all_patch_tokens = torch.cat(all_patch_tokens, dim=0)  # Shape: [Total, N, D]
    all_cls_tokens = torch.cat(all_cls_tokens, dim=0)      # Shape: [Total, 1, D]
    all_labels = torch.cat(all_labels, dim=0)              # Shape: [Total]
    
    # Concatenate hidden layers
    for layer_key in all_hidden_layers.keys():
        all_hidden_layers[layer_key] = torch.cat(all_hidden_layers[layer_key], dim=0)  # Shape: [Total, N, D]

    print(f"[Info] Extracted Patch Tokens shape: {all_patch_tokens.shape}")
    print(f"[Info] Extracted Class Tokens shape: {all_cls_tokens.shape}")
    for layer_key, tensor in all_hidden_layers.items():
        print(f"[Info] Extracted {layer_key} shape: {tensor.shape}")

    # Optionally save features
    if args.save_features:
        save_dir = os.path.dirname(args.save_features)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        torch.save({
            'patch_tokens': all_patch_tokens,
            'cls_tokens': all_cls_tokens,
            'hidden_states': all_hidden_layers,
            'labels': all_labels
        }, args.save_features)
        print(f"[Success] Features saved to {args.save_features} in PyTorch format.")
    else:
        print("[Info] --save-features not provided. Skipping saving features.")

    print("[Info] Feature Extraction Pipeline Completed Successfully.")

if __name__ == "__main__":
    main()

