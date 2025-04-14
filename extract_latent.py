import os
import gc
import torch
import numpy as np
import argparse
from datetime import datetime
from glob import glob
from PIL import ImageFile
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
from safetensors.torch import save_file
from diffusers.models import AutoencoderKL
from utils import load_encoders
from torchvision.transforms import Normalize 
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from dataset import CustomDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True
CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)

@torch.no_grad()
def sample_posterior(moments, latents_scale=1., latents_bias=0.):
    """Sample from posterior distribution"""
    device = moments.device
    mean, std = torch.chunk(moments, 2, dim=1)
    z = mean + std * torch.randn_like(mean)
    z = (z * latents_scale + latents_bias)
    return z

def preprocess_raw_image(x, enc_type):
    resolution = x.shape[-1]
    if 'clip' in enc_type:
        x = x / 255.
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
        x = Normalize(CLIP_DEFAULT_MEAN, CLIP_DEFAULT_STD)(x)
    elif 'mocov3' in enc_type or 'mae' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'dinov2' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
    elif 'dinov1' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'jepa' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
    return x

def preprocess_dataset(args):
    try:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        print(f"Initialized distributed training: rank={rank}, world_size={world_size}")
        distributed = True
    except Exception as e:
        print(f"Distributed initialization failed: {e}. Using single process.")
        rank = 0
        world_size = 1
        distributed = False
    
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    
    output_dir = os.path.join(args.output_dir, f"{args.data_split}_{args.resolution}")
    if rank == 0:
        print(f"Saving precomputed features to {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
    vae = AutoencoderKL.from_pretrained(args.vae_path).to(device)
    vae.eval()
    
    encoders, encoder_types, architectures = load_encoders(args.enc_type, device, args.resolution)
    for enc in encoders:
        enc.eval()

    # Create dataset and properly shard it for distributed processing
    dataset = CustomDataset(args.data_path)
    
    if distributed:
        # Use DistributedSampler to ensure each process gets a unique subset
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed)
        loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, 
                          num_workers=args.num_workers, pin_memory=True, drop_last=False)
        if rank == 0:
            print(f"Dataset sharded across {world_size} processes")
    else:
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, 
                          num_workers=args.num_workers, pin_memory=True, drop_last=False)
    
    total_data = len(dataset)
    if rank == 0:
        print(f"Total images: {total_data}")
    
    latents = []
    encoder_features = []  
    labels = []

    latents_scale = torch.tensor(
        [0.18215, 0.18215, 0.18215, 0.18215]
        ).view(1, 4, 1, 1).to(device)
    latents_bias = torch.tensor(
        [0., 0., 0., 0.]
        ).view(1, 4, 1, 1).to(device)
    
    for batch_idx, (raw_image, x, y) in enumerate(loader):
        if rank == 0 and batch_idx % 10 == 0:
            print(f"{datetime.now()} Processing batch {batch_idx+1}/{len(loader)}")
        raw_image = raw_image.to(device)
        x = x.squeeze(dim=1).to(device)
        y = y.to(device)
        
        with torch.no_grad():
            latent = sample_posterior(x, latents_scale=latents_scale, latents_bias=latents_bias)
            zs = []
            for encoder, enc_type in zip(encoders, encoder_types):
                processed_image = preprocess_raw_image(raw_image, enc_type)
                z = encoder.forward_features(processed_image)
                if 'mocov3' in enc_type:
                    z = z[:, 1:]
                if 'dinov2' in enc_type:
                    z = z['x_norm_patchtokens']
                zs.append(z)
        
        latents.append(latent.cpu())
        labels.append(y.cpu())
        encoder_features.append(zs)
    
    try:
        latents_cat = torch.cat(latents, dim=0)
        labels_cat = torch.cat(labels, dim=0)
        
        num_encoders = len(encoders)
        processed_features = []
        for i in range(num_encoders):
            feats = torch.cat([batch[i] for batch in encoder_features], dim=0)
            processed_features.append(feats)
        
        save_dict = {
            'latents': latents_cat,
            'labels': labels_cat,
        }
        for i in range(num_encoders):
            save_dict[f'encoder_{i}_features'] = processed_features[i]

        # Save with shard information in metadata and filename
        save_filename = os.path.join(output_dir, f"latents_shard{rank:02d}_of_{world_size:02d}.safetensors")
        save_file(save_dict, save_filename, metadata={
            'total_size': str(latents_cat.shape[0]),  
            'dtype': str(latents_cat.dtype),
            'encoder_types': ','.join(encoder_types),
            'shard': str(rank),
            'num_shards': str(world_size)
        })
        print(f"Rank {rank}: Saved {latents_cat.shape[0]} samples to {save_filename}")
        
        # Optionally, create an index file on rank 0
        if rank == 0:
            index_path = os.path.join(output_dir, "dataset_index.json")
            import json
            with open(index_path, 'w') as f:
                json.dump({
                    'num_shards': world_size,
                    'total_samples': total_data,
                    'encoder_types': encoder_types
                }, f)
            print(f"Saved dataset index to {index_path}")
        
        torch.cuda.empty_cache()
        gc.collect()
    except Exception as e:
        print(f"Error saving file: {e}")
    
    try:
        dist.barrier()
    except:
        pass
    
    if rank == 0:
        print("Preprocessing complete!")
    
    try:
        dist.barrier()
        dist.destroy_process_group()
    except:
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess dataset to extract latents and encoder features")
    parser.add_argument("--data-path", type=str, required=True, help="Path to image folder")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save precomputed features")
    parser.add_argument("--data-split", type=str, default="train", help="Dataset split (train/val)")
    parser.add_argument("--resolution", type=int, default=256, help="Image resolution")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for processing")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of dataloader workers")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--enc-type", type=str, default="dinov2-vit-b", help="Encoder type")
    parser.add_argument("--vae-path", type=str, default="stabilityai/sd-vae-ft-mse", help="VAE model path")
    args = parser.parse_args()
    preprocess_dataset(args)
