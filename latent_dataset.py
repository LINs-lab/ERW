import os
import torch
import numpy as np
import json
from glob import glob
from safetensors import safe_open
from torch.utils.data import Dataset

class PrecomputedLatentDataset(Dataset):
    def __init__(self, 
                 data_dir, 
                 use_encoder_features=True,
                 use_all_shards=True):
        self.data_dir = data_dir
        self.use_encoder_features = use_encoder_features
        self.use_all_shards = use_all_shards
        
        # Check for dataset index
        self.index_path = os.path.join(data_dir, "dataset_index.json")
        if os.path.exists(self.index_path):
            with open(self.index_path, 'r') as f:
                self.index_info = json.load(f)
            print(f"Found dataset index: {self.index_info}")
        
        # Get the appropriate files
        self.files = self._get_data_files()
        
        if len(self.files) == 0:
            raise ValueError(f"No valid safetensors files found in {data_dir}")
            
        self.sample_map = self._build_sample_map()
        self.num_encoders = self.get_num_encoders()
    
    def _get_data_files(self):
        """Get the appropriate safetensor files based on configuration"""
        all_files = sorted([f for f in glob(os.path.join(self.data_dir, "*.safetensors")) 
                          if "latents_flip" not in os.path.basename(f) and "latents" in os.path.basename(f)])
        
        # Handle differently based on file naming patterns
        shard_files = [f for f in all_files if "_shard" in os.path.basename(f)]
        rank_files = [f for f in all_files if "_rank" in os.path.basename(f)]
        
        # If we have sharded files
        if shard_files:
            print(f"Found {len(shard_files)} sharded dataset files")
            
            if self.use_all_shards:
                print(f"Using all {len(shard_files)} shards")
                return shard_files
            else:
                # Use only shard00
                shard00_files = [f for f in shard_files if "shard00_" in os.path.basename(f)]
                if shard00_files:
                    print(f"Using only first shard: {shard00_files}")
                    return shard00_files
                print("No shard00 file found, using first available shard")
                return [shard_files[0]] if shard_files else []
                
        # If we have rank-based files
        elif rank_files:
            print(f"Found {len(rank_files)} rank-based dataset files")
            
            if self.use_all_shards:
                print(f"Using all {len(rank_files)} rank files")
                return rank_files
            else:
                # Use only rank00
                rank00_files = [f for f in rank_files if "rank00_" in os.path.basename(f)]
                if rank00_files:
                    print(f"Using only rank00 file: {rank00_files}")
                    return rank00_files
                print("No rank00 file found, using first available rank file")
                return [rank_files[0]] if rank_files else []
        
        # Default case
        print(f"Using all {len(all_files)} files (no shard/rank pattern detected)")
        return all_files
    
    def _build_sample_map(self):
        """Build a map of dataset indices to file locations and offsets"""
        sample_map = {}
        current_idx = 0
        sample_ids_seen = set()
        total_possible = 0
        
        for file_idx, file_path in enumerate(self.files):
            try:
                with safe_open(file_path, framework="pt", device="cpu") as f:
                    metadata = f.metadata() or {}
                    
                    # Get the number of samples in this file
                    if 'total_size' in metadata:
                        num_samples = int(metadata['total_size'])
                    else:
                        try:
                            labels = f.get_tensor("labels")
                            num_samples = labels.shape[0]
                        except:
                            print(f"Warning: Could not determine sample count in {file_path}, assuming 1")
                            num_samples = 1
                    
                    total_possible += num_samples
                    
                    # Get shard/rank info for better sample tracking
                    file_name = os.path.basename(file_path)
                    is_sharded = "_shard" in file_name
                    
                    # For each sample in this file
                    for i in range(num_samples):
                        if is_sharded:
                            shard_id = metadata.get('shard', -1)
                            if shard_id == -1:
                                try:
                                    shard_id = file_name.split("_shard")[1].split("_")[0]
                                except:
                                    shard_id = file_idx  
                        
                            sample_id = f"shard{shard_id}_{i}"
                        else:
                            sample_id = f"{file_name}_{i}"
                        
                        # Skip if we've seen this sample ID before (handles duplicates)
                        if sample_id in sample_ids_seen:
                            continue
                            
                        # Add to our dataset
                        sample_map[current_idx] = {
                            'file_path': file_path,
                            'idx_in_file': i
                        }
                        sample_ids_seen.add(sample_id)
                        current_idx += 1
                        
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
        
        print(f"Built sample map with {len(sample_map)} samples (from potential {total_possible})")
        return sample_map
    
    def get_num_encoders(self):
        """Determine the number of encoder feature sets in the dataset"""
        try:
            with safe_open(self.files[0], framework="pt", device="cpu") as f:
                encoder_count = 0
                keys = list(f.keys())
                while f"encoder_{encoder_count}_features" in keys:
                    encoder_count += 1
            return encoder_count
        except Exception as e:
            print(f"Error getting number of encoders: {e}")
            return 1

    def __len__(self):
        return len(self.sample_map)
    
    def __getitem__(self, idx):
        info = self.sample_map[idx]
        with safe_open(info['file_path'], framework="pt", device="cpu") as f:
            i = info['idx_in_file']
            latent = f.get_tensor("latents")[i:i+1]    # shape: (1, 4, H, W)
            label = f.get_tensor("labels")[i:i+1]      # shape: (1,)

            encoder_features = []
            if self.use_encoder_features:
                for encoder_idx in range(self.num_encoders):
                    feat_key = f"encoder_{encoder_idx}_features"
                    feat = f.get_tensor(feat_key)[i:i+1]
                    encoder_features.append(feat.squeeze(0))       
                    
            latent = latent.squeeze(0)
            label = label.squeeze(0)
            
            return latent, label, encoder_features
