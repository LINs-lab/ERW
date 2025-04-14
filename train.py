import argparse
import copy
from copy import deepcopy
import logging
import os
from pathlib import Path
from collections import OrderedDict
import json
import math
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from diffusers.models import AutoencoderKL
import wandb
from torchvision.utils import make_grid
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize

from models.sit import SiT_models
from loss import SILoss
from utils import load_encoders
from dataset import CustomDataset

CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)

# ================ Configuration ================

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training with warmup and full training phases")
    parser.add_argument("--output-dir", type=str, default="exps")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--logging-dir", type=str, default="logs")
    parser.add_argument("--report-to", type=str, default="wandb")
    parser.add_argument("--sampling-steps", type=int, default=10000)
    parser.add_argument("--resume-step", type=int, default=0)
    parser.add_argument("--warmup-steps", type=int, default=10000, help="Number of steps for warmup phase")

    parser.add_argument("--model", type=str)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--encoder-depth", type=int, default=8)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qk-norm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_rope", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--data-dir", type=str, default="../data/imagenet256")
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--batch-size", type=int, default=256)

    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])

    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--max-train-steps", type=int, default=400000)
    parser.add_argument("--checkpointing-steps", type=int, default=10000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-weight-decay", type=float, default=0.)
    parser.add_argument("--adam-epsilon", type=float, default=1e-08)
    parser.add_argument("--max-grad-norm", default=1.0, type=float)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--prediction", type=str, default="v", choices=["v"])
    parser.add_argument("--cfg-prob", type=float, default=0.1)
    parser.add_argument("--enc-type", type=str, default='dinov2-vit-b')
    parser.add_argument("--proj-coeff", type=float, default=0.5)
    parser.add_argument("--weighting", default="uniform", type=str)
    parser.add_argument("--legacy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--l2r-depth", type=int, default=4)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    return args

# ================ Data Processing ================

def preprocess_raw_image(x, enc_type):
    """Preprocess raw images based on encoder type"""
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

def array2grid(x):
    """Convert batch of images to grid for visualization"""
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return x

@torch.no_grad()
def sample_posterior(moments, latents_scale=1., latents_bias=0.):
    """Sample from posterior distribution"""
    device = moments.device
    mean, std = torch.chunk(moments, 2, dim=1)
    z = mean + std * torch.randn_like(mean)
    z = (z * latents_scale + latents_bias)
    return z

def process_batch(raw_image, x, y, device, encoders, encoder_types, architectures, 
                  latents_scale, latents_bias, cfg_prob, num_classes, legacy=False):
    """Process a batch of data for model input"""
    raw_image = raw_image.to(device)
    x = x.squeeze(dim=1).to(device)
    y = y.to(device)
    
    # Handle labels for classifier-free guidance
    if legacy:
        drop_ids = torch.rand(y.shape[0], device=y.device) < cfg_prob
        labels = torch.where(drop_ids, num_classes, y)
    else:
        labels = y
        
    # Get latents and encoder features
    with torch.no_grad():
        x = sample_posterior(x, latents_scale=latents_scale, latents_bias=latents_bias)
        zs = []
        for encoder, encoder_type, arch in zip(encoders, encoder_types, architectures):
            raw_image_ = preprocess_raw_image(raw_image, encoder_type)
            z = encoder.forward_features(raw_image_)
            if 'mocov3' in encoder_type: z = z[:, 1:]
            if 'dinov2' in encoder_type: z = z['x_norm_patchtokens']
            zs.append(z)
            
    return x, labels, zs

# ================ Model Management ================

def get_dynamic_proj_coeff(args, step):
    """Calculate dynamic projection coefficient based on training step"""
    model_lower = args.model.lower()
    if "xl" in model_lower:
        c0 = 6.0
    elif "l" in model_lower:
        c0 = 3.0
    else:
        c0 = 1.0
    
    full_step = max(step - args.warmup_steps, 0)
    effective_step = min(full_step, args.max_train_steps - args.warmup_steps)
    decay_factor = 2 ** (-effective_step / 50000.0)
    proj_coeff = c0 * decay_factor
    return proj_coeff

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """Update exponential moving average model"""
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def requires_grad(model, flag=True):
    """Set requires_grad for all model parameters"""
    for p in model.parameters():
        p.requires_grad = flag

def init_model(args, latent_size, z_dims, is_warmup=True):
    """Initialize model based on configuration"""
    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    
    if is_warmup:
        model = SiT_models[args.model](
            input_size=latent_size,
            num_classes=args.num_classes,
            use_cfg=(args.cfg_prob > 0),
            class_dropout_prob=0.1,
            z_dims=z_dims,
            encoder_depth=args.l2r_depth, 
            trainable_depth=args.l2r_depth, 
            use_rope=False,
            warmup=True, 
            **block_kwargs
        )
    else:
        model = SiT_models[args.model](
            input_size=latent_size,
            num_classes=args.num_classes,
            use_cfg=(args.cfg_prob > 0),
            z_dims=z_dims,
            encoder_depth=args.encoder_depth,
            warmup=False,  
            use_rope=args.use_rope,
            trainable_depth=28, 
            **block_kwargs
        )
    
    return model

# ================ Checkpoint Management ================

def load_checkpoint(checkpoint_path, device="cpu", target_model=None):
    """Load checkpoint with robust error handling"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if target_model is not None:
            if 'model' in checkpoint:
                checkpoint['model'] = checkpoint['model']
            if 'ema' in checkpoint:
                checkpoint['ema'] = checkpoint['ema']
                
        return checkpoint
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None

def save_checkpoint(checkpoint_dir, name, model, ema, optimizer=None, args=None, step=0):
    """Save model checkpoint"""
    checkpoint = {
        "model": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
        "ema": ema.state_dict(),
        "args": args,
        "steps": step,
    }
    
    if optimizer is not None:
        checkpoint["opt"] = optimizer.state_dict()
        
    checkpoint_path = os.path.join(checkpoint_dir, f"{name}.pt")
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path

# ================ Logging and Metrics ================

def create_logger(logging_dir):
    """Create logger for experiment tracking"""
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger

# ================ Training Loops ================

def training_step(model, x, model_kwargs, zs, loss_fn, optimizer, accelerator, 
                  ema_model, is_warmup, proj_coeff=None, step=None, args=None):
    """Execute a single training step"""
    with accelerator.accumulate(model):
        if is_warmup:
            _, proj_loss = loss_fn(model, x, model_kwargs, zs=zs, warmup=True)
            proj_loss_mean = proj_loss.mean()
            loss = proj_loss_mean * args.proj_coeff
            logs = {
                "proj_loss": accelerator.gather(proj_loss_mean).mean().detach().item(),
            }
        else:
            loss, proj_loss = loss_fn(model, x, model_kwargs, zs=zs, warmup=False)
            loss_mean = loss.mean()
            proj_loss_mean = proj_loss.mean()
            current_proj_coeff = get_dynamic_proj_coeff(args, step)
            loss = loss_mean + proj_loss_mean * current_proj_coeff
            logs = {
                "loss": accelerator.gather(loss_mean).mean().detach().item(),
                "proj_loss": accelerator.gather(proj_loss_mean).mean().detach().item(),
                "proj_coeff": current_proj_coeff
            }

        accelerator.backward(loss)
        
        if accelerator.sync_gradients:
            params_to_clip = model.parameters()
            grad_norm = accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
            logs["grad_norm"] = accelerator.gather(grad_norm).mean().detach().item()
        
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        if accelerator.sync_gradients:
            update_ema(ema_model, model)
            
        return logs

def train_phase(accelerator, model, ema_model, optimizer, train_dataloader, loss_fn,
               args, phase_name, start_step, max_steps, checkpoint_dir,
               encoders, encoder_types, architectures, latents_scale, latents_bias,
               logger=None, is_warmup=False):
    """Execute a full training phase (warmup or full training)"""
    device = accelerator.device
    model.train()
    
    # Calculate steps to run in this phase
    steps_to_run = max_steps - start_step
    
    # Create progress bar
    progress_bar = tqdm(
        range(steps_to_run),
        initial=0,
        desc=f"{phase_name} Steps",
        disable=not accelerator.is_local_main_process,
    )
    
    global_step = start_step
    phase_complete = False
    
    while not phase_complete:
        for raw_image, x, y in train_dataloader:
            # Skip iteration if phase is already complete
            if global_step >= max_steps:
                phase_complete = True
                break
                
            # Process batch
            x, labels, zs = process_batch(
                raw_image, x, y, device, 
                encoders, encoder_types, architectures,
                latents_scale, latents_bias, 
                args.cfg_prob, args.num_classes, args.legacy
            )
            
            # Perform training step
            model_kwargs = dict(y=labels)
            logs = training_step(
                model, x, model_kwargs, zs, loss_fn, optimizer, accelerator,
                ema_model, is_warmup, args.proj_coeff, global_step, args
            )
            
            if accelerator.sync_gradients:
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                progress_bar.update(1)
                global_step += 1

            # Save periodic checkpoints
            if global_step % args.checkpointing_steps == 0 and global_step > 0:
                if accelerator.is_main_process:
                    prefix = "warmup_" if is_warmup else ""
                    save_path = save_checkpoint(
                        checkpoint_dir, 
                        f"{prefix}{global_step:07d}", 
                        model, ema_model, optimizer, args, global_step
                    )
                    if logger:
                        logger.info(f"Saved checkpoint to {save_path}")
            
            # Check if we've completed all training
            if global_step >= max_steps:
                if accelerator.is_main_process and logger:
                    prefix = "warmup_" if is_warmup else ""
                    save_path = save_checkpoint(
                        checkpoint_dir, 
                        f"{prefix}final", 
                        model, ema_model, optimizer, args, global_step
                    )
                    logger.info(f"Saved final {phase_name} checkpoint to {save_path}")
                phase_complete = True
                break
        
        # Log epoch completion if training isn't complete
        if not phase_complete and accelerator.is_main_process and logger:
            logger.info(f"Completed epoch in {phase_name} phase. Step: {global_step}/{max_steps}")

    # Close the progress bar when done
    progress_bar.close()
    return global_step

# ================ Main Pipeline ================

def main(args):
    # Initialize accelerator
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Create directories and setup logging
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        save_dir = os.path.join(args.output_dir, args.exp_name)
        os.makedirs(save_dir, exist_ok=True)
        
        # Save args
        args_dict = vars(args)
        json_dir = os.path.join(save_dir, "args.json")
        with open(json_dir, 'w') as f:
            json.dump(args_dict, f, indent=4)
            
        checkpoint_dir = os.path.join(save_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")
    else:
        logger = None
        save_dir = os.path.join(args.output_dir, args.exp_name)
        checkpoint_dir = os.path.join(save_dir, "checkpoints")

    # Initialize random seed
    device = accelerator.device
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)

    # Validate resolution
    assert args.resolution % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.resolution // 8

    # Initialize encoders
    if args.enc_type != None:
        encoders, encoder_types, architectures = load_encoders(args.enc_type, device, args.resolution)
    else:
        raise NotImplementedError("Encoder type must be specified")
        
    z_dims = [encoder.embed_dim for encoder in encoders] if args.enc_type != 'None' else [0]
    
    # Initialize VAE
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse"
    ).to(device)
    
    # Set latent scaling factors
    latents_scale = torch.tensor(
        [0.18215, 0.18215, 0.18215, 0.18215]
    ).view(1, 4, 1, 1).to(device)
    
    latents_bias = torch.tensor(
        [0., 0., 0., 0.]
    ).view(1, 4, 1, 1).to(device)

    # Initialize dataset and dataloader
    train_dataset = CustomDataset(args.data_dir)
    local_batch_size = int(args.batch_size // accelerator.num_processes)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    if accelerator.is_main_process and logger:
        logger.info(f"Dataset contains {len(train_dataset):,} images ({args.data_dir})")

    # Determine training phase based on resume step
    start_from_warmup = args.resume_step < args.warmup_steps
    warmup_complete = args.resume_step >= args.warmup_steps
    global_step = args.resume_step

    # Initialize WandB tracker if main process
    if accelerator.is_main_process:
        wandb.init(mode="offline")
        tracker_config = vars(copy.deepcopy(args))
        accelerator.init_trackers(
            project_name="ERW",
            config=tracker_config,
            init_kwargs={"wandb": {"name": f"{args.exp_name}"}}
        )

    # ====================== WARMUP PHASE ======================
    if not warmup_complete:
        if accelerator.is_main_process and logger:
            logger.info("Setting up warmup phase...")
            
        # Initialize warmup model and optimizer
        model_warmup = init_model(args, latent_size, z_dims, is_warmup=True).to(device)
        ema_warmup = deepcopy(model_warmup).to(device)
        requires_grad(ema_warmup, False)
        
        optimizer_warmup = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model_warmup.parameters()), 
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
        
        # Resume from checkpoint if specified
        if args.resume_step > 0:
            resume_ckpt_path = os.path.join(
                checkpoint_dir, 
                f"warmup_{args.resume_step:07d}.pt"
            )
            
            if accelerator.is_main_process and logger:
                logger.info(f"Resuming warmup from checkpoint: {resume_ckpt_path}")
                
            if os.path.exists(resume_ckpt_path):
                ckpt = load_checkpoint(
                    resume_ckpt_path, 
                    device='cpu',
                    target_model=model_warmup
                )
                
                if ckpt:
                    model_warmup.load_state_dict(ckpt['model'], strict=False)
                    ema_warmup.load_state_dict(ckpt['ema'], strict=False)
                    optimizer_warmup.load_state_dict(ckpt['opt'], strict=False)
                    global_step = ckpt['steps']
                    
                    if accelerator.is_main_process and logger:
                        logger.info(f"Successfully loaded warmup checkpoint from step {global_step}")
                else:
                    if accelerator.is_main_process and logger:
                        logger.warning(f"Failed to load warmup checkpoint, starting from scratch")
            else:
                if accelerator.is_main_process and logger:
                    logger.warning(f"Checkpoint file {resume_ckpt_path} not found, starting from scratch")
        
        # Initialize EMA with current model
        update_ema(ema_warmup, model_warmup, decay=0)
        
        # Prepare models for distributed training
        model_warmup, optimizer_warmup, train_dataloader = accelerator.prepare(
            model_warmup, optimizer_warmup, train_dataloader
        )
        
        # Setup loss function
        loss_fn = SILoss(
            prediction=args.prediction,
            path_type=args.path_type,
            encoders=encoders,
            accelerator=accelerator,
            latents_scale=latents_scale,
            latents_bias=latents_bias,
            weighting=args.weighting
        )
        
        # Run warmup training phase
        global_step = train_phase(
            accelerator=accelerator,
            model=model_warmup,
            ema_model=ema_warmup,
            optimizer=optimizer_warmup,
            train_dataloader=train_dataloader,
            loss_fn=loss_fn,
            args=args,
            phase_name="Warmup",
            start_step=global_step,
            max_steps=args.warmup_steps,
            checkpoint_dir=checkpoint_dir,
            encoders=encoders, 
            encoder_types=encoder_types, 
            architectures=architectures,
            latents_scale=latents_scale,
            latents_bias=latents_bias,
            logger=logger,
            is_warmup=True
        )
        
        # Set model to eval mode after training
        model_warmup.eval()
        
        # Save the final warmup checkpoint for transition
        if accelerator.is_main_process and logger:
            logger.info("Warmup phase complete. Preparing for transition to full training phase...")
            warmup_checkpoint_path = save_checkpoint(
                checkpoint_dir, 
                "warmup_final", 
                model_warmup, ema_warmup, optimizer_warmup, args, global_step
            )
    
    # ====================== FULL TRAINING PHASE ======================
    if accelerator.is_main_process and logger:
        logger.info("Setting up full training phase model...")
        
    # Initialize the full training model
    model_full = init_model(args, latent_size, z_dims, is_warmup=False).to(device)
    ema_full = deepcopy(model_full).to(device) 
    requires_grad(ema_full, False)
    
    if accelerator.is_main_process and logger:
        logger.info(f"Full model parameters: {sum(p.numel() for p in model_full.parameters()):,}")

    # Initialize optimizer for full training phase
    optimizer_full = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model_full.parameters()), 
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    # Handle transition from warmup to full training or resume from full training checkpoint
    if warmup_complete and args.resume_step == args.warmup_steps:
        # Just completed warmup phase or resuming from warmup boundary
        final_warmup_path = os.path.join(checkpoint_dir, "warmup_final.pt")
        if not os.path.exists(final_warmup_path):
            final_warmup_path = os.path.join(checkpoint_dir, f"warmup_{args.warmup_steps:07d}.pt")
            
        if os.path.exists(final_warmup_path):
            if accelerator.is_main_process and logger:
                logger.info(f"Loading warmup checkpoint from {final_warmup_path}")
                
            ckpt = load_checkpoint(
                final_warmup_path, 
                device='cpu',
                target_model=model_full
            )
            
            if ckpt:
                # Only transfer model weights, not optimizer state
                model_full.load_state_dict(ckpt['model'], strict=False)
                ema_full.load_state_dict(ckpt['model'], strict=False)  # Initialize EMA from model
                global_step = args.warmup_steps
                
                if accelerator.is_main_process and logger:
                    logger.info(f"Successfully loaded model weights from warmup checkpoint")
            else:
                if accelerator.is_main_process and logger:
                    logger.warning(f"Failed to load warmup checkpoint for transition")
        else:
            if accelerator.is_main_process and logger:
                logger.warning(f"No warmup checkpoint found at {final_warmup_path}")
    elif args.resume_step > args.warmup_steps:
        # Resuming from a specific full training checkpoint
        full_ckpt_path = os.path.join(checkpoint_dir, f"{args.resume_step:07d}.pt")
        
        if os.path.exists(full_ckpt_path):
            if accelerator.is_main_process and logger:
                logger.info(f"Resuming full training from checkpoint {full_ckpt_path}")
                
            ckpt = load_checkpoint(
                full_ckpt_path,
                device='cpu',
                target_model=model_full
            )
            
            if ckpt:
                model_full.load_state_dict(ckpt['model'])
                ema_full.load_state_dict(ckpt['ema'])
                optimizer_full.load_state_dict(ckpt['opt'])
                global_step = ckpt['steps']
                
                if accelerator.is_main_process and logger:
                    logger.info(f"Successfully resumed from full training checkpoint at step {global_step}")
            else:
                if accelerator.is_main_process and logger:
                    logger.warning(f"Failed to load full training checkpoint for resume")
        else:
            if accelerator.is_main_process and logger:
                logger.warning(f"Checkpoint {full_ckpt_path} not found for resume")
    
    # Initialize EMA with current model
    update_ema(ema_full, model_full, decay=0)
    
    # Prepare models and dataloaders for distributed training
    model_full, optimizer_full, train_dataloader = accelerator.prepare(
        model_full, optimizer_full, train_dataloader
    )
    
    # Setup loss function for full training
    loss_fn = SILoss(
        prediction=args.prediction,
        path_type=args.path_type,
        encoders=encoders,
        accelerator=accelerator,
        latents_scale=latents_scale,
        latents_bias=latents_bias,
        weighting=args.weighting
    )
    
    # Run full training phase
    if accelerator.is_main_process and logger:
        logger.info("Starting full training phase...")
    
    global_step = train_phase(
        accelerator=accelerator,
        model=model_full,
        ema_model=ema_full,
        optimizer=optimizer_full,
        train_dataloader=train_dataloader,
        loss_fn=loss_fn,
        args=args,
        phase_name="Full Training",
        start_step=global_step,
        max_steps=args.max_train_steps,
        checkpoint_dir=checkpoint_dir,
        encoders=encoders, 
        encoder_types=encoder_types, 
        architectures=architectures,
        latents_scale=latents_scale,
        latents_bias=latents_bias,
        logger=logger,
        is_warmup=False
    )
    
    # Final cleanup and completion
    model_full.eval()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process and logger:
        logger.info("Training complete!")
    accelerator.end_training()

# ================ Main Entry Point ================

if __name__ == "__main__":
    args = parse_args()
    main(args)
