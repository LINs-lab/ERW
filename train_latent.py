import argparse
import copy
from copy import deepcopy
import logging
import os
from pathlib import Path
from collections import OrderedDict
import json
import math
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed

import wandb
from torchvision.utils import make_grid

from models.sit import SiT_models
from loss import SILoss 
from latent_dataset import PrecomputedLatentDataset

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training with precomputed latents and encoder features")
    parser.add_argument("--output-dir", type=str, default="exps")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--logging-dir", type=str, default="logs")
    parser.add_argument("--report-to", type=str, default="wandb")
    parser.add_argument("--resume-step", type=int, default=0)
    parser.add_argument("--warmup-steps", type=int, default=10000, help="Number of steps for warmup phase")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--encoder-depth", type=int, default=8)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qk-norm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_rope", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing precomputed latent safetensors")
    parser.add_argument("--data-split", type=str, default="train", help="数据集分割名称")
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--max-train-steps", type=int, default=400000)
    parser.add_argument("--checkpointing-steps", type=int, default=10000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-weight-decay", type=float, default=0.)
    parser.add_argument("--adam-epsilon", type=float, default=1e-08)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--prediction", type=str, default="v", choices=["v"])
    parser.add_argument("--cfg-prob", type=float, default=0.1)
    parser.add_argument("--enc-type", type=str, default='dinov2-vit-b')
    parser.add_argument("--proj-coeff", type=float, default=0.5)
    parser.add_argument("--weighting", type=str, default="uniform")
    parser.add_argument("--legacy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--l2r-depth", type=int, default=4)
    parser.add_argument("--use-precomputed", action="store_true", default=True, help="使用预计算的特征")
    return parser.parse_args(input_args)

def get_dynamic_proj_coeff(args, step):
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
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        name = name.replace("module.", "")
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def init_model(args, latent_size, z_dims, is_warmup=True):
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

def load_checkpoint(checkpoint_path, device="cpu", target_model=None):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        return checkpoint
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None

def save_checkpoint(checkpoint_dir, name, model, ema, optimizer=None, args=None, step=0):
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

def create_logger(logging_dir):
    os.makedirs(logging_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(logging_dir, "log.txt"))]
    )
    logger = logging.getLogger(__name__)
    return logger

def training_step(model, x, model_kwargs, encoder_features, loss_fn, optimizer, accelerator, 
                  ema_model, is_warmup, proj_coeff=None, step=None, args=None):
    with accelerator.accumulate(model):
        if is_warmup:
            _, proj_loss = loss_fn(model, x, model_kwargs, zs=encoder_features, warmup=True)
            proj_loss_mean = proj_loss.mean()
            loss = proj_loss_mean * args.proj_coeff
            logs = {"proj_loss": accelerator.gather(proj_loss_mean).mean().detach().item()}
        else:
            loss, proj_loss = loss_fn(model, x, model_kwargs, zs=encoder_features, warmup=False)
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
            grad_norm = accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            logs["grad_norm"] = accelerator.gather(grad_norm).mean().detach().item()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if accelerator.sync_gradients:
            update_ema(ema_model, model)
        return logs

def train_phase(accelerator, model, ema_model, optimizer, train_dataloader, loss_fn,
                args, phase_name, start_step, max_steps, checkpoint_dir,
                logger=None, is_warmup=False):
    device = accelerator.device
    model.train()
    steps_to_run = max_steps - start_step
    progress_bar = tqdm(range(steps_to_run), initial=0, desc=f"{phase_name} Steps",
                        disable=not accelerator.is_local_main_process)
    global_step = start_step
    phase_complete = False

    while not phase_complete:
        for batch in train_dataloader:
            if global_step >= max_steps:
                phase_complete = True
                break
            x, y, encoder_features = batch
            x = x.to(device)
            y = y.to(device)
            encoder_features = [feat.to(device) for feat in encoder_features]
            if args.legacy:
                drop_ids = torch.rand(y.shape[0], device=y.device) < args.cfg_prob
                labels = torch.where(drop_ids, args.num_classes, y)
            else:
                labels = y

            model_kwargs = {"y": labels}
            logs = training_step(model, x, model_kwargs, encoder_features, loss_fn, optimizer, accelerator,
                                 ema_model, is_warmup, args.proj_coeff, global_step, args)

            if accelerator.sync_gradients:
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                progress_bar.update(1)
                global_step += 1

            if global_step % args.checkpointing_steps == 0 and global_step > 0:
                if accelerator.is_main_process:
                    prefix = "warmup_" if is_warmup else ""
                    save_path = save_checkpoint(checkpoint_dir, f"{prefix}{global_step:07d}", model, ema_model, optimizer, args, global_step)
                    if logger:
                        logger.info(f"Saved checkpoint to {save_path}")

            if global_step >= max_steps:
                if accelerator.is_main_process and logger:
                    prefix = "warmup_" if is_warmup else ""
                    save_path = save_checkpoint(checkpoint_dir, f"{prefix}final", model, ema_model, optimizer, args, global_step)
                    logger.info(f"Saved final {phase_name} checkpoint to {save_path}")
                phase_complete = True
                break

        if not phase_complete and accelerator.is_main_process and logger:
            logger.info(f"Completed an epoch in {phase_name} phase. Step: {global_step}/{max_steps}")

    progress_bar.close()
    return global_step

def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps,
                              mixed_precision=args.mixed_precision,
                              log_with=args.report_to,
                              project_config=accelerator_project_config)
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        save_dir = os.path.join(args.output_dir, args.exp_name)
        os.makedirs(save_dir, exist_ok=True)
        args_dict = vars(args)
        with open(os.path.join(save_dir, "args.json"), 'w') as f:
            json.dump(args_dict, f, indent=4)
        checkpoint_dir = os.path.join(save_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")
    else:
        save_dir = os.path.join(args.output_dir, args.exp_name)
        checkpoint_dir = os.path.join(save_dir, "checkpoints")
        logger = None

    device = accelerator.device
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)
    assert args.resolution % 8 == 0, "Image resolution must be divisible by 8."
    latent_size = args.resolution // 8
    precomputed_data_dir = os.path.join(args.data_dir, f"{args.data_split}_{args.resolution}")
    train_dataset = PrecomputedLatentDataset(
        data_dir=precomputed_data_dir,
        use_encoder_features=True
    )
    local_batch_size = int(args.batch_size // accelerator.num_processes)
    train_dataloader = DataLoader(train_dataset, batch_size=local_batch_size,
                                  shuffle=True, num_workers=args.num_workers,
                                  pin_memory=True, drop_last=True)
    if accelerator.is_main_process and logger:
        logger.info(f"Dataset contains {len(train_dataset):,} samples from {precomputed_data_dir}")
        logger.info(f"Using precomputed latents with {train_dataset.num_encoders} encoders")
    z_dims = [768]
    global_step = args.resume_step
    warmup_complete = args.resume_step >= args.warmup_steps
    if accelerator.is_main_process:
        wandb.init(mode="offline")
        tracker_config = vars(copy.deepcopy(args))
        accelerator.init_trackers(project_name="ERW", config=tracker_config,
                                  init_kwargs={"wandb": {"name": f"{args.exp_name}"}})
    # ====================== WARMUP PHASE ======================
    if not warmup_complete:
        if accelerator.is_main_process and logger:
            logger.info("Setting up warmup phase...")
        model_warmup = init_model(args, latent_size, z_dims, is_warmup=True).to(device)
        ema_warmup = deepcopy(model_warmup).to(device)
        requires_grad(ema_warmup, False)
        optimizer_warmup = torch.optim.AdamW(filter(lambda p: p.requires_grad, model_warmup.parameters()),
                                             lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2),
                                             weight_decay=args.adam_weight_decay, eps=args.adam_epsilon)
        if args.resume_step > 0:
            resume_ckpt_path = os.path.join(checkpoint_dir, f"warmup_{args.resume_step:07d}.pt")
            if accelerator.is_main_process and logger:
                logger.info(f"Resuming warmup from checkpoint: {resume_ckpt_path}")
            if os.path.exists(resume_ckpt_path):
                ckpt = load_checkpoint(resume_ckpt_path, device='cpu', target_model=model_warmup)
                if ckpt:
                    model_warmup.load_state_dict(ckpt['model'], strict=False)
                    ema_warmup.load_state_dict(ckpt['ema'], strict=False)
                    optimizer_warmup.load_state_dict(ckpt['opt'])
                    global_step = ckpt['steps']
                    if accelerator.is_main_process and logger:
                        logger.info(f"Loaded warmup checkpoint from step {global_step}")
                else:
                    if accelerator.is_main_process and logger:
                        logger.warning("Failed to load warmup checkpoint, starting from scratch")
            else:
                if accelerator.is_main_process and logger:
                    logger.warning(f"Warmup checkpoint {resume_ckpt_path} not found, starting from scratch")
        update_ema(ema_warmup, model_warmup, decay=0)
        model_warmup, optimizer_warmup, train_dataloader = accelerator.prepare(model_warmup, optimizer_warmup, train_dataloader)
        loss_fn = SILoss(prediction=args.prediction,
                         path_type=args.path_type,
                         weighting=args.weighting,
                         precomputed_features=True)
        global_step = train_phase(accelerator, model_warmup, ema_warmup, optimizer_warmup,
                                  train_dataloader, loss_fn, args, phase_name="Warmup",
                                  start_step=global_step, max_steps=args.warmup_steps,
                                  checkpoint_dir=checkpoint_dir, logger=logger, is_warmup=True)
        model_warmup.eval()
        if accelerator.is_main_process and logger:
            logger.info("Warmup phase complete. Saving warmup checkpoint for transition...")
            warmup_checkpoint_path = save_checkpoint(checkpoint_dir, "warmup_final", model_warmup, ema_warmup, optimizer_warmup, args, global_step)
    # ====================== FULL TRAINING PHASE ======================
    if accelerator.is_main_process and logger:
        logger.info("Setting up full training phase model...")
    model_full = init_model(args, latent_size, z_dims, is_warmup=False).to(device)
    ema_full = deepcopy(model_full).to(device)
    requires_grad(ema_full, False)
    if accelerator.is_main_process and logger:
        logger.info(f"Full model parameters: {sum(p.numel() for p in model_full.parameters()):,}")
    optimizer_full = torch.optim.AdamW(filter(lambda p: p.requires_grad, model_full.parameters()),
                                       lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2),
                                       weight_decay=args.adam_weight_decay, eps=args.adam_epsilon)
    if (args.resume_step >= args.warmup_steps) and (args.resume_step != args.warmup_steps):
        full_ckpt_path = os.path.join(checkpoint_dir, f"{args.resume_step:07d}.pt")
        if os.path.exists(full_ckpt_path):
            if accelerator.is_main_process and logger:
                logger.info(f"Resuming full training from checkpoint {full_ckpt_path}")
            ckpt = load_checkpoint(full_ckpt_path, device='cpu', target_model=model_full)
            if ckpt:
                model_full.load_state_dict(ckpt['model'])
                ema_full.load_state_dict(ckpt['ema'])
                optimizer_full.load_state_dict(ckpt['opt'])
                global_step = ckpt['steps']
                if accelerator.is_main_process and logger:
                    logger.info(f"Resumed full training checkpoint from step {global_step}")
            else:
                if accelerator.is_main_process and logger:
                    logger.warning("Failed to load full training checkpoint, starting from scratch")
        else:
            if accelerator.is_main_process and logger:
                logger.warning(f"Full training checkpoint {full_ckpt_path} not found, starting from scratch")
    elif args.resume_step == args.warmup_steps:
        final_warmup_path = os.path.join(checkpoint_dir, "warmup_final.pt")
        if not os.path.exists(final_warmup_path):
            final_warmup_path = os.path.join(checkpoint_dir, f"warmup_{args.warmup_steps:07d}.pt")
        if os.path.exists(final_warmup_path):
            if accelerator.is_main_process and logger:
                logger.info(f"Loading warmup checkpoint from {final_warmup_path} for transition")
            ckpt = load_checkpoint(final_warmup_path, device='cpu', target_model=model_full)
            if ckpt:
                model_full.load_state_dict(ckpt['model'], strict=False)
                ema_full.load_state_dict(ckpt['model'], strict=False)
                global_step = args.warmup_steps
                if accelerator.is_main_process and logger:
                    logger.info("Warmup checkpoint loaded for transition")
            else:
                if accelerator.is_main_process and logger:
                    logger.warning("Failed to load warmup checkpoint for transition")
        else:
            if accelerator.is_main_process and logger:
                logger.warning("No warmup checkpoint found for transition")
    update_ema(ema_full, model_full, decay=0)
    model_full, optimizer_full, train_dataloader = accelerator.prepare(model_full, optimizer_full, train_dataloader)
    loss_fn = SILoss(prediction=args.prediction,
                     path_type=args.path_type,
                     weighting=args.weighting,
                     precomputed_features=True)
    if accelerator.is_main_process and logger:
        logger.info("Starting full training phase...")
    global_step = train_phase(accelerator, model_full, ema_full, optimizer_full,
                              train_dataloader, loss_fn, args, phase_name="Full Training",
                              start_step=global_step, max_steps=args.max_train_steps,
                              checkpoint_dir=checkpoint_dir, logger=logger, is_warmup=False)
    model_full.eval()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process and logger:
        logger.info("Training complete!")
    accelerator.end_training()

if __name__ == "__main__":
    args = parse_args()
    main(args)
