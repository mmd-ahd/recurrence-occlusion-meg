"""Stage 2 of HTRN training: end-to-end fine-tuning of the whole network on ImageNet-1k.

Loads the stage-1 checkpoint (``train_htrn_stage1.py``), unfreezes the backbone and continues
training with differential learning rates (small for the ResNet backbone and ``fc``, larger for the
feedback modules), cosine decay and gradient clipping. Clean validation uses the same images every
epoch; occluded validation reseeds the random erasing so the occlusion patterns are identical
across epochs.

Checkpoints: ``latest_checkpoint.pt``, ``best_checkpoint_clean.pt`` and ``best_checkpoint_occluded.pt``
under ``--checkpoint_dir``. The best clean checkpoint is the paper's ``finetune_clean_checkpoint.pt``.

Usage:
    python scripts/04_training/train_htrn_finetune.py --data_dir ImageNet \
        --checkpoint_in results/checkpoints/htrn_stage1/best_checkpoint_clean.pt \
        --checkpoint_dir results/checkpoints/htrn_finetuned [--resume]
"""
import argparse
import logging
import math
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import DataLoader

from dfbmodels.models.resnet import ResNet50
from htrn import setup_logging
from htrn.imagenet import (collate_fn, run_deterministic_occluded_evaluation, run_evaluation,
                           transform_train_fn, transform_val_clean_fn, transform_val_occluded_fn)

log = logging.getLogger(__name__)

# Parameters whose name contains one of these belong to the feedback pathway
FEEDBACK_KEYWORDS = ['compress', 'feedback', 'layernorm', 'fb_']


def parse_args():
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])

    parser.add_argument('--data_dir', default='ImageNet', help='Directory with the ImageNet Parquet files')
    parser.add_argument('--checkpoint_in', default=os.path.join(
        'results', 'checkpoints', 'htrn_stage1', 'best_checkpoint_clean.pt'), help='Stage-1 checkpoint')
    parser.add_argument('--checkpoint_dir', default=os.path.join(
        'results', 'checkpoints', 'htrn_finetuned'))
    parser.add_argument('--resume', action='store_true', help='Resume from latest_checkpoint.pt')

    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=715)
    parser.add_argument('--iterations', '-T', type=int, default=5, help='Feedback iterations per forward pass')
    parser.add_argument('--tau', type=float, default=1.0)
    parser.add_argument('--fb_filters', type=int, default=256, help='Size of the feedback state')

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr_backbone', type=float, default=1e-5, help='Learning rate of backbone and fc')
    parser.add_argument('--lr_feedback', type=float, default=1e-4, help='Learning rate of the feedback modules')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)

    parser.add_argument('--batch_size', type=int, default=64, help='Micro-batch size per step')
    parser.add_argument('--accumulation_steps', type=int, default=2, help='Effective batch = batch_size * this')
    parser.add_argument('--workers', type=int, default=4)
    return parser.parse_args()


def main():
    """Fine-tune the network and checkpoint after every epoch."""
    args = parse_args()
    setup_logging()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.checkpoint_dir, 'latest_checkpoint.pt')
    best_clean_path = os.path.join(args.checkpoint_dir, 'best_checkpoint_clean.pt')
    best_occluded_path = os.path.join(args.checkpoint_dir, 'best_checkpoint_occluded.pt')

    torch.set_float32_matmul_precision('high')

    train_dataset = load_dataset('parquet', data_dir=args.data_dir, split='train')
    val_dataset = load_dataset('parquet', data_dir=args.data_dir, split='val')
    train_dataset.set_transform(transform_train_fn)
    val_dataset.set_transform(transform_val_clean_fn)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, collate_fn=collate_fn,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True, collate_fn=collate_fn)

    model = ResNet50(fb_filters=args.fb_filters, n_classes=1000, shape=(224, 224), tau=args.tau,
                     freeze_backbone=False)  # whole network co-adapts

    if not os.path.exists(args.checkpoint_in):
        raise FileNotFoundError(f"Initial checkpoint '{args.checkpoint_in}' not found")
    ckpt = torch.load(args.checkpoint_in, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    log.info('Loaded stage-1 weights (clean top-1 %.2f%%, occluded top-1 %.2f%%)',
             ckpt.get('best_clean_top1', ckpt.get('clean_top1', 0.0)),
             ckpt.get('best_occluded_top1', ckpt.get('occluded_top1', 0.0)))
    model.to(device)

    # Feedback modules and backbone/head get different learning rates
    feedback_params, backbone_params = [], []
    for name, param in model.named_parameters():
        param.requires_grad = True
        if any(kw in name for kw in FEEDBACK_KEYWORDS):
            feedback_params.append(param)
        else:
            backbone_params.append(param)
    log.info('Backbone + head: %s params (lr %g) | feedback: %s params (lr %g)',
             f'{sum(p.numel() for p in backbone_params):,}', args.lr_backbone,
             f'{sum(p.numel() for p in feedback_params):,}', args.lr_feedback)

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': args.lr_backbone},
        {'params': feedback_params, 'lr': args.lr_feedback},
    ], weight_decay=args.weight_decay)

    # No warm-up: the weights are already converged
    steps_per_epoch = math.ceil(len(train_loader) / args.accumulation_steps)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * steps_per_epoch, eta_min=1e-7)

    scaler = torch.amp.GradScaler('cuda')
    start_epoch = 0
    best_clean_top1 = 0.0
    best_occluded_top1 = 0.0

    if args.resume and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_clean_top1 = checkpoint.get('best_clean_top1', 0.0)
        best_occluded_top1 = checkpoint.get('best_occluded_top1', 0.0)
        log.info('Resumed at epoch %d (best clean %.2f%%, best occluded %.2f%%)',
                 start_epoch + 1, best_clean_top1, best_occluded_top1)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()
        optimizer.zero_grad()

        for i, (imgs, targets) in enumerate(train_loader):
            imgs, targets = imgs.to(device), targets.to(device)

            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(imgs, T=args.iterations)
                loss = loss_fn(outputs, targets) / args.accumulation_steps
            scaler.scale(loss).backward()

            if (i + 1) % args.accumulation_steps == 0 or (i + 1) == len(train_loader):
                # Unscale before clipping so the norm is computed on the true gradients
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            epoch_loss += loss.item() * args.accumulation_steps
            if (i + 1) % 200 == 0:
                log.info('Epoch %d/%d step %d/%d lr %.6f loss %.4f', epoch + 1, args.epochs, i + 1,
                         len(train_loader), optimizer.param_groups[0]['lr'],
                         loss.item() * args.accumulation_steps)

        log.info('Epoch %d done: train loss %.4f (%.0fs)', epoch + 1, epoch_loss / len(train_loader),
                 time.time() - start_time)
        torch.cuda.empty_cache()

        val_dataset.set_transform(transform_val_clean_fn)
        clean_top1, clean_top5 = run_evaluation(model, val_loader, device, args.iterations)
        val_dataset.set_transform(transform_val_occluded_fn)
        occluded_top1, occluded_top5 = run_deterministic_occluded_evaluation(
            model, val_loader, device, args.iterations, eval_seed=715)
        log.info('Clean top-1/top-5: %.2f / %.2f | Occluded top-1/top-5: %.2f / %.2f',
                 clean_top1, clean_top5, occluded_top1, occluded_top5)

        checkpoint_state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'clean_top1': clean_top1,
            'occluded_top1': occluded_top1,
            'best_clean_top1': max(clean_top1, best_clean_top1),
            'best_occluded_top1': max(occluded_top1, best_occluded_top1),
        }
        torch.save(checkpoint_state, checkpoint_path)

        if clean_top1 > best_clean_top1:
            best_clean_top1 = clean_top1
            torch.save(checkpoint_state, best_clean_path)
            log.info('New best clean top-1: %.2f%%', best_clean_top1)
        if occluded_top1 > best_occluded_top1:
            best_occluded_top1 = occluded_top1
            torch.save(checkpoint_state, best_occluded_path)
            log.info('New best occluded top-1: %.2f%%', best_occluded_top1)

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
