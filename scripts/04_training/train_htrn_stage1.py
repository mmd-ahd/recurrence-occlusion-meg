"""Stage 1 of HTRN training: learn the top-down feedback pathway on ImageNet-1k.

A ResNet-50 with the 256-D feedback loop (``dfbmodels.models.resnet.ResNet50``) is initialised
from torchvision's IMAGENET1K_V2 weights. The convolutional backbone is frozen; the feedback
modules and the ``fc`` readout are trained with AdamW (linear warm-up, cosine decay). After every
epoch the model is validated on clean images and on images with synthetic occlusion (random
erasing).

ImageNet is read from local Parquet files (``--data_dir`` with ``train`` and ``val`` splits).
Checkpoints: ``latest_checkpoint.pt``, ``best_checkpoint_clean.pt`` and ``best_checkpoint_occluded.pt``
under ``--checkpoint_dir``. The paper's ``DFM_only_checkpoint.pt`` comes from this stage; stage 2
(``train_htrn_finetune.py``) produces ``finetune_clean_checkpoint.pt``.

Usage:
    python scripts/04_training/train_htrn_stage1.py --data_dir ImageNet --checkpoint_dir results/checkpoints/htrn_stage1 [--resume]
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
from torchvision.models import ResNet50_Weights

from dfbmodels.models.resnet import ResNet50
from htrn import setup_logging
from htrn.imagenet import (collate_fn, run_evaluation, transform_train_fn, transform_val_clean_fn,
                           transform_val_occluded_fn)

log = logging.getLogger(__name__)


def parse_args():
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])

    parser.add_argument('--data_dir', default='ImageNet', help='Directory with the ImageNet Parquet files')
    parser.add_argument('--checkpoint_dir', default=os.path.join('results', 'checkpoints', 'htrn_stage1'))
    parser.add_argument('--resume', action='store_true', help='Resume from latest_checkpoint.pt')

    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=715)
    parser.add_argument('--iterations', '-T', type=int, default=5, help='Feedback iterations per forward pass')
    parser.add_argument('--tau', type=float, default=1.0)
    parser.add_argument('--fb_filters', type=int, default=256, help='Size of the feedback state')

    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--warmup_epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=5e-3, help='Learning rate of the trainable modules')
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    parser.add_argument('--batch_size', type=int, default=64, help='Micro-batch size per step')
    parser.add_argument('--accumulation_steps', type=int, default=2, help='Effective batch = batch_size * this')
    parser.add_argument('--workers', type=int, default=4)
    return parser.parse_args()


def main():
    """Train the feedback modules and checkpoint after every epoch."""
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
                     freeze_backbone=True)  # only the feedback modules and fc are trained

    # Initialise the backbone from torchvision's IMAGENET1K_V2 ResNet-50
    pretrained_state = ResNet50_Weights.IMAGENET1K_V2.get_state_dict(progress=True)
    model_state = model.state_dict()
    matching = {k: v for k, v in pretrained_state.items()
                if k in model_state and model_state[k].shape == v.shape}
    model_state.update(matching)
    model.load_state_dict(model_state, strict=True)
    log.info('Loaded %d pretrained backbone tensors', len(matching))
    model.to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    log.info('Parameters: %s total, %s trainable',
             f'{sum(p.numel() for p in model.parameters()):,}',
             f'{sum(p.numel() for p in trainable_params):,}')

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    # Linear warm-up into cosine decay, stepped per optimizer step
    steps_per_epoch = math.ceil(len(train_loader) / args.accumulation_steps)
    warmup_steps = args.warmup_epochs * steps_per_epoch
    total_steps = args.epochs * steps_per_epoch
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps),
                    optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps,
                                                         eta_min=1e-6)],
        milestones=[warmup_steps])

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
        occluded_top1, occluded_top5 = run_evaluation(model, val_loader, device, args.iterations)
        log.info('Clean top-1/top-5: %.2f / %.2f | Occluded top-1/top-5: %.2f / %.2f',
                 clean_top1, clean_top5, occluded_top1, occluded_top5)

        checkpoint_state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
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
