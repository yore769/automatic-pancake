"""Detection solver: main training and validation loop."""

import os
import time
import math
import torch
import torch.nn as nn
import torch.distributed as dist
from contextlib import nullcontext

from src.misc.dist_utils import is_main_process, reduce_dict, get_world_size
from .solver_utils import (
    save_checkpoint, load_checkpoint, load_tuning_state,
    MetricLogger, SmoothedValue
)


class DetSolver:
    """
    Solver for RT-DETRv2 detection training.
    Wraps model, optimizer, scheduler, and evaluation.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        # Build components lazily via cfg properties
        self.model = cfg.model.to(self.device)
        self.criterion = cfg.criterion.to(self.device)
        self.postprocessor = cfg.postprocessor

        # Distributed
        world_size = get_world_size()
        if world_size > 1:
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[int(os.environ.get('LOCAL_RANK', 0))],
                find_unused_parameters=cfg.yaml_cfg.get('find_unused_parameters', False),
            )
            if cfg.yaml_cfg.get('sync_bn', False):
                self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

        self.optimizer = cfg.optimizer
        self.lr_scheduler = cfg.lr_scheduler
        self.lr_warmup_scheduler = cfg.lr_warmup_scheduler
        self.ema = cfg.ema
        self.scaler = cfg.scaler

        self.output_dir = cfg.yaml_cfg.get('output_dir', './output')
        self.print_freq = cfg.yaml_cfg.get('print_freq', 100)
        self.checkpoint_freq = cfg.yaml_cfg.get('checkpoint_freq', 1)
        self.epoches = cfg.yaml_cfg.get('epoches', 72)
        self.clip_max_norm = cfg.yaml_cfg.get('clip_max_norm', 0.1)
        self.use_amp = cfg.yaml_cfg.get('use_amp', False)
        self.eval_spatial_size = cfg.yaml_cfg.get('eval_spatial_size', [640, 640])

        self.last_epoch = 0

        # Handle resume / tuning
        resume = cfg.yaml_cfg.get('resume', None)
        tuning = cfg.yaml_cfg.get('tuning', None)
        if resume:
            self.last_epoch = load_checkpoint(
                resume, self._get_model(), self.optimizer, self.ema
            )
        elif tuning:
            load_tuning_state(tuning, self._get_model())

    def _get_model(self) -> nn.Module:
        """Return unwrapped model."""
        if isinstance(self.model, nn.parallel.DistributedDataParallel):
            return self.model.module
        return self.model

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self):
        """Run full training loop."""
        train_loader = self.cfg.train_dataloader
        val_loader = self.cfg.val_dataloader

        for epoch in range(self.last_epoch, self.epoches):
            if hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)
            # Propagate epoch to collate_fn for multi-scale stop
            if hasattr(train_loader.collate_fn, 'set_epoch'):
                train_loader.collate_fn.set_epoch(epoch)

            self._train_one_epoch(epoch, train_loader)
            self.lr_scheduler.step()

            if is_main_process():
                if (epoch + 1) % self.checkpoint_freq == 0:
                    self._save(epoch)

        # Final validation
        self.val()

    def _train_one_epoch(self, epoch: int, loader):
        self.model.train()
        self.criterion.train()
        metric_logger = MetricLogger()

        amp_ctx = (
            torch.cuda.amp.autocast() if self.use_amp and torch.cuda.is_available()
            else nullcontext()
        )

        for i, (samples, targets) in enumerate(loader):
            # Move to device
            samples = samples.to(self.device)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            # Warmup step
            if self.lr_warmup_scheduler is not None and not self.lr_warmup_scheduler.finished:
                self.lr_warmup_scheduler.step()

            with amp_ctx:
                outputs = self.model(samples, targets=targets)
                loss_dict = self.criterion(outputs, targets)
                weight_dict = self.criterion.weight_dict
                total_loss = sum(
                    loss_dict[k] * weight_dict[k]
                    for k in loss_dict if k in weight_dict
                )

            self.optimizer.zero_grad()
            if self.scaler is not None:
                self.scaler.scale(total_loss).backward()
                if self.clip_max_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_max_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                if self.clip_max_norm > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_max_norm)
                self.optimizer.step()

            if self.ema is not None:
                self.ema.update(self._get_model())

            # Logging
            loss_dict_reduced = reduce_dict({k: v.detach() for k, v in loss_dict.items()})
            total_loss_reduced = sum(
                loss_dict_reduced[k] * weight_dict.get(k, 1.0)
                for k in loss_dict_reduced if k in weight_dict
            )
            metric_logger.update(loss=total_loss_reduced, **loss_dict_reduced)

            if i % self.print_freq == 0 and is_main_process():
                lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch [{epoch}][{i}/{len(loader)}]  {metric_logger}  lr: {lr:.6f}")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def val(self):
        """Run validation."""
        eval_model = self.ema.module if self.ema else self._get_model()
        eval_model.eval()

        val_loader = self.cfg.val_dataloader
        evaluator = self.cfg.evaluator

        with torch.no_grad():
            for samples, targets in val_loader:
                samples = samples.to(self.device)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                outputs = eval_model(samples)
                orig_sizes = torch.stack([t['orig_size'] for t in targets], dim=0)
                results = self.postprocessor(outputs, orig_sizes)

                res = {t['image_id'].item(): r for t, r in zip(targets, results)}
                evaluator.update(res)

        evaluator.synchronize_between_processes()
        evaluator.accumulate()
        evaluator.summarize()

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def _save(self, epoch: int):
        state = {
            'epoch': epoch + 1,
            'model': self._get_model().state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        if self.ema:
            state['ema'] = self.ema.state_dict()
        if self.scaler:
            state['scaler'] = self.scaler.state_dict()
        save_checkpoint(state, self.output_dir, f'checkpoint{epoch:04d}.pth')
        save_checkpoint(state, self.output_dir, 'last.pth')
