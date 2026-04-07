"""Detection solver for RT-DETR training and evaluation."""

import os
import sys
import re
import json
import time
import datetime
import torch
import torch.nn as nn
from pathlib import Path

from src.misc.dist_utils import (
    is_main_process, get_rank, get_world_size,
    reduce_dict, save_on_master,
)
from src.misc.logger import MetricLogger

__all__ = ['DetSolver']

# Pattern to identify auxiliary/encoder/denoising loss keys (excluded from step log)
_AUX_PAT = re.compile(r'.*(_aux_\d+|_enc|_dn_\d+)$')


class DetSolver:
    """Manages training and evaluation for detection models."""

    def __init__(self, cfg):
        self.cfg = cfg

        # Build components
        self.model = cfg.model
        self.criterion = cfg.criterion
        self.postprocessor = cfg.postprocessor
        self.optimizer = cfg.optimizer
        self.lr_scheduler = cfg.lr_scheduler
        self.lr_warmup_scheduler = cfg.lr_warmup_scheduler

        self.train_dataloader = cfg.train_dataloader
        self.val_dataloader = cfg.val_dataloader
        self.ema = cfg.ema

        yaml_cfg = cfg.yaml_cfg
        self.epoches = yaml_cfg.get('epoches', 72)
        self.output_dir = Path(yaml_cfg.get('output_dir', './output'))
        self.summary_dir = yaml_cfg.get('summary_dir', None)
        self.clip_max_norm = yaml_cfg.get('clip_max_norm', 0.1)
        self.use_amp = yaml_cfg.get('use_amp', False)
        self.device = yaml_cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = yaml_cfg.get('seed', None)

        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        self.model.to(self.device)
        self.criterion.to(self.device)

        self.start_epoch = 0
        self.global_step = 0

        if is_main_process():
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.log_file = open(self.output_dir / 'log.txt', 'a')
        else:
            self.log_file = None

    def resume(self, path):
        ckpt = torch.load(path, map_location='cpu')
        self.model.load_state_dict(ckpt['model'])
        if self.ema and 'ema' in ckpt:
            self.ema.load_state_dict(ckpt['ema'])
        if 'optimizer' in ckpt:
            self.optimizer.load_state_dict(ckpt['optimizer'])
        if 'lr_scheduler' in ckpt:
            self.lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
        self.start_epoch = ckpt.get('epoch', 0) + 1
        self.global_step = ckpt.get('global_step', 0)
        print(f'Resumed from {path}, epoch={self.start_epoch}')

    def fit(self):
        print(f'Starting training for {self.epoches} epochs')
        start_time = time.time()

        for epoch in range(self.start_epoch, self.epoches):
            self._train_epoch(epoch)
            stats = self._val_epoch(epoch)
            self._save_checkpoint(epoch, stats)

        total_time = time.time() - start_time
        print(f'Training completed in {datetime.timedelta(seconds=int(total_time))}')

    def _train_epoch(self, epoch):
        self.model.train()
        self.criterion.train()

        metric_logger = MetricLogger()

        if hasattr(self.train_dataloader.sampler, 'set_epoch'):
            self.train_dataloader.sampler.set_epoch(epoch)

        for samples, targets in self.train_dataloader:
            samples = samples.to(self.device)
            targets = [{k: v.to(self.device) for k, v in t.items()}
                       for t in targets]

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(samples, targets)
                loss_dict = self.criterion(outputs, targets)
                total_loss = sum(loss_dict.values())

            if self.use_amp:
                self.scaler.scale(total_loss).backward()
                if self.clip_max_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.clip_max_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                if self.clip_max_norm > 0:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.clip_max_norm)
                self.optimizer.step()

            if self.lr_warmup_scheduler is not None:
                self.lr_warmup_scheduler.step()

            if self.ema is not None:
                self.ema.update(self.model)

            loss_dict_reduced = reduce_dict(
                {k: v.detach() for k, v in loss_dict.items()})
            metric_logger.update(
                train_loss=sum(loss_dict_reduced.values()).item(),
                **{k: v.item() for k, v in loss_dict_reduced.items()
                   if not _AUX_PAT.match(k)})

            self.global_step += 1

        self.lr_scheduler.step()

        if is_main_process():
            log_entry = {
                'epoch': epoch,
                'train_loss': metric_logger.meters['train_loss'].global_avg,
            }
            for k, v in metric_logger.meters.items():
                if k != 'train_loss':
                    log_entry[k] = v.global_avg
            if self.log_file:
                self.log_file.write(json.dumps(log_entry) + '\n')
                self.log_file.flush()
            print(f'Epoch {epoch}: {log_entry}')

    def _val_epoch(self, epoch):
        model = self.ema.module if self.ema else self.model
        model.eval()

        results = {}
        try:
            from faster_coco_eval import COCO as CocoAPI
            from faster_coco_eval.core.faster_eval_api import COCOeval_faster as COCOeval
            coco_gt = self.val_dataloader.dataset.coco
            coco_dt_list = []

            with torch.no_grad():
                for samples, targets in self.val_dataloader:
                    samples = samples.to(self.device)
                    orig_sizes = torch.stack(
                        [t['orig_size'] for t in targets]).to(self.device)
                    outputs = model(samples)
                    preds = self.postprocessor(outputs, orig_sizes)

                    for pred, tgt in zip(preds, targets):
                        img_id = tgt['image_id'].item()
                        for box, score, label in zip(
                                pred['boxes'], pred['scores'], pred['labels']):
                            x1, y1, x2, y2 = box.cpu().tolist()
                            coco_dt_list.append({
                                'image_id': img_id,
                                'category_id': int(label.item()) + 1,
                                'bbox': [x1, y1, x2 - x1, y2 - y1],
                                'score': float(score.item()),
                            })

            if coco_dt_list:
                coco_dt = coco_gt.loadRes(coco_dt_list)
                coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
                stats = coco_eval.stats.tolist()
            else:
                stats = [0.0] * 12

        except Exception as e:
            print(f'Validation error: {e}')
            stats = [0.0] * 12

        if is_main_process() and self.log_file:
            log_entry = {
                'epoch': epoch,
                'test_coco_eval_bbox': stats,
            }
            self.log_file.write(json.dumps(log_entry) + '\n')
            self.log_file.flush()
            print(f'Epoch {epoch} eval: AP={stats[0]:.4f}')

        return stats

    def _save_checkpoint(self, epoch, stats):
        if not is_main_process():
            return
        ckpt = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'epoch': epoch,
            'global_step': self.global_step,
        }
        if self.ema:
            ckpt['ema'] = self.ema.state_dict()
        save_on_master(ckpt, self.output_dir / f'checkpoint{epoch:04d}.pth')
        save_on_master(ckpt, self.output_dir / 'checkpoint_last.pth')

    def val(self):
        stats = self._val_epoch(epoch=0)
        print(f'Validation stats: {stats}')
