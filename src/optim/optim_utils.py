"""Optimizer utilities: build optimizer, LR scheduler, and warmup scheduler."""

import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR


# -----------------------------------------------------------------------
# Optimizer
# -----------------------------------------------------------------------

def build_optimizer(cfg: dict, model: nn.Module) -> optim.Optimizer:
    """Build an optimizer from config dict."""
    cfg = dict(cfg)
    opt_type = cfg.pop('type', 'AdamW')
    param_groups_cfg = cfg.pop('params', [])

    # Build param groups with per-group lr / weight_decay
    param_groups = _build_param_groups(model, param_groups_cfg, cfg)

    opt_cls = getattr(optim, opt_type)
    return opt_cls(param_groups, **cfg)


def _build_param_groups(model: nn.Module, groups_cfg: list, base_cfg: dict) -> list:
    """Build parameter groups for the optimizer."""
    base_lr = base_cfg.get('lr', 1e-4)
    base_wd = base_cfg.get('weight_decay', 1e-4)

    if not groups_cfg:
        return [{'params': model.parameters(), 'lr': base_lr, 'weight_decay': base_wd}]

    all_param_names = {name for name, _ in model.named_parameters()}
    assigned = set()
    groups = []

    for g in groups_cfg:
        g = dict(g)
        pattern = g.pop('params')
        regex = re.compile(pattern)
        matched = {
            name: param
            for name, param in model.named_parameters()
            if regex.search(name) and name not in assigned
        }
        if matched:
            group = {
                'params': list(matched.values()),
                'lr': g.get('lr', base_lr),
                'weight_decay': g.get('weight_decay', base_wd),
            }
            groups.append(group)
            assigned.update(matched.keys())

    # Remaining params
    remaining = [p for name, p in model.named_parameters() if name not in assigned]
    if remaining:
        groups.append({'params': remaining, 'lr': base_lr, 'weight_decay': base_wd})

    return groups


# -----------------------------------------------------------------------
# LR Schedulers
# -----------------------------------------------------------------------

class LinearWarmup:
    """Linear LR warmup over warmup_duration steps."""

    def __init__(self, optimizer: optim.Optimizer, warmup_duration: int):
        self.optimizer = optimizer
        self.warmup_duration = warmup_duration
        self._step = 0
        self._base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        # Start near zero
        self._set_lr(0.0)

    def _set_lr(self, factor: float):
        for pg, base_lr in zip(self.optimizer.param_groups, self._base_lrs):
            pg['lr'] = base_lr * factor

    def step(self):
        self._step += 1
        if self._step <= self.warmup_duration:
            factor = self._step / self.warmup_duration
            self._set_lr(factor)

    @property
    def finished(self) -> bool:
        return self._step >= self.warmup_duration


def build_lr_scheduler(cfg: dict, optimizer: optim.Optimizer):
    """Build a learning rate scheduler from config dict."""
    cfg = dict(cfg)
    sched_type = cfg.pop('type')
    if sched_type == 'MultiStepLR':
        return MultiStepLR(optimizer, **cfg)
    elif sched_type == 'CosineAnnealingLR':
        return CosineAnnealingLR(optimizer, **cfg)
    else:
        raise ValueError(f"Unknown lr_scheduler type: {sched_type}")


def build_lr_warmup_scheduler(cfg: dict, optimizer: optim.Optimizer):
    """Build a LR warmup scheduler from config dict."""
    cfg = dict(cfg)
    sched_type = cfg.pop('type', 'LinearWarmup')
    if sched_type == 'LinearWarmup':
        return LinearWarmup(optimizer, **cfg)
    else:
        raise ValueError(f"Unknown lr_warmup_scheduler type: {sched_type}")
