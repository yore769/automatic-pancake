"""Linear warmup LR scheduler."""

import torch.optim as optim
from src.core.workspace import register

__all__ = ['LinearWarmup']


@register
class LinearWarmup:
    """Linearly warm up LR from 0 to base_lr over ``warmup_duration`` steps."""

    def __init__(self, optimizer: optim.Optimizer, warmup_duration: int = 2000):
        self.optimizer = optimizer
        self.warmup_duration = warmup_duration
        self._step = 0
        self._base_lrs = [pg['lr'] for pg in optimizer.param_groups]

    def step(self):
        self._step += 1
        if self._step <= self.warmup_duration:
            scale = self._step / self.warmup_duration
            for pg, base_lr in zip(self.optimizer.param_groups, self._base_lrs):
                pg['lr'] = base_lr * scale

    def state_dict(self):
        return {'step': self._step, 'base_lrs': self._base_lrs}

    def load_state_dict(self, state):
        self._step = state['step']
        self._base_lrs = state['base_lrs']
