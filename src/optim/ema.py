"""Exponential Moving Average of model weights."""

import math
import copy
import torch
import torch.nn as nn


class ModelEMA:
    """
    Model Exponential Moving Average.
    Maintains a shadow copy of model weights updated with EMA.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999, warmups: int = 2000):
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.warmups = warmups
        self.updates = 0

    def _get_decay(self) -> float:
        # Ramp up decay during warmup
        return self.decay * (1 - math.exp(-self.updates / self.warmups))

    @torch.no_grad()
    def update(self, model: nn.Module):
        self.updates += 1
        d = self._get_decay()
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach() * (1 - d))

    def state_dict(self):
        return {'module': self.module.state_dict(), 'updates': self.updates}

    def load_state_dict(self, state_dict: dict):
        self.module.load_state_dict(state_dict['module'])
        self.updates = state_dict.get('updates', 0)
