"""Model EMA (Exponential Moving Average) for RT-DETR training."""

import copy
import torch
import torch.nn as nn

from src.core.workspace import register

__all__ = ['ModelEMA']


@register
class ModelEMA:
    """Maintains an exponential moving average of model parameters.

    Args:
        model:   the model to track
        decay:   EMA decay factor (e.g. 0.9999)
        warmups: number of update steps before reaching target decay
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999,
                 warmups: int = 2000):
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.warmups = warmups
        self._updates = 0

    def update(self, model: nn.Module):
        self._updates += 1
        d = self.decay * (1 - (1 - self.decay) ** (
            self._updates / max(self._updates, self.warmups)))
        with torch.no_grad():
            for ema_p, model_p in zip(self.module.parameters(),
                                       model.parameters()):
                ema_p.copy_(d * ema_p + (1 - d) * model_p.data)
            for ema_b, model_b in zip(self.module.buffers(),
                                       model.buffers()):
                ema_b.copy_(model_b.data)

    def state_dict(self):
        return {
            'module': self.module.state_dict(),
            'updates': self._updates,
        }

    def load_state_dict(self, state):
        self.module.load_state_dict(state['module'])
        self._updates = state.get('updates', 0)
