"""Solver utilities: checkpoint saving/loading and metric tracking."""

import os
import math
import torch
import torch.nn as nn
from src.misc.dist_utils import is_main_process


def save_checkpoint(state: dict, output_dir: str, filename: str = 'checkpoint.pth'):
    """Save checkpoint to output_dir/filename."""
    if is_main_process():
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)
        torch.save(state, path)
    return os.path.join(output_dir, filename)


def load_checkpoint(path: str, model: nn.Module, optimizer=None, ema=None):
    """Load checkpoint from path, return epoch number."""
    ckpt = torch.load(path, map_location='cpu')
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    if optimizer is not None and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    if ema is not None and 'ema' in ckpt:
        ema.load_state_dict(ckpt['ema'])
    return ckpt.get('epoch', 0)


def load_tuning_state(path: str, model: nn.Module):
    """Load weights for tuning (partial load, ignoring size mismatches)."""
    ckpt = torch.load(path, map_location='cpu')
    state = ckpt.get('model', ckpt)
    model_state = model.state_dict()
    loaded = {}
    for k, v in state.items():
        if k in model_state and model_state[k].shape == v.shape:
            loaded[k] = v
        else:
            print(f"Skipping {k}: shape mismatch or not found")
    model_state.update(loaded)
    model.load_state_dict(model_state)
    print(f"Loaded {len(loaded)}/{len(state)} parameters from {path}")


class SmoothedValue:
    """Track a series of values and compute smoothed statistics."""

    def __init__(self, window_size: int = 20):
        from collections import deque
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1):
        self.deque.append(value)
        self.total += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return sum(self.deque) / max(len(self.deque), 1)

    @property
    def global_avg(self) -> float:
        return self.total / max(self.count, 1)

    def __str__(self) -> str:
        return f"{self.avg:.4f} ({self.global_avg:.4f})"


class MetricLogger:
    """Log and aggregate training metrics."""

    def __init__(self, delimiter: str = '  '):
        from collections import defaultdict
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.meters[k].update(v)

    def __str__(self) -> str:
        return self.delimiter.join(f"{k}: {v}" for k, v in self.meters.items())

    def log_every(self, iterable, print_freq: int, header: str = ''):
        i = 0
        import time
        start = time.time()
        for obj in iterable:
            yield obj
            i += 1
            if i % print_freq == 0:
                elapsed = time.time() - start
                print(f"{header} [{i}]  {self}  time: {elapsed/i:.3f}s")
