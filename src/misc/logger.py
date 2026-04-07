"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
import sys
import datetime
import torch


__all__ = ['MetricLogger', 'SmoothedValue', 'setup_default_logging']


class SmoothedValue:
    """Track a series of values and provide access to smoothed values."""

    def __init__(self, window_size=20, fmt='{median:.4f} ({global_avg:.4f})'):
        self.deque = []
        self.window_size = window_size
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        if len(self.deque) > self.window_size:
            self.deque.pop(0)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        return sorted(self.deque)[len(self.deque) // 2] if self.deque else 0.0

    @property
    def avg(self):
        return sum(self.deque) / len(self.deque) if self.deque else 0.0

    @property
    def global_avg(self):
        return self.total / self.count if self.count else 0.0

    @property
    def max(self):
        return max(self.deque) if self.deque else 0.0

    @property
    def value(self):
        return self.deque[-1] if self.deque else 0.0

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger:
    def __init__(self, delimiter='\t'):
        self.meters = {}
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            if k not in self.meters:
                self.meters[k] = SmoothedValue(window_size=20, fmt='{value:.4f}')
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.__dict__.get('meters', {}):
            return self.meters[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        parts = [f'{k}: {str(v)}' for k, v in self.meters.items()]
        return self.delimiter.join(parts)

    def log_every(self, iterable, print_freq, header=''):
        i = 0
        start_time = datetime.datetime.now()
        for obj in iterable:
            yield obj
            i += 1
            if i % print_freq == 0 or i == len(iterable):
                elapsed = (datetime.datetime.now() - start_time).total_seconds()
                print(f'{header} [{i}/{len(iterable)}]  '
                      f'elapsed: {elapsed:.1f}s  {str(self)}')


def setup_default_logging(output_dir=None):
    pass
