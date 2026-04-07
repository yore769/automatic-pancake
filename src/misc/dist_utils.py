"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
import sys
import time
import datetime
import functools
import torch
import torch.distributed as dist


__all__ = ['setup_distributed', 'cleanup', 'is_dist_available_and_initialized',
           'get_rank', 'get_world_size', 'is_main_process', 'save_on_master',
           'all_reduce_dict', 'reduce_dict']


def setup_distributed(print_rank=0, print_method='builtin', seed=None):
    """Initialize distributed training if available."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))

        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank,
        )
        dist.barrier()
    elif seed is not None:
        torch.manual_seed(seed)


def cleanup():
    if is_dist_available_and_initialized():
        dist.destroy_process_group()


def is_dist_available_and_initialized():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    if not is_dist_available_and_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    if not is_dist_available_and_initialized():
        return 1
    return dist.get_world_size()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def reduce_dict(input_dict, average=True):
    """Reduce a dict of tensors from all processes."""
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = sorted(input_dict.keys())
        values = torch.stack([input_dict[k] for k in names])
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced = {k: v for k, v in zip(names, values)}
    return reduced


def all_reduce_dict(input_dict):
    return reduce_dict(input_dict, average=False)
