"""Distributed training utilities."""

import os
import torch
import torch.distributed as dist


def setup_distributed(print_rank: int = 0, print_method: str = 'builtin', seed: int = None):
    """Initialize distributed training if RANK env var is set."""
    rank = int(os.environ.get('RANK', -1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    if rank >= 0 and world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
    else:
        rank = 0
        local_rank = 0
        world_size = 1

    if seed is not None:
        import random
        import numpy as np
        seed = seed + rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(local_rank)
    os.environ['WORLD_SIZE'] = str(world_size)


def cleanup():
    """Clean up distributed training."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def get_rank() -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()


def is_main_process() -> bool:
    return get_rank() == 0


def reduce_dict(input_dict: dict, average: bool = True) -> dict:
    """Reduce dict of tensors across all processes."""
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        keys = sorted(input_dict.keys())
        values = torch.stack([input_dict[k] for k in keys])
        dist.all_reduce(values)
        if average:
            values /= world_size
        return {k: v for k, v in zip(keys, values)}


def all_gather(data):
    """All-gather a list of data across all processes."""
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    output = [None] * world_size
    dist.all_gather_object(output, data)
    return output
