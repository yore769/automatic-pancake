"""Distributed training helpers."""

import torch
import torch.distributed as dist


def is_dist_available_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    if not is_dist_available_and_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    if not is_dist_available_and_initialized():
        return 1
    return dist.get_world_size()


def reduce_dict(input_dict: dict, average: bool = True) -> dict:
    """Reduce the values in *input_dict* across all processes.

    In single-GPU training this is a no-op.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = sorted(input_dict.keys())
        values = torch.stack([input_dict[k] for k in names])
        dist.all_reduce(values)
        if average:
            values /= world_size
        return {k: v for k, v in zip(names, values)}
