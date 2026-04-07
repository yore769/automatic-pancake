"""Data loading utilities."""

import copy
import torch
from torch.utils.data import DataLoader, DistributedSampler

from .coco import CocoDetection
from .transforms import (
    Compose, BatchImageCollateFuncion,
    Resize, RandomHorizontalFlip, RandomPhotometricDistort,
    RandomZoomOut, RandomIoUCrop, SanitizeBoundingBoxes,
    ConvertPILImage, ConvertBoxes,
)

# Transform name -> class mapping
TRANSFORM_REGISTRY = {
    'Resize': Resize,
    'RandomHorizontalFlip': RandomHorizontalFlip,
    'RandomPhotometricDistort': RandomPhotometricDistort,
    'RandomZoomOut': RandomZoomOut,
    'RandomIoUCrop': RandomIoUCrop,
    'SanitizeBoundingBoxes': SanitizeBoundingBoxes,
    'ConvertPILImage': ConvertPILImage,
    'ConvertBoxes': ConvertBoxes,
    'Compose': Compose,
}


def _build_transforms(transforms_cfg: dict):
    """Build a Compose transform from config dict."""
    if transforms_cfg is None:
        return None
    ops_cfg = transforms_cfg.get('ops') or []
    policy = transforms_cfg.get('policy', None)
    ops = []
    for op_cfg in (ops_cfg or []):
        op_cfg = dict(op_cfg)
        op_type = op_cfg.pop('type')
        cls = TRANSFORM_REGISTRY[op_type]
        ops.append(cls(**op_cfg))
    return Compose(ops=ops, policy=policy)


def _build_collate_fn(collate_cfg: dict):
    """Build a collate function from config dict."""
    if not collate_cfg:
        return None
    collate_cfg = dict(collate_cfg)
    cls_name = collate_cfg.pop('type', 'BatchImageCollateFuncion')
    if cls_name == 'BatchImageCollateFuncion':
        return BatchImageCollateFuncion(**collate_cfg)
    raise ValueError(f"Unknown collate_fn type: {cls_name}")


def build_dataloader(cfg: dict, split: str = 'train', num_classes: int = 80) -> DataLoader:
    """Build a DataLoader from config dict."""
    cfg = copy.deepcopy(cfg)
    dataset_cfg = cfg.pop('dataset', {})
    collate_cfg = cfg.pop('collate_fn', {})
    shuffle = cfg.pop('shuffle', split == 'train')
    total_batch_size = cfg.pop('total_batch_size', 4)
    num_workers = cfg.pop('num_workers', 2)
    drop_last = cfg.pop('drop_last', split == 'train')
    _ = cfg.pop('type', None)

    # Build transforms
    transforms_cfg = (dataset_cfg or {}).pop('transforms', {})
    transforms = _build_transforms(transforms_cfg)

    dataset_type = (dataset_cfg or {}).pop('type', 'CocoDetection')
    ds_kwargs = {k: v for k, v in (dataset_cfg or {}).items()}
    ds_kwargs['transforms'] = transforms

    if dataset_type == 'CocoDetection':
        dataset = CocoDetection(**ds_kwargs)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    # Distributed sampler
    from src.misc.dist_utils import get_world_size, get_rank
    world_size = get_world_size()
    rank = get_rank()
    batch_size = max(1, total_batch_size // world_size)

    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                      shuffle=shuffle, drop_last=drop_last)
        shuffle = False

    collate_fn = _build_collate_fn(collate_cfg)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )


__all__ = ['build_dataloader', 'CocoDetection']
