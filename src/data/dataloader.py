"""DataLoader builder for RT-DETR training."""

import torch
from torch.utils.data import DataLoader, DistributedSampler

from src.data.dataset import CocoDetection
from src.data.transforms import Compose
from src.core.workspace import create as ws_create
from src.misc.dist_utils import is_dist_available_and_initialized

__all__ = ['build_dataloader', 'BatchImageCollateFuncion']


def _build_transforms(transforms_cfg):
    """Build transform pipeline from config."""
    ops_cfg = transforms_cfg.get('ops', [])
    from src.data import transforms as T
    ops = []
    for op_cfg in ops_cfg:
        op_cfg = op_cfg.copy()
        op_type = op_cfg.pop('type')
        op_cls = getattr(T, op_type)
        ops.append(op_cls(**op_cfg))
    return Compose(ops)


class BatchImageCollateFuncion:
    """Collate function supporting multi-scale training."""

    def __init__(self, scales=None, stop_epoch=None):
        self.scales = scales
        self.stop_epoch = stop_epoch
        self._epoch = 0

    def set_epoch(self, epoch):
        self._epoch = epoch

    def __call__(self, batch):
        imgs, targets = zip(*batch)
        imgs = torch.stack(imgs, dim=0)
        return imgs, list(targets)


def build_dataloader(dataset_cfg, loader_cfg, global_cfg):
    """Build a DataLoader from config dicts."""
    img_folder = dataset_cfg.get('img_folder', '')
    ann_file = dataset_cfg.get('ann_file', '')
    remap = global_cfg.get('remap_mscoco_category', False)
    transforms_cfg = dataset_cfg.get('transforms', {})
    transforms = _build_transforms(transforms_cfg)

    dataset = CocoDetection(
        img_folder=img_folder,
        ann_file=ann_file,
        transforms=transforms,
        remap_mscoco_category=remap,
    )

    shuffle = loader_cfg.get('shuffle', False)
    batch_size = loader_cfg.get('total_batch_size', 4)
    num_workers = loader_cfg.get('num_workers', 2)

    collate_cfg = loader_cfg.get('collate_fn', {})
    collate_type = collate_cfg.pop('type', 'BatchImageCollateFuncion') \
        if isinstance(collate_cfg, dict) else 'BatchImageCollateFuncion'
    collate_fn = BatchImageCollateFuncion(
        **collate_cfg) if isinstance(collate_cfg, dict) else BatchImageCollateFuncion()

    if is_dist_available_and_initialized():
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False
    else:
        sampler = None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
