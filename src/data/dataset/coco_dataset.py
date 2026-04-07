"""Dataset implementations for RT-DETR training.

Supports COCO-format datasets (including VisDrone converted to COCO format).
"""

import os
import json
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image

__all__ = ['CocoDetection']


class CocoDetection(Dataset):
    """COCO-format detection dataset.

    Args:
        img_folder:  path to image directory
        ann_file:    path to COCO-format annotation JSON
        transforms:  callable transform applied to (image, target) pairs
        remap_mscoco_category: whether to remap MS-COCO 91-class IDs to 80
    """

    MSCOCO_91_TO_80 = {
        1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9,
        11: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17,
        20: 18, 21: 19, 22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25,
        31: 26, 32: 27, 33: 28, 34: 29, 35: 30, 36: 31, 37: 32, 38: 33,
        39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39, 46: 40, 47: 41,
        48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49,
        56: 50, 57: 51, 58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57,
        64: 58, 65: 59, 67: 60, 70: 61, 72: 62, 73: 63, 74: 64, 75: 65,
        76: 66, 77: 67, 78: 68, 79: 69, 80: 70, 81: 71, 82: 72, 84: 73,
        85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79,
    }

    def __init__(self, img_folder, ann_file, transforms=None,
                 remap_mscoco_category=False):
        from faster_coco_eval import COCO as CocoAPI
        self.img_folder = Path(img_folder)
        self.coco = CocoAPI(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms
        self.remap = remap_mscoco_category

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.imgs[img_id]
        img_path = self.img_folder / img_info['file_name']
        img = Image.open(img_path).convert('RGB')

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        for ann in anns:
            if ann.get('iscrowd', 0):
                continue
            x, y, w, h = ann['bbox']
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            cat_id = ann['category_id']
            if self.remap:
                cat_id = self.MSCOCO_91_TO_80.get(cat_id, cat_id)
            labels.append(cat_id)

        target = {
            'image_id': torch.tensor([img_id]),
            'boxes': torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            'labels': torch.as_tensor(labels, dtype=torch.long),
            'orig_size': torch.tensor([img.height, img.width]),
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target
