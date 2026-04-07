"""Data transforms for RT-DETR training.

Implements transforms compatible with torchvision.transforms.v2 API.
"""

import random
import torch
import torchvision.transforms.functional as TF
from PIL import Image

__all__ = [
    'RandomPhotometricDistort',
    'RandomZoomOut',
    'RandomIoUCrop',
    'RandomHorizontalFlip',
    'Resize',
    'SanitizeBoundingBoxes',
    'ConvertPILImage',
    'ConvertBoxes',
    'Compose',
]


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target


class RandomPhotometricDistort:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            img = TF.adjust_brightness(img, random.uniform(0.5, 1.5))
            img = TF.adjust_contrast(img, random.uniform(0.5, 1.5))
            img = TF.adjust_saturation(img, random.uniform(0.5, 1.5))
            img = TF.adjust_hue(img, random.uniform(-0.1, 0.1))
        return img, target


class RandomZoomOut:
    def __init__(self, fill=0, max_scale=4.0, p=0.5):
        self.fill = fill
        self.max_scale = max_scale
        self.p = p

    def __call__(self, img, target):
        if random.random() > self.p:
            return img, target
        W, H = img.size
        scale = random.uniform(1.0, self.max_scale)
        nW, nH = int(W * scale), int(H * scale)
        new_img = Image.new('RGB', (nW, nH), (self.fill,) * 3)
        left = random.randint(0, nW - W)
        top = random.randint(0, nH - H)
        new_img.paste(img, (left, top))

        if 'boxes' in target and len(target['boxes']) > 0:
            boxes = target['boxes'].clone()
            boxes[:, [0, 2]] += left
            boxes[:, [1, 3]] += top
            target['boxes'] = boxes
        target['orig_size'] = torch.tensor([nH, nW])
        return new_img, target


class RandomIoUCrop:
    def __init__(self, min_scale=0.3, max_scale=1.0, min_aspect_ratio=0.5,
                 max_aspect_ratio=2.0, sampler_options=None, trials=40, p=0.8):
        self.p = p
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.trials = trials

    def __call__(self, img, target):
        if random.random() > self.p or 'boxes' not in target:
            return img, target
        W, H = img.size
        for _ in range(self.trials):
            scale = random.uniform(self.min_scale, self.max_scale)
            aspect = random.uniform(self.min_aspect_ratio, self.max_aspect_ratio)
            nW = int(W * scale)
            nH = int(nW / aspect)
            if nH > H:
                nH = H
                nW = int(nH * aspect)
            if nW > W or nH > H:
                continue
            left = random.randint(0, W - nW)
            top = random.randint(0, H - nH)
            crop_box = torch.tensor([left, top, left + nW, top + nH], dtype=torch.float32)

            boxes = target['boxes']
            if len(boxes) == 0:
                break
            inter_x1 = torch.max(boxes[:, 0], crop_box[0])
            inter_y1 = torch.max(boxes[:, 1], crop_box[1])
            inter_x2 = torch.min(boxes[:, 2], crop_box[2])
            inter_y2 = torch.min(boxes[:, 3], crop_box[3])
            inter_w = (inter_x2 - inter_x1).clamp(min=0)
            inter_h = (inter_y2 - inter_y1).clamp(min=0)
            inter_area = inter_w * inter_h
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            iou = inter_area / (area + 1e-6)
            if iou.max() < 0.1:
                continue

            keep = iou > 0.0
            new_boxes = boxes[keep].clone()
            new_boxes[:, [0, 2]] = new_boxes[:, [0, 2]].clamp(left, left + nW) - left
            new_boxes[:, [1, 3]] = new_boxes[:, [1, 3]].clamp(top, top + nH) - top

            img = img.crop((left, top, left + nW, top + nH))
            target['boxes'] = new_boxes
            target['labels'] = target['labels'][keep]
            target['orig_size'] = torch.tensor([nH, nW])
            return img, target
        return img, target


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            W, H = img.size
            img = TF.hflip(img)
            if 'boxes' in target and len(target['boxes']) > 0:
                boxes = target['boxes'].clone()
                boxes[:, [0, 2]] = W - boxes[:, [2, 0]]
                target['boxes'] = boxes
        return img, target


class Resize:
    def __init__(self, size):
        if isinstance(size, int):
            size = (size, size)
        self.size = size  # (H, W)

    def __call__(self, img, target):
        orig_W, orig_H = img.size
        new_H, new_W = self.size
        img = img.resize((new_W, new_H), Image.BILINEAR)

        if 'boxes' in target and len(target['boxes']) > 0:
            boxes = target['boxes'].clone().float()
            sx = new_W / orig_W
            sy = new_H / orig_H
            boxes[:, [0, 2]] *= sx
            boxes[:, [1, 3]] *= sy
            target['boxes'] = boxes
        target['orig_size'] = torch.tensor([new_H, new_W])
        return img, target


class SanitizeBoundingBoxes:
    def __init__(self, min_size=1):
        self.min_size = min_size

    def __call__(self, img, target):
        if 'boxes' not in target or len(target['boxes']) == 0:
            return img, target
        boxes = target['boxes']
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        keep = (w >= self.min_size) & (h >= self.min_size)
        target['boxes'] = boxes[keep]
        target['labels'] = target['labels'][keep]
        return img, target


class ConvertPILImage:
    def __init__(self, dtype='float32', scale=True):
        self.dtype = dtype
        self.scale = scale

    def __call__(self, img, target):
        import numpy as np
        arr = np.array(img, dtype=np.float32)
        if self.scale:
            arr /= 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        return tensor, target


class ConvertBoxes:
    """Convert boxes between formats and optionally normalise."""

    def __init__(self, fmt='cxcywh', normalize=True):
        self.fmt = fmt
        self.normalize = normalize

    def __call__(self, img, target):
        if 'boxes' not in target or len(target['boxes']) == 0:
            return img, target
        boxes = target['boxes'].clone().float()

        # Input is assumed to be (x1, y1, x2, y2)
        if self.fmt == 'cxcywh':
            boxes = torch.cat([
                (boxes[:, :2] + boxes[:, 2:]) / 2,   # cx, cy
                boxes[:, 2:] - boxes[:, :2],           # w, h
            ], dim=1)

        if self.normalize:
            if isinstance(img, torch.Tensor):
                _, H, W = img.shape
            else:
                W, H = img.size
            boxes[:, [0, 2]] /= W
            boxes[:, [1, 3]] /= H

        target['boxes'] = boxes
        return img, target
