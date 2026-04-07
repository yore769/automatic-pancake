"""Image and box transforms for object detection."""

import random
import math
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image


def _box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    x0, y0, x1, y1 = boxes.unbind(-1)
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    w = x1 - x0
    h = y1 - y0
    return torch.stack([cx, cy, w, h], dim=-1)


def _box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


class Resize:
    """Resize image and scale boxes accordingly."""

    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = tuple(size)  # (H, W)

    def __call__(self, img, target):
        orig_w, orig_h = img.size
        new_h, new_w = self.size
        img = img.resize((new_w, new_h), Image.BILINEAR)

        if 'boxes' in target and len(target['boxes']) > 0:
            boxes = target['boxes']
            scale_x = new_w / orig_w
            scale_y = new_h / orig_h
            boxes = boxes * torch.tensor([scale_x, scale_y, scale_x, scale_y])
            target['boxes'] = boxes
        target['size'] = torch.tensor([new_h, new_w])
        return img, target


class RandomHorizontalFlip:
    """Randomly flip image and boxes horizontally."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            w, h = img.size
            img = TF.hflip(img)
            if 'boxes' in target and len(target['boxes']) > 0:
                boxes = target['boxes'].clone()
                # xyxy format: flip x coords
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                target['boxes'] = boxes
        return img, target


class RandomPhotometricDistort:
    """Random photometric distortion (brightness, contrast, saturation, hue)."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            img = TF.adjust_brightness(img, random.uniform(0.5, 1.5))
        if random.random() < self.p:
            img = TF.adjust_contrast(img, random.uniform(0.5, 1.5))
        if random.random() < self.p:
            img = TF.adjust_saturation(img, random.uniform(0.5, 1.5))
        if random.random() < self.p:
            img = TF.adjust_hue(img, random.uniform(-0.1, 0.1))
        return img, target


class RandomZoomOut:
    """Randomly zoom out (paste image on larger canvas)."""

    def __init__(self, fill=0, p: float = 0.5, max_scale: float = 4.0):
        self.fill = fill
        self.p = p
        self.max_scale = max_scale

    def __call__(self, img, target):
        if random.random() > self.p:
            return img, target
        w, h = img.size
        scale = random.uniform(1.0, self.max_scale)
        new_w = int(w * scale)
        new_h = int(h * scale)
        new_img = Image.new(img.mode, (new_w, new_h), self.fill)
        left = random.randint(0, new_w - w)
        top = random.randint(0, new_h - h)
        new_img.paste(img, (left, top))
        if 'boxes' in target and len(target['boxes']) > 0:
            boxes = target['boxes'].clone()
            boxes[:, [0, 2]] += left
            boxes[:, [1, 3]] += top
            target['boxes'] = boxes
        target['size'] = torch.tensor([new_h, new_w])
        return new_img, target


class RandomIoUCrop:
    """Random crop based on IoU thresholds with original boxes."""

    def __init__(self, p: float = 0.8, min_scale: float = 0.3, max_scale: float = 1.0,
                 min_aspect_ratio: float = 0.5, max_aspect_ratio: float = 2.0,
                 sampler_options=None, num_trials: int = 40):
        self.p = p
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.options = sampler_options or [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        self.num_trials = num_trials

    def __call__(self, img, target):
        if random.random() > self.p:
            return img, target

        w, h = img.size
        boxes = target.get('boxes', torch.zeros(0, 4))

        for _ in range(self.num_trials):
            iou_threshold = random.choice(self.options)
            if iou_threshold >= 1.0:
                return img, target

            scale = random.uniform(self.min_scale, self.max_scale)
            aspect = random.uniform(self.min_aspect_ratio, self.max_aspect_ratio)
            cw = int(w * scale)
            ch = int(h * scale)
            if cw > w or ch > h:
                continue

            left = random.randint(0, w - cw)
            top = random.randint(0, h - ch)
            crop_box = torch.tensor([[left, top, left + cw, top + ch]], dtype=torch.float32)

            if len(boxes) == 0:
                img = img.crop((left, top, left + cw, top + ch))
                target['size'] = torch.tensor([ch, cw])
                return img, target

            # Compute IoU
            inter_x1 = torch.max(boxes[:, 0], crop_box[:, 0])
            inter_y1 = torch.max(boxes[:, 1], crop_box[:, 1])
            inter_x2 = torch.min(boxes[:, 2], crop_box[:, 2])
            inter_y2 = torch.min(boxes[:, 3], crop_box[:, 3])
            inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
            box_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            iou = inter / box_area.clamp(min=1e-6)

            if iou.min() < iou_threshold:
                continue

            # Apply crop
            img = img.crop((left, top, left + cw, top + ch))
            new_boxes = boxes.clone()
            new_boxes[:, [0, 2]] -= left
            new_boxes[:, [1, 3]] -= top
            new_boxes = new_boxes.clamp(min=0)
            new_boxes[:, 2] = new_boxes[:, 2].clamp(max=cw)
            new_boxes[:, 3] = new_boxes[:, 3].clamp(max=ch)
            keep = (new_boxes[:, 2] > new_boxes[:, 0]) & (new_boxes[:, 3] > new_boxes[:, 1])
            target['boxes'] = new_boxes[keep]
            if 'labels' in target:
                target['labels'] = target['labels'][keep]
            target['size'] = torch.tensor([ch, cw])
            return img, target

        return img, target


class SanitizeBoundingBoxes:
    """Remove degenerate bounding boxes (too small or invalid)."""

    def __init__(self, min_size: float = 1.0):
        self.min_size = min_size

    def __call__(self, img, target):
        if 'boxes' not in target or len(target['boxes']) == 0:
            return img, target
        boxes = target['boxes']
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        keep = (w >= self.min_size) & (h >= self.min_size)
        target['boxes'] = boxes[keep]
        if 'labels' in target:
            target['labels'] = target['labels'][keep]
        return img, target


class ConvertPILImage:
    """Convert PIL image to float tensor."""

    def __init__(self, dtype: str = 'float32', scale: bool = True):
        self.dtype = dtype
        self.scale = scale

    def __call__(self, img, target):
        img = TF.to_tensor(img)  # [C, H, W], float32, [0, 1]
        if not self.scale:
            img = img * 255.0
        return img, target


class ConvertBoxes:
    """Convert boxes between formats and optionally normalize."""

    def __init__(self, fmt: str = 'cxcywh', normalize: bool = True):
        self.fmt = fmt
        self.normalize = normalize

    def __call__(self, img, target):
        if 'boxes' not in target or len(target['boxes']) == 0:
            return img, target
        boxes = target['boxes']
        if self.fmt == 'cxcywh':
            boxes = _box_xyxy_to_cxcywh(boxes)
        if self.normalize:
            h, w = target['size'].tolist()
            boxes = boxes / torch.tensor([w, h, w, h], dtype=boxes.dtype)
        target['boxes'] = boxes
        return img, target
