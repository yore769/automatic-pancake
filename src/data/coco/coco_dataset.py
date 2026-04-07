"""COCO detection dataset."""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image


class CocoDetection(Dataset):
    """
    COCO-format detection dataset.
    Returns (image, target) where target is a dict with boxes and labels.
    """

    def __init__(
        self,
        img_folder: str,
        ann_file: str,
        return_masks: bool = False,
        transforms=None,
    ):
        from pycocotools.coco import COCO

        self.img_folder = img_folder
        self.ann_file = ann_file
        self.return_masks = return_masks
        self._transforms = transforms

        self.coco = COCO(ann_file)
        self.ids = sorted(self.coco.imgs.keys())

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        img_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_folder, img_info['file_name'])
        img = Image.open(img_path).convert('RGB')

        w, h = img.size
        boxes = []
        labels = []
        for ann in anns:
            if ann.get('iscrowd', 0):
                continue
            x, y, bw, bh = ann['bbox']
            # Convert to xyxy
            boxes.append([x, y, x + bw, y + bh])
            labels.append(ann['category_id'])

        target = {
            'image_id': torch.tensor([img_id]),
            'boxes': torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            'labels': torch.tensor(labels, dtype=torch.long),
            'orig_size': torch.tensor([h, w], dtype=torch.long),
            'size': torch.tensor([h, w], dtype=torch.long),
        }

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target
