"""Transform container with epoch-based policy support."""

import torch


class Compose:
    """Compose a list of transforms. Each transform takes (img, target)."""

    def __init__(self, ops=None, policy=None):
        self.ops = ops or []
        self.policy = policy  # dict with 'name', 'epoch', 'ops'
        self._current_epoch = 0

    def set_epoch(self, epoch: int):
        self._current_epoch = epoch

    def _get_active_ops(self):
        if self.policy is None:
            return self.ops
        policy = self.policy
        stop_epoch = policy.get('epoch', float('inf'))
        excluded = set(policy.get('ops', []))
        if self._current_epoch >= stop_epoch:
            return [op for op in self.ops if type(op).__name__ not in excluded]
        return self.ops

    def __call__(self, img, target):
        for op in self._get_active_ops():
            img, target = op(img, target)
        return img, target


class BatchImageCollateFuncion:
    """
    Collate function that pads images to the same size within a batch.
    Supports multi-scale training by randomly selecting a size from `scales`.
    """

    def __init__(self, scales=None, stop_epoch: int = None):
        self.scales = scales
        self.stop_epoch = stop_epoch
        self._epoch = 0

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def __call__(self, batch):
        imgs, targets = zip(*batch)
        # Determine target size
        if self.scales and (self.stop_epoch is None or self._epoch < self.stop_epoch):
            size = self.scales[torch.randint(len(self.scales), (1,)).item()]
        else:
            size = None

        if size is not None:
            # Resize all images to size x size
            import torch.nn.functional as F
            resized = []
            for img in imgs:
                if isinstance(img, torch.Tensor):
                    # img: [C, H, W]
                    resized.append(
                        F.interpolate(img.unsqueeze(0), size=(size, size), mode='bilinear',
                                      align_corners=False).squeeze(0)
                    )
                else:
                    resized.append(img)
            imgs = resized

        if isinstance(imgs[0], torch.Tensor):
            imgs = torch.stack(imgs, dim=0)
        return imgs, list(targets)
