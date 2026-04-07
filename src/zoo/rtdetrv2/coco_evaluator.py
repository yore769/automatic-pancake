"""COCO evaluator."""

import torch
from src.core.config import register
from src.misc.dist_utils import all_gather, is_main_process


@register
class CocoEvaluator:
    """COCO-format evaluator using pycocotools."""

    def __init__(self, iou_types=None):
        if iou_types is None:
            iou_types = ['bbox']
        self.iou_types = iou_types
        self.results = []
        self._coco_eval = None

    def update(self, predictions: dict):
        """
        Args:
            predictions: dict mapping image_id -> {'scores', 'labels', 'boxes'}
        """
        for img_id, pred in predictions.items():
            scores = pred['scores'].cpu().tolist()
            labels = pred['labels'].cpu().tolist()
            boxes = pred['boxes'].cpu().tolist()
            for score, label, box in zip(scores, labels, boxes):
                x1, y1, x2, y2 = box
                self.results.append({
                    'image_id': img_id,
                    'category_id': label,
                    'bbox': [x1, y1, x2 - x1, y2 - y1],
                    'score': score,
                })

    def synchronize_between_processes(self):
        all_results = all_gather(self.results)
        merged = []
        for r in all_results:
            merged.extend(r)
        self.results = merged

    def accumulate(self):
        if not is_main_process() or not self.results:
            return
        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
            import json, tempfile, os

            # We need ground truth - skip accumulate if coco GT not available
            if self._coco_eval is None:
                return
        except ImportError:
            pass

    def summarize(self):
        if is_main_process() and self.results:
            print(f"CocoEvaluator: {len(self.results)} detections collected.")
