"""RT-DETRv2 package: registers all model components."""

from src.core.config import register

# Import all model components to trigger registration
from .hgnetv2 import HGNetv2
from .hybrid_encoder import HybridEncoder
from .rtdetrv2_decoder import RTDETRTransformerv2
from .rtdetrv2_criterion import RTDETRCriterionv2
from .rtdetrv2_postprocessor import RTDETRPostProcessor
from .rtdetrv2 import RTDETR
from .matcher import HungarianMatcher
from .coco_evaluator import CocoEvaluator

__all__ = [
    'HGNetv2',
    'HybridEncoder',
    'RTDETRTransformerv2',
    'RTDETRCriterionv2',
    'RTDETRPostProcessor',
    'RTDETR',
    'HungarianMatcher',
    'CocoEvaluator',
]
