"""YAMLConfig: loads YAML config and instantiates model/training objects."""

import importlib
import copy
from pathlib import Path
import torch

from .yaml_utils import load_yaml, merge_dict


# Registry: class_name -> class
GLOBAL_REGISTRY: dict = {}


def register(cls):
    """Decorator to register a class in the global registry."""
    GLOBAL_REGISTRY[cls.__name__] = cls
    return cls


def _import_zoo():
    """Lazily import all zoo modules to populate the registry."""
    import src.zoo.rtdetrv2  # noqa: F401


def get_class(name: str):
    if not GLOBAL_REGISTRY:
        _import_zoo()
    if name not in GLOBAL_REGISTRY:
        _import_zoo()
    if name not in GLOBAL_REGISTRY:
        raise KeyError(f"Class '{name}' not found in registry. "
                       f"Available: {list(GLOBAL_REGISTRY.keys())}")
    return GLOBAL_REGISTRY[name]


def instantiate(cfg_dict: dict, **extra_kwargs):
    """
    Instantiate an object from a config dict that has a 'type' key.
    Remaining keys are passed as kwargs.
    """
    cfg = dict(cfg_dict)
    cls_name = cfg.pop('type')
    cfg.update(extra_kwargs)
    cls = get_class(cls_name)
    return cls(**cfg)


class YAMLConfig:
    """
    Loads a YAML config file (with includes), stores the merged dict,
    and provides convenient properties to instantiate model components.
    """

    def __init__(self, cfg_path: str, **overrides):
        self.cfg_path = cfg_path
        self.yaml_cfg: dict = load_yaml(cfg_path)
        if overrides:
            self.yaml_cfg = merge_dict(self.yaml_cfg, overrides)

        # Apply device
        device = self.yaml_cfg.get('device', None)
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        # Populate registry
        _import_zoo()

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def _get_section(self, key: str) -> dict:
        return self.yaml_cfg.get(key, {}) or {}

    def _build_from_key(self, key: str, **extra):
        """Build an object whose class name is yaml_cfg[key] and whose
        kwargs come from yaml_cfg[class_name]."""
        cls_name = self.yaml_cfg[key]
        cls_cfg = copy.deepcopy(self._get_section(cls_name))
        # Recursively instantiate nested 'type' dicts
        cls_cfg = self._resolve_nested(cls_cfg)
        cls_cfg.update(extra)
        cls = get_class(cls_name)
        return cls(**cls_cfg)

    def _resolve_nested(self, cfg):
        """Recursively resolve dicts with 'type' key into objects."""
        if isinstance(cfg, dict):
            if 'type' in cfg:
                sub = copy.deepcopy(cfg)
                return instantiate(sub)
            return {k: self._resolve_nested(v) for k, v in cfg.items()}
        if isinstance(cfg, list):
            return [self._resolve_nested(v) for v in cfg]
        return cfg

    # ------------------------------------------------------------------
    # Main properties
    # ------------------------------------------------------------------

    @property
    def model(self):
        if not hasattr(self, '_model'):
            model_name = self.yaml_cfg['model']
            model_cfg = copy.deepcopy(self._get_section(model_name))
            # Resolve backbone / encoder / decoder references
            model_cfg = self._resolve_model_cfg(model_cfg)
            cls = get_class(model_name)
            self._model = cls(**model_cfg)
        return self._model

    def _resolve_model_cfg(self, model_cfg: dict) -> dict:
        """Build backbone, encoder, decoder from their config sections."""
        num_classes = self.yaml_cfg.get('num_classes', 80)
        resolved = {}
        for k, v in model_cfg.items():
            if isinstance(v, str) and v in self.yaml_cfg:
                # v is a class name like 'HGNetv2'
                sub_cfg = copy.deepcopy(self._get_section(v))
                sub_cfg = self._resolve_nested(sub_cfg)
                # Pass num_classes to decoder so heads are built with correct size
                if k == 'decoder':
                    sub_cfg.setdefault('num_classes', num_classes)
                cls = get_class(v)
                resolved[k] = cls(**sub_cfg)
            elif isinstance(v, dict) and 'type' in v:
                resolved[k] = instantiate(copy.deepcopy(v))
            else:
                resolved[k] = v
        return resolved

    @property
    def criterion(self):
        if not hasattr(self, '_criterion'):
            crit_name = self.yaml_cfg['criterion']
            crit_cfg = copy.deepcopy(self._get_section(crit_name))
            crit_cfg = self._resolve_nested(crit_cfg)
            crit_cfg['num_classes'] = self.yaml_cfg.get('num_classes', 80)
            cls = get_class(crit_name)
            self._criterion = cls(**crit_cfg)
        return self._criterion

    @property
    def postprocessor(self):
        if not hasattr(self, '_postprocessor'):
            pp_name = self.yaml_cfg['postprocessor']
            pp_cfg = copy.deepcopy(self._get_section(pp_name))
            pp_cfg = self._resolve_nested(pp_cfg)
            pp_cfg['num_classes'] = self.yaml_cfg.get('num_classes', 80)
            pp_cfg['use_focal_loss'] = self.yaml_cfg.get('use_focal_loss', True)
            cls = get_class(pp_name)
            self._postprocessor = cls(**pp_cfg)
        return self._postprocessor

    @property
    def optimizer(self):
        if not hasattr(self, '_optimizer'):
            opt_cfg = copy.deepcopy(self.yaml_cfg.get('optimizer', {}))
            self._optimizer = self._build_optimizer(opt_cfg, self.model)
        return self._optimizer

    def _build_optimizer(self, cfg: dict, model):
        from src.optim import build_optimizer
        return build_optimizer(cfg, model)

    @property
    def lr_scheduler(self):
        if not hasattr(self, '_lr_scheduler'):
            sched_cfg = copy.deepcopy(self.yaml_cfg.get('lr_scheduler', {}))
            self._lr_scheduler = self._build_lr_scheduler(sched_cfg, self.optimizer)
        return self._lr_scheduler

    def _build_lr_scheduler(self, cfg: dict, optimizer):
        from src.optim import build_lr_scheduler
        return build_lr_scheduler(cfg, optimizer)

    @property
    def lr_warmup_scheduler(self):
        if not hasattr(self, '_lr_warmup_scheduler'):
            cfg = copy.deepcopy(self.yaml_cfg.get('lr_warmup_scheduler', None))
            if cfg:
                from src.optim import build_lr_warmup_scheduler
                self._lr_warmup_scheduler = build_lr_warmup_scheduler(cfg, self.optimizer)
            else:
                self._lr_warmup_scheduler = None
        return self._lr_warmup_scheduler

    @property
    def train_dataloader(self):
        if not hasattr(self, '_train_dataloader'):
            from src.data import build_dataloader
            cfg = copy.deepcopy(self.yaml_cfg.get('train_dataloader', {}))
            self._train_dataloader = build_dataloader(cfg, split='train',
                                                       num_classes=self.yaml_cfg.get('num_classes', 80))
        return self._train_dataloader

    @property
    def val_dataloader(self):
        if not hasattr(self, '_val_dataloader'):
            from src.data import build_dataloader
            cfg = copy.deepcopy(self.yaml_cfg.get('val_dataloader', {}))
            self._val_dataloader = build_dataloader(cfg, split='val',
                                                     num_classes=self.yaml_cfg.get('num_classes', 80))
        return self._val_dataloader

    @property
    def ema(self):
        if not hasattr(self, '_ema'):
            if self.yaml_cfg.get('use_ema', False):
                from src.optim import ModelEMA
                ema_cfg = copy.deepcopy(self.yaml_cfg.get('ema', {}))
                ema_cfg.pop('type', None)
                self._ema = ModelEMA(self.model, **ema_cfg)
            else:
                self._ema = None
        return self._ema

    @property
    def scaler(self):
        if not hasattr(self, '_scaler'):
            if self.yaml_cfg.get('use_amp', False):
                from torch.cuda.amp import GradScaler
                scaler_cfg = copy.deepcopy(self.yaml_cfg.get('scaler', {}))
                scaler_cfg.pop('type', None)
                self._scaler = GradScaler(**scaler_cfg)
            else:
                self._scaler = None
        return self._scaler

    @property
    def evaluator(self):
        if not hasattr(self, '_evaluator'):
            ev_cfg = copy.deepcopy(self.yaml_cfg.get('evaluator', {}))
            ev_cfg = self._resolve_nested(ev_cfg)
            cls = get_class(ev_cfg.pop('type') if isinstance(ev_cfg, dict) and 'type' in ev_cfg else 'CocoEvaluator')
            self._evaluator = cls(**ev_cfg)
        return self._evaluator

    def __repr__(self):
        return f"YAMLConfig(path={self.cfg_path})"
