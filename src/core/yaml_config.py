"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import re
import copy
import yaml
from pathlib import Path

from .workspace import GLOBAL_CONFIG, create


__all__ = ['YAMLConfig', 'yaml_utils']


class yaml_utils:
    @staticmethod
    def parse_cli(update_list):
        """Parse CLI overrides like key=value pairs."""
        update_dict = {}
        if update_list is None:
            return update_dict
        for item in update_list:
            k, v = item.split('=', 1)
            try:
                import ast
                v = ast.literal_eval(v)
            except Exception:
                pass
            update_dict[k] = v
        return update_dict


def _deep_merge(base, override):
    """Recursively merge override into base dict."""
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def _load_yaml(path):
    """Load a YAML file, resolving __include__ directives."""
    path = Path(path)
    with open(path, 'r') as f:
        data = yaml.safe_load(f) or {}

    includes = data.pop('__include__', [])
    if isinstance(includes, str):
        includes = [includes]

    merged = {}
    for inc in includes:
        inc_path = path.parent / inc
        inc_data = _load_yaml(inc_path)
        merged = _deep_merge(merged, inc_data)

    merged = _deep_merge(merged, data)
    return merged


class YAMLConfig:
    """Configuration class that loads YAML files and builds model/optimizer/etc."""

    def __init__(self, cfg_path, **kwargs):
        self.yaml_cfg = _load_yaml(cfg_path)

        # Apply CLI/kwarg overrides
        for k, v in kwargs.items():
            if k in ('config', 'device', 'output_dir', 'summary_dir',
                     'resume', 'tuning', 'seed', 'use_amp',
                     'print_rank', 'print_method', 'local_rank', 'test_only'):
                continue
            if v is not None:
                self.yaml_cfg[k] = v

        # Apply nested key overrides (e.g. 'RTDETRCriterionv2.weight_dict.loss_nwd')
        flat_overrides = {}
        for k, v in kwargs.items():
            if '.' in k:
                flat_overrides[k] = v
        for key_path, val in flat_overrides.items():
            parts = key_path.split('.')
            d = self.yaml_cfg
            for p in parts[:-1]:
                d = d.setdefault(p, {})
            d[parts[-1]] = val

        self._model = None
        self._postprocessor = None
        self._criterion = None
        self._optimizer = None
        self._lr_scheduler = None
        self._lr_warmup_scheduler = None
        self._train_dataloader = None
        self._val_dataloader = None
        self._ema = None

    def _build(self, key):
        """Build a component by key from yaml_cfg."""
        cfg = copy.deepcopy(self.yaml_cfg.get(key, {}))
        cfg_type = cfg.pop('type', key)
        # Inject global config values into sub-components
        self._inject_globals(cfg)
        return create(cfg_type, **cfg)

    def _inject_globals(self, cfg):
        """Inject top-level config values into nested configs."""
        pass  # Handled by the workspace create() mechanism

    @property
    def model(self):
        if self._model is None:
            self._model = self._build_model()
        return self._model

    def _build_model(self):
        from src.nn.model.rtdetr import RTDETR
        model_cfg = copy.deepcopy(self.yaml_cfg)

        rtdetr_cfg = model_cfg.get('RTDETR', {})
        backbone_type = rtdetr_cfg.get('backbone', 'HGNetv2')
        encoder_type = rtdetr_cfg.get('encoder', 'HybridEncoder')
        decoder_type = rtdetr_cfg.get('decoder', 'RTDETRTransformerv2')

        backbone = self._build_component(backbone_type, model_cfg.get(backbone_type, {}))
        encoder = self._build_component(encoder_type, model_cfg.get(encoder_type, {}))
        decoder = self._build_component(decoder_type, model_cfg.get(decoder_type, {}),
                                        num_classes=model_cfg.get('num_classes', 80))

        return RTDETR(backbone=backbone, encoder=encoder, decoder=decoder)

    def _build_component(self, cls_name, cfg, **extra):
        cfg = copy.deepcopy(cfg)
        cfg.update(extra)
        return create(cls_name, **cfg)

    @property
    def postprocessor(self):
        if self._postprocessor is None:
            self._postprocessor = self._build_component(
                'RTDETRPostProcessor',
                self.yaml_cfg.get('RTDETRPostProcessor', {}),
                num_classes=self.yaml_cfg.get('num_classes', 80),
            )
        return self._postprocessor

    @property
    def criterion(self):
        if self._criterion is None:
            cfg = copy.deepcopy(self.yaml_cfg.get('RTDETRCriterionv2', {}))
            cfg['num_classes'] = self.yaml_cfg.get('num_classes', 80)
            self._criterion = create('RTDETRCriterionv2', **cfg)
        return self._criterion

    @property
    def optimizer(self):
        if self._optimizer is None:
            self._optimizer = self._build_optimizer()
        return self._optimizer

    def _build_optimizer(self):
        import torch.optim as optim
        cfg = copy.deepcopy(self.yaml_cfg.get('optimizer', {}))
        opt_type = cfg.pop('type', 'AdamW')
        param_groups_cfg = cfg.pop('params', [])

        model = self.model
        param_dict = {n: p for n, p in model.named_parameters() if p.requires_grad}

        if param_groups_cfg:
            groups = []
            assigned = set()
            for group_cfg in param_groups_cfg:
                group_cfg = copy.deepcopy(group_cfg)
                pattern = group_cfg.pop('params')
                matched = {n: p for n, p in param_dict.items()
                           if re.search(pattern, n)}
                assigned.update(matched.keys())
                groups.append({'params': list(matched.values()), **group_cfg})
            remaining = [p for n, p in param_dict.items() if n not in assigned]
            groups.append({'params': remaining})
            params = groups
        else:
            params = list(param_dict.values())

        opt_cls = getattr(optim, opt_type)
        return opt_cls(params, **cfg)

    @property
    def lr_scheduler(self):
        if self._lr_scheduler is None:
            self._lr_scheduler = self._build_lr_scheduler()
        return self._lr_scheduler

    def _build_lr_scheduler(self):
        import torch.optim.lr_scheduler as lr_sched
        cfg = copy.deepcopy(self.yaml_cfg.get('lr_scheduler', {}))
        sched_type = cfg.pop('type', 'MultiStepLR')
        sched_cls = getattr(lr_sched, sched_type)
        return sched_cls(self.optimizer, **cfg)

    @property
    def lr_warmup_scheduler(self):
        if self._lr_warmup_scheduler is None:
            cfg = copy.deepcopy(self.yaml_cfg.get('lr_warmup_scheduler', {}))
            if cfg:
                self._lr_warmup_scheduler = create(cfg.pop('type'), self.optimizer, **cfg)
        return self._lr_warmup_scheduler

    @property
    def train_dataloader(self):
        if self._train_dataloader is None:
            self._train_dataloader = self._build_dataloader('train_dataloader')
        return self._train_dataloader

    @property
    def val_dataloader(self):
        if self._val_dataloader is None:
            self._val_dataloader = self._build_dataloader('val_dataloader')
        return self._val_dataloader

    def _build_dataloader(self, key):
        from src.data.dataloader import build_dataloader
        cfg = copy.deepcopy(self.yaml_cfg.get(key, {}))
        dataset_cfg = cfg.pop('dataset', {})
        return build_dataloader(dataset_cfg, cfg, self.yaml_cfg)

    @property
    def ema(self):
        if self._ema is None:
            ema_cfg = copy.deepcopy(self.yaml_cfg.get('ema', {}))
            if ema_cfg:
                ema_type = ema_cfg.pop('type', 'ModelEMA')
                self._ema = create(ema_type, self.model, **ema_cfg)
        return self._ema
