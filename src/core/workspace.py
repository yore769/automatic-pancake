"""Registry and workspace utilities for RT-DETR."""

import copy
import importlib

GLOBAL_CONFIG = {}


class Registry:
    """A simple registry for class/function registration."""

    def __init__(self, name: str):
        self._name = name
        self._module_dict: dict = {}

    def __repr__(self):
        return f"Registry(name={self._name}, items={list(self._module_dict.keys())})"

    def register(self, cls=None, name: str = None):
        if cls is None:
            return lambda c: self.register(c, name=name)
        key = name if name is not None else cls.__name__
        if key in self._module_dict:
            raise ValueError(f"'{key}' is already registered in {self._name}.")
        self._module_dict[key] = cls
        return cls

    def __contains__(self, key):
        return key in self._module_dict

    def __getitem__(self, key):
        return self._module_dict[key]


_REGISTRY = Registry("global")


def register(cls=None, name: str = None):
    """Register a class globally."""
    return _REGISTRY.register(cls, name=name)


def create(cfg: dict, **kwargs):
    """Instantiate a registered class from a config dict.

    The config dict must have a ``type`` key specifying the class name.
    All other keys are forwarded as keyword arguments to the constructor.
    """
    if cfg is None:
        return None
    cfg = copy.deepcopy(cfg)
    cls_name = cfg.pop("type")
    if cls_name not in _REGISTRY:
        raise ValueError(f"'{cls_name}' is not registered. Available: {list(_REGISTRY._module_dict.keys())}")
    cls = _REGISTRY[cls_name]
    cfg.update(kwargs)
    return cls(**cfg)


def setup_global_cfg(cfg: dict):
    """Merge a flat config dict into GLOBAL_CONFIG."""
    GLOBAL_CONFIG.update(cfg)
