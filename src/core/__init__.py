"""Core registry and factory utilities."""

from .workspace import GLOBAL_CONFIG, register, create, setup_global_cfg

__all__ = ['GLOBAL_CONFIG', 'register', 'create', 'setup_global_cfg']
