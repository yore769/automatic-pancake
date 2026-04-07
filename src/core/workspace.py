"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

__all__ = ['register', 'create', 'GLOBAL_CONFIG']

GLOBAL_CONFIG = {}


def register(cls=None, *, name=None):
    """Class decorator to register a class in the global config."""
    def _register(cls):
        _name = cls.__name__ if name is None else name
        GLOBAL_CONFIG[_name] = cls
        return cls

    if cls is None:
        return _register
    return _register(cls)


def create(name_or_dict, **kwargs):
    """Create an instance from a registered class name or config dict."""
    if isinstance(name_or_dict, dict):
        cfg = name_or_dict.copy()
        _type = cfg.pop('type')
        cfg.update(kwargs)
        return create(_type, **cfg)
    assert isinstance(name_or_dict, str)
    if name_or_dict not in GLOBAL_CONFIG:
        raise RuntimeError(f'`{name_or_dict}` is not registered in GLOBAL_CONFIG. '
                           f'Available: {list(GLOBAL_CONFIG.keys())}')
    return GLOBAL_CONFIG[name_or_dict](**kwargs)
