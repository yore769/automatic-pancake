"""YAML utilities: loading with includes, merging, CLI parsing."""

import re
import yaml
from pathlib import Path


def load_yaml(path: str) -> dict:
    """Load YAML file, processing __include__ directives recursively."""
    path = Path(path)
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        cfg = {}

    includes = cfg.pop('__include__', [])
    if isinstance(includes, str):
        includes = [includes]

    merged = {}
    for inc in includes:
        inc_path = path.parent / inc
        inc_cfg = load_yaml(inc_path)
        merged = merge_dict(merged, inc_cfg)

    return merge_dict(merged, cfg)


def merge_dict(base: dict, override: dict) -> dict:
    """Deep merge override into base. override wins on conflicts."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = merge_dict(result[k], v)
        else:
            result[k] = v
    return result


def parse_cli(updates) -> dict:
    """Parse CLI update strings of form 'key=value' or 'key.sub=value'."""
    result = {}
    if not updates:
        return result
    for item in updates:
        if '=' not in item:
            continue
        key, value = item.split('=', 1)
        try:
            value = yaml.safe_load(value)
        except Exception:
            pass
        keys = key.split('.')
        d = result
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
    return result
