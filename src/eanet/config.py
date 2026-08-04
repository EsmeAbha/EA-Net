"""Config loading with dotted-path overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> dict:
    """Load a YAML config and apply ``key.sub=value`` overrides."""
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"override must look like key.sub=value, got {override!r}")
        key, raw = override.split("=", 1)
        _set_dotted(cfg, key.strip(), _coerce(raw.strip()))
    return cfg


def _set_dotted(cfg: dict, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = cfg
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def _coerce(raw: str) -> Any:
    """Parse a CLI scalar using YAML rules, so `3`, `1e-4`, `true`, `[1,2]` all work."""
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def merge(base: dict, extra: dict) -> dict:
    """Recursively merge ``extra`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = value
    return out
