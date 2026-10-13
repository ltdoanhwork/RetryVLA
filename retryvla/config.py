"""Tiny YAML config system.

- A config file may contain ``defaults: [env/libero_spatial, policy/smolvla_libero, ...]``;
  each entry is a path relative to ``configs/`` (without ``.yaml``) and is merged in order,
  then the file's own keys are merged on top.
- CLI overrides are ``dotted.key=value`` strings (parsed as YAML scalars).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from retryvla import CONFIG_DIR

REQUIRED_KEYS = ("milestone", "tag")


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if p.suffix != ".yaml":
        p = p.with_suffix(".yaml")
    if p.is_file():
        return p
    q = CONFIG_DIR / p
    if q.is_file():
        return q
    raise FileNotFoundError(f"config not found: {path}")


def _load(path: str | Path) -> DictConfig:
    p = _resolve(path)
    cfg = OmegaConf.load(p)
    defaults = cfg.pop("defaults", []) or []
    merged = OmegaConf.create({})
    for d in defaults:
        merged = OmegaConf.merge(merged, _load(d))
    return OmegaConf.merge(merged, cfg)


def load_config(path: str | Path, overrides: list[str] | None = None) -> DictConfig:
    cfg = _load(path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    for k in REQUIRED_KEYS:
        if k not in cfg:
            raise KeyError(f"experiment config must define '{k}'")
    return cfg


def cli_config(argv: list[str] | None = None) -> DictConfig:
    """``python -m retryvla.x configs/exp/foo.yaml a.b=1 c=2``"""
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        raise SystemExit("usage: <config.yaml> [key=value ...]")
    return load_config(argv[0], argv[1:])


def to_dict(cfg: DictConfig) -> dict[str, Any]:
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
