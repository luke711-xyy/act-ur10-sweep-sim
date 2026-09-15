"""Configuration handling.

A configuration is a plain nested mapping loaded from YAML, wrapped in
:class:`Config` so that it can be accessed either as ``cfg.controller.desired_force``
or as ``cfg["controller"]["desired_force"]``.

Keeping the config as data (rather than a rigid dataclass hierarchy) makes it
cheap to add experiment knobs and to serialise the *exact* configuration that
produced a run next to its results, which is what reproducibility needs.
"""

from __future__ import annotations

import copy
import json
import os
from typing import Any, Iterable, Mapping

import yaml

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "default.yaml"
)


class Config(Mapping):
    """Read-mostly nested config with attribute access."""

    __slots__ = ("_data",)

    def __init__(self, data: Mapping[str, Any] | None = None):
        object.__setattr__(self, "_data", {})
        for key, value in (data or {}).items():
            self._data[key] = Config(value) if isinstance(value, Mapping) else value

    # -- Mapping protocol ---------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    # -- attribute access ---------------------------------------------------
    def __getattr__(self, key: str) -> Any:
        try:
            return self._data[key]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(f"no config entry {key!r}") from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self._data[key] = Config(value) if isinstance(value, Mapping) else value

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Config({json.dumps(self.to_dict(), indent=2, default=str)})"

    # -- helpers ------------------------------------------------------------
    def to_dict(self) -> dict:
        out = {}
        for key, value in self._data.items():
            out[key] = value.to_dict() if isinstance(value, Config) else value
        return out

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if isinstance(node, Config) and part in node:
                node = node[part]
            else:
                return default
        return node

    def set_path(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node = self
        for part in parts[:-1]:
            if part not in node._data or not isinstance(node._data[part], Config):
                node._data[part] = Config({})
            node = node._data[part]
        node._data[parts[-1]] = Config(value) if isinstance(value, Mapping) else value

    def copy(self) -> "Config":
        return Config(copy.deepcopy(self.to_dict()))


def _deep_merge(base: dict, override: Mapping) -> dict:
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, Mapping):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _coerce(text: str) -> Any:
    """Best-effort conversion of a CLI override string into a Python value."""
    lowered = text.strip().lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("none", "null"):
        return None
    try:
        return yaml.safe_load(text)
    except Exception:  # pragma: no cover - yaml is very permissive
        return text


def load_config(
    path: str | None = None,
    overrides: Iterable[str] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Config:
    """Load a YAML config, apply a dict overlay then ``key.path=value`` overrides."""
    path = path or DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if extra:
        data = _deep_merge(data, extra)
    cfg = Config(data)
    for item in overrides or ():
        if "=" not in item:
            raise ValueError(f"override {item!r} must look like key.path=value")
        key, _, raw = item.partition("=")
        cfg.set_path(key.strip(), _coerce(raw))
    cfg.set_path("_source_config", os.path.abspath(path))
    return cfg


def save_config(cfg: Config, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg.to_dict(), handle, sort_keys=False, default_flow_style=False)
