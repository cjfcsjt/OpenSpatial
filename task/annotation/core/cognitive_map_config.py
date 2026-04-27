"""
Configuration helpers for the cognitive map feature.

Parses the ``cognitive_map`` section of a task-level YAML config into a
concrete ``CognitiveMapSettings`` dataclass, with sensible defaults that
preserve backward compatibility (i.e. disabled by default).
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Mapping


@dataclass
class CognitiveMapSettings:
    """Runtime settings for the cognitive map feature."""
    enable: bool = False
    enable_visualization: bool = True
    dump_samples: bool = True
    dump_sample_count: int = 20
    grid_size: int = 10
    padding_ratio: float = 0.10

    @property
    def active(self) -> bool:
        """True iff we need to build cognitive maps at all."""
        return bool(self.enable)


def _coerce_to_mapping(value: Any) -> Mapping[str, Any]:
    """Convert SimpleNamespace/dict/None into a plain mapping."""
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return value
    if isinstance(value, SimpleNamespace):
        return vars(value)
    if hasattr(value, "__dict__"):
        return vars(value)
    raise TypeError(
        f"Unsupported cognitive_map config type: {type(value).__name__}"
    )


def parse_cognitive_map_settings(args: Mapping[str, Any]) -> CognitiveMapSettings:
    """Parse the ``cognitive_map`` sub-dict from task args into settings.

    Missing keys fall back to dataclass defaults. When the ``cognitive_map``
    key is absent from the YAML the feature is implicitly **disabled** — this
    preserves the pre-feature parquet schema for all existing configs.
    """
    raw = args.get("cognitive_map") if isinstance(args, Mapping) else None
    cfg = _coerce_to_mapping(raw)
    settings = CognitiveMapSettings()

    if "enable" in cfg:
        settings.enable = bool(cfg["enable"])
    if "enable_visualization" in cfg:
        settings.enable_visualization = bool(cfg["enable_visualization"])
    if "dump_samples" in cfg:
        settings.dump_samples = bool(cfg["dump_samples"])
    if "dump_sample_count" in cfg:
        settings.dump_sample_count = int(cfg["dump_sample_count"])
    if "grid_size" in cfg:
        settings.grid_size = int(cfg["grid_size"])
    if "padding_ratio" in cfg:
        settings.padding_ratio = float(cfg["padding_ratio"])

    return settings
