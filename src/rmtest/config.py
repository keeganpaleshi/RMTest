"""Lightweight configuration helpers for :mod:`rmtest`."""

from __future__ import annotations

from os import PathLike
from typing import Any, Mapping

import io_utils as _io_utils

from .fitting.emg_config import synchronize_emg_stable_mode

ConfigMapping = Mapping[str, Any]

__all__ = ["ConfigMapping", "load_config"]


def load_config(config: PathLike[str] | str | ConfigMapping) -> dict[str, Any]:
    """Load a configuration mapping and keep EMG state in sync."""

    cfg = _io_utils.load_config(config)
    synchronize_emg_stable_mode(cfg)
    return cfg
