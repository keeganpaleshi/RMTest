from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable


def _get_formats(cfg: dict | None, default: str) -> Iterable[str]:
    if cfg is None:
        return [default]
    if isinstance(cfg, dict):
        plots = cfg.get("plots")
        if isinstance(plots, dict) and plots.get("formats"):
            fmts = plots["formats"]
        else:
            fmts = cfg.get("plot_save_formats")
            if fmts is None:
                plotting = cfg.get("plotting")
                if isinstance(plotting, dict):
                    fmts = plotting.get("plot_save_formats")
        if fmts:
            if isinstance(fmts, str):
                return [fmts]
            return list(fmts)
    return [default]


def get_targets(cfg: dict | None, stem: str | os.PathLike[str]) -> Dict[str, Path]:
    """Return output paths mapped by extension."""
    base_path = Path(stem)
    dirpath = base_path.parent if base_path.parent != Path("") else Path(".")
    os.makedirs(dirpath, exist_ok=True)
    base = base_path.with_suffix("")
    default_ext = base_path.suffix.lstrip(".") or "png"
    formats = _get_formats(cfg, default_ext)
    return {
        fmt.lstrip("."): base.with_suffix(f".{fmt.lstrip('.')}") for fmt in formats
    }
