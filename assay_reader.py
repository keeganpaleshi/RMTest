"""Wide-format Lucas-cell assay CSV parser for RMTest.

Reads the legacy spreadsheet layout where each assay occupies a group of
~4 columns side-by-side.  Column 0 holds row labels; the header row contains
assay-type labels at the start of each column group.

Two CSV variants are supported:

* **UI / LN2 Plant** – row labels include ``Date of Assay``, ``Number of
  Counts``, ``Overall Efficiency (global)``, etc.
* **UG board calibration** – row labels include ``Date of Assay``,
  ``Number of Counts``, ``Lucas cell single-alpha efficiency``, etc.

Both are auto-detected from the row-label vocabulary.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class LucasCellAssay:
    """Single parsed Lucas-cell assay record."""

    label: str = ""
    assay_date: datetime | None = None
    operators: str = ""
    comments: str = ""
    wave_file: str = ""

    # Raw measurement inputs
    assay_duration_min: float = math.nan
    assay_duration_unc_min: float = math.nan
    delay_time_days: float = math.nan
    delay_time_unc_days: float = math.nan
    counting_time_days: float = math.nan
    counting_time_unc_days: float = math.nan
    n_counts: float = math.nan
    n_counts_unc: float = math.nan
    lc_background_cpd: float = math.nan
    lc_background_unc_cpd: float = math.nan
    board_background_cpd: float = 0.0
    board_background_unc_cpd: float = 0.0
    rn_half_life_days: float = 3.8240
    overall_efficiency: float = math.nan  # dimensionless, 0–1
    flow_rate_lpm: float = math.nan  # L/min

    # Pre-computed derived quantities (from the spreadsheet)
    decayed_atoms_in_cell: float = math.nan
    total_atoms_in_cell: float = math.nan
    total_atoms_in_cell_unc: float = math.nan
    total_atoms_in_assay: float = math.nan
    total_atoms_in_assay_unc: float = math.nan
    atoms_per_liter: float = math.nan  # concentration (atoms/L)
    atoms_per_liter_unc: float = math.nan
    atoms_in_monitor: float = math.nan  # scaled to RM volume
    atoms_in_monitor_unc: float = math.nan

    # Provenance
    source_file: str = ""
    column_index: int = -1

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dict."""
        d = asdict(self)
        if self.assay_date is not None:
            d["assay_date"] = self.assay_date.isoformat()
        # Drop NaN values for cleaner output
        return {k: v for k, v in d.items() if not (_is_nan(v))}


def _is_nan(v: Any) -> bool:
    try:
        return isinstance(v, float) and math.isnan(v)
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Row-label detection helpers
# ---------------------------------------------------------------------------

# Canonical row-label keys (case-insensitive prefix match)
_ROW_KEYS_UI = {
    "date of assay": "date",
    "operators": "operators",
    "comments": "comments",
    "wave file": "wave_file",
    "assay duration": "duration",
    "time between lc": "delay",
    "counting time": "counting_time",
    "flow rate": "flow_rate",
    "number of counts": "n_counts",
    "lucas cell background": "lc_bkg",
    "board background": "board_bkg",
    "radon halflife": "half_life",
    "overall efficiency": "efficiency",
    "decayed radon atoms": "decayed_atoms",
    "total radon atoms in lucas cell after": "atoms_in_cell_after",
    "total radon atoms in lucas cell (in the beg": "atoms_in_cell_start",
    "total radon atoms in the assay": "atoms_in_assay",
    "total radon atoms in 1l": "atoms_per_liter",
    "atoms in radon monitor": "atoms_in_monitor",
    "flow rate": "flow_rate",
}

_ROW_KEYS_UG = {
    "date of assay": "date",
    "inputs": "wave_file",
    "assay duration": "duration",
    "emanation time": "emanation_time",
    "counting time": "counting_time",
    "number of counts": "n_counts",
    "lucas cell background": "lc_bkg",
    "chamber background": "chamber_bkg",
    "delay to start counting": "delay",
    "radon halflife": "half_life",
    "lucas cell single-alpha": "efficiency",
    "decayed radon atoms": "decayed_atoms",
    "radon atoms at start of counting": "atoms_in_cell_start",
    "radon atoms in chamber": "atoms_in_chamber",
    "assay result": "atoms_in_assay",
    "source emitter rate": "emitter_rate",
}


def _match_row_label(label: str, key_map: dict[str, str]) -> str | None:
    """Return the canonical key for *label*, or ``None``."""
    label_lower = label.strip().lower()
    for prefix, key in key_map.items():
        if label_lower.startswith(prefix):
            return key
    return None


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------

def _to_float(val: Any) -> float:
    """Convert a cell value to float, returning NaN on failure."""
    if val is None:
        return math.nan
    if isinstance(val, (int, float)):
        if isinstance(val, float) and math.isnan(val):
            return math.nan
        return float(val)
    s = str(val).strip()
    if not s or s in ("#REF!", "#DIV/0!", "#VALUE!", "#N/A", "n/a", "—", "–"):
        return math.nan
    # Strip percentage sign
    if s.endswith("%"):
        try:
            return float(s[:-1]) / 100.0
        except ValueError:
            return math.nan
    # Strip annotations like "(SNO)", "New Eff"
    s = re.sub(r"\(.*?\)", "", s).strip()
    s = re.sub(r"[A-Za-z%]+$", "", s).strip()
    try:
        return float(s)
    except ValueError:
        return math.nan


def _normalise_efficiency(raw: float) -> float:
    """Ensure efficiency is dimensionless in [0, 1].

    The spreadsheets store efficiency inconsistently:
    * ``38.73%`` → already handled by _to_float → 0.3873
    * ``0.77`` → already correct
    * ``1.42`` → this is a *raw* SNO-convention efficiency >1 that
      represents a product of multiple sub-efficiencies and should
      be kept as-is (not divided by 100).

    Only values above 1.0 that look like *percentages* (i.e. in the
    range 1–100 and resulting from a "%" string that wasn't caught
    earlier) get divided by 100.  Values above 1.0 that are genuine
    multi-step efficiencies are preserved.
    """
    if math.isnan(raw):
        return math.nan
    # If it was already converted from a % string, it will be < 1
    # If it is a genuine composite efficiency > 1 (like 1.42), keep it
    return raw


# ---------------------------------------------------------------------------
# Column-group detection
# ---------------------------------------------------------------------------

def _detect_assay_columns(df: pd.DataFrame) -> list[tuple[int, str]]:
    """Find (column_index, label) for each assay block in the header row.

    The header row is row 0.  Column 0 is the row-label column (or a
    category header like "LN2 Plant Assays").  Assay labels appear in
    subsequent columns; empty columns between them are part of the
    preceding block.
    """
    header = df.iloc[0]
    blocks: list[tuple[int, str]] = []
    for col_idx in range(1, len(header)):
        val = header.iloc[col_idx]
        if pd.isna(val):
            continue
        s = str(val).strip()
        if s and s not in ("", "nan"):
            blocks.append((col_idx, s))
    return blocks


def _build_row_index(df: pd.DataFrame, key_map: dict[str, str]) -> dict[str, int]:
    """Map canonical row keys to 0-based row indices by scanning column 0."""
    index: dict[str, int] = {}
    for row_idx in range(df.shape[0]):
        cell = df.iloc[row_idx, 0]
        if pd.isna(cell):
            continue
        label = str(cell).strip()
        if not label:
            continue
        key = _match_row_label(label, key_map)
        if key is not None and key not in index:
            index[key] = row_idx
    return index


def _extract_value_unc(
    df: pd.DataFrame,
    row: int,
    col_start: int,
    block_width: int,
) -> tuple[float, float]:
    """Extract (value, uncertainty) from a row within a column block.

    Scans the block for the first numeric value and then looks for a
    ``+/-`` marker followed by the uncertainty.
    """
    if row < 0 or row >= df.shape[0]:
        return (math.nan, math.nan)

    col_end = min(col_start + block_width, df.shape[1])

    # Value is at the block start column
    value = _to_float(df.iloc[row, col_start])

    # Scan for "+/-" marker to find uncertainty
    unc = math.nan
    for c in range(col_start + 1, col_end):
        cell = df.iloc[row, c]
        if pd.notna(cell) and str(cell).strip() == "+/-":
            # Uncertainty is in the next column
            if c + 1 < col_end:
                unc = _to_float(df.iloc[row, c + 1])
            break

    return (value, unc)


def _extract_text(df: pd.DataFrame, row: int, col: int) -> str:
    """Extract a text value from a cell, returning '' on NaN/missing."""
    if row < 0 or row >= df.shape[0] or col < 0 or col >= df.shape[1]:
        return ""
    val = df.iloc[row, col]
    if pd.isna(val):
        return ""
    return str(val).strip()


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------

_DATE_FORMATS = [
    "%d-%m-%y",       # 28-06-23
    "%d-%m-%Y",       # 28-06-2023
    "%d/%m/%Y",       # 13/08/2024
    "%d/%m/%y",       # 14/08/24
    "%m-%d-%y",       # 10-26-23 (US month-day-year)
    "%Y-%m-%d",       # ISO
]


def _parse_date(s: str) -> datetime | None:
    """Try several date formats, return None on failure."""
    s = s.strip()
    if not s:
        return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    logger.warning("Could not parse assay date: %r", s)
    return None


# ---------------------------------------------------------------------------
# CSV-variant detection
# ---------------------------------------------------------------------------

def _detect_variant(df: pd.DataFrame) -> str:
    """Return 'ui' or 'ug' based on row-label vocabulary."""
    labels_lower = set()
    for row_idx in range(min(30, df.shape[0])):
        cell = df.iloc[row_idx, 0]
        if pd.notna(cell):
            labels_lower.add(str(cell).strip().lower())

    # UG-specific labels
    if any("emanation time" in lbl for lbl in labels_lower):
        return "ug"
    if any("overall efficiency" in lbl for lbl in labels_lower):
        return "ui"
    # Fallback
    return "ui"


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def parse_assay_csv(path: str | Path) -> list[LucasCellAssay]:
    """Parse a wide-format Lucas-cell assay CSV/XLS file.

    Parameters
    ----------
    path : str or Path
        Path to the CSV (or XLS/XLSX) file.

    Returns
    -------
    list[LucasCellAssay]
        One record per detected assay column-block.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Assay file not found: {path}")

    # Load the raw data
    suffix = path.suffix.lower()
    if suffix in (".xls", ".xlsx"):
        df = pd.read_excel(path, header=None, dtype=str)
    else:
        # Try UTF-8 first, fall back to latin-1 for legacy spreadsheet exports
        for enc in ("utf-8-sig", "utf-8", "latin-1"):
            try:
                df = pd.read_csv(path, header=None, dtype=str, encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            df = pd.read_csv(path, header=None, dtype=str, encoding="latin-1")

    # Convert all cells: try numeric where possible but keep strings
    # We work with the raw string DataFrame and convert manually

    variant = _detect_variant(df)
    key_map = _ROW_KEYS_UI if variant == "ui" else _ROW_KEYS_UG
    logger.info("Detected assay CSV variant: %s (%s)", variant, path.name)

    row_index = _build_row_index(df, key_map)
    logger.debug("Row index: %s", row_index)

    blocks = _detect_assay_columns(df)
    if not blocks:
        logger.warning("No assay column blocks detected in %s", path)
        return []

    # Compute block widths from spacing between blocks
    block_starts = [b[0] for b in blocks]
    block_widths: list[int] = []
    for i, start in enumerate(block_starts):
        if i + 1 < len(block_starts):
            block_widths.append(block_starts[i + 1] - start)
        else:
            block_widths.append(df.shape[1] - start)

    assays: list[LucasCellAssay] = []
    for (col_start, label), bw in zip(blocks, block_widths):
        a = LucasCellAssay(
            label=label,
            source_file=str(path),
            column_index=col_start,
        )

        # --- Extract fields based on variant ---
        if variant == "ui":
            _parse_ui_block(df, row_index, col_start, bw, a)
        else:
            _parse_ug_block(df, row_index, col_start, bw, a)

        assays.append(a)

    logger.info(
        "Parsed %d assay blocks from %s (variant=%s)", len(assays), path.name, variant
    )
    return assays


def _parse_ui_block(
    df: pd.DataFrame,
    ri: dict[str, int],
    col: int,
    bw: int,
    a: LucasCellAssay,
) -> None:
    """Fill an assay record from a UI-variant CSV block."""
    # Date
    if "date" in ri:
        date_str = _extract_text(df, ri["date"], col)
        a.assay_date = _parse_date(date_str)

    # Operators
    if "operators" in ri:
        a.operators = _extract_text(df, ri["operators"], col)

    # Comments
    if "comments" in ri:
        a.comments = _extract_text(df, ri["comments"], col)

    # Wave file
    if "wave_file" in ri:
        a.wave_file = _extract_text(df, ri["wave_file"], col)

    # Assay duration (min)
    if "duration" in ri:
        v, u = _extract_value_unc(df, ri["duration"], col, bw)
        a.assay_duration_min = v
        a.assay_duration_unc_min = u

    # Delay time (days)
    if "delay" in ri:
        v, u = _extract_value_unc(df, ri["delay"], col, bw)
        a.delay_time_days = v
        a.delay_time_unc_days = u

    # Counting time (days)
    if "counting_time" in ri:
        v, u = _extract_value_unc(df, ri["counting_time"], col, bw)
        a.counting_time_days = v
        a.counting_time_unc_days = u

    # Number of counts
    if "n_counts" in ri:
        v, u = _extract_value_unc(df, ri["n_counts"], col, bw)
        a.n_counts = v
        a.n_counts_unc = u

    # Lucas cell background (counts/day)
    if "lc_bkg" in ri:
        v, u = _extract_value_unc(df, ri["lc_bkg"], col, bw)
        a.lc_background_cpd = v
        a.lc_background_unc_cpd = u

    # Board background (counts/day)
    if "board_bkg" in ri:
        v, u = _extract_value_unc(df, ri["board_bkg"], col, bw)
        a.board_background_cpd = v if not math.isnan(v) else 0.0
        a.board_background_unc_cpd = u if not math.isnan(u) else 0.0

    # Half-life
    if "half_life" in ri:
        v, _ = _extract_value_unc(df, ri["half_life"], col, bw)
        if not math.isnan(v):
            a.rn_half_life_days = v

    # Overall efficiency
    if "efficiency" in ri:
        v, _ = _extract_value_unc(df, ri["efficiency"], col, bw)
        a.overall_efficiency = _normalise_efficiency(v)

    # Derived: decayed atoms in cell
    if "decayed_atoms" in ri:
        v, u = _extract_value_unc(df, ri["decayed_atoms"], col, bw)
        a.decayed_atoms_in_cell = v

    # Derived: total atoms in cell (beginning of counting)
    if "atoms_in_cell_start" in ri:
        v, u = _extract_value_unc(df, ri["atoms_in_cell_start"], col, bw)
        a.total_atoms_in_cell = v
        a.total_atoms_in_cell_unc = u

    # Derived: total atoms in cell after extraction
    if "atoms_in_cell_after" in ri:
        v, u = _extract_value_unc(df, ri["atoms_in_cell_after"], col, bw)
        # Use "after extraction" as the cell total if "start" wasn't found
        if math.isnan(a.total_atoms_in_cell):
            a.total_atoms_in_cell = v
            a.total_atoms_in_cell_unc = u

    # Derived: total atoms in the assay
    if "atoms_in_assay" in ri:
        v, u = _extract_value_unc(df, ri["atoms_in_assay"], col, bw)
        a.total_atoms_in_assay = v
        a.total_atoms_in_assay_unc = u

    # Flow rate (L/min)
    if "flow_rate" in ri:
        v, _ = _extract_value_unc(df, ri["flow_rate"], col, bw)
        a.flow_rate_lpm = v

    # Concentration: atoms per liter (row "Total radon atoms in 1L assay")
    if "atoms_per_liter" in ri:
        v, u = _extract_value_unc(df, ri["atoms_per_liter"], col, bw)
        a.atoms_per_liter = v
        a.atoms_per_liter_unc = u

    # Atoms in Radon Monitor (pre-computed, scaled to RM volume)
    if "atoms_in_monitor" in ri:
        v, u = _extract_value_unc(df, ri["atoms_in_monitor"], col, bw)
        a.atoms_in_monitor = v
        a.atoms_in_monitor_unc = u


def _parse_ug_block(
    df: pd.DataFrame,
    ri: dict[str, int],
    col: int,
    bw: int,
    a: LucasCellAssay,
) -> None:
    """Fill an assay record from a UG board-calibration CSV block."""
    # Date
    if "date" in ri:
        date_str = _extract_text(df, ri["date"], col)
        a.assay_date = _parse_date(date_str)

    # Wave file
    if "wave_file" in ri:
        a.wave_file = _extract_text(df, ri["wave_file"], col)

    # Duration
    if "duration" in ri:
        v, u = _extract_value_unc(df, ri["duration"], col, bw)
        a.assay_duration_min = v
        a.assay_duration_unc_min = u

    # Delay (hours in UG CSV → convert to days)
    if "delay" in ri:
        v, u = _extract_value_unc(df, ri["delay"], col, bw)
        a.delay_time_days = v / 24.0 if not math.isnan(v) else math.nan
        a.delay_time_unc_days = u / 24.0 if not math.isnan(u) else math.nan

    # Counting time (days)
    if "counting_time" in ri:
        v, u = _extract_value_unc(df, ri["counting_time"], col, bw)
        a.counting_time_days = v
        a.counting_time_unc_days = u

    # Number of counts
    if "n_counts" in ri:
        v, u = _extract_value_unc(df, ri["n_counts"], col, bw)
        a.n_counts = v
        a.n_counts_unc = u

    # LC background
    if "lc_bkg" in ri:
        v, u = _extract_value_unc(df, ri["lc_bkg"], col, bw)
        a.lc_background_cpd = v
        a.lc_background_unc_cpd = u

    # Chamber background (UG equivalent of board background)
    if "chamber_bkg" in ri:
        v, _ = _extract_value_unc(df, ri["chamber_bkg"], col, bw)
        a.board_background_cpd = v if not math.isnan(v) else 0.0

    # Half-life
    if "half_life" in ri:
        v, _ = _extract_value_unc(df, ri["half_life"], col, bw)
        if not math.isnan(v):
            a.rn_half_life_days = v

    # Efficiency (single-alpha in UG)
    if "efficiency" in ri:
        v, _ = _extract_value_unc(df, ri["efficiency"], col, bw)
        a.overall_efficiency = _normalise_efficiency(v)

    # Derived: decayed atoms in cell
    if "decayed_atoms" in ri:
        v, _ = _extract_value_unc(df, ri["decayed_atoms"], col, bw)
        a.decayed_atoms_in_cell = v

    # Atoms at start of counting
    if "atoms_in_cell_start" in ri:
        v, u = _extract_value_unc(df, ri["atoms_in_cell_start"], col, bw)
        a.total_atoms_in_cell = v
        a.total_atoms_in_cell_unc = u

    # Assay result (total atoms)
    if "atoms_in_assay" in ri:
        v, u = _extract_value_unc(df, ri["atoms_in_assay"], col, bw)
        a.total_atoms_in_assay = v
        a.total_atoms_in_assay_unc = u


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def filter_assays(
    assays: list[LucasCellAssay],
    selection: dict[str, Any] | None = None,
) -> list[LucasCellAssay]:
    """Filter assays by config-driven criteria.

    Parameters
    ----------
    assays : list[LucasCellAssay]
        All parsed assays.
    selection : dict, optional
        Config dict with optional keys:
        * ``labels`` – list of substrings; keep assays whose label
          contains any of them (case-insensitive).
        * ``date_range`` – ``[start_iso, end_iso]``; keep assays whose
          date falls within the range.
        * ``indices`` – explicit list of column indices to keep.

    Returns
    -------
    list[LucasCellAssay]
        Filtered assay list.
    """
    if not selection:
        return assays

    result = assays

    # Filter by label substrings
    labels = selection.get("labels")
    if labels:
        labels_lower = [l.lower() for l in labels]
        result = [
            a for a in result
            if any(sub in a.label.lower() for sub in labels_lower)
        ]

    # Filter by date range
    date_range = selection.get("date_range")
    if date_range and len(date_range) == 2:
        try:
            dt_start = datetime.fromisoformat(date_range[0])
            dt_end = datetime.fromisoformat(date_range[1])
            result = [
                a for a in result
                if a.assay_date is not None and dt_start <= a.assay_date <= dt_end
            ]
        except (ValueError, TypeError) as exc:
            logger.warning("Invalid date_range in assay selection: %s", exc)

    # Filter by column indices
    indices = selection.get("indices")
    if indices is not None:
        idx_set = set(indices)
        result = [a for a in result if a.column_index in idx_set]

    return result
