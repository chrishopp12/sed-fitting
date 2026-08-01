"""
table.py

SED Table Schema
---------------------------------------------------------

The assembled SED table: write, read, validate. Validation runs on every
read and is the authoritative unknown-band and error check. SPHEREx rows
carry per-object channel geometry (wave_um, bandwidth_um) plus
scatter_uJy and n_exp; broadband rows leave all four empty and take
their bandpasses from the registry. qa_flags carries the emitting
measurement's QA token string on broadband rows (empty when the source
supplies none); SPHEREx rows leave it empty.

Requirements:
  - numpy, pandas
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from sedfit.core.registry import Registry

# ------------------------------------
# Constants
# ------------------------------------

COLUMNS = ("band", "flux_uJy", "flux_err_uJy", "scatter_uJy",
           "wave_um", "bandwidth_um", "n_exp", "qa_flags")
SPHEREX_ONLY_COLUMNS = ("scatter_uJy", "wave_um", "bandwidth_um", "n_exp")


# ------------------------------------
# Validation
# ------------------------------------

def validate_sed_table(
        frame: pd.DataFrame,
        registry: Registry,
        *,
        label: str = "sed table",
    ) -> None:
    """Hard error on any schema or content violation.

    Checks: exact column set and order; non-empty; unique bands; every
    band registry-identifiable; finite flux; finite positive errors;
    SPHEREx rows with positive channel geometry, non-negative finite
    scatter, and integer n_exp >= 1; broadband rows with the SPHEREx-only
    columns empty; qa_flags empty on SPHEREx rows and empty or a
    parseable ';'-joined key=value token string on broadband rows.
    """
    if tuple(frame.columns) != COLUMNS:
        raise ValueError(f"{label}: columns {list(frame.columns)} != "
                         f"required {list(COLUMNS)}")
    if frame.empty:
        raise ValueError(f"{label}: empty table")

    bands = [str(b) for b in frame["band"]]
    duplicates = sorted(b for b, count in Counter(bands).items() if count > 1)
    if duplicates:
        raise ValueError(f"{label}: duplicate bands {duplicates}")
    registry.require_known(bands)

    flux = frame["flux_uJy"].to_numpy(dtype=float)
    err = frame["flux_err_uJy"].to_numpy(dtype=float)
    if not np.isfinite(flux).all():
        bad = sorted(np.asarray(bands)[~np.isfinite(flux)])
        raise ValueError(f"{label}: non-finite flux_uJy: {bad}")
    bad_err = ~np.isfinite(err) | (err <= 0)
    if bad_err.any():
        raise ValueError(f"{label}: non-finite or non-positive flux_err_uJy: "
                         f"{sorted(np.asarray(bands)[bad_err])}")

    is_spherex = np.array([registry.is_spherex(b) for b in bands])
    broadband_labels = np.asarray(bands)[~is_spherex]
    spherex_labels = np.asarray(bands)[is_spherex]

    for column in SPHEREX_ONLY_COLUMNS:
        values = frame[column].to_numpy(dtype=float)[~is_spherex]
        filled = np.isfinite(values)
        if filled.any():
            raise ValueError(f"{label}: broadband rows must leave {column} "
                             f"empty: {sorted(broadband_labels[filled])}")

    for column in ("wave_um", "bandwidth_um"):
        values = frame[column].to_numpy(dtype=float)[is_spherex]
        bad = ~np.isfinite(values) | (values <= 0)
        if bad.any():
            raise ValueError(f"{label}: SPHEREx rows need positive {column}: "
                             f"{sorted(spherex_labels[bad])}")

    scatter = frame["scatter_uJy"].to_numpy(dtype=float)[is_spherex]
    bad_scatter = ~np.isfinite(scatter) | (scatter < 0)
    if bad_scatter.any():
        raise ValueError(f"{label}: SPHEREx rows need finite non-negative "
                         f"scatter_uJy: {sorted(spherex_labels[bad_scatter])}")

    n_exp = frame["n_exp"].to_numpy(dtype=float)[is_spherex]
    bad_n = ~np.isfinite(n_exp) | (n_exp < 1) | (n_exp != np.round(n_exp))
    if bad_n.any():
        raise ValueError(f"{label}: SPHEREx rows need integer n_exp >= 1: "
                         f"{sorted(spherex_labels[bad_n])}")

    has_qa = frame["qa_flags"].notna().to_numpy()
    if has_qa[is_spherex].any():
        filled = np.asarray(bands)[is_spherex & has_qa]
        raise ValueError(f"{label}: SPHEREx rows must leave qa_flags empty: "
                         f"{sorted(filled)}")
    qa_values = frame["qa_flags"].to_numpy(dtype=object)
    for band, value in zip(np.asarray(bands)[has_qa], qa_values[has_qa]):
        segments = [s for s in str(value).split(";") if s]
        if not segments or any("=" not in s or not s.split("=", 1)[0]
                               for s in segments):
            raise ValueError(f"{label}: unparseable qa_flags on {band}: "
                             f"{value!r}")


# ------------------------------------
# Write / read
# ------------------------------------

def write_sed_table(
        frame: pd.DataFrame,
        path: str | Path,
        registry: Registry,
    ) -> None:
    """Validate and write; empty CSV fields encode the NaN cells."""
    path = Path(path)
    validate_sed_table(frame, registry, label=str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, na_rep="")


def read_sed_table(path: str | Path, registry: Registry) -> pd.DataFrame:
    """Read an SED table and validate it against the registry."""
    path = Path(path)
    frame = pd.read_csv(path)
    validate_sed_table(frame, registry, label=str(path))
    return frame
