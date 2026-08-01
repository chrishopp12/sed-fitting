"""
sources.py

Source-Table Consumption
---------------------------------------------------------

Reading and row selection for acquisition source tables. Tables must
carry band, flux_uJy, flux_err_uJy, and source; arbitrary extra columns
are preserved. Row selection filters by band and by the provider token's
source-string prefix; zero or multiple matches per band are hard errors.
Flux and error validity is enforced on selected rows only. Tables that
carry target coordinates are cross-checked against the roster position.

Requirements:
  - numpy, pandas, astropy
"""

from __future__ import annotations

import warnings
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord

# ------------------------------------
# Constants
# ------------------------------------

# Provenance vocabulary for source-CSV rows: a row belongs to a provider
# token when its source column starts with that token's prefix. The
# strings appear in data files on disk -- append new tokens; never rename
# an existing prefix. The set must stay prefix-free so startswith
# matching is unambiguous.
SOURCE_PREFIXES = {
    "legacy":    "Legacy_",
    "unwise":    "unWISE_Legacy_",
    "allwise":   "AllWISE",
    "galex":     "GALEX_",
    "sdss":      "SDSS_",
    "panstarrs": "PanSTARRS_",
    "jplus":     "JPLUS_",
    "hst":       "HST_HAP_",
    "aperture":  "sedphot_aperture_",
    "sersic":    "sedphot_sersic_",
}

REQUIRED_COLUMNS = ("band", "flux_uJy", "flux_err_uJy", "source")
POSITION_TOL_ARCSEC = 0.5


# ------------------------------------
# Reading
# ------------------------------------

def read_source_table(path: str | Path) -> pd.DataFrame:
    """Read a source table, enforcing the required columns file-wide.

    Parameters
    ----------
    path : str or Path
        CSV with at least REQUIRED_COLUMNS; extra columns are preserved.

    Returns
    -------
    frame : pd.DataFrame
        The table as read, unmodified.
    """
    path = Path(path)
    frame = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"{path}: missing required columns {missing}")
    if frame.empty:
        raise ValueError(f"{path}: empty source table")
    return frame


def provider_token(provider: str) -> str:
    """Leading token of a roster provider string."""
    return provider.split(":", 1)[0].strip()


def source_prefix(provider: str) -> str:
    """Source-string prefix for a roster provider entry's leading token."""
    token = provider_token(provider)
    if token not in SOURCE_PREFIXES:
        raise ValueError(f"unknown provider token {token!r} "
                         f"(known: {sorted(SOURCE_PREFIXES)})")
    return SOURCE_PREFIXES[token]


# ------------------------------------
# Row selection
# ------------------------------------

def match_counts(
        frame: pd.DataFrame,
        bands: list[str] | tuple[str, ...],
        *,
        provider: str,
    ) -> dict[str, int]:
    """Rows matching each band under the provider's source prefix.

    Presence and uniqueness checks at roster load; the error-validating
    selection at build time is select_rows.
    """
    prefix = source_prefix(provider)
    in_provider = frame["source"].astype(str).str.startswith(prefix)
    return {band: int(((frame["band"] == band) & in_provider).sum())
            for band in bands}


def select_rows(
        frame: pd.DataFrame,
        bands: list[str] | tuple[str, ...],
        *,
        provider: str,
        label: str,
    ) -> pd.DataFrame:
    """Select exactly one row per band under the provider's source prefix.

    Parameters
    ----------
    bands : list of str
        Bands to select, each exactly once.
    provider : str
        Roster provider string; its leading token keys SOURCE_PREFIXES.
    label : str
        Context for error messages (e.g. "BCG/legacy").

    Returns
    -------
    rows : pd.DataFrame
        One row per requested band, in request order, index reset.
    """
    if len(set(bands)) != len(bands):
        raise ValueError(f"{label}: duplicate bands in selection: {list(bands)}")
    prefix = source_prefix(provider)
    in_provider = frame["source"].astype(str).str.startswith(prefix)

    picked = []
    for band in bands:
        hits = frame[(frame["band"] == band) & in_provider]
        if len(hits) == 0:
            raise ValueError(f"{label}: band {band!r} matches no row with "
                             f"source prefix {prefix!r}")
        if len(hits) > 1:
            raise ValueError(f"{label}: band {band!r} is ambiguous: "
                             f"{len(hits)} rows with source prefix {prefix!r}")
        picked.append(hits)
    rows = pd.concat(picked, ignore_index=True)
    _validate_consumed(rows, label)
    return rows


def _validate_consumed(rows: pd.DataFrame, label: str) -> None:
    """Hard error when any selected row has invalid flux or error."""
    flux = rows["flux_uJy"].to_numpy(dtype=float)
    err = rows["flux_err_uJy"].to_numpy(dtype=float)
    bad = ~np.isfinite(flux) | ~np.isfinite(err) | (err <= 0)
    if bad.any():
        bad_bands = sorted(set(rows.loc[bad, "band"].astype(str)))
        raise ValueError(f"{label}: consumed rows with non-finite flux or "
                         f"non-positive error: {bad_bands}")


# ------------------------------------
# Position cross-check
# ------------------------------------

def check_position(
        frame: pd.DataFrame,
        ra_deg: float,
        dec_deg: float,
        *,
        label: str,
        tol_arcsec: float = POSITION_TOL_ARCSEC,
    ) -> None:
    """Cross-check table target coordinates against the roster position.

    A table carrying no target_ra/target_dec columns is exempt from the
    check and raises a UserWarning rather than passing silently; rows
    with empty coordinates are skipped.

    Parameters
    ----------
    frame : pd.DataFrame
        The source table to check.
    ra_deg, dec_deg : float
        The roster target's position.
    label : str
        Prefix identifying the table in messages.
    tol_arcsec : float
        Maximum allowed separation. [default: 0.5]
    """
    if "target_ra" not in frame.columns or "target_dec" not in frame.columns:
        warnings.warn(
            f"{label}: no target_ra/target_dec columns, so the position "
            f"cross-check is SKIPPED for this table",
            UserWarning, stacklevel=2)
        return
    ra = np.asarray(frame["target_ra"], dtype=float)
    dec = np.asarray(frame["target_dec"], dtype=float)
    have = np.isfinite(ra) & np.isfinite(dec)
    if not have.any():
        return
    expected = SkyCoord(ra_deg * u.deg, dec_deg * u.deg)
    table_pos = SkyCoord(ra[have] * u.deg, dec[have] * u.deg)
    worst = float(expected.separation(table_pos).arcsec.max())
    if worst > tol_arcsec:
        raise ValueError(f"{label}: table target position is {worst:.3f} arcsec "
                         f"from the roster position (tolerance {tol_arcsec})")
