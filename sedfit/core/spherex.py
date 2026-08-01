"""
spherex.py

SPHEREx Ingest
---------------------------------------------------------

Reads the IRSA Spectrophotometry Tool per-exposure table (one row per
visit x LVF channel), applies quality cuts, and bins repeat visits into
a per-channel spectrum.

Binning is model-matched: blue of the C3K -> BaSeL handoff only
near-duplicate visits merge (the blue side stays dense); red of it,
visits adaptive-bin to about one local resolution element. Each channel
reports both the inverse-variance statistical error (bins down with N)
and the visit-to-visit scatter (does not bin down); fit policy chooses
between them.

Requirements:
  - numpy, pandas

Notes:
  - Strict contract: lambda, lambda_width, flux, flux_err, flags must
    all be present, and a requested cut whose column is absent (fit_ql,
    local_bkg_flg) is a hard error.
  - Default flag policy rejects any visit with any flag bit set other
    than SOURCE (bit 21, set on every forced-photometry row by
    construction). Pass flags_reject=0 to keep every visit.
  - A single-visit bin has no scatter to measure and reports its
    statistical error as its scatter.
  - bandwidth_um is the larger of the channel width and the bin's
    wavelength span.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------------
# Constants
# ------------------------------------

# FSPS C3K high-resolution spectra end at this rest wavelength; BaSeL
# (coarse) fills beyond it. The observed-frame binning split rides at
# (1 + z) times this.
C3K_BASEL_HANDOFF_UM = 2.40

# SPHEREx flag bits: 0-21 are the Level-2 image-pixel flags (Explanatory
# Supplement v1.0, Table 9, sampled at the source); 32 is the
# forced-photometry catalog bad-pixel roll-up. Reference table -- only
# SOURCE is consumed below.
FLAG_BITS = {
    "TRANSIENT": 0,
    "OVERFLOW": 1,
    "SUR_ERROR": 2,
    "NONFUNC": 6,
    "DICHROIC": 7,
    "MISSING_DATA": 9,
    "HOT": 10,
    "COLD": 11,
    "FULLSAMPLE": 12,
    "PHANMISS": 14,
    "NONLINEAR": 15,
    "PERSIST": 17,
    "OUTLIER": 19,
    "SOURCE": 21,
    "CONTAINS_BAD_PIXEL": 32,
}
SOURCE_FLAG = 1 << FLAG_BITS["SOURCE"]
# Reject a visit if any of bits 0-39 other than SOURCE is set: the
# documented bits plus any undocumented bit in that range.
DEFAULT_FLAGS_REJECT = ((1 << 40) - 1) & ~SOURCE_FLAG

REQUIRED_COLUMNS = ("lambda", "lambda_width", "flux", "flux_err", "flags")


# ------------------------------------
# Loading and quality cuts
# ------------------------------------

def load_visits(path: str | Path) -> pd.DataFrame:
    """Read a raw IRSA per-exposure table, enforcing the column contract."""
    path = Path(path)
    frame = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"{path}: missing required SPHEREx columns {missing}")
    if frame.empty:
        raise ValueError(f"{path}: empty SPHEREx table")
    return frame


def apply_quality_cuts(
        frame: pd.DataFrame,
        *,
        fit_ql_max: float | None = None,
        flags_reject: int = DEFAULT_FLAGS_REJECT,
        drop_local_bkg: bool = False,
    ) -> pd.DataFrame:
    """Drop per-visit rows failing the requested quality cuts.

    Parameters
    ----------
    fit_ql_max : float or None
        Drop rows with fit_ql above this. [default: no cut]
    flags_reject : int
        Bitmask; drop rows where flags & flags_reject is non-zero.
        [default: DEFAULT_FLAGS_REJECT -- every bit except SOURCE]
    drop_local_bkg : bool
        Drop rows with local_bkg_flg set. local_bkg_flg marks the
        background method, not data quality, so it is independent of
        flags_reject. [default: False]
    """
    keep = np.ones(len(frame), dtype=bool)
    if fit_ql_max is not None:
        if "fit_ql" not in frame.columns:
            raise ValueError("fit_ql cut requested but the table has no "
                             "fit_ql column")
        keep &= frame["fit_ql"].to_numpy() <= fit_ql_max
    if flags_reject:
        if "flags" not in frame.columns:
            raise ValueError("flag cut requested but the table has no "
                             "flags column")
        flags = frame["flags"].to_numpy().astype(np.int64)
        keep &= (flags & int(flags_reject)) == 0
    if drop_local_bkg:
        if "local_bkg_flg" not in frame.columns:
            raise ValueError("local_bkg cut requested but the table has no "
                             "local_bkg_flg column")
        keep &= ~frame["local_bkg_flg"].to_numpy().astype(bool)
    return frame[keep].reset_index(drop=True)


# ------------------------------------
# Binning
# ------------------------------------

def default_split_um(z_ref: float) -> float:
    """Observed-frame binning split: (1 + z_ref) x the C3K/BaSeL handoff."""
    return (1.0 + float(z_ref)) * C3K_BASEL_HANDOFF_UM


def _combine(
        group: pd.DataFrame,
    ) -> tuple[float, float, float, float, float, int]:
    """Inverse-variance combine one bin's visits into a single channel.

    Returns
    -------
    channel : tuple
        (wave_um, bandwidth_um, flux_uJy, flux_err_uJy, scatter_uJy,
        n_exp), in the column order bin_spectrum assembles its frame
        with. bandwidth_um is the larger of the widest channel width in
        the bin and the bin's wavelength span; a single-visit bin
        reports its statistical error as scatter_uJy.
    """
    lam = group["lambda"].to_numpy(dtype=float)
    flux = group["flux"].to_numpy(dtype=float)
    err = group["flux_err"].to_numpy(dtype=float)
    weight = 1.0 / np.square(err)
    wsum = weight.sum()
    fbar = float((weight * flux).sum() / wsum)
    lbar = float((weight * lam).sum() / wsum)
    stat = float(1.0 / np.sqrt(wsum))
    n = len(group)
    scatter = float(np.std(flux, ddof=1)) if n > 1 else stat
    width = float(max(group["lambda_width"].max(), lam.max() - lam.min()))
    return lbar, width, fbar, stat, scatter, n


def bin_spectrum(
        frame: pd.DataFrame,
        *,
        split_um: float,
        blue_merge_dlam_um: float = 0.001,
        red_bin_resolution: float = 1.0,
    ) -> pd.DataFrame:
    """Model-matched split binning of a quality-cut per-visit table.

    Parameters
    ----------
    split_um : float
        Observed-frame blue/red boundary (default_split_um for the
        rest-frame handoff at a target redshift).
    blue_merge_dlam_um : float
        Blue of the split, merge only visits spanning at most this
        range. [default: 0.001]
    red_bin_resolution : float
        Red of the split, bin visits to this many local resolution
        elements (lambda_width at the bin start). [default: 1.0]

    Returns
    -------
    spectrum : pd.DataFrame
        Columns band, flux_uJy, flux_err_uJy, scatter_uJy, wave_um,
        bandwidth_um, n_exp; channels labeled SPHEREx_NNN in ascending
        wavelength order.
    """
    ordered = frame.sort_values("lambda").reset_index(drop=True)
    lam = ordered["lambda"].to_numpy(dtype=float)
    width = ordered["lambda_width"].to_numpy(dtype=float)

    bin_id = np.empty(len(ordered), dtype=int)
    current = 0
    i = 0
    while i < len(ordered):
        if lam[i] <= split_um:
            j = i
            while (j + 1 < len(ordered) and lam[j + 1] <= split_um
                   and lam[j + 1] - lam[i] <= blue_merge_dlam_um):
                j += 1
        else:
            span = width[i] * red_bin_resolution
            j = i
            while j + 1 < len(ordered) and lam[j + 1] - lam[i] <= span:
                j += 1
        bin_id[i:j + 1] = current
        current += 1
        i = j + 1

    rows = [_combine(ordered[bin_id == b]) for b in range(current)]
    spectrum = pd.DataFrame(
        rows, columns=["wave_um", "bandwidth_um", "flux_uJy",
                       "flux_err_uJy", "scatter_uJy", "n_exp"])
    spectrum = spectrum.sort_values("wave_um").reset_index(drop=True)
    spectrum.insert(0, "band",
                    [f"SPHEREx_{k:03d}" for k in range(len(spectrum))])
    return spectrum[["band", "flux_uJy", "flux_err_uJy", "scatter_uJy",
                     "wave_um", "bandwidth_um", "n_exp"]]


# ------------------------------------
# Ingest
# ------------------------------------

def ingest(
        path: str | Path,
        *,
        split_um: float,
        fit_ql_max: float | None = None,
        flags_reject: int = DEFAULT_FLAGS_REJECT,
        drop_local_bkg: bool = False,
        blue_merge_dlam_um: float = 0.001,
        red_bin_resolution: float = 1.0,
    ) -> tuple[pd.DataFrame, dict]:
    """Full SPHEREx ingest: load -> quality cuts -> split binning.

    Returns
    -------
    spectrum : pd.DataFrame
        Binned per-channel spectrum with columns band, flux_uJy,
        flux_err_uJy, scatter_uJy, wave_um, bandwidth_um, n_exp.
    counts : dict
        visits_total, visits_kept and channels: rows read, rows
        surviving the quality cuts, and channels in the spectrum.
    """
    visits = load_visits(path)
    kept = apply_quality_cuts(visits, fit_ql_max=fit_ql_max,
                              flags_reject=flags_reject,
                              drop_local_bkg=drop_local_bkg)
    if kept.empty:
        raise ValueError(f"{path}: quality cuts removed every visit")
    spectrum = bin_spectrum(kept, split_um=split_um,
                            blue_merge_dlam_um=blue_merge_dlam_um,
                            red_bin_resolution=red_bin_resolution)
    counts = {"visits_total": int(len(visits)),
              "visits_kept": int(len(kept)),
              "channels": int(len(spectrum))}
    return spectrum, counts
