"""
policy.py

Shared Data Policy
---------------------------------------------------------

Band inclusion, signal-to-noise gating, lensing magnification, and the
error floor, applied between the SED table and either backend. Both
backends consume this module's output and run with their internal
floors zeroed.

Requirements:
  - numpy, pandas

Notes:
  - Order: magnification first (flux and error divided by mu), then the
    clamped floor err' = sqrt(err^2 + (floor * max(flux, 0))^2). The two
    commute.
  - min_snr_broadband is evaluated on statistical (pre-floor) errors;
    SPHEREx channels are exempt. 0 disables the gate.
  - MISSING_SENTINEL marks a row a backend must treat as unobserved. It
    sits far below any physical flux, and each backend maps it onto its
    own not-observed convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from sedfit.core.registry import Registry

# ------------------------------------
# Constants
# ------------------------------------

MISSING_SENTINEL = -1.0e30
DEFAULT_MIN_VALID_BANDS = 5
DEFAULT_MIN_SNR_BROADBAND = 2.0

# The numeric qa_flags tokens a gate may name. The emitting
# measurement's vocabulary is append-only; unknown tokens in data are
# ignored, and a gate naming a token outside this set is a hard error.
QA_GATE_TOKENS = ("art", "atbound", "bg", "conv", "cov", "excess", "far",
                  "leash", "maskfrac", "mesh", "nbsub", "pedb", "refit",
                  "reg", "twinfrac")


# ------------------------------------
# Data model
# ------------------------------------

@dataclass(frozen=True)
class PolicyResult:
    """Backend-ready photometry with the full inclusion accounting.

    flux_uJy / flux_err_uJy are the transformed vectors (mu divided,
    floor applied), full table length; fittable marks the rows a backend
    fits. Excluded rows keep transformed values -- backends replace them
    with their own missing representation (sentinel or omission).
    floor_by_band is the fraction applied to each band.
    """
    bands: tuple[str, ...]
    flux_uJy: np.ndarray
    flux_err_uJy: np.ndarray
    fittable: np.ndarray
    included: np.ndarray
    excluded_by_set: tuple[str, ...]
    low_snr: tuple[str, ...]
    floor_by_band: dict[str, float]
    n_fit: int
    qa_rejected: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def accounting(self) -> dict:
        """Per-band lists and counts for the run manifest."""
        return {
            "n_bands_table": len(self.bands),
            "n_bands_fit": self.n_fit,
            "n_excluded": len(self.excluded_by_set),
            "n_low_snr": len(self.low_snr),
            "n_qa_rejected": len(self.qa_rejected),
            "included": [b for b, ok in zip(self.bands, self.fittable) if ok],
            "excluded_by_set": list(self.excluded_by_set),
            "low_snr": list(self.low_snr),
            "qa_rejected": list(self.qa_rejected),
            "warnings": list(self.warnings),
        }


# ------------------------------------
# Inclusion
# ------------------------------------

def _inclusion_mask(
        bands: list[str],
        bands_include: list[str] | tuple[str, ...] | None,
        registry: Registry,
    ) -> np.ndarray:
    if bands_include is None:
        return np.ones(len(bands), dtype=bool)
    if not bands_include:
        raise ValueError("bands_include: empty list (use null for all bands)")

    instruments = [registry.instrument_of(b) for b in bands]
    mask = np.zeros(len(bands), dtype=bool)
    for entry in bands_include:
        if entry in registry.instruments:
            hits = np.array([inst == entry for inst in instruments])
        elif entry in registry or registry.is_spherex(entry):
            hits = np.array([b == entry for b in bands])
        else:
            raise ValueError(f"bands_include: {entry!r} is neither a registry "
                             f"instrument nor a known band")
        if not hits.any():
            raise ValueError(f"bands_include: {entry!r} matches no band in "
                             f"this table")
        mask |= hits
    return mask


# ------------------------------------
# QA gates
# ------------------------------------

def parse_qa_tokens(value: str | float | None) -> dict[str, float]:
    """Numeric tokens of one row's qa_flags string.

    Parameters
    ----------
    value : str or float or None
        A ';'-joined key=value string, or the NaN or None a row without
        tokens carries.

    Returns
    -------
    tokens : dict
        Token name -> numeric value; non-numeric segments are skipped.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return {}
    tokens = {}
    for segment in str(value).split(";"):
        if not segment:
            continue
        key, _, text = segment.partition("=")
        try:
            tokens[key] = float(text)
        except ValueError:
            continue
    return tokens


def validate_qa_gates(qa_gates: dict) -> None:
    """Hard error on any structural or vocabulary violation.

    Parameters
    ----------
    qa_gates : dict
        Non-empty map of qa_flags token -> bounds, where the token is
        one of QA_GATE_TOKENS and bounds is a non-empty map holding a
        'min' and/or a 'max' number.
    """
    if not isinstance(qa_gates, dict) or not qa_gates:
        raise ValueError("qa_gates must be null or a non-empty map")
    unknown = sorted(set(qa_gates) - set(QA_GATE_TOKENS))
    if unknown:
        raise ValueError(f"qa_gates: unrecognized tokens {unknown} (known: "
                         f"{list(QA_GATE_TOKENS)})")
    for token, bounds in qa_gates.items():
        if not isinstance(bounds, dict) or not bounds:
            raise ValueError(f"qa_gates[{token}] must be a map with 'min' "
                             f"and/or 'max'")
        extra = sorted(set(bounds) - {"min", "max"})
        if extra:
            raise ValueError(f"qa_gates[{token}]: unrecognized keys {extra}")
        for edge, value in bounds.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"qa_gates[{token}].{edge} must be a number")


def _qa_gate_mask(
        qa_gates: dict,
        frame: pd.DataFrame,
        included: np.ndarray,
    ) -> tuple[np.ndarray, list[str]]:
    """Rows failing any gate; a gate fires only on rows carrying its
    token (absence means the condition never triggered)."""
    validate_qa_gates(qa_gates)
    tokens_by_row = [parse_qa_tokens(v) for v in frame["qa_flags"]]
    rejected = np.zeros(len(tokens_by_row), dtype=bool)
    warnings = []
    for token in sorted(qa_gates):
        bounds = qa_gates[token]
        values = np.array([row.get(token, np.nan) for row in tokens_by_row])
        carries = np.isfinite(values)
        if not carries.any():
            warnings.append(f"qa_gates token {token!r} appears on no row "
                            f"in this fit")
            continue
        fail = np.zeros(len(tokens_by_row), dtype=bool)
        if "min" in bounds:
            fail |= carries & (values < bounds["min"])
        if "max" in bounds:
            fail |= carries & (values > bounds["max"])
        rejected |= included & fail
    return rejected, warnings


# ------------------------------------
# Error floor
# ------------------------------------

def resolve_floors(
        err_floor: float | dict,
        bands: list[str],
        included: np.ndarray,
        registry: Registry,
    ) -> tuple[dict[str, float], list[str]]:
    """Per-band floor fractions from a scalar or per-instrument map.

    Parameters
    ----------
    err_floor : float or dict
        Fractional calibration floor: a non-negative scalar for every
        band, or a map of registry instrument name -> fraction with an
        optional 'default'.
    bands : list of str
        Table bands, in table order.
    included : np.ndarray
        Boolean mask of the bands this fit includes; an included band
        the map does not reach is a hard error.
    registry : Registry
        Instrument authority for the map keys and the band names.

    Returns
    -------
    floor_by_band : dict
        Band -> floor fraction, every table band; an excluded band the
        map does not reach gets 0.0.
    warnings : list of str
        Valid map keys that match no band in this fit.
    """
    if isinstance(err_floor, (int, float)):
        if err_floor < 0:
            raise ValueError(f"err_floor: negative floor {err_floor}")
        return {b: float(err_floor) for b in bands}, []

    if not isinstance(err_floor, dict) or not err_floor:
        raise ValueError("err_floor must be a number or a non-empty map")
    allowed = set(registry.instruments) | {"default"}
    unknown = sorted(set(err_floor) - allowed)
    if unknown:
        raise ValueError(f"err_floor: unrecognized keys {unknown} (keys must "
                         f"be registry instruments or 'default')")
    for key, value in err_floor.items():
        if not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"err_floor[{key}]: must be a non-negative "
                             f"number")

    instruments = [registry.instrument_of(b) for b in bands]
    warnings = [f"err_floor key {key!r} matches no band in this fit"
                for key in sorted(set(err_floor) - {"default"})
                if key not in set(instruments)]

    floor_by_band = {}
    floorless = []
    for band, instrument, is_included in zip(bands, instruments, included):
        if instrument in err_floor:
            floor_by_band[band] = float(err_floor[instrument])
        elif "default" in err_floor:
            floor_by_band[band] = float(err_floor["default"])
        else:
            floor_by_band[band] = 0.0
            if is_included:
                floorless.append(band)
    if floorless:
        raise ValueError(f"err_floor: included bands with no applicable "
                         f"floor and no 'default': {sorted(set(floorless))}")
    return floor_by_band, warnings


# ------------------------------------
# Policy
# ------------------------------------

def apply_policy(
        frame: pd.DataFrame,
        *,
        registry: Registry,
        bands_include: list[str] | tuple[str, ...] | None = None,
        min_valid_bands: int = DEFAULT_MIN_VALID_BANDS,
        min_snr_broadband: float = DEFAULT_MIN_SNR_BROADBAND,
        err_floor: float | dict = 0.05,
        mu_lensing: float = 1.0,
        qa_gates: dict | None = None,
    ) -> PolicyResult:
    """Apply the shared data policy to a validated SED table.

    Parameters
    ----------
    frame : pd.DataFrame
        SED table (validated by core.table on read).
    registry : Registry
        Band authority: instrument names, SPHEREx membership, and the
        membership tests behind bands_include.
    bands_include : list or None
        Registry instrument names and/or band names; None fits every
        table band. An entry matching nothing is a hard error.
    min_valid_bands : int
        Hard error when fewer fittable rows survive. [default: 5]
    min_snr_broadband : float
        Mark broadband rows below this statistical S/N missing; SPHEREx
        exempt; 0 disables. [default: 2.0]
    err_floor : float or dict
        Fractional calibration floor, scalar or per-instrument map with
        optional 'default'. [default: 0.05]
    mu_lensing : float
        Lensing magnification; flux and error are divided by it.
        [default: 1.0]
    qa_gates : dict or None
        Map of qa_flags token -> {'min' and/or 'max'} bounds; rows
        failing a gate are marked missing like low-S/N rows. A gate
        fires only on rows carrying its token. None disables.
        [default: None]

    Returns
    -------
    result : PolicyResult
        Transformed flux and error vectors at full table length, the
        fittable mask, and the per-band inclusion accounting.
    """
    if not (isinstance(mu_lensing, (int, float)) and mu_lensing > 0):
        raise ValueError(f"mu_lensing must be positive, got {mu_lensing!r}")
    if min_snr_broadband < 0:
        raise ValueError("min_snr_broadband must be >= 0")

    bands = [str(b) for b in frame["band"]]
    included = _inclusion_mask(bands, bands_include, registry)

    flux = frame["flux_uJy"].to_numpy(dtype=float) / mu_lensing
    err_stat = frame["flux_err_uJy"].to_numpy(dtype=float) / mu_lensing

    is_spherex = np.array([registry.is_spherex(b) for b in bands])
    low_snr = np.zeros(len(bands), dtype=bool)
    if min_snr_broadband > 0:
        snr = flux / err_stat
        low_snr = included & ~is_spherex & (snr < min_snr_broadband)

    qa_rejected = np.zeros(len(bands), dtype=bool)
    qa_warnings: list[str] = []
    if qa_gates is not None:
        qa_rejected, qa_warnings = _qa_gate_mask(qa_gates, frame, included)

    floor_by_band, warnings = resolve_floors(err_floor, bands, included,
                                             registry)
    floors = np.array([floor_by_band[b] for b in bands])
    err = np.sqrt(err_stat**2 + (floors * np.maximum(flux, 0.0))**2)

    fittable = included & ~low_snr & ~qa_rejected
    n_fit = int(fittable.sum())
    if n_fit < min_valid_bands:
        raise ValueError(f"only {n_fit} fittable bands after inclusion and "
                         f"gates (min_valid_bands={min_valid_bands})")

    return PolicyResult(
        bands=tuple(bands),
        flux_uJy=flux,
        flux_err_uJy=err,
        fittable=fittable,
        included=included,
        excluded_by_set=tuple(b for b, ok in zip(bands, included) if not ok),
        low_snr=tuple(b for b, m in zip(bands, low_snr) if m),
        floor_by_band=floor_by_band,
        n_fit=n_fit,
        qa_rejected=tuple(b for b, m in zip(bands, qa_rejected) if m),
        warnings=tuple(warnings) + tuple(qa_warnings),
    )
