"""
spectrum.py

Observed-Spectrum Input
---------------------------------------------------------

Read and validate a prepared observed spectrum for a joint
photometry+spectrum fit, and apply the shared data policy to it.
Preparation (flux calibration, telluric handling, error inflation,
frame conversion) happens upstream in target-specific tooling; this
module only enforces the file contract and the policy transforms, so
the fit consumes the spectrum exactly as prepared.

The file is a CSV with exactly the columns wave_A, flux_uJy,
flux_err_uJy, mask: observed-frame VACUUM wavelengths in Angstroms,
f_nu flux densities in microjanskys, and a 0/1 fit-inclusion mask.
A provenance sidecar (<stem>.provenance.json) is required and must
declare wave_frame "vacuum" and flux_unit "uJy"; its remaining content
is free-form preparation provenance, staged verbatim into the run
directory.

Requirements:
  - numpy, pandas
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from sedfit.core.provenance import sha256_bytes

# ------------------------------------
# Constants
# ------------------------------------

SPECTRUM_COLUMNS = ("wave_A", "flux_uJy", "flux_err_uJy", "mask")
SIDECAR_SUFFIX = ".provenance.json"
REQUIRED_WAVE_FRAME = "vacuum"
REQUIRED_FLUX_UNIT = "uJy"

# Standard-air dispersion relation: Edlen (1966) as revised by Birch & Downs
# (1994, Metrologia 31, 315), the IAU standard. These are the same five
# constants musetools.cube and the RM J0019 reference fit both carry, verified
# equal to 0.0 across 4750-9350 A; the linear backend's air Chebyshev only
# stays bit-identical to the reference while they agree.
_N_AIR_TERMS = (8.34254e-5, 2.406147e-2, 130.0, 1.5998e-4, 38.9)


# ------------------------------------
# Data model
# ------------------------------------

@dataclass(frozen=True)
class Spectrum:
    """One prepared observed spectrum, read once, with its provenance.

    wave_A is observed-frame vacuum; flux and error are f_nu in
    microjanskys; mask is the file's 0/1 column as booleans (True =
    fit). csv_bytes and sha256 are the file's exact content, for
    staging and run identity.
    """

    path: Path
    wave_A: np.ndarray
    flux_uJy: np.ndarray
    flux_err_uJy: np.ndarray
    mask: np.ndarray
    provenance: dict = field(repr=False)
    csv_bytes: bytes = field(repr=False)
    sha256: str = ""


# ------------------------------------
# Wavelength frames
# ------------------------------------

def n_air(wave_vac: np.ndarray | float) -> np.ndarray:
    """Refractive index of standard air at a vacuum wavelength [Angstrom]."""
    a, b1, c1, b2, c2 = _N_AIR_TERMS
    s2 = (1e4 / np.asarray(wave_vac, float)) ** 2
    return 1.0 + a + b1 / (c1 - s2) + b2 / (c2 - s2)


def vac_to_air(wave_vac: np.ndarray | float) -> np.ndarray:
    """Air wavelengths [Angstrom] for vacuum ones. Exact, not iterative.

    This module requires spectra in vacuum (REQUIRED_WAVE_FRAME), so nothing
    here reads air data. The conversion exists because the linear backend's
    multiplicative polynomial may be built on an air coordinate while its
    design matrix stays in vacuum -- a free choice of smooth coordinate, but
    not a neutral one: the offset runs 1.3 A at 4750 A to 2.6 A at 9350 A.
    """
    return np.asarray(wave_vac, float) / n_air(wave_vac)


# ------------------------------------
# Reading
# ------------------------------------

def sidecar_path(path: str | Path) -> Path:
    """The provenance sidecar path for a spectrum CSV."""
    path = Path(path)
    return path.parent / (path.stem + SIDECAR_SUFFIX)


def read_spectrum(path: str | Path) -> Spectrum:
    """Read a prepared spectrum, enforcing the file contract.

    Parameters
    ----------
    path : str or Path
        Spectrum CSV; its sidecar must sit alongside.

    Returns
    -------
    spectrum : Spectrum
        Validated arrays, the sidecar's content, and the file's bytes
        and sha256.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"spectrum not found: {path}")
    label = str(path)
    csv_bytes = path.read_bytes()               # read once
    frame = pd.read_csv(io.BytesIO(csv_bytes))

    if tuple(frame.columns) != SPECTRUM_COLUMNS:
        raise ValueError(f"{label}: columns {list(frame.columns)} != "
                         f"required {list(SPECTRUM_COLUMNS)}")
    if frame.empty:
        raise ValueError(f"{label}: empty spectrum")

    wave = frame["wave_A"].to_numpy(dtype=float)
    flux = frame["flux_uJy"].to_numpy(dtype=float)
    err = frame["flux_err_uJy"].to_numpy(dtype=float)
    mask_raw = frame["mask"].to_numpy(dtype=float)

    if not np.isfinite(wave).all():
        raise ValueError(f"{label}: non-finite wave_A")
    if not (np.diff(wave) > 0).all():
        raise ValueError(f"{label}: wave_A must be strictly increasing")
    if not np.isin(mask_raw, (0.0, 1.0)).all():
        raise ValueError(f"{label}: mask must be 0 or 1")
    mask = mask_raw.astype(bool)
    if not mask.any():
        raise ValueError(f"{label}: every channel is masked")

    bad_flux = mask & ~np.isfinite(flux)
    if bad_flux.any():
        raise ValueError(f"{label}: non-finite flux_uJy on "
                         f"{int(bad_flux.sum())} unmasked channels")
    bad_err = mask & (~np.isfinite(err) | (err <= 0))
    if bad_err.any():
        raise ValueError(f"{label}: non-finite or non-positive flux_err_uJy "
                         f"on {int(bad_err.sum())} unmasked channels")

    sidecar = sidecar_path(path)
    if not sidecar.is_file():
        raise FileNotFoundError(f"{label}: no provenance sidecar at "
                                f"{sidecar.name}")
    provenance = json.loads(sidecar.read_text(encoding="utf-8"))
    for key, required in (("wave_frame", REQUIRED_WAVE_FRAME),
                          ("flux_unit", REQUIRED_FLUX_UNIT)):
        found = provenance.get(key)
        if found != required:
            raise ValueError(f"{sidecar}: {key} is {found!r}; this package "
                             f"fits only {required!r} spectra")

    return Spectrum(path=path, wave_A=wave, flux_uJy=flux, flux_err_uJy=err,
                    mask=mask, provenance=provenance, csv_bytes=csv_bytes,
                    sha256=sha256_bytes(csv_bytes))


# ------------------------------------
# Policy
# ------------------------------------

def apply_spectrum_policy(
        spectrum: Spectrum,
        *,
        mu_lensing: float = 1.0,
        err_floor: float = 0.0,
        mask_windows: list | tuple | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The policy transforms for a spectrum, mirroring core.policy.

    Order: magnification first (flux and error divided by mu), then the
    clamped floor err' = sqrt(err^2 + (floor * max(flux, 0))^2); the
    two commute. mask_windows are observed-frame vacuum [lo, hi]
    intervals excluded on top of the file's own mask.

    Parameters
    ----------
    spectrum : Spectrum
        core.spectrum.read_spectrum output.
    mu_lensing : float
        Lensing magnification; flux and error are divided by it.
        [default: 1.0]
    err_floor : float
        Fractional calibration floor on the spectrum. [default: 0.0]
    mask_windows : list of [lo, hi] or None
        Additional exclusion windows [Angstrom, vacuum, observed].
        [default: None]

    Returns
    -------
    flux_uJy, flux_err_uJy, mask : np.ndarray
        Transformed vectors at full channel count and the combined
        fit-inclusion mask.
    """
    if not (isinstance(mu_lensing, (int, float)) and mu_lensing > 0):
        raise ValueError(f"mu_lensing must be positive, got {mu_lensing!r}")
    if err_floor < 0:
        raise ValueError(f"err_floor must be non-negative, got {err_floor}")

    flux = spectrum.flux_uJy / mu_lensing
    err = spectrum.flux_err_uJy / mu_lensing
    err = np.sqrt(err**2 + (err_floor * np.maximum(flux, 0.0))**2)

    mask = spectrum.mask.copy()
    for window in (mask_windows or ()):
        lo, hi = window
        mask &= ~((spectrum.wave_A >= lo) & (spectrum.wave_A <= hi))
    if not mask.any():
        raise ValueError("no unmasked channels remain after mask_windows")
    return flux, err, mask
