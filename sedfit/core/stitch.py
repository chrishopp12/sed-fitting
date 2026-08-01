"""
stitch.py

Cross-Instrument Scale Measurement
---------------------------------------------------------

One achromatic factor places an instrument on the SPHEREx total-flux
scale: synthesize the SPHEREx spectrum through the instrument's reddest
fully-covered bandpass and divide by the measured flux there. The scale
is an aperture-fraction correction in the forward-model sense; the
synthetic photometry is photon-counting AB, the same convention the
fitting backends use. An instrument with no SPHEREx overlap gets None:
its level floats at fit time.

Requirements:
  - numpy
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from sedfit.core.synth import pivot_wave_AA, synth_fnu

if TYPE_CHECKING:
    from sedfit.core.registry import Registry

# ------------------------------------
# Constants
# ------------------------------------

DEFAULT_MIN_COVERAGE = 0.98


# ------------------------------------
# Scale measurement
# ------------------------------------

def measure_scale(
        bands: list[str] | tuple[str, ...],
        fluxes_uJy: list[float] | tuple[float, ...] | np.ndarray,
        *,
        spec_wave_um: np.ndarray,
        spec_fnu_uJy: np.ndarray,
        registry: Registry,
        min_coverage: float = DEFAULT_MIN_COVERAGE,
    ) -> tuple[float | None, dict]:
    """One achromatic scale lifting an instrument onto the SPHEREx scale.

    Parameters
    ----------
    bands, fluxes_uJy : sequences
        Band labels (registry members) and measured fluxes [uJy],
        parallel.
    spec_wave_um, spec_fnu_uJy : array
        The SPHEREx spectrum [micron, uJy].
    registry : Registry
        Bandpass authority for the bands.
    min_coverage : float
        Minimum transmission-weighted coverage for a band to be
        scale-eligible. [default: 0.98]

    Returns
    -------
    scale : float or None
        Multiplies the instrument's fluxes onto the SPHEREx scale; None
        when no band is covered (the level must float at fit time). The
        anchor is the reddest band that clears min_coverage with a
        finite synthetic flux and a positive measured flux; covered
        bands with no positive measured flux among them are a hard
        error.
    info : dict
        anchor band, scale, and per-band diagnostics (pivot_AA, flux,
        synth, coverage, ratio); the ratios on the other covered bands
        are a built-in cross-check. ratio is NaN where the measured
        flux is not positive.
    """
    rows = []
    for band, measured in zip(bands, fluxes_uJy, strict=True):
        wave, thru = registry.get_bandpass(band)
        synth, coverage = synth_fnu(wave, thru, spec_wave_um, spec_fnu_uJy)
        flux = float(measured)
        rows.append(dict(band=band,
                         pivot_AA=pivot_wave_AA(wave, thru),
                         flux=flux,
                         synth=synth,
                         coverage=coverage,
                         ratio=(synth / flux if flux > 0 else float("nan"))))
    covered = [r for r in rows
               if r["coverage"] >= min_coverage and np.isfinite(r["synth"])]
    eligible = [r for r in covered if r["flux"] > 0]
    if not eligible:
        if covered:
            detail = ", ".join(f"{r['band']}={r['flux']:.6g}" for r in covered)
            raise ValueError(f"no covered band has a positive measured flux, "
                             f"so no achromatic scale can be measured: "
                             f"{detail}")
        return None, {"anchor": None, "scale": None, "bands": rows}
    anchor = max(eligible, key=lambda r: r["pivot_AA"])
    scale = anchor["synth"] / anchor["flux"]
    return float(scale), {"anchor": anchor["band"], "scale": float(scale),
                          "bands": rows}
