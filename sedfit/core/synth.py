"""
synth.py

Synthetic Photometry
---------------------------------------------------------

Photon-counting AB integration of a spectrum through a bandpass, the
pivot wavelength of a bandpass, and the SPHEREx tophat generator.

Requirements:
  - numpy
"""

from __future__ import annotations

import numpy as np

# ------------------------------------
# Constants
# ------------------------------------

SPHEREX_TOPHAT_SAMPLES = 25


# ------------------------------------
# Synthetic photometry
# ------------------------------------

def synth_fnu(
        wave_AA: np.ndarray,
        throughput: np.ndarray,
        spec_wave_um: np.ndarray,
        spec_fnu_uJy: np.ndarray,
    ) -> tuple[float, float]:
    """Spectrum flux density through a bandpass, photon-counting AB.

    <f_nu> = integral(f_nu T dlam / lam) / integral(T dlam / lam)

    Parameters
    ----------
    wave_AA, throughput : array
        Bandpass arrays [Angstrom, dimensionless].
    spec_wave_um, spec_fnu_uJy : array
        Spectrum wavelength [micron] and flux density [uJy].

    Returns
    -------
    synth_fnu_uJy : float
        Band flux density [uJy]; NaN when fewer than two bandpass samples
        land on the spectrum.
    coverage : float
        Transmission-weighted fraction of the band covered by the
        spectrum (1.0 = fully covered). Bandpass samples off the
        spectrum are dropped before integrating, so the trapezoid rule
        bridges an interior gap and counts it as covered.
    """
    fw = np.asarray(wave_AA, dtype=float)
    ft = np.asarray(throughput, dtype=float)
    sw = np.asarray(spec_wave_um, dtype=float) * 1e4
    order = np.argsort(sw)
    fnu = np.interp(fw, sw[order], np.asarray(spec_fnu_uJy, dtype=float)[order],
                    left=np.nan, right=np.nan)
    good = np.isfinite(fnu)
    if good.sum() < 2:
        return float("nan"), 0.0
    coverage = float(np.trapezoid(ft[good], fw[good]) / np.trapezoid(ft, fw))
    num = np.trapezoid(fnu[good] * ft[good] / fw[good], fw[good])
    den = np.trapezoid(ft[good] / fw[good], fw[good])
    return float(num / den), coverage


def pivot_wave_AA(wave_AA: np.ndarray, throughput: np.ndarray) -> float:
    """Pivot wavelength [Angstrom] of a bandpass.

    lambda_pivot = sqrt( integral(T lam dlam) / integral(T dlam / lam) )
    """
    wave = np.asarray(wave_AA, dtype=float)
    thru = np.asarray(throughput, dtype=float)
    return float(np.sqrt(np.trapezoid(thru * wave, wave)
                         / np.trapezoid(thru / wave, wave)))


# ------------------------------------
# SPHEREx tophats
# ------------------------------------

def make_spherex_tophat(
        wave_um: float,
        bandwidth_um: float,
        *,
        n_samples: int = SPHEREX_TOPHAT_SAMPLES,
    ) -> tuple[np.ndarray, np.ndarray]:
    """Rectangular tophat bandpass for one SPHEREx channel.

    Parameters
    ----------
    wave_um : float
        Channel center wavelength [micron].
    bandwidth_um : float
        Channel full width [micron].
    n_samples : int
        Samples across the open interval. [default: 25]

    Returns
    -------
    wave_AA : np.ndarray
        Wavelength [Angstrom], with zero-throughput endpoints just
        outside the band so interpolation cannot extrapolate the edge.
    throughput : np.ndarray
        1 inside the channel, 0 at the padded endpoints.
    """
    center = float(wave_um) * 1e4
    width = float(bandwidth_um) * 1e4
    lo, hi = center - width / 2.0, center + width / 2.0
    pad = max(width * 1e-3, 1e-3)
    wave = np.concatenate([[lo - pad], np.linspace(lo, hi, n_samples), [hi + pad]])
    throughput = np.concatenate([[0.0], np.ones(n_samples), [0.0]])
    return wave, throughput
