"""
dered.py

Galactic Dereddening
---------------------------------------------------------

Foreground Milky Way extinction correction for an assembled photometry
set, applied per source in its native frame before any reference or
stitch transform. The extinction curve is O'Donnell (1994) in the
optical with the Cardelli, Clayton & Mathis (1989) near-IR and
ultraviolet extensions at R_V = 3.1; E(B-V) is a Schlafly & Finkbeiner
(2011) value for the target position. Broadband factors are
photon-counting bandpass integrals (matching ``synth``); SPHEREx channels
are corrected at their center wavelength.

Requirements:
  - numpy, pandas
  - astropy, astroquery (fetch_ebv_sfd only)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sedfit.core.registry import Registry

# ------------------------------------
# Constants
# ------------------------------------

RV_MW = 3.1
LAW = "odonnell94+ccm89"

# Branch boundaries in x = 1/lambda [1/micron].
X_IR_OPTICAL = 1.1
X_OPTICAL_UV = 3.3
X_UV_MAX = 8.0

# O'Donnell (1994) optical coefficients, 1.1 <= x <= 3.3 [1/micron].
_ODONNELL_A = (1.0, 0.104, -0.609, 0.701, 1.137, -1.718, -0.827, 1.647, -0.505)
_ODONNELL_B = (0.0, 1.952, 2.908, -3.989, -7.985, 11.102, 5.491, -10.805, 3.347)


# ------------------------------------
# Extinction curve
# ------------------------------------

def k_lambda(wave_um: np.ndarray | float, *,
             r_v: float = RV_MW) -> np.ndarray:
    """A_lambda / E(B-V) at the given wavelength(s) [micron].

    Three branches in x = 1/lambda: the CCM (1989) near-IR power law
    below x = 1.1, the O'Donnell (1994) optical polynomial to x = 3.3,
    and the CCM (1989) ultraviolet form to x = 8. The near-IR power law
    is extrapolated redward of its stated range to cover the SPHEREx
    tail.

    Parameters
    ----------
    wave_um : np.ndarray or float
        Wavelength(s) in micron.
    r_v : float
        Total-to-selective extinction ratio. [default: 3.1]

    Returns
    -------
    k : np.ndarray
        A_lambda / E(B-V), always an array even for scalar input.

    Raises
    ------
    ValueError
        For wavelengths shortward of 1/8 micron, where the curve is
        undefined.
    """
    x = 1.0 / np.atleast_1d(np.asarray(wave_um, dtype=float))
    if np.any(x > X_UV_MAX):
        raise ValueError(
            f"extinction curve is undefined below "
            f"{1.0 / X_UV_MAX:.4f} micron; got "
            f"{1.0 / float(np.max(x)):.4f} micron")
    a = np.empty_like(x)
    b = np.empty_like(x)

    ir = x < X_IR_OPTICAL
    a[ir] = 0.574 * x[ir] ** 1.61
    b[ir] = -0.527 * x[ir] ** 1.61

    opt = (x >= X_IR_OPTICAL) & (x <= X_OPTICAL_UV)
    y = x[opt] - 1.82
    a[opt] = np.polynomial.polynomial.polyval(y, _ODONNELL_A)
    b[opt] = np.polynomial.polynomial.polyval(y, _ODONNELL_B)

    uv = x > X_OPTICAL_UV
    xu = x[uv]
    drop = np.where(xu >= 5.9, xu - 5.9, 0.0)
    a[uv] = (1.752 - 0.316 * xu - 0.104 / ((xu - 4.67) ** 2 + 0.341)
             - 0.04473 * drop ** 2 - 0.009779 * drop ** 3)
    b[uv] = (-3.090 + 1.825 * xu + 1.206 / ((xu - 4.62) ** 2 + 0.263)
             + 0.2130 * drop ** 2 + 0.1207 * drop ** 3)

    return r_v * a + b


def band_factor(
        wave_AA: np.ndarray,
        throughput: np.ndarray,
        ebv: float,
        *,
        r_v: float = RV_MW,
    ) -> tuple[float, float]:
    """Photon-counting dereddening factor and A_band [mag] for a bandpass.

    Parameters
    ----------
    wave_AA : np.ndarray
        Bandpass wavelength grid in Angstrom.
    throughput : np.ndarray
        Bandpass throughput on that grid.
    ebv : float
        Foreground E(B-V).
    r_v : float
        Total-to-selective extinction ratio. [default: 3.1]

    Returns
    -------
    factor : float
        Multiplies an observed band flux to remove foreground
        extinction: the ratio of the unattenuated to the attenuated
        photon-counting band average (``synth`` weight T dlam / lam)
        for a flat-f_nu source.
    a_band : float
        The same correction in magnitudes.
    """
    wave = np.asarray(wave_AA, dtype=float)
    thru = np.asarray(throughput, dtype=float)
    atten = 10.0 ** (-0.4 * k_lambda(wave / 1e4, r_v=r_v) * ebv)
    weight = thru / wave
    mean_atten = (np.trapezoid(atten * weight, wave)
                  / np.trapezoid(weight, wave))
    factor = 1.0 / mean_atten
    return float(factor), float(2.5 * np.log10(factor))


# ------------------------------------
# E(B-V) lookup
# ------------------------------------

def fetch_ebv_sfd(ra_deg: float, dec_deg: float) -> float:
    """Schlafly & Finkbeiner (2011) E(B-V) at the position, via IRSA."""
    from astropy.coordinates import SkyCoord
    try:
        from astroquery.ipac.irsa.irsa_dust import IrsaDust
    except ImportError:
        raise RuntimeError(
            "auto E(B-V) lookup needs astroquery (pip install astroquery); "
            "or pass an explicit value with --mw-ebv") from None
    table = IrsaDust.get_query_table(
        SkyCoord(ra_deg, dec_deg, unit="deg"), section="ebv")
    return float(table["ext SandF mean"][0])


# ------------------------------------
# Application
# ------------------------------------

def deredden(selected: dict[str, pd.DataFrame], spectrum: pd.DataFrame | None,
             ebv: float, registry: Registry) -> dict:
    """Deredden every selected source frame and the SPHEREx spectrum in place.

    Broadband rows are scaled by their photon-counting band factor
    (flux and error); SPHEREx rows by their center-wavelength factor
    (flux, error, and scatter). Returns a provenance record.
    """
    a_band: dict[str, float] = {}
    for frame in selected.values():
        factors = np.ones(len(frame))
        for i, band in enumerate(frame["band"]):
            wave_AA, thru = registry.get_bandpass(str(band))
            factors[i], a_band[str(band)] = band_factor(wave_AA, thru, ebv)
        for column in ("flux_uJy", "flux_err_uJy"):
            frame[column] = frame[column].to_numpy(dtype=float) * factors

    spherex_a = None
    if spectrum is not None and len(spectrum):
        factor = 10.0 ** (0.4 * k_lambda(spectrum["wave_um"].to_numpy()) * ebv)
        for column in ("flux_uJy", "flux_err_uJy", "scatter_uJy"):
            spectrum[column] = spectrum[column].to_numpy(dtype=float) * factor
        mags = 2.5 * np.log10(factor)
        spherex_a = {"min_mag": float(mags.min()), "max_mag": float(mags.max())}

    return {
        "ebv": float(ebv),
        "law": LAW,
        "r_v": RV_MW,
        "broadband_A_mag": {b: round(a, 5) for b, a in a_band.items()},
        "spherex_A_mag": spherex_a,
    }
