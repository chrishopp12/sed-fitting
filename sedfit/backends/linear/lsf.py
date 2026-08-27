"""
lsf.py

Instrument Line Spread and Template Library Resolution
---------------------------------------------------------

The model carries one velocity width for the whole band, which is an
instrument assumption. This supplies the real one and takes the quadrature
difference against the template library's own resolution:

    sigma_kernel(l_obs)^2 = sigma_LSF,data(l_obs)^2
                            - [sigma_lib(l_obs / (1 + z)) * (1 + z)]^2

Applied in the observed frame, on the design matrix, after the redshift
shift. When the difference is constant in velocity it folds into the cached
rest-frame broadening instead, for free.

Requirements:
    numpy, scipy

Notes:
    Rationale in DESIGN.md section 17.2.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

from sedfit.backends.linear.basis import C_KMS
from sedfit.core.spectrum import vac_to_air

PACKAGED_TEMPLATE_DIR = (Path(__file__).resolve().parents[2] / "data"
                         / "templates")
RESOLUTION_NAME = "resolution.txt"

# Matches tools/make_ssp_templates.py, which wrote the shipped curves.
FWHM_PER_SIGMA = 2.3548

# Gaussian half-width the banded convolution carries, and the relative spread
# below which a per-pixel kernel counts as one velocity width.
_HALF_WIDTHS = 4.0
_FLAT_RTOL = 1e-9

# Fractional slack allowed when a query leaves a tabulated curve.
_EDGE_RTOL = 1e-6


def to_sigma_kms(value: np.ndarray | float, wave_A: np.ndarray | float,
                 unit: str) -> np.ndarray:
    """A resolution in any declared unit as sigma [km/s] at `wave_A`."""
    value = np.asarray(value, float)
    wave_A = np.asarray(wave_A, float)
    if unit == "R":
        return C_KMS / (value * FWHM_PER_SIGMA)
    if unit == "fwhm_kms":
        return value / FWHM_PER_SIGMA
    if unit == "sigma_kms":
        return value * np.ones_like(wave_A)
    if unit == "fwhm_A":
        return C_KMS * value / (FWHM_PER_SIGMA * wave_A)
    if unit == "sigma_A":
        return C_KMS * value / wave_A
    raise ValueError(f"unknown resolution unit {unit!r}")


def resolve_resolution_file(spec: str | Path) -> Path:
    """The curve a `file` field names, or a packaged set's shipped one."""
    path = Path(spec).expanduser()
    if path.is_file():
        return path
    packaged = PACKAGED_TEMPLATE_DIR / str(spec) / RESOLUTION_NAME
    if packaged.is_file():
        return packaged
    raise ValueError(f"{str(spec)!r} is not a resolution file, and no "
                     f"packaged template set of that name ships a "
                     f"{RESOLUTION_NAME}")


class Resolution:
    """One line-spread width as a function of wavelength.

    Parameters
    ----------
    unit : str
        One of R, fwhm_A, sigma_A, fwhm_kms, sigma_kms.
    constant : float or None
        A single value in `unit`, for a curve that has no table.
    wave : np.ndarray or None
        Tabulated wavelengths [A] in `frame`, increasing.
    value : np.ndarray or None
        Tabulated values in `unit`, one per wavelength.
    frame : str
        'vacuum' or 'air', the frame `wave` is expressed in. Required for a
        table for the same reason poly_wave_frame is: nothing about the file
        says which it was prepared in.
    """

    def __init__(self, *, unit: str, constant: float | None = None,
                 wave: np.ndarray | None = None,
                 value: np.ndarray | None = None,
                 frame: str = "vacuum") -> None:
        if (constant is None) == (wave is None):
            raise ValueError("a resolution takes exactly one of a constant "
                             "or a table")
        self.unit = unit
        self.frame = frame
        self.constant = None if constant is None else float(constant)
        self.wave = None if wave is None else np.asarray(wave, float)
        self.value = None if value is None else np.asarray(value, float)
        if self.wave is not None:
            if self.value is None or self.value.shape != self.wave.shape:
                raise ValueError("a tabulated resolution needs one value per "
                                 "wavelength")
            if not np.all(np.diff(self.wave) > 0):
                raise ValueError("resolution wavelengths must increase "
                                 "strictly")
        # to_sigma_kms rejects an unknown unit before anything reads the curve
        to_sigma_kms(1.0, 5000.0, unit)

    @classmethod
    def from_file(cls, path: str | Path, *, unit: str, frame: str
                  ) -> "Resolution":
        table = np.loadtxt(path, usecols=(0, 1))
        if table.ndim != 2 or len(table) < 2:
            raise ValueError(f"{path}: need at least two (wavelength, value) "
                             f"rows")
        return cls(unit=unit, wave=table[:, 0], value=table[:, 1], frame=frame)

    def sigma_kms(self, wave_vac: np.ndarray) -> np.ndarray:
        """sigma [km/s] at the given vacuum wavelengths, in this curve's frame.

        Parameters
        ----------
        wave_vac : np.ndarray
            Query wavelengths [A], vacuum, in the same rest-or-observed sense
            the curve was built for.

        Returns
        -------
        sigma_kms : np.ndarray
            One width per query wavelength.
        """
        wave_vac = np.asarray(wave_vac, float)
        query = wave_vac if self.frame == "vacuum" else vac_to_air(wave_vac)
        if self.constant is not None:
            return to_sigma_kms(self.constant, query, self.unit)
        span = self.wave[-1] - self.wave[0]
        if (query.min() < self.wave[0] - _EDGE_RTOL * span
                or query.max() > self.wave[-1] + _EDGE_RTOL * span):
            raise ValueError(
                f"resolution curve spans {self.wave[0]:.1f}-"
                f"{self.wave[-1]:.1f} A but was asked for "
                f"{query.min():.1f}-{query.max():.1f} A")
        return to_sigma_kms(np.interp(query, self.wave, self.value), query,
                            self.unit)


class LineSpread:
    """The data's line spread against the template library's own resolution.

    Parameters
    ----------
    wave_vac_obs : np.ndarray
        The spectrum's observed vacuum wavelengths [A], increasing.
    data : Resolution
        The instrument LSF, in the observed frame.
    library : Resolution
        The template set's resolution, in the rest frame.
    on_undersampled : str
        What to do where the library is broader than the data: 'raise',
        'degrade_data' or 'ignore'. No default; all three are defensible and
        the wrong one is silent.
    fitted : np.ndarray or None
        Which channels the policy is enforced over; None means all.
        [default: None]
    """

    def __init__(self, wave_vac_obs: np.ndarray, data: Resolution,
                 library: Resolution, *, on_undersampled: str,
                 fitted: np.ndarray | None = None) -> None:
        self.wave = np.asarray(wave_vac_obs, float)
        self.data = data
        self.library = library
        self.on_undersampled = on_undersampled
        self.fitted = (np.ones(self.wave.size, bool) if fitted is None
                       else np.asarray(fitted, bool))
        self.data_sigma_kms = data.sigma_kms(self.wave)
        self.dispersion = np.gradient(self.wave)

    def library_sigma_kms(self, redshift: float) -> np.ndarray:
        """The library width at the observed wavelengths, in km/s.

        A velocity width is invariant under the redshift stretch, so the
        library's rest-frame km/s curve is read at the rest wavelengths and
        used as it stands.
        """
        return self.library.sigma_kms(self.wave / (1.0 + redshift))

    def _difference(self, redshift: float) -> np.ndarray:
        return self.data_sigma_kms ** 2 - self.library_sigma_kms(redshift) ** 2

    def plan(self, redshift: float) -> tuple[float | None, np.ndarray | None]:
        """The broadening to apply at this redshift.

        Returns
        -------
        flat_kms : float or None
            A single velocity width, to fold into the cached rest-frame
            broadening. None when the kernel varies across the band.
        sigma_A : np.ndarray or None
            Per-pixel widths [A] for an observed-frame convolution, when the
            kernel is not flat.
        """
        difference = self._difference(redshift)
        short = difference < 0.0
        if short.any() and self.on_undersampled == "raise":
            where = self.wave[short & self.fitted]
            if where.size:
                raise ValueError(
                    f"the template library is broader than the data over "
                    f"{where.min():.1f}-{where.max():.1f} A at z={redshift:.4f} "
                    f"({where.size} of {int(self.fitted.sum())} fitted "
                    f"channels); set lsf.on_undersampled to 'degrade_data' "
                    f"or 'ignore' to proceed")
        kernel = np.sqrt(np.where(short, 0.0, difference))
        if np.ptp(kernel) <= _FLAT_RTOL * max(float(kernel.max()), 1.0):
            return float(kernel[0]), None
        return None, kernel * self.wave / C_KMS

    def smear(self, rows: np.ndarray, sigma_A: np.ndarray, *,
              variance: bool = False) -> np.ndarray:
        """Convolve `rows` on this spectrum's own pixel grid."""
        return smear(rows, sigma_A, self.dispersion, variance=variance)

    def degrade(self, flux: np.ndarray, error: np.ndarray,
                fitted: np.ndarray, redshift: float
                ) -> tuple[np.ndarray, np.ndarray]:
        """Smooth the DATA up to the library resolution where that is coarser.

        A normalized convolution over the fitted channels, so a mask edge
        does not bleed masked values into the fit. Returns the inputs
        unchanged unless the policy is 'degrade_data' and the library is
        broader somewhere.

        The result's errors are correlated and only the diagonal is carried,
        which is why a degraded run's chi-square does not compare with an
        undegraded one's.
        """
        widths = self.degradation_sigma_A(redshift)
        if widths is None:
            return flux, error
        weight = self.smear(np.asarray(fitted, float), widths)
        safe = np.where(weight > 0, weight, 1.0)
        smoothed = self.smear(np.where(fitted, flux, 0.0), widths) / safe
        variance = self.smear(np.where(fitted, np.asarray(error, float) ** 2,
                                       0.0), widths, variance=True) / safe ** 2
        return (np.where(fitted, smoothed, flux),
                np.where(fitted, np.sqrt(variance), error))

    def degradation_sigma_A(self, redshift: float) -> np.ndarray | None:
        """Widths [A] to smooth the DATA by, where the library is broader.

        None unless `on_undersampled` is 'degrade_data'. The result is
        z-dependent whenever the library's velocity width is, which
        `assert_degradation_is_fixed` is what refuses.
        """
        if self.on_undersampled != "degrade_data":
            return None
        deficit = -self._difference(redshift)
        if not (deficit > 0).any():
            return None
        return np.sqrt(np.maximum(deficit, 0.0)) * self.wave / C_KMS

    def assert_degradation_is_fixed(self, z_min: float, z_max: float) -> None:
        """Refuse a degradation that would have to be redone at every z.

        Degrading the data is a transform of the data, so it happens once
        before the fit. That is exact only when the library's width in km/s
        does not move with redshift -- true of a constant-R library like
        `ssp_c3k_a` inside its window, false of a constant-FWHM one like
        `ssp_miles`.
        """
        if self.on_undersampled != "degrade_data":
            return
        low = self.library_sigma_kms(z_min)
        high = self.library_sigma_kms(z_max)
        spread = float(np.max(np.abs(high - low)))
        if spread > _FLAT_RTOL * max(float(low.max()), 1.0):
            raise ValueError(
                f"lsf.on_undersampled 'degrade_data' needs a library whose "
                f"width in km/s does not move with redshift, but it moves by "
                f"{spread:.3f} km/s between z={z_min:.3f} and z={z_max:.3f}; "
                f"use a constant-R library or 'ignore'")


def smear(rows: np.ndarray, sigma_A: np.ndarray, dispersion: np.ndarray,
          *, variance: bool = False) -> np.ndarray:
    """Convolve along the pixel axis with a per-pixel Gaussian width.

    Parameters
    ----------
    rows : np.ndarray
        (n, npix) design rows, or a single (npix,) spectrum.
    sigma_A : np.ndarray
        (npix,) kernel widths [A].
    dispersion : np.ndarray
        (npix,) local dlambda/dpixel, which carries a non-uniform grid
        without a special case.
    variance : bool
        Propagate variances rather than values: sum(w^2 v) / sum(w)^2 in
        place of sum(w x) / sum(w). Off-diagonal terms are dropped, which is
        what makes a degraded chi-square incomparable. [default: False]

    Returns
    -------
    smeared : np.ndarray
        The same shape as `rows`, edges extended.
    """
    flat = rows.ndim == 1
    rows = np.atleast_2d(rows)
    sigma_pix = np.asarray(sigma_A, float) / np.asarray(dispersion, float)
    if sigma_pix.max() <= 0:
        return rows[0] if flat else rows
    if not variance and np.ptp(sigma_pix) <= _FLAT_RTOL * float(
            sigma_pix.max()):
        out = gaussian_filter1d(rows, float(sigma_pix[0]), axis=1,
                                mode="nearest")
        return out[0] if flat else out

    half = int(np.ceil(_HALF_WIDTHS * float(sigma_pix.max())))
    index = np.arange(rows.shape[1])
    safe = np.where(sigma_pix > 0, sigma_pix, 1.0)
    out = np.zeros_like(rows)
    norm = np.zeros(rows.shape[1])
    for offset in range(-half, half + 1):
        weight = np.where(sigma_pix > 0,
                          np.exp(-0.5 * (offset / safe) ** 2),
                          float(offset == 0))
        shifted = rows[:, np.clip(index + offset, 0, rows.shape[1] - 1)]
        out += (weight ** 2 if variance else weight) * shifted
        norm += weight
    out /= norm ** 2 if variance else norm
    return out[0] if flat else out
