"""
basis.py

Template Library on a Uniform ln-Lambda Grid
---------------------------------------------------------

Loads a template set onto a constant-velocity grid, broadens it by a line-of-
sight velocity dispersion, and projects it onto a spectrum's own wavelengths.

Requirements:
    numpy, scipy

Notes:
    Templates are rest VACUUM. Rationale in DESIGN.md sections 16 and 17.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

C_KMS = 299792.458

# Margin on the loaded file rows, and the coverage margin in ln-lambda that
# keeps the broadening kernel inside the grid.
_LOAD_MARGIN_A = 50.0
_KERNEL_MARGIN_LN = 4e-3

_CONV_CACHE_MAX = 8

# gaussian_filter1d truncates at 4 sigma, so a kernel wider than a quarter of
# the grid spans the whole of it. Cost is linear in sigma with no natural
# bound -- 3.8 ms at 250 km/s against 6 s at 3.4e5 on a 15 km/s grid -- so a
# degenerate fit that walks sigma up stalls rather than failing.
_MAX_KERNEL_FRACTION = 0.25


class TemplateBasis:
    """A template set resampled onto a uniform ln-lambda grid.

    Each template is normalized to mean 1 over `normalize_range`, so NNLS
    amplitudes read directly as light fractions.
    """

    def __init__(self, paths: list[Path], *, wave_range: tuple[float, float],
                 dv_kms: float, normalize_range: tuple[float, float]) -> None:
        if not paths:
            raise ValueError("no template files given")
        lo, hi = wave_range
        if not lo < hi:
            raise ValueError(f"wave_range must be increasing, got {wave_range}")
        self.paths = [Path(p) for p in paths]
        self.names = [p.name for p in self.paths]
        self.wave_range = (float(lo), float(hi))
        self.normalize_range = (float(normalize_range[0]),
                                float(normalize_range[1]))
        self.dln = float(dv_kms) / C_KMS
        self.lnw = np.arange(np.log(lo), np.log(hi), self.dln)

        rows = []
        for path in self.paths:
            table = np.loadtxt(path, usecols=(0, 1))
            wave, flam = table[:, 0], table[:, 1]
            keep = ((wave > lo - _LOAD_MARGIN_A)
                    & (wave < hi + _LOAD_MARGIN_A))
            if keep.sum() < 2:
                raise ValueError(f"{path.name} does not cover {wave_range}")
            rows.append(np.interp(np.exp(self.lnw), wave[keep], flam[keep]))
        flam = np.asarray(rows)

        grid = np.exp(self.lnw)
        band = (grid > normalize_range[0]) & (grid < normalize_range[1])
        if not band.any():
            raise ValueError(f"normalize_range {normalize_range} is outside "
                             f"the grid {wave_range}")
        self.flam = flam / flam[:, band].mean(axis=1)[:, None]
        self._conv_cache: dict[float, np.ndarray] = {}

    @property
    def n_templates(self) -> int:
        return self.flam.shape[0]

    def convolved(self, sigma_kms: float) -> np.ndarray:
        """The basis broadened to `sigma_kms`, cached."""
        samples = float(sigma_kms) / (self.dln * C_KMS)
        if samples > _MAX_KERNEL_FRACTION * self.lnw.size:
            raise ValueError(
                f"sigma {float(sigma_kms):.0f} km/s is a broadening kernel of "
                f"{samples:.0f} samples on a {self.lnw.size}-sample grid; it "
                f"must stay well inside the grid it is applied to")
        key = round(float(sigma_kms), 3)
        if key not in self._conv_cache:
            if len(self._conv_cache) > _CONV_CACHE_MAX:
                self._conv_cache.clear()
            self._conv_cache[key] = gaussian_filter1d(
                self.flam, sigma_kms / (self.dln * C_KMS), axis=1,
                mode="nearest")
        return self._conv_cache[key]

    def design(self, wave_vac_obs: np.ndarray, redshift: float,
               sigma_kms: float, *, check: np.ndarray | None = None,
               lsf=None) -> np.ndarray:
        """(NTEMP, npix) template rows at the observed vacuum wavelengths.

        Coverage is asserted over `check` — the fit mask, or every pixel when
        None. Pixels outside it may leave the grid and are clamped, so masked
        diagnostics do not raise.

        `lsf` is a LineSpread, or None to take `sigma_kms` as the whole
        broadening. When it is given, `sigma_kms` is the intrinsic dispersion
        and the instrument kernel is applied on top: folded into the cached
        rest-frame convolution when it is constant in velocity, and applied
        in the observed frame after the shift when it is not.
        """
        x = np.log(wave_vac_obs) - np.log1p(redshift)
        inside = x if check is None else x[check]
        if (inside.min() < self.lnw[0] + _KERNEL_MARGIN_LN
                or inside.max() > self.lnw[-1] - _KERNEL_MARGIN_LN):
            raise ValueError(
                f"rest coverage {np.exp(inside.min()):.0f}-"
                f"{np.exp(inside.max()):.0f} A of the fitted pixels leaves the "
                f"template grid {self.wave_range} plus its kernel margin")
        flat, sigma_A = (None, None) if lsf is None else lsf.plan(redshift)
        conv = self.convolved(sigma_kms if flat is None
                              else float(np.hypot(sigma_kms, flat)))
        index = np.clip((x - self.lnw[0]) / self.dln, 0.0,
                        len(self.lnw) - 1.001)
        low = index.astype(int)
        frac = index - low
        rows = conv[:, low] * (1 - frac) + conv[:, low + 1] * frac
        if sigma_A is None:
            return rows
        return lsf.smear(rows, sigma_A)
