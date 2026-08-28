"""
fitting.py

Linear Spectral Fit — NNLS Amplitudes, Chebyshev Continuum, (z, sigma)
---------------------------------------------------------

    F(lambda) = Cheb(lambda; c) * T(lambda) * sum_k a_k B_k(lambda; z, sigma)

`a_k >= 0` by NNLS; `c` by fixed-point alternation with that solve; `(z, sigma)`
by coarse grid then Nelder-Mead. Deterministic, no priors, no sampler.

Requirements:
    numpy, scipy

Notes:
    Rationale in DESIGN.md sections 16 and 17.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.polynomial import chebyshev
from scipy.optimize import minimize

from sedfit.backends.linear.basis import C_KMS, TemplateBasis
from sedfit.backends.linear.gas import GasBasis
from sedfit.backends.linear.physics import check_ratios, velocity_floor_kms
from sedfit.core.nnls import nnls_fit

# Amplitude of the template sum below which a pixel is dropped from the
# Chebyshev refit, as a fraction of that sum's median over the fitted pixels.
_POLY_FLOOR = 0.1

# Nelder-Mead works in (z, sigma/1e4) so one xatol serves both axes; the
# default 5% simplex would move z off the grid-scan basin entirely.
_SIGMA_SCALE = 1e4
_SIMPLEX_DZ = 3e-4
_SIMPLEX_DSIGMA = 15.0

# Coarse-pass defaults. The continuum polynomial does not need to be
# converged to decide which basin the redshift is in, and the sigma grid buys
# nothing there.
_Z_STEP_COARSE = 1.5e-3
_N_POLY_ITER_COARSE = 1
_WINDOW_STEPS = 10

# Coarse steps per gas line width. An emission line is narrower than any
# stellar feature by an order of magnitude, so a step sized against the LOSVD
# walks straight past it: at two widths off centre the column takes no
# amplitude and the true redshift stops looking special, while a line aliased
# onto a different transition elsewhere is hit exactly. Half a step must stay
# inside one width. Measured on synthetic emission-line spectra at five
# redshifts: two widths recovers all five, three recovers three, four
# recovers two.
_COARSE_GAS_WIDTHS = 2.0

# Velocity separation below which two grid minima are one basin. A threshold
# on dz alone would be a different criterion at each end of the grid.
_MINIMUM_DV_KMS = 1000.0


@dataclass(frozen=True)
class Minimum:
    """One distinct local minimum of a grid scan's chi-square profile."""

    redshift: float
    sigma_kms: float
    chi2: float
    delta_chi2: float


def rank_minima(redshift_grid: np.ndarray, sigma_grid: np.ndarray,
                chi2_grid: np.ndarray, *, dv_kms: float = _MINIMUM_DV_KMS
                ) -> list[Minimum]:
    """Distinct local minima of the chi-square profile, best first.

    Parameters
    ----------
    redshift_grid : np.ndarray
        The scanned redshifts, increasing.
    sigma_grid : np.ndarray
        The scanned velocity dispersions, one per row of `chi2_grid`.
    chi2_grid : np.ndarray
        Chi-square over (sigma, redshift).
    dv_kms : float
        Velocity separation below which two minima are one basin.
        [default: 1000.0]

    Returns
    -------
    minima : list[Minimum]
        Ranked by chi-square, each `delta_chi2` measured from the best.
    """
    redshift_grid = np.asarray(redshift_grid, float)
    sigma_grid = np.asarray(sigma_grid, float)
    chi2_grid = np.atleast_2d(np.asarray(chi2_grid, float))
    profile = chi2_grid.min(axis=0)
    at_sigma = chi2_grid.argmin(axis=0)

    # Strict on the left, so a flat run marks only the point that descends
    # into it rather than every point along it.
    local = np.zeros(profile.size, bool)
    if profile.size > 1:
        local[1:-1] = ((profile[1:-1] < profile[:-2])
                       & (profile[1:-1] <= profile[2:]))
        local[0] = profile[0] < profile[1]
        local[-1] = profile[-1] < profile[-2]
    # A wholly flat profile marks nothing, so the winner is forced in.
    local[profile.argmin()] = True

    kept: list[Minimum] = []
    best = profile.min()
    for index in np.flatnonzero(local)[np.argsort(profile[local],
                                                  kind="stable")]:
        redshift = float(redshift_grid[index])
        if any(abs(redshift - m.redshift) / (1.0 + m.redshift)
               <= dv_kms / C_KMS for m in kept):
            continue
        kept.append(Minimum(redshift=redshift,
                            sigma_kms=float(sigma_grid[at_sigma[index]]),
                            chi2=float(profile[index]),
                            delta_chi2=float(profile[index] - best)))
    return kept


@dataclass(frozen=True)
class LinearFit:
    """One converged fit."""

    redshift: float
    sigma_kms: float
    chi2: float
    dof: int
    amplitudes: np.ndarray
    chebyshev_coefficients: np.ndarray
    model: np.ndarray
    fitted: np.ndarray
    redshift_error: float | None = None
    sigma_error: float | None = None
    n_clipped: int = 0
    sigma_pinned: bool = False
    names: list[str] = field(default_factory=list)
    gas_names: list[str] = field(default_factory=list)
    redshift_grid: np.ndarray | None = None
    sigma_grid: np.ndarray | None = None
    chi2_grid: np.ndarray | None = None
    minima: list[Minimum] = field(default_factory=list)
    grid_n_poly_iter: int | None = None

    @property
    def delta_chi2(self) -> float | None:
        """Chi-square separation from the best grid minimum to the next.

        None when the grid holds fewer than two distinct minima. A
        discriminant, not a probability: the variance is diagonal and a
        spectrum is thousands of correlated channels.
        """
        return self.minima[1].delta_chi2 if len(self.minima) > 1 else None

    @property
    def stellar_amplitudes(self) -> np.ndarray:
        """The leading amplitudes, one per template; gas columns follow."""
        return self.amplitudes[:len(self.names)]

    @property
    def light_fractions(self) -> dict[str, float]:
        """Stellar amplitudes as fractions of their sum, zeros dropped.

        Gas columns are excluded: they carry a line flux, not a share of a
        continuum normalized over `normalize_range`.
        """
        amplitudes = self.stellar_amplitudes
        total = float(amplitudes.sum())
        if total <= 0:
            return {}
        return {name: float(a / total)
                for name, a in zip(self.names, amplitudes) if a > 0}

    @property
    def gas_fluxes(self) -> dict[str, float]:
        """Line fluxes, in the fit's flux unit times Angstrom, zeros kept."""
        return {name: float(a) for name, a in
                zip(self.gas_names, self.amplitudes[len(self.names):])}

    @property
    def stellar_basis_empty(self) -> bool:
        """NNLS gave every stellar template zero amplitude.

        A state, not an error: the spectrum has no continuum this basis can
        represent, which for an over-subtracted background is the correct
        answer rather than a failed one. Any dispersion or light fraction
        reported alongside it is meaningless.
        """
        return not bool((self.stellar_amplitudes > 0).any())

    @property
    def physics_violations(self) -> list:
        """Physical bounds the fitted line set does not respect.

        Derived from `gas_fluxes`, so it changes no fitted value and an
        empty list means nothing was CHECKABLE, not that all is well.
        """
        return check_ratios(self.gas_fluxes)

    @property
    def velocity_floor_kms(self) -> float | None:
        """Worst rest-wavelength floor among the lines carrying flux.

        This fit weights every line alike, so a velocity from it cannot beat
        its worst-known line.
        """
        return velocity_floor_kms(
            [n for n, f in self.gas_fluxes.items() if f > 0])


@dataclass(frozen=True)
class SpectrumScan:
    """A coarse blind pass and the refined fit it selected.

    The coarse chi-square lives here and the final one in `fit`, never in a
    shared field: they come from different `n_poly_iter` and differencing
    them is meaningless.
    """

    fit: LinearFit
    redshift_grid: np.ndarray
    sigma_kms: float
    chi2_grid: np.ndarray
    z_step: float = _Z_STEP_COARSE
    minima: list[Minimum] = field(default_factory=list)
    n_poly_iter: int = _N_POLY_ITER_COARSE

    @property
    def delta_chi2(self) -> float | None:
        """Coarse separation from the winning basin to the next distinct one.

        The blind scan's confidence statistic: a catastrophic redshift is a
        minimum-selection failure, which the fit's own error cannot express.
        """
        return self.minima[1].delta_chi2 if len(self.minima) > 1 else None


def _chebyshev_domain(wave: np.ndarray,
                      domain: tuple[float, float]) -> np.ndarray:
    middle = 0.5 * (domain[0] + domain[1])
    half = 0.5 * (domain[1] - domain[0])
    return (wave - middle) / half


def fit_at(redshift: float, sigma_kms: float, wave_vac: np.ndarray,
           flux: np.ndarray, var: np.ndarray, fitted: np.ndarray,
           basis: TemplateBasis, transmission: np.ndarray,
           cheb_x: np.ndarray, poly_order: int, n_iter: int, *,
           gas: GasBasis | None = None, lsf=None
           ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Chebyshev/NNLS fixed point at one (z, sigma).

    Gas columns follow the stellar rows in the amplitude vector and multiply
    through the Chebyshev like every other row, which is correct rather than
    incidental: the polynomial corrects the data's flux calibration, and that
    affects lines and continuum alike.

    Returns (chi2, amplitudes, chebyshev coefficients, model over all pixels).
    """
    rows = basis.design(wave_vac, redshift, sigma_kms, check=fitted,
                        lsf=lsf)
    if gas is not None:
        # The stellar rows take the DIFFERENCE against the library's own
        # width; an analytic line carries no resolution, so it takes the
        # whole instrument kernel.
        rows = np.vstack([rows, gas.design(
            wave_vac, redshift,
            lsf_sigma_kms=None if lsf is None else lsf.data_sigma_kms)])
    rows = rows * transmission[None, :]
    coefficients = np.zeros(poly_order + 1)
    coefficients[0] = 1.0
    poly = np.ones_like(flux)
    chi2, amps, model = np.inf, None, None
    for _ in range(n_iter):
        chi2, amps, model = nnls_fit(rows * poly[None, :], flux, var, fitted)
        star = amps @ rows
        good = fitted & (star > _POLY_FLOOR * np.median(star[fitted]))
        # NNLS can zero the ENTIRE basis, and it does so robustly rather than
        # marginally: no non-negative combination of positive templates has a
        # negative or zero mean, so a spectrum whose background was
        # over-subtracted drives every amplitude to exactly 0. The floor is
        # then a fraction of zero, `good` is empty, and chebfit raises. This
        # is not exotic -- a continuum-free population scatters symmetrically
        # about zero, so roughly half of it lands there, and during a blind
        # scan most trial redshifts put no line under the gas columns either.
        # A polynomial needs poly_order + 1 points; with fewer, there is no
        # continuum to correct and the honest move is to leave it alone.
        # The polynomial corrects a CONTINUUM. With no stellar amplitude
        # there is no continuum to correct, and the pixels that survive the
        # floor are the emission lines alone -- a few clustered windows of a
        # wide domain. Fitting order-8 to those and evaluating it everywhere
        # extrapolates by four orders of magnitude. Leave the polynomial at
        # whatever it was; the gas columns still fit the lines, which is the
        # whole of what such a spectrum has to offer.
        if not bool((amps[:basis.n_templates] > 0).any()):
            break
        if int(good.sum()) <= poly_order:
            break
        ratio = np.where(star > 0, flux / np.where(star > 0, star, 1.0), 1.0)
        weight = np.where(good, star / np.sqrt(var), 0.0)
        coefficients = chebyshev.chebfit(cheb_x[good], ratio[good], poly_order,
                                         w=weight[good])
        poly = chebyshev.chebval(cheb_x, coefficients)
    chi2, amps, model = nnls_fit(rows * poly[None, :], flux, var, fitted)
    return chi2, amps, coefficients, model


def _polish(redshift: float, sigma_kms: float, cost,
            sigma_bounds: tuple[float, float] | None = None
            ) -> tuple[float, float, float]:
    """Nelder-Mead from one grid point, optionally held to a sigma range.

    The redshift is left unbounded: its range is a search window the minimum
    may legitimately sit just outside, and `TemplateBasis.design` already
    refuses an excursion that leaves the template grid. A dispersion range is
    a declared physical bound, and unbounded the simplex will walk out of it.
    """
    start = [redshift, sigma_kms / _SIGMA_SCALE]
    simplex = np.array([start,
                        [redshift + _SIMPLEX_DZ, sigma_kms / _SIGMA_SCALE],
                        [redshift,
                         (sigma_kms + _SIMPLEX_DSIGMA) / _SIGMA_SCALE]])
    bounds = None
    if sigma_bounds is not None:
        low, high = (bound / _SIGMA_SCALE for bound in sigma_bounds)
        bounds = [(-np.inf, np.inf), (low, high)]
        # Starting outside a declared bound means nothing, and scipy would
        # only warn and clip. It cannot happen through a config, whose grid
        # is generated from the same two numbers.
        start[1] = min(max(start[1], low), high)
        simplex[:, 1] = np.clip(simplex[:, 1], low, high)
    result = minimize(lambda u: cost(u[0], u[1] * _SIGMA_SCALE), start,
                      method="Nelder-Mead", bounds=bounds,
                      options=dict(initial_simplex=simplex, xatol=3e-6,
                                   fatol=0.05, maxiter=400))
    return result.x[0], result.x[1] * _SIGMA_SCALE, float(result.fun)


def _hessian_errors(redshift: float, sigma_kms: float, chi2_min: float,
                    dof: int, cost) -> tuple[float, float]:
    """Symmetric 1-sigma errors from the (z, sigma) curvature, chi2/dof scaled.

    Not a delta-chi-square profile bound: this inverts the finite-difference
    Hessian, so the errors are symmetric by construction.
    """
    dz, ds = 2e-5, 4.0
    fzz = (cost(redshift + dz, sigma_kms) - 2 * chi2_min
           + cost(redshift - dz, sigma_kms)) / dz ** 2
    fss = (cost(redshift, sigma_kms + ds) - 2 * chi2_min
           + cost(redshift, sigma_kms - ds)) / ds ** 2
    fzs = (cost(redshift + dz, sigma_kms + ds)
           - cost(redshift + dz, sigma_kms - ds)
           - cost(redshift - dz, sigma_kms + ds)
           + cost(redshift - dz, sigma_kms - ds)) / (4 * dz * ds)
    hessian = 0.5 * np.array([[fzz, fzs], [fzs, fss]])
    covariance = np.linalg.inv(hessian)
    scale = np.sqrt(max(chi2_min / dof, 1.0))
    return (float(np.sqrt(covariance[0, 0]) * scale),
            float(np.sqrt(covariance[1, 1]) * scale))


def fit_spectrum(wave_vac: np.ndarray, flux: np.ndarray, error: np.ndarray,
                 fitted: np.ndarray, basis: TemplateBasis, *,
                 redshift_grid: np.ndarray, sigma_grid: np.ndarray,
                 poly_order: int, poly_domain: tuple[float, float],
                 transmission: np.ndarray | None = None,
                 poly_wave: np.ndarray | None = None,
                 n_poly_iter: int = 4, clip_sigma: float | None = 4.0,
                 errors: bool = True, gas: GasBasis | None = None,
                 lsf=None, sigma_bounds: tuple[float, float] | None = None,
                 minima_dv_kms: float = _MINIMUM_DV_KMS) -> LinearFit:
    """Grid, polish, optionally clip and refit, and return the converged fit.

    `sigma_bounds` holds the Nelder-Mead polish to a declared dispersion
    range; None leaves it unbounded, which is what the grid alone does. A fit
    that comes back against a bound reports `sigma_pinned`, because a clamped
    dispersion is a different claim from a fitted one and its Hessian error
    describes a curvature the bound overrode.

    `poly_wave` is the wavelength scale the multiplicative Chebyshev is built
    on, defaulting to `wave_vac`. It is a free choice — the polynomial is a
    smooth function of any monotonic coordinate — but it is not a neutral one:
    the same fit run on air rather than vacuum wavelengths moves the fitted
    amplitudes in the fifth decimal. Pass the scale the reference used.
    """
    var = np.asarray(error, float) ** 2
    fitted = np.asarray(fitted, bool).copy()
    if transmission is None:
        transmission = np.ones_like(flux)
    cheb_x = _chebyshev_domain(
        wave_vac if poly_wave is None else np.asarray(poly_wave, float),
        poly_domain)

    def cost(z: float, s: float) -> float:
        return fit_at(z, s, wave_vac, flux, var, fitted, basis, transmission,
                      cheb_x, poly_order, n_poly_iter, gas=gas, lsf=lsf)[0]

    grid = np.empty((len(sigma_grid), len(redshift_grid)))
    best = (np.inf, float(redshift_grid[0]), float(sigma_grid[0]))
    for i, sigma in enumerate(sigma_grid):
        for j, redshift in enumerate(redshift_grid):
            chi2 = cost(float(redshift), float(sigma))
            grid[i, j] = chi2
            if chi2 < best[0]:
                best = (chi2, float(redshift), float(sigma))
    redshift, sigma = _polish(best[1], best[2], cost, sigma_bounds)[:2]

    chi2, amps, coefficients, model = fit_at(
        redshift, sigma, wave_vac, flux, var, fitted, basis, transmission,
        cheb_x, poly_order, n_poly_iter, gas=gas, lsf=lsf)

    n_clipped = 0
    if clip_sigma is not None:
        residual = (flux - model) / np.sqrt(var)
        clip = fitted & (np.abs(residual) > clip_sigma)
        n_clipped = int(clip.sum())
        if n_clipped:
            fitted &= ~clip
            redshift, sigma = _polish(redshift, sigma, cost,
                                      sigma_bounds)[:2]
            chi2, amps, coefficients, model = fit_at(
                redshift, sigma, wave_vac, flux, var, fitted, basis,
                transmission, cheb_x, poly_order, n_poly_iter, gas=gas,
                lsf=lsf)

    # Every amplitude counts, not the active ones, so gas columns lower dof
    # on every fit -- including those with no line flux.
    n_columns = basis.n_templates + (0 if gas is None else gas.n_columns)
    dof = int(fitted.sum()) - (2 + poly_order + 1 + n_columns)
    pinned = False
    if sigma_bounds is not None:
        edge = 1e-9 * max(abs(sigma_bounds[1]),
                          sigma_bounds[1] - sigma_bounds[0])
        pinned = bool(sigma <= sigma_bounds[0] + edge
                      or sigma >= sigma_bounds[1] - edge)
    z_err = s_err = None
    if errors:
        z_err, s_err = _hessian_errors(redshift, sigma, chi2, dof, cost)
    if pinned:
        # A dispersion held at a bound is not a stationary point in sigma at
        # all -- chi2 is still descending when the bound stops it -- so the
        # Hessian there measures curvature where no extremum exists. Whether
        # that returns nan or a plausible number is decided by the SIGN of a
        # second difference 14 orders of magnitude below the redshift
        # curvature, i.e. by rounding. The finite branch is the dangerous
        # one: it looks quotable. Report neither.
        s_err = None
    return LinearFit(redshift=float(redshift), sigma_kms=float(sigma),
                     chi2=float(chi2), dof=dof, amplitudes=amps,
                     chebyshev_coefficients=coefficients, model=model,
                     fitted=fitted, redshift_error=z_err, sigma_error=s_err,
                     n_clipped=n_clipped, sigma_pinned=pinned,
                     names=list(basis.names),
                     gas_names=[] if gas is None else list(gas.names),
                     redshift_grid=np.asarray(redshift_grid, float),
                     sigma_grid=np.asarray(sigma_grid, float), chi2_grid=grid,
                     minima=rank_minima(redshift_grid, sigma_grid, grid,
                                        dv_kms=minima_dv_kms),
                     grid_n_poly_iter=n_poly_iter)


def coarse_step(gas: GasBasis | None, redshift: float) -> float:
    """The widest coarse step that still resolves the basis at `redshift`.

    Parameters
    ----------
    gas : GasBasis or None
        The emission-line columns, if any. None gives the absorption-only
        step, which is what it was measured against.
    redshift : float
        The low end of the scan, where a velocity width is narrowest in z.

    Returns
    -------
    z_step : float
        Never coarser than the absorption-only step.
    """
    if gas is None:
        return _Z_STEP_COARSE
    return min(_Z_STEP_COARSE,
               _COARSE_GAS_WIDTHS * (1.0 + redshift) * gas.sigma_kms / C_KMS)


def scan_spectrum(wave_vac: np.ndarray, flux: np.ndarray, error: np.ndarray,
                  fitted: np.ndarray, basis: TemplateBasis, *,
                  redshift_grid: np.ndarray, sigma_grid: np.ndarray,
                  poly_order: int, poly_domain: tuple[float, float],
                  transmission: np.ndarray | None = None,
                  poly_wave: np.ndarray | None = None,
                  z_step_coarse: float | None = None,
                  n_poly_iter_coarse: int = _N_POLY_ITER_COARSE,
                  sigma_coarse: float | None = None,
                  window_steps: int = _WINDOW_STEPS,
                  gas: GasBasis | None = None, lsf=None,
                  minima_dv_kms: float = _MINIMUM_DV_KMS,
                  **fit_kwargs) -> SpectrumScan:
    """Blind two-stage redshift scan: coarse basin, then a full refine.

    A single-stage scan at production settings over a wide redshift range is
    tens of thousands of evaluations. The coarse pass drops the polynomial
    iteration and the sigma axis to pick a basin, and `fit_spectrum` then runs
    unmodified over a window of `redshift_grid` around it.

    Parameters
    ----------
    redshift_grid : np.ndarray
        The full fine grid, increasing. The coarse pass spans its ends and the
        refine takes the window of it around the coarse winner, so the refine
        is always on the grid the caller asked for.
    sigma_grid : np.ndarray
        The refine's dispersion grid; the coarse pass uses one value from it.
    z_step_coarse : float or None
        Coarse redshift step. None derives it from the basis: 1.5e-3 with no
        gas, and two gas line widths when there is, since an emission line is
        far narrower than a stellar one. [default: None]
    n_poly_iter_coarse : int
        Chebyshev iterations in the coarse pass. [default: 1]
    sigma_coarse : float or None
        The coarse pass's single dispersion; None takes the median of
        `sigma_grid`. [default: None]
    window_steps : int
        Refine half-width, in coarse steps. [default: 10]
    minima_dv_kms : float
        Velocity separation below which two minima are one basin.
        [default: 1000.0]
    **fit_kwargs
        Passed to `fit_spectrum` for the refine.

    Returns
    -------
    scan : SpectrumScan
        The coarse grid and its ranked minima, plus the refined `LinearFit`.
    """
    redshift_grid = np.asarray(redshift_grid, float)
    sigma_grid = np.asarray(sigma_grid, float)
    if z_step_coarse is None:
        z_step_coarse = coarse_step(gas, float(redshift_grid[0]))
    if sigma_coarse is None:
        sigma_coarse = float(np.median(sigma_grid))

    var = np.asarray(error, float) ** 2
    fitted = np.asarray(fitted, bool)
    if transmission is None:
        transmission = np.ones_like(flux)
    cheb_x = _chebyshev_domain(
        wave_vac if poly_wave is None else np.asarray(poly_wave, float),
        poly_domain)

    coarse = np.arange(redshift_grid[0],
                       redshift_grid[-1] + 0.5 * z_step_coarse, z_step_coarse)
    # The rest range is monotonic in redshift, so the two ends bracket the
    # scan: a coverage failure surfaces here, not eleven seconds into it.
    for probe in (coarse[0], coarse[-1]):
        basis.design(wave_vac, float(probe), sigma_coarse, check=fitted,
                     lsf=lsf)

    chi2 = np.array([fit_at(float(z), sigma_coarse, wave_vac, flux, var,
                            fitted, basis, transmission, cheb_x, poly_order,
                            n_poly_iter_coarse, gas=gas, lsf=lsf)[0]
                     for z in coarse])
    minima = rank_minima(coarse, np.array([sigma_coarse]), chi2[None, :],
                         dv_kms=minima_dv_kms)

    half = window_steps * z_step_coarse
    centre = minima[0].redshift
    window = redshift_grid[(redshift_grid >= centre - half)
                           & (redshift_grid <= centre + half)]
    if not window.size:
        window = redshift_grid[[int(np.abs(redshift_grid - centre).argmin())]]

    fit = fit_spectrum(wave_vac, flux, error, fitted, basis,
                       redshift_grid=window, sigma_grid=sigma_grid,
                       poly_order=poly_order, poly_domain=poly_domain,
                       transmission=transmission, poly_wave=poly_wave,
                       gas=gas, lsf=lsf, minima_dv_kms=minima_dv_kms,
                       **fit_kwargs)
    return SpectrumScan(fit=fit, redshift_grid=coarse,
                        sigma_kms=sigma_coarse, chi2_grid=chi2,
                        z_step=float(z_step_coarse), minima=minima,
                        n_poly_iter=n_poly_iter_coarse)
