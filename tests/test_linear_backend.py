"""Linear spectral engine: recovery, guards, and the conventions it carries.

The bit-identity check against the reference implementation
(`bcg_template_fit.py`, RM J0019 MUSE workspace) is not reproducible here — it
needs that workspace and its 2.9 GB cube. It was run on 2026-08-25 and agreed
to 0.00e+00 on redshift, sigma, chi2, clip count, amplitudes and the model
array; DESIGN.md section 16 records it. These tests pin the behavior that
equivalence rested on.
"""
from __future__ import annotations

import numpy as np
import pytest

from sedfit.backends.linear import (
    TemplateBasis,
    fit_spectrum,
    rank_minima,
    scan_spectrum,
)
from sedfit.core.nnls import nnls_fit

DV_KMS = 15.0
WAVE_RANGE = (3600.0, 7400.0)
NORMALIZE = (3750.0, 7350.0)


def _write_templates(directory, n: int = 4) -> list:
    """A small basis with distinguishable absorption features."""
    wave = np.arange(3400.0, 7600.0, 0.9)
    paths = []
    for k in range(n):
        flux = np.ones_like(wave) * (1.0 + 0.1 * k)
        for centre in (4200.0 + 300.0 * k, 5100.0 + 200.0 * k, 6400.0):
            flux -= (0.3 + 0.05 * k) * np.exp(
                -0.5 * ((wave - centre) / 6.0) ** 2)
        path = directory / f"ssp_test_t{k:02d}.dat"
        np.savetxt(path, np.c_[wave, flux], fmt="%14.4f %.6e")
        paths.append(path)
    return paths


@pytest.fixture(scope="module")
def basis(tmp_path_factory):
    directory = tmp_path_factory.mktemp("templates")
    return TemplateBasis(_write_templates(directory), wave_range=WAVE_RANGE,
                         dv_kms=DV_KMS, normalize_range=NORMALIZE)


def _observed(basis, redshift, sigma_kms, weights, wave_obs):
    rows = basis.design(wave_obs, redshift, sigma_kms)
    return np.asarray(weights) @ rows


def test_templates_normalize_to_unit_mean(basis) -> None:
    grid = np.exp(basis.lnw)
    band = (grid > NORMALIZE[0]) & (grid < NORMALIZE[1])
    assert np.allclose(basis.flam[:, band].mean(axis=1), 1.0)


def test_broadening_conserves_and_widens(basis) -> None:
    narrow = basis.convolved(50.0)
    wide = basis.convolved(300.0)
    assert np.allclose(narrow.mean(axis=1), wide.mean(axis=1), rtol=1e-3)
    assert wide.std(axis=1).max() < narrow.std(axis=1).max()


def test_design_raises_outside_the_grid(basis) -> None:
    # rest 3600-7400 at z = 0.5 is observed 5400-11100; ask for redder
    wave = np.linspace(11000.0, 12000.0, 50)
    with pytest.raises(ValueError, match="leaves the template grid"):
        basis.design(wave, 0.5, 200.0)


def test_recovers_a_known_combination(basis) -> None:
    truth_z, truth_sigma = 0.2690, 250.0
    weights = np.array([0.5, 0.0, 0.3, 0.2])
    wave = np.arange(4800.0, 9300.0, 1.25)
    flux = _observed(basis, truth_z, truth_sigma, weights, wave)
    error = np.full_like(flux, flux.mean() * 1e-3)
    fit = fit_spectrum(
        wave, flux, error, np.ones_like(flux, bool), basis,
        redshift_grid=np.arange(0.264, 0.2741, 4e-4),
        sigma_grid=np.arange(150.0, 401.0, 50.0),
        poly_order=8, poly_domain=(4750.0, 9350.2), clip_sigma=None,
        errors=False)
    assert fit.redshift == pytest.approx(truth_z, abs=2e-5)
    assert fit.sigma_kms == pytest.approx(truth_sigma, abs=2.0)
    recovered = fit.amplitudes / fit.amplitudes.sum()
    assert np.allclose(recovered, weights, atol=5e-3)


def test_light_fractions_drop_zeros(basis) -> None:
    truth = np.array([0.6, 0.0, 0.4, 0.0])
    wave = np.arange(4800.0, 9300.0, 1.25)
    flux = _observed(basis, 0.269, 250.0, truth, wave)
    fit = fit_spectrum(
        wave, flux, np.full_like(flux, flux.mean() * 1e-3),
        np.ones_like(flux, bool), basis,
        redshift_grid=np.array([0.269]), sigma_grid=np.array([250.0]),
        poly_order=4, poly_domain=(4750.0, 9350.2), clip_sigma=None,
        errors=False)
    assert set(fit.light_fractions) <= set(basis.names)
    assert all(value > 0 for value in fit.light_fractions.values())
    assert sum(fit.light_fractions.values()) == pytest.approx(1.0)


def test_poly_wave_changes_the_answer(basis) -> None:
    """The Chebyshev coordinate is a free choice but not a neutral one.

    Air and vacuum differ by ~2.6 A at 9000 A; the reference builds the
    polynomial on air while the design matrix uses vacuum. Matching that was
    the difference between agreeing to 5e-5 and agreeing exactly.
    """
    wave = np.arange(4800.0, 9300.0, 1.25)
    flux = _observed(basis, 0.269, 250.0, np.array([0.5, 0.2, 0.3, 0.0]), wave)
    error = np.full_like(flux, flux.mean() * 1e-3)
    common = dict(redshift_grid=np.array([0.269]),
                  sigma_grid=np.array([250.0]), poly_order=8,
                  poly_domain=(4750.0, 9350.2), clip_sigma=None, errors=False)
    default = fit_spectrum(wave, flux, error, np.ones_like(flux, bool), basis,
                           **common)
    shifted = fit_spectrum(wave, flux, error, np.ones_like(flux, bool), basis,
                           poly_wave=wave - 2.6, **common)
    assert not np.array_equal(default.amplitudes, shifted.amplitudes)


def test_nnls_is_the_shared_solver() -> None:
    """core.nnls is what the eazy backend uses too -- one solver, not two."""
    from sedfit.backends.eazy import quick
    assert quick._nnls_fit is nnls_fit
    rng = np.random.default_rng(0)
    design = np.abs(rng.normal(size=(3, 40))) + 0.1
    truth = np.array([1.5, 0.0, 0.7])
    data = truth @ design
    var = np.full(40, 1e-6)
    chi2, coeffs, model = nnls_fit(design, data, var, np.ones(40, bool))
    assert np.allclose(coeffs, truth, atol=1e-6)
    assert chi2 < 1e-6
    assert np.allclose(model, data, atol=1e-6)


def _two_wells(redshift_grid: np.ndarray, first: float, second: float,
               depths: tuple[float, float]) -> np.ndarray:
    """A chi-square profile with two resolved parabolic basins."""
    width = 8e-4
    return (100.0
            - depths[0] * np.exp(-0.5 * ((redshift_grid - first) / width) ** 2)
            - depths[1] * np.exp(-0.5 * ((redshift_grid - second) / width) ** 2))


def test_minima_are_ranked_by_chi_square() -> None:
    grid = np.arange(0.20, 0.24, 2e-4)
    chi2 = _two_wells(grid, 0.210, 0.230, (30.0, 50.0))
    minima = rank_minima(grid, np.array([200.0]), chi2[None, :])
    assert [round(m.redshift, 4) for m in minima[:2]] == [0.2300, 0.2100]
    assert minima[0].delta_chi2 == 0.0
    assert minima[1].delta_chi2 == pytest.approx(20.0, abs=0.5)


def test_distinctness_is_a_velocity_not_a_redshift() -> None:
    """The same dz is one basin at high z and two at low z."""
    separation = 6e-3
    near = np.arange(0.0, 0.04, 2e-4)
    far = np.arange(3.0, 3.04, 2e-4)
    resolved = rank_minima(
        near, np.array([200.0]),
        _two_wells(near, 0.014, 0.014 + separation, (30.0, 50.0))[None, :])
    merged = rank_minima(
        far, np.array([200.0]),
        _two_wells(far, 3.014, 3.014 + separation, (30.0, 50.0))[None, :])
    assert len(resolved) == 2
    assert len(merged) == 1


def test_a_single_point_grid_has_one_minimum() -> None:
    minima = rank_minima(np.array([0.3]), np.array([200.0]),
                         np.array([[7.0]]))
    assert len(minima) == 1
    assert minima[0].redshift == 0.3 and minima[0].delta_chi2 == 0.0


def test_the_scan_grid_survives_the_fit(basis) -> None:
    """The grid fit_spectrum used to pick a basin, kept rather than dropped."""
    wave = np.arange(4800.0, 9300.0, 1.25)
    flux = _observed(basis, 0.269, 250.0, np.array([0.5, 0.0, 0.3, 0.2]), wave)
    redshifts = np.arange(0.264, 0.2741, 4e-4)
    sigmas = np.arange(150.0, 401.0, 50.0)
    fit = fit_spectrum(
        wave, flux, np.full_like(flux, flux.mean() * 1e-3),
        np.ones_like(flux, bool), basis, redshift_grid=redshifts,
        sigma_grid=sigmas, poly_order=8, poly_domain=(4750.0, 9350.2),
        clip_sigma=None, errors=False, n_poly_iter=2)
    assert fit.chi2_grid.shape == (sigmas.size, redshifts.size)
    assert np.array_equal(fit.redshift_grid, redshifts)
    assert np.array_equal(fit.sigma_grid, sigmas)
    # the grid picked the basin; the polish then improved on it
    assert fit.chi2 <= fit.chi2_grid.min()
    # ... which is why the two must never be differenced
    assert fit.grid_n_poly_iter == 2
    assert fit.minima[0].chi2 == pytest.approx(fit.chi2_grid.min())


def test_delta_chi2_needs_a_runner_up(basis) -> None:
    wave = np.arange(4800.0, 9300.0, 1.25)
    flux = _observed(basis, 0.269, 250.0, np.array([0.5, 0.0, 0.3, 0.2]), wave)
    fit = fit_spectrum(
        wave, flux, np.full_like(flux, flux.mean() * 1e-3),
        np.ones_like(flux, bool), basis, redshift_grid=np.array([0.269]),
        sigma_grid=np.array([250.0]), poly_order=4,
        poly_domain=(4750.0, 9350.2), clip_sigma=None, errors=False)
    assert fit.delta_chi2 is None


BLIND_WAVE = np.arange(5500.0, 7000.0, 1.25)
BLIND_Z, BLIND_SIGMA = 0.3100, 250.0
BLIND_GRID = np.arange(0.15, 0.45, 4e-4)


@pytest.fixture(scope="module")
def blind(basis):
    """A spectrum over a range narrow enough to scan a wide redshift span."""
    flux = _observed(basis, BLIND_Z, BLIND_SIGMA,
                     np.array([0.5, 0.0, 0.3, 0.2]), BLIND_WAVE)
    return flux, np.full_like(flux, flux.mean() * 2e-3)


def _scan(basis, blind, **overrides):
    flux, error = blind
    kwargs = dict(redshift_grid=BLIND_GRID, sigma_grid=np.array([200.0, 300.0]),
                  poly_order=4, poly_domain=(5500.0, 7000.0),
                  z_step_coarse=3e-3, window_steps=4, clip_sigma=None,
                  errors=False)
    kwargs.update(overrides)
    return scan_spectrum(BLIND_WAVE, flux, error,
                         np.ones_like(flux, bool), basis, **kwargs)


def test_a_blind_scan_finds_the_redshift(basis, blind) -> None:
    """No prior, no input redshift: 0.15 to 0.45 with nothing to go on."""
    scan = _scan(basis, blind)
    assert scan.fit.redshift == pytest.approx(BLIND_Z, abs=1e-3)
    assert scan.minima[0].redshift == pytest.approx(BLIND_Z, abs=3e-3)


def test_the_refine_touches_a_fraction_of_the_blind_grid(basis, blind) -> None:
    """What makes the verb usable: the fine grid is never scanned whole."""
    scan = _scan(basis, blind)
    assert scan.fit.redshift_grid.size < BLIND_GRID.size / 5
    assert set(scan.fit.redshift_grid.tolist()) <= set(BLIND_GRID.tolist())
    assert abs(scan.fit.redshift_grid.mean() - scan.minima[0].redshift) < 0.012


def test_the_two_grids_are_never_on_the_same_footing(basis, blind) -> None:
    scan = _scan(basis, blind)
    assert scan.n_poly_iter == 1
    assert scan.fit.grid_n_poly_iter == 4
    assert scan.chi2_grid.size == scan.redshift_grid.size
    assert scan.redshift_grid.size < BLIND_GRID.size


def test_the_coarse_pass_uses_one_representative_sigma(basis, blind) -> None:
    assert _scan(basis, blind).sigma_kms == 250.0
    assert _scan(basis, blind, sigma_coarse=180.0).sigma_kms == 180.0


def test_coverage_is_checked_before_the_scan_not_during(basis, blind) -> None:
    """template_wave_range must cover lambda_min / (1 + z_max)."""
    with pytest.raises(ValueError, match="template grid"):
        _scan(basis, blind, redshift_grid=np.arange(0.15, 0.85, 4e-4))


def test_the_broadening_kernel_must_fit_the_grid(basis) -> None:
    """Cost is linear in sigma with no natural bound, so a runaway stalls.

    3.8 ms at 250 km/s against six seconds at 3.4e5 on a 15 km/s grid; the
    ceiling turns that into an immediate, legible error.
    """
    assert basis.convolved(400.0).shape[0] == basis.n_templates
    with pytest.raises(ValueError, match="must stay well inside"):
        basis.convolved(3.4e5)


def _bounded(basis, sigma_bounds):
    wave = np.arange(4800.0, 9300.0, 1.25)
    flux = _observed(basis, 0.269, 250.0, np.array([0.5, 0.0, 0.3, 0.2]), wave)
    error = np.full_like(flux, flux.mean() * 0.01)
    flux = flux + np.random.default_rng(2).normal(0.0, error)
    return fit_spectrum(
        wave, flux, error, np.ones_like(flux, bool), basis,
        redshift_grid=np.arange(0.264, 0.2741, 4e-4),
        sigma_grid=np.arange(150.0, 401.0, 50.0), poly_order=8,
        poly_domain=(4750.0, 9350.2), clip_sigma=None, errors=False,
        sigma_bounds=sigma_bounds)


def test_a_bound_that_does_not_bind_changes_nothing(basis) -> None:
    """What lets the runner declare one on every fit."""
    free, held = _bounded(basis, None), _bounded(basis, (150.0, 400.0))
    assert held.sigma_kms == free.sigma_kms
    assert held.redshift == free.redshift
    assert held.chi2 == free.chi2
    assert not held.sigma_pinned and not free.sigma_pinned


def test_a_bound_that_binds_holds_sigma_and_says_so(basis) -> None:
    """Unbounded, the simplex is the one part of the fit that never sees
    sigma_min and sigma_max."""
    held = _bounded(basis, (150.0, 200.0))
    assert held.sigma_kms == pytest.approx(200.0)
    assert held.sigma_pinned
    # a clamped dispersion is a different claim from a fitted one
    assert _bounded(basis, None).sigma_kms > 200.0
