"""The instrument LSF and the template library's own resolution.

The kernel is a quadrature difference between the two, applied in the
observed frame after the redshift shift. These pin the unit decoding, the
two application paths -- one velocity width folded into the cached
broadening, or a per-pixel convolution -- and the undersampling policy, which
has no default because all three answers are defensible.
"""
from __future__ import annotations

import numpy as np
import pytest

from sedfit.backends.linear import TemplateBasis, fit_spectrum
from sedfit.backends.linear.basis import C_KMS
from sedfit.backends.linear.lsf import (
    FWHM_PER_SIGMA,
    LineSpread,
    Resolution,
    resolve_resolution_file,
    smear,
    to_sigma_kms,
)

WAVE = np.arange(5000.0, 6500.0, 0.6)
Z, STAR_SIGMA, LIB_KMS = 0.1000, 80.0, 10.0
C3K_SIGMA_KMS = 42.4345


@pytest.fixture(scope="module")
def c3k():
    return Resolution.from_file(resolve_resolution_file("ssp_c3k_a"),
                                unit="sigma_kms", frame="vacuum")


def _narrow_templates(directory, n: int = 3) -> list:
    """Sharp features, so the LOSVD is what sets the observed line width."""
    wave = np.arange(4300.0, 6200.0, 0.2)
    rng = np.random.default_rng(3)
    paths = []
    for k in range(n):
        flux = np.ones_like(wave) * (1.0 + 0.1 * k)
        for centre in rng.uniform(4400.0, 6100.0, 40):
            flux -= 0.25 * np.exp(-0.5 * ((wave - centre) / 1.0) ** 2)
        path = directory / f"ssp_narrow_t{k:02d}.dat"
        np.savetxt(path, np.c_[wave, flux], fmt="%14.4f %.6e")
        paths.append(path)
    return paths


@pytest.fixture(scope="module")
def basis(tmp_path_factory):
    directory = tmp_path_factory.mktemp("lsf_templates")
    return TemplateBasis(_narrow_templates(directory),
                         wave_range=(4350.0, 6150.0), dv_kms=5.0,
                         normalize_range=(4400.0, 6100.0))


# ------------------------------------
# Units and curves
# ------------------------------------

def test_every_unit_decodes_to_the_same_width() -> None:
    """Five spellings of one physical width."""
    wave, sigma_kms = 5000.0, 60.0
    sigma_A = sigma_kms * wave / C_KMS
    for value, unit in ((C_KMS / (sigma_kms * FWHM_PER_SIGMA), "R"),
                        (sigma_kms * FWHM_PER_SIGMA, "fwhm_kms"),
                        (sigma_kms, "sigma_kms"),
                        (sigma_A * FWHM_PER_SIGMA, "fwhm_A"),
                        (sigma_A, "sigma_A")):
        assert to_sigma_kms(value, wave, unit) == pytest.approx(sigma_kms)


def test_an_unknown_unit_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown resolution unit"):
        Resolution(unit="angstroms_ish", constant=1.0)


def test_a_resolution_takes_a_constant_or_a_table_not_both() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        Resolution(unit="R", constant=3000.0, wave=WAVE, value=WAVE)
    with pytest.raises(ValueError, match="exactly one"):
        Resolution(unit="R")


def test_the_packaged_curve_resolves_by_set_name(c3k) -> None:
    assert resolve_resolution_file("ssp_c3k_a").name == "resolution.txt"
    # DESIGN 17.2a: R = 3000 over 2750-9100 A rest, not full C3K's 10000
    assert c3k.sigma_kms(np.array([5000.0]))[0] == pytest.approx(
        C3K_SIGMA_KMS, abs=1e-3)


def test_a_query_off_the_table_is_refused(c3k) -> None:
    with pytest.raises(ValueError, match="but was asked for"):
        c3k.sigma_kms(np.array([2e5]))


def test_the_shipped_curve_spans_the_shipped_templates(c3k) -> None:
    """Same grid, so a scan can never leave the curve before the basis."""
    grid = np.loadtxt(sorted(
        resolve_resolution_file("ssp_c3k_a").parent.glob("*.dat"))[0])[:, 0]
    assert np.array_equal(c3k.wave, grid)


def test_a_curve_is_read_in_the_frame_it_declares(tmp_path) -> None:
    """2.5 A at 9155 A is what guessing costs, so the frame is required."""
    path = tmp_path / "curve.txt"
    np.savetxt(path, np.c_[np.array([4000.0, 10000.0]),
                           np.array([2.0, 3.0])])
    query = np.array([9155.0])
    vacuum = Resolution.from_file(path, unit="fwhm_A", frame="vacuum")
    air = Resolution.from_file(path, unit="fwhm_A", frame="air")
    assert vacuum.sigma_kms(query)[0] != air.sigma_kms(query)[0]


# ------------------------------------
# The kernel
# ------------------------------------

def test_a_constant_r_lsf_folds_into_one_velocity_width(c3k) -> None:
    """Both constant in velocity, so the difference is too -- and cached."""
    lsf = LineSpread(WAVE, Resolution(unit="R", constant=2000.0), c3k,
                     on_undersampled="raise")
    flat, sigma_A = lsf.plan(0.3)
    assert sigma_A is None
    expected = np.sqrt((C_KMS / (2000.0 * FWHM_PER_SIGMA)) ** 2
                       - C3K_SIGMA_KMS ** 2)
    assert flat == pytest.approx(expected, rel=1e-3)
    # z-invariant, which is what lets one convolution serve a whole scan
    assert lsf.plan(0.6)[0] == pytest.approx(flat)


def test_a_constant_wavelength_lsf_needs_a_per_pixel_kernel(c3k) -> None:
    lsf = LineSpread(WAVE, Resolution(unit="fwhm_A", constant=2.6), c3k,
                     on_undersampled="ignore")
    flat, sigma_A = lsf.plan(0.3)
    assert flat is None
    assert sigma_A.shape == WAVE.shape
    # constant in A means falling in velocity, so the kernel narrows redward
    assert (sigma_A / WAVE)[0] > (sigma_A / WAVE)[-1]


def test_an_undersampled_band_is_refused_by_name(c3k) -> None:
    """Neither shipped set matches MUSE over ~43% of the band at z = 0."""
    lsf = LineSpread(WAVE, Resolution(unit="R", constant=8000.0), c3k,
                     on_undersampled="raise")
    with pytest.raises(ValueError, match="broader than the data"):
        lsf.plan(0.3)


def test_ignore_clamps_rather_than_refusing(c3k) -> None:
    lsf = LineSpread(WAVE, Resolution(unit="R", constant=8000.0), c3k,
                     on_undersampled="ignore")
    flat, sigma_A = lsf.plan(0.3)
    assert (flat == 0.0) or (sigma_A is not None and sigma_A.min() >= 0.0)


def test_degrade_data_refuses_a_library_that_moves_with_redshift(c3k) -> None:
    """A constant-FWHM library needs the degradation redone at every z."""
    miles = Resolution.from_file(resolve_resolution_file("ssp_miles"),
                                 unit="sigma_kms", frame="vacuum")
    LineSpread(WAVE, Resolution(unit="R", constant=3000.0), c3k,
               on_undersampled="degrade_data"
               ).assert_degradation_is_fixed(0.0, 0.5)
    with pytest.raises(ValueError, match="does not move with redshift"):
        LineSpread(WAVE, Resolution(unit="R", constant=3000.0), miles,
                   on_undersampled="degrade_data"
                   ).assert_degradation_is_fixed(0.0, 0.5)


# ------------------------------------
# Smearing
# ------------------------------------

def test_smearing_conserves_and_widens() -> None:
    line = np.exp(-0.5 * ((WAVE - 5750.0) / 2.0) ** 2)
    wide = smear(line, np.full_like(WAVE, 3.0), np.gradient(WAVE))
    assert np.trapezoid(wide, WAVE) == pytest.approx(
        np.trapezoid(line, WAVE), rel=1e-3)
    assert wide.max() < line.max()


def test_variance_propagation_matches_smoothed_noise() -> None:
    """sum(w^2 v) / sum(w)^2, which for white noise is 1 / (2 sqrt(pi) s)."""
    n, width = 20000, 3.0
    noise = np.random.default_rng(0).normal(0.0, 1.0, n)
    dispersion, sigma = np.ones(n), np.full(n, width)
    smoothed = smear(noise, sigma, dispersion)
    propagated = smear(np.ones(n), sigma, dispersion, variance=True)
    analytic = 1.0 / (2.0 * np.sqrt(np.pi) * width)
    assert propagated[100:-100].mean() == pytest.approx(analytic, rel=1e-3)
    assert smoothed[100:-100].var() == pytest.approx(analytic, rel=0.05)


def test_degrading_leaves_masked_channels_out_of_the_average(c3k) -> None:
    """A normalized convolution, so a mask edge does not bleed into the fit."""
    lsf = LineSpread(WAVE, Resolution(unit="R", constant=8000.0), c3k,
                     on_undersampled="degrade_data")
    flux = np.ones_like(WAVE)
    fitted = np.ones_like(WAVE, bool)
    fitted[:50] = False
    flux[:50] = 1e6
    out_flux, out_error = lsf.degrade(flux, np.ones_like(WAVE), fitted, 0.3)
    assert out_flux[fitted].max() < 1.001
    assert np.array_equal(out_flux[~fitted], flux[~fitted])
    # smoothing correlates neighbours, so the diagonal variance falls
    assert out_error[fitted].max() < 1.0


# ------------------------------------
# What it buys
# ------------------------------------

def test_a_supplied_lsf_returns_the_intrinsic_dispersion(basis, tmp_path
                                                         ) -> None:
    """Ignoring a real LSF folds the instrument into the reported sigma."""
    path = tmp_path / "rising.txt"
    np.savetxt(path, np.c_[np.linspace(4900.0, 6600.0, 50),
                           np.linspace(1500.0, 4000.0, 50)])
    lsf = LineSpread(WAVE, Resolution.from_file(path, unit="R",
                                                frame="vacuum"),
                     Resolution(unit="sigma_kms", constant=LIB_KMS),
                     on_undersampled="raise")
    flux = np.array([0.5, 0.3, 0.2]) @ basis.design(WAVE, Z, STAR_SIGMA,
                                                    lsf=lsf)
    error = np.full_like(flux, flux.mean() * 0.01)
    flux = flux + np.random.default_rng(5).normal(0.0, error)

    def run(arg):
        return fit_spectrum(WAVE, flux, error, np.ones_like(flux, bool), basis,
                            redshift_grid=np.arange(Z - 1e-3, Z + 1e-3, 2e-4),
                            sigma_grid=np.arange(40.0, 181.0, 20.0),
                            poly_order=4, poly_domain=(5000.0, 6500.0),
                            lsf=arg, clip_sigma=None, errors=False)

    supplied, ignored = run(lsf), run(None)
    assert supplied.sigma_kms == pytest.approx(STAR_SIGMA, abs=4.0)
    assert ignored.sigma_kms > supplied.sigma_kms + 10.0
    # a symmetric kernel does not move a centroid: z survives either way
    assert ignored.redshift == pytest.approx(supplied.redshift, abs=2e-5)
