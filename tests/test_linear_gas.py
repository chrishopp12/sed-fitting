"""Emission-line columns: what the basis is, and what it changes downstream.

The lines are built analytically at observed wavelength rather than resampled
from files, so these pin the column construction itself -- unit integrated
flux, locked ratios in one column, zero outside the band -- and then the two
things gas changes about a fit: dof, and which basin a blind scan lands in.
"""
from __future__ import annotations

import numpy as np
import pytest

from sedfit.backends.linear import (
    TemplateBasis,
    coarse_step,
    fit_spectrum,
    scan_spectrum,
)
from sedfit.backends.linear.basis import C_KMS
from sedfit.backends.linear.gas import (
    GasBasis,
    packaged_line_lists,
    read_line_list,
    resolve_line_list,
)
from sedfit.core.fitconfig import LINEAR_GAS_RATIO_LOCKED
from test_linear_backend import (
    DV_KMS,
    NORMALIZE,
    WAVE_RANGE,
    _write_templates,
)

WAVE = np.arange(5500.0, 7000.0, 1.25)
TRUTH_Z, TRUTH_SIGMA, GAS_SIGMA = 0.3100, 250.0, 100.0
WEIGHTS = np.array([0.5, 0.0, 0.3, 0.2])
FLUXES = {"[OIII]": 30.0, "Hbeta": 10.0, "Hgamma": 4.0}


@pytest.fixture(scope="module")
def lines():
    return read_line_list(resolve_line_list("optical"))


@pytest.fixture(scope="module")
def gas(lines):
    return GasBasis(lines, sigma_kms=GAS_SIGMA)


@pytest.fixture(scope="module")
def basis(tmp_path_factory):
    directory = tmp_path_factory.mktemp("gas_templates")
    return TemplateBasis(_write_templates(directory), wave_range=WAVE_RANGE,
                         dv_kms=DV_KMS, normalize_range=NORMALIZE)


@pytest.fixture(scope="module")
def spectrum(basis, lines):
    """Continuum plus emission lines at the truth redshift."""
    truth = GasBasis(lines, sigma_kms=GAS_SIGMA)
    amplitudes = np.array([FLUXES.get(name, 0.0) for name in truth.names])
    flux = (WEIGHTS @ basis.design(WAVE, TRUTH_Z, TRUTH_SIGMA)
            + amplitudes @ truth.design(WAVE, TRUTH_Z))
    error = np.full_like(flux, flux.mean() * 0.03)
    return flux + np.random.default_rng(0).normal(0.0, error), error


# ------------------------------------
# The line list
# ------------------------------------

def test_the_packaged_list_resolves_by_bare_name() -> None:
    assert "optical" in packaged_line_lists()
    assert resolve_line_list("optical").name == "optical.txt"


def test_an_unknown_list_names_what_is_available() -> None:
    with pytest.raises(ValueError, match="packaged lists are"):
        resolve_line_list("no_such_list")


def test_the_list_is_rest_vacuum(lines) -> None:
    # Hbeta air 4861.33 -> vacuum 4862.68; a list in air would fit 83 km/s off
    assert lines["Hbeta"] == pytest.approx(4862.683, abs=0.01)
    assert lines["Halpha"] == pytest.approx(6564.608, abs=0.01)
    assert all(value > 0 for value in lines.values())


@pytest.mark.parametrize("text,message", [
    ("Hbeta 4862.683\nHbeta 4862.683\n", "listed twice"),
    ("Hbeta\n", "expected 'name wavelength'"),
    ("Hbeta -1.0\n", "wavelength"),
    ("# only a comment\n", "no lines"),
])
def test_a_malformed_list_is_refused(tmp_path, text: str, message: str) -> None:
    path = tmp_path / "bad.txt"
    path.write_text(text)
    with pytest.raises(ValueError, match=message):
        read_line_list(path)


# ------------------------------------
# The columns
# ------------------------------------

def test_locked_pairs_are_one_column(lines, gas) -> None:
    locked = sum(len(group["lines"]) for group in LINEAR_GAS_RATIO_LOCKED)
    assert gas.n_columns == len(lines) - locked + len(LINEAR_GAS_RATIO_LOCKED)
    assert "[OIII]" in gas.names and "[OIII]5007" not in gas.names


def test_a_locked_pair_holds_its_ratio(lines) -> None:
    """One column, both members, at the ratio atomic physics fixes."""
    gas = GasBasis(lines, sigma_kms=GAS_SIGMA, ratio_locked=[
        {"name": "[OIII]", "lines": ["[OIII]4959", "[OIII]5007"],
         "ratios": [1.0, 2.98]}])
    wave = np.arange(4900.0, 5100.0, 0.2)
    row = gas.design(wave, 0.0)[gas.names.index("[OIII]")]

    def flux(centre: float) -> float:
        near = np.abs(wave - centre) < 10.0
        return float(np.trapezoid(row[near], wave[near]))

    # a flux ratio, not a peak ratio: a constant-velocity width makes the
    # redder line broader in Angstroms and therefore shallower
    assert (flux(lines["[OIII]5007"]) / flux(lines["[OIII]4959"])
            == pytest.approx(2.98, rel=1e-3))


def test_each_column_carries_unit_integrated_flux(gas) -> None:
    """So an amplitude reads as a line flux rather than an arbitrary scale."""
    wave = np.arange(3000.0, 10000.0, 0.5)
    integrals = np.trapezoid(gas.design(wave, 0.0), wave, axis=1)
    assert np.allclose(integrals, 1.0, atol=1e-4)


def test_a_line_outside_the_band_is_a_zero_column(gas) -> None:
    rows = gas.design(WAVE, TRUTH_Z)
    empty = [name for name, row in zip(gas.names, rows) if not row.any()]
    assert "Halpha" in empty
    # the column stays in the matrix so dof and chi2 do not step with z
    assert rows.shape[0] == gas.n_columns


def test_the_width_is_the_quadrature_sum_with_the_lsf(lines) -> None:
    gas = GasBasis({"Hbeta": lines["Hbeta"]}, sigma_kms=GAS_SIGMA)
    wave = np.arange(4700.0, 5000.0, 0.1)
    lsf = np.full_like(wave, 150.0)
    peaks = [gas.design(wave, 0.0, lsf_sigma_kms=arg).max()
             for arg in (None, lsf)]
    # a normalized Gaussian's peak falls as 1 / sigma
    assert peaks[0] / peaks[1] == pytest.approx(
        np.hypot(GAS_SIGMA, 150.0) / GAS_SIGMA, rel=1e-3)


@pytest.mark.parametrize("groups,message", [
    ([{"name": "X", "lines": ["Hbeta", "nope"], "ratios": [1.0, 2.0]}],
     "not in the line list"),
    ([{"name": "X", "lines": ["Hbeta"], "ratios": [1.0, 2.0]}], "against"),
    ([{"name": "X", "lines": ["Hbeta"], "ratios": [-1.0]}], "positive"),
    ([{"name": "A", "lines": ["Hbeta"], "ratios": [1.0]},
      {"name": "B", "lines": ["Hbeta"], "ratios": [1.0]}], "in both"),
    ([{"name": "Hbeta", "lines": ["Halpha"], "ratios": [1.0]}],
     "both a line and a group"),
])
def test_a_malformed_group_is_refused(lines, groups, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        GasBasis(lines, ratio_locked=groups)


def test_a_group_no_line_in_the_list_belongs_to_is_skipped(lines) -> None:
    """A short list missing a species is a choice; half a locked pair is not."""
    without = {k: v for k, v in lines.items() if not k.startswith("[OIII]4")
               and not k.startswith("[OIII]5")}
    gas = GasBasis(without, sigma_kms=GAS_SIGMA)
    assert "[OIII]" not in gas.names
    assert "[NII]" in gas.names


# ------------------------------------
# What gas changes about a fit
# ------------------------------------

def _fit(basis, spectrum, gas, **overrides):
    flux, error = spectrum
    kwargs = dict(redshift_grid=np.array([TRUTH_Z]),
                  sigma_grid=np.array([TRUTH_SIGMA]), poly_order=4,
                  poly_domain=(5500.0, 7000.0), gas=gas, clip_sigma=None,
                  errors=False)
    kwargs.update(overrides)
    return fit_spectrum(WAVE, flux, error, np.ones_like(flux, bool), basis,
                        **kwargs)


def test_gas_recovers_the_line_fluxes(basis, spectrum, gas) -> None:
    fit = _fit(basis, spectrum, gas)
    assert fit.chi2 / fit.dof == pytest.approx(1.0, abs=0.15)
    for name, truth in FLUXES.items():
        assert fit.gas_fluxes[name] == pytest.approx(truth, rel=0.05)


def test_a_wrong_gas_width_costs_flux_and_not_redshift(basis, lines) -> None:
    """Why gas.sigma_kms is fixed by config rather than freed.

    A symmetric kernel does not move a line centroid, so the redshift
    survives a wrong width to first order while the amplitude does not.
    """
    truth = GasBasis(lines, sigma_kms=180.0)
    amplitudes = np.array([FLUXES.get(name, 0.0) for name in truth.names])
    flux = (WEIGHTS @ basis.design(WAVE, TRUTH_Z, TRUTH_SIGMA)
            + amplitudes @ truth.design(WAVE, TRUTH_Z))
    error = np.full_like(flux, flux.mean() * 0.03)
    flux = flux + np.random.default_rng(0).normal(0.0, error)

    narrow = _fit(basis, (flux, error), GasBasis(lines, sigma_kms=GAS_SIGMA),
                  redshift_grid=np.arange(TRUTH_Z - 2e-3, TRUTH_Z + 2e-3, 2e-4))
    matched = _fit(basis, (flux, error), truth,
                   redshift_grid=np.arange(TRUTH_Z - 2e-3, TRUTH_Z + 2e-3, 2e-4))
    assert narrow.redshift == pytest.approx(matched.redshift, abs=1e-4)
    assert narrow.gas_fluxes["[OIII]"] < 0.95 * matched.gas_fluxes["[OIII]"]


def test_light_fractions_stay_stellar(basis, spectrum, gas) -> None:
    """A line flux is not a share of a continuum, so it is reported apart."""
    fit = _fit(basis, spectrum, gas)
    assert set(fit.light_fractions) <= set(basis.names)
    assert sum(fit.light_fractions.values()) == pytest.approx(1.0)
    assert set(fit.gas_fluxes) == set(gas.names)
    assert fit.stellar_amplitudes.size == basis.n_templates


def test_gas_columns_cost_degrees_of_freedom(basis, spectrum, gas) -> None:
    """Every amplitude counts, not the active ones -- so dof falls always."""
    assert (_fit(basis, spectrum, None).dof - _fit(basis, spectrum, gas).dof
            == gas.n_columns)


def test_the_coarse_step_tightens_when_gas_is_present(gas) -> None:
    assert coarse_step(None, 0.15) == pytest.approx(1.5e-3)
    tight = coarse_step(gas, 0.15)
    assert tight == pytest.approx(2.0 * 1.15 * GAS_SIGMA / C_KMS)
    assert tight < coarse_step(None, 0.15)


def test_a_blind_scan_finds_an_emission_line_redshift(basis, spectrum, gas
                                                      ) -> None:
    """The step has to resolve the narrowest feature, which gas now sets."""
    flux, error = spectrum
    scan = scan_spectrum(WAVE, flux, error, np.ones_like(flux, bool), basis,
                         gas=gas, redshift_grid=np.arange(0.15, 0.45, 4e-4),
                         sigma_grid=np.array([200.0, 300.0]), poly_order=4,
                         poly_domain=(5500.0, 7000.0), window_steps=10,
                         errors=False)
    assert scan.z_step == pytest.approx(coarse_step(gas, 0.15))
    assert scan.fit.redshift == pytest.approx(TRUTH_Z, abs=1e-3)
