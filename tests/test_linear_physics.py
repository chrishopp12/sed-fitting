"""Physical coherence of a fitted line set, and the bounds it is judged by.

Nothing here changes a fitted value: the checks derive from `gas_fluxes`, so
a fit's numbers are identical whether or not they run.
"""
from __future__ import annotations

import numpy as np
import pytest

from sedfit.backends.linear.gas import GasBasis, read_line_list, resolve_line_list
from sedfit.backends.linear.physics import (
    BALMER_MINIMUM, REST_FLOOR_KMS, check_presence, check_ratios,
    velocity_floor_kms)
from sedfit.backends.linear.runner import band_coverage

MUSE = np.arange(4750.0, 9350.0, 1.25)


def _kinds(violations) -> set:
    return {v.kind for v in violations}


def test_a_clean_set_passes() -> None:
    assert check_ratios({"Halpha": 300.0, "Hbeta": 100.0, "Hgamma": 46.0,
                         "[OIII]4959": 100.0, "[OIII]5007": 298.0}) == []


def test_an_inverted_balmer_decrement_is_caught() -> None:
    got = check_ratios({"Halpha": 100.0, "Hbeta": 100.0})
    assert _kinds(got) == {"balmer"}
    assert got[0].observed == pytest.approx(1.0)


def test_dust_can_only_raise_the_decrement_so_the_bound_is_one_sided() -> None:
    # Every case B ratio is a floor: an arbitrarily reddened set stays legal.
    for reddening in (1.0, 3.0, 10.0):
        assert check_ratios({"Halpha": 2.86 * reddening * 100.0,
                             "Hbeta": 100.0}) == []


def test_a_source_at_the_low_density_limit_is_not_flagged() -> None:
    # A real RM J0019 arc fits to 1.511 against a theoretical limit of 1.50.
    # A hard bound there would reject correct physics.
    assert check_ratios({"[OII]3727": 325.3, "[OII]3730": 491.4}) == []


def test_a_grossly_broken_doublet_is_still_caught() -> None:
    assert _kinds(check_ratios({"[SII]6716": 3.0,
                                "[SII]6731": 1.0})) == {"density"}


def test_an_auroral_line_rivalling_its_nebular_partner_is_caught() -> None:
    assert _kinds(check_ratios({"[OIII]4363": 50.0,
                                "[OIII]5007": 100.0})) == {"temperature"}


def test_absent_and_negligible_lines_are_skipped_not_failed() -> None:
    # A line that was not measured cannot violate anything.
    assert check_ratios({"Halpha": 100.0}) == []
    assert check_ratios({"Halpha": 100.0, "Hbeta": 1e-9}) == []


def test_presence_check_is_flux_free() -> None:
    # Hbeta is at least 2.13x Hgamma for ANY reddening, so detecting the
    # fainter without the brighter is incoherent whatever the fluxes were.
    got = check_presence(["Hdelta", "Hgamma"],
                         ["Hdelta", "Hgamma", "Hbeta", "Halpha"])
    assert _kinds(got) == {"absence"}
    assert any("Hbeta" in v.lines for v in got)


def test_presence_check_forgives_a_line_out_of_band() -> None:
    assert check_presence(["Hgamma"], ["Hgamma", "Hdelta"]) == []


def test_every_balmer_bound_exceeds_unity() -> None:
    # The redder member is always the brighter one; a bound below 1 would
    # make the ordering unenforceable.
    assert all(floor > 1.0 for _, _, floor in BALMER_MINIMUM)


def test_the_velocity_floor_is_the_worst_line_not_the_mean() -> None:
    assert velocity_floor_kms(["[OIII]4959", "[OIII]5007"]) == pytest.approx(0.2)
    assert velocity_floor_kms(["Halpha", "[NII]6584"]) == pytest.approx(2.9)
    assert velocity_floor_kms(["nothing_known"]) is None


def test_every_packaged_optical_line_has_a_floor() -> None:
    packaged = read_line_list(resolve_line_list("optical"))
    assert set(packaged) <= set(REST_FLOOR_KMS)


def test_band_coverage_finds_the_inert_lock_on_muse() -> None:
    gas = GasBasis(read_line_list(resolve_line_list("optical")))
    got = band_coverage(gas, MUSE, 0.0, 1.6)
    # [SIII]9531 sits past MUSE's red edge at z = 0 and only moves redder.
    assert got["[SIII]"]["inert"] is True
    assert got["[SIII]"]["members_ever_in_band"] == 1
    for name in ("[OIII]", "[NII]", "[OI]"):
        assert got[name]["inert"] is False
        assert got[name]["both_in_band_over"][0] == 0.0


def test_band_coverage_is_a_property_of_the_band_not_the_package() -> None:
    # The same lock is live on an instrument that reaches redder.
    gas = GasBasis(read_line_list(resolve_line_list("optical")))
    redder = np.arange(4750.0, 11000.0, 1.25)
    assert band_coverage(gas, redder, 0.0, 1.6)["[SIII]"]["inert"] is False


def test_band_coverage_is_silent_without_gas() -> None:
    assert band_coverage(None, MUSE, 0.0, 1.6) == {}
