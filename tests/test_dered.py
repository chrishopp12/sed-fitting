from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sedfit.core import dered
from sedfit.core.registry import load_registry

REG = load_registry()


def test_klambda_normalization_is_rv():
    # At x = 1.82 /micron the O'Donnell polynomials give a=1, b=0, so
    # A_lambda / E(B-V) is exactly R_V.
    assert dered.k_lambda(1.0 / 1.82)[0] == pytest.approx(dered.RV_MW, abs=1e-6)


def test_klambda_decreases_to_the_red():
    k = dered.k_lambda([0.44, 0.55, 0.90, 2.0])
    assert np.all(np.diff(k) < 0)


def test_band_factor_zero_ebv_is_identity():
    wave, thru = REG.get_bandpass("JPLUS_gSDSS")
    factor, a_band = dered.band_factor(wave, thru, 0.0)
    assert factor == pytest.approx(1.0)
    assert a_band == pytest.approx(0.0)


def test_band_factor_bluer_band_reddened_more():
    ebv = 0.05
    a = {}
    for band in ("JPLUS_J0430", "JPLUS_gSDSS", "JPLUS_rSDSS", "JPLUS_zSDSS"):
        wave, thru = REG.get_bandpass(band)
        factor, a[band] = dered.band_factor(wave, thru, ebv)
        assert factor > 1.0
    assert a["JPLUS_J0430"] > a["JPLUS_gSDSS"] > a["JPLUS_rSDSS"] > a["JPLUS_zSDSS"]


def test_deredden_scales_flux_and_error_and_records_meta():
    selected = {"jplus": pd.DataFrame({
        "band": ["JPLUS_gSDSS", "JPLUS_zSDSS"],
        "flux_uJy": [100.0, 100.0], "flux_err_uJy": [1.0, 2.0],
        "source": ["J-PLUS_DR3", "J-PLUS_DR3"]})}
    spectrum = pd.DataFrame({
        "band": ["SPHEREx_000", "SPHEREx_001"],
        "flux_uJy": [10.0, 10.0], "flux_err_uJy": [0.1, 0.1],
        "scatter_uJy": [0.2, 0.2], "wave_um": [0.75, 4.0],
        "bandwidth_um": [0.01, 0.01], "n_exp": [1.0, 1.0]})

    meta = dered.deredden(selected, spectrum, 0.05, REG)

    g, z = selected["jplus"]["flux_uJy"]
    assert g > z > 100.0                                   # bluer scaled more
    assert selected["jplus"]["flux_err_uJy"][0] == pytest.approx(
        1.0 * g / 100.0)                                   # error tracks flux
    assert spectrum["flux_uJy"][0] > spectrum["flux_uJy"][1] > 10.0
    assert spectrum["scatter_uJy"][0] > 0.2                # scatter scaled too
    assert meta["ebv"] == 0.05 and meta["law"] == dered.LAW
    assert set(meta["broadband_A_mag"]) == {"JPLUS_gSDSS", "JPLUS_zSDSS"}


def test_deredden_zero_ebv_is_noop():
    selected = {"legacy": pd.DataFrame({
        "band": ["Legacy_r"], "flux_uJy": [50.0], "flux_err_uJy": [0.5],
        "source": ["Legacy_DR9"]})}
    dered.deredden(selected, None, 0.0, REG)
    assert selected["legacy"]["flux_uJy"][0] == pytest.approx(50.0)
