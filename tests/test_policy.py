from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sedfit.core.policy import apply_policy, resolve_floors
from sedfit.core.registry import load_registry
from sedfit.core.table import COLUMNS

REG = load_registry()


def _frame() -> pd.DataFrame:
    rows = [
        ("CFHT_u", 10.0, 0.5, np.nan, np.nan, np.nan, np.nan),
        ("CFHT_g", 1.0, 1.0, np.nan, np.nan, np.nan, np.nan),
        ("WISE_W1", -50.0, 20.0, np.nan, np.nan, np.nan, np.nan),
        ("SPHEREx_000", 0.5, 1.0, 6.0, 0.75, 0.02, 3.0),
        ("SPHEREx_001", 400.0, 4.0, 5.5, 0.77, 0.02, 2.0),
        ("SPHEREx_002", 410.0, 4.1, 5.5, 0.79, 0.02, 2.0),
    ]
    frame = pd.DataFrame(rows, columns=list(COLUMNS[:-1]))
    frame["qa_flags"] = np.full(len(frame), np.nan, dtype=object)
    return frame


def test_gates_and_floor() -> None:
    result = apply_policy(_frame(), registry=REG, min_valid_bands=4)
    assert result.n_fit == 4
    assert set(result.low_snr) == {"CFHT_g", "WISE_W1"}
    assert result.excluded_by_set == ()

    by_band = dict(zip(result.bands, result.flux_err_uJy))
    assert by_band["CFHT_u"] == pytest.approx(
        np.sqrt(0.5**2 + (0.05 * 10.0)**2), rel=1e-12)
    assert by_band["WISE_W1"] == pytest.approx(20.0, rel=1e-12)
    assert by_band["SPHEREx_000"] == pytest.approx(
        np.sqrt(1.0**2 + (0.05 * 0.5)**2), rel=1e-12)


def test_default_min_valid_bands_gate() -> None:
    with pytest.raises(ValueError, match="fittable bands"):
        apply_policy(_frame(), registry=REG)


def test_snr_gate_disable_and_exemption() -> None:
    result = apply_policy(_frame(), registry=REG, min_snr_broadband=0.0,
                          min_valid_bands=4)
    assert result.low_snr == ()
    assert result.n_fit == 6

    marked = apply_policy(_frame(), registry=REG, min_valid_bands=1,
                          bands_include=["SPHEREx"])
    assert marked.low_snr == ()
    assert marked.n_fit == 3


def test_mu_applied_once() -> None:
    mu = 2.0
    result = apply_policy(_frame(), registry=REG, min_valid_bands=4,
                          mu_lensing=mu)
    raw = _frame()
    flux = raw["flux_uJy"].to_numpy(float) / mu
    err_stat = raw["flux_err_uJy"].to_numpy(float) / mu
    expected = np.sqrt(err_stat**2 + (0.05 * np.maximum(flux, 0.0))**2)
    assert result.flux_uJy == pytest.approx(flux, rel=1e-15)
    assert result.flux_err_uJy == pytest.approx(expected, rel=1e-15)


def test_bands_include() -> None:
    result = apply_policy(_frame(), registry=REG, min_valid_bands=1,
                          bands_include=["CFHT"])
    assert set(result.excluded_by_set) == {"WISE_W1", "SPHEREx_000",
                                           "SPHEREx_001", "SPHEREx_002"}
    assert result.n_fit == 1

    explicit = apply_policy(_frame(), registry=REG, min_valid_bands=1,
                            bands_include=["CFHT_u", "SPHEREx_001"])
    assert explicit.n_fit == 2

    with pytest.raises(ValueError, match="matches no band"):
        apply_policy(_frame(), registry=REG, min_valid_bands=1,
                     bands_include=["GALEX"])
    with pytest.raises(ValueError, match="neither a registry instrument"):
        apply_policy(_frame(), registry=REG, min_valid_bands=1,
                     bands_include=["CHFT"])


def test_floor_map() -> None:
    result = apply_policy(_frame(), registry=REG, min_valid_bands=4,
                          err_floor={"CFHT": 0.10, "default": 0.02})
    assert result.floor_by_band["CFHT_u"] == 0.10
    assert result.floor_by_band["SPHEREx_001"] == 0.02

    with pytest.raises(ValueError, match="unrecognized keys"):
        apply_policy(_frame(), registry=REG, min_valid_bands=4,
                     err_floor={"CHFT": 0.08, "default": 0.05})

    with pytest.raises(ValueError, match="no applicable floor"):
        apply_policy(_frame(), registry=REG, min_valid_bands=4,
                     err_floor={"CFHT": 0.08})

    scoped = apply_policy(_frame(), registry=REG, min_valid_bands=1,
                          bands_include=["CFHT"], err_floor={"CFHT": 0.08})
    assert scoped.n_fit == 1

    warned = apply_policy(_frame(), registry=REG, min_valid_bands=4,
                          err_floor={"GALEX": 0.10, "default": 0.05})
    assert any("GALEX" in w for w in warned.warnings)


def test_accounting_shape() -> None:
    result = apply_policy(_frame(), registry=REG, min_valid_bands=4)
    acct = result.accounting()
    assert acct["n_bands_table"] == 6
    assert acct["n_bands_fit"] == 4
    assert acct["n_low_snr"] == 2
    assert "CFHT_u" in acct["included"]


def _qa_frame() -> pd.DataFrame:
    frame = _frame()
    frame.loc[0, "qa_flags"] = "cov=1.000;maskfrac=0.031;conv=37;bg=-0.0033"
    frame.loc[1, "qa_flags"] = "cov=0.950;maskfrac=0.600;conv=-1;atbound=2"
    return frame


def test_qa_gate_max_min_and_sentinel() -> None:
    result = apply_policy(_qa_frame(), registry=REG, min_valid_bands=3,
                          min_snr_broadband=0,
                          qa_gates={"maskfrac": {"max": 0.5}})
    assert result.qa_rejected == ("CFHT_g",)
    assert "CFHT_g" not in result.accounting()["included"]

    result = apply_policy(_qa_frame(), registry=REG, min_valid_bands=3,
                          min_snr_broadband=0, qa_gates={"conv": {"min": 0}})
    assert result.qa_rejected == ("CFHT_g",)


def test_qa_gate_absent_token_passes_and_missing_warns() -> None:
    result = apply_policy(_qa_frame(), registry=REG, min_valid_bands=3,
                          min_snr_broadband=0,
                          qa_gates={"atbound": {"max": 0}})
    assert result.qa_rejected == ("CFHT_g",)

    result = apply_policy(_qa_frame(), registry=REG, min_valid_bands=3,
                          min_snr_broadband=0,
                          qa_gates={"mesh": {"max": 10.0}})
    assert result.qa_rejected == ()
    assert any("mesh" in w for w in result.warnings)


def test_qa_gate_vocabulary_and_shape() -> None:
    frame = _qa_frame()
    with pytest.raises(ValueError, match="unrecognized tokens"):
        apply_policy(frame, registry=REG, qa_gates={"CHFT": {"max": 1}})
    with pytest.raises(ValueError, match="'min'"):
        apply_policy(frame, registry=REG, qa_gates={"cov": {}})
    with pytest.raises(ValueError, match="unrecognized keys"):
        apply_policy(frame, registry=REG, qa_gates={"cov": {"lo": 1}})
    with pytest.raises(ValueError, match="must be a number"):
        apply_policy(frame, registry=REG, qa_gates={"cov": {"min": True}})
    with pytest.raises(ValueError, match="non-empty"):
        apply_policy(frame, registry=REG, qa_gates={})


def test_qa_accounting() -> None:
    result = apply_policy(_qa_frame(), registry=REG, min_valid_bands=3,
                          min_snr_broadband=0,
                          qa_gates={"maskfrac": {"max": 0.5}})
    acc = result.accounting()
    assert acc["n_qa_rejected"] == 1
    assert acc["qa_rejected"] == ["CFHT_g"]
