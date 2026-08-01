from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sedfit.core.sources import (
    SOURCE_PREFIXES,
    check_position,
    read_source_table,
    select_rows,
    source_prefix,
)

RA, DEC = 216.988087, 56.987586


def _catalog() -> pd.DataFrame:
    rows = [
        ("Legacy_g", 120.0, 1.2, "Legacy_DR9"),
        ("Legacy_r", 250.0, 2.0, "Legacy_DR9"),
        ("WISE_W1", 900.0, 9.0, "unWISE_Legacy_DR9"),
        ("WISE_W1", 780.0, 30.0, "AllWISE"),
        ("WISE_W3", -50.0, np.nan, "AllWISE"),
        ("GALEX_NUV", 5.0, 0.8, "GALEX_GUVcat_AIS"),
    ]
    frame = pd.DataFrame(rows, columns=["band", "flux_uJy", "flux_err_uJy",
                                        "source"])
    frame["mag_AB"] = 20.0
    frame["target_ra"] = RA
    frame["target_dec"] = DEC
    return frame


def test_prefixes_frozen() -> None:
    assert SOURCE_PREFIXES == {
        "legacy": "Legacy_",
        "unwise": "unWISE_Legacy_",
        "allwise": "AllWISE",
        "galex": "GALEX_",
        "sdss": "SDSS_",
        "panstarrs": "PanSTARRS_",
        "jplus": "JPLUS_",
        "hst": "HST_HAP_",
        "aperture": "sedphot_aperture_",
        "sersic": "sedphot_sersic_",
    }


def test_prefixes_prefix_free() -> None:
    prefixes = list(SOURCE_PREFIXES.values())
    for a in prefixes:
        for b in prefixes:
            assert a == b or not b.startswith(a)


def test_source_prefix_tokens() -> None:
    assert source_prefix("legacy: DR9 North Tractor via Datalab TAP") == "Legacy_"
    assert source_prefix("unwise") == "unWISE_Legacy_"
    with pytest.raises(ValueError, match="unknown provider token"):
        source_prefix("legacy_tractor: retired vocabulary")


def test_read_source_table(tmp_path) -> None:
    path = tmp_path / "catalog.csv"
    _catalog().to_csv(path, index=False)
    frame = read_source_table(path)
    assert "mag_AB" in frame.columns

    bad = tmp_path / "bad.csv"
    _catalog().drop(columns=["source"]).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="missing required"):
        read_source_table(bad)


def test_select_disambiguates_wise() -> None:
    frame = _catalog()
    unwise = select_rows(frame, ["WISE_W1"], provider="unwise", label="t/wise")
    assert float(unwise["flux_uJy"].iloc[0]) == 900.0
    allwise = select_rows(frame, ["WISE_W1"], provider="allwise", label="t/wise")
    assert float(allwise["flux_uJy"].iloc[0]) == 780.0


def test_select_order_and_errors() -> None:
    frame = _catalog()
    rows = select_rows(frame, ["Legacy_r", "Legacy_g"], provider="legacy",
                       label="t")
    assert list(rows["band"]) == ["Legacy_r", "Legacy_g"]

    with pytest.raises(ValueError, match="matches no row"):
        select_rows(frame, ["Legacy_z"], provider="legacy", label="t")
    with pytest.raises(ValueError, match="duplicate bands"):
        select_rows(frame, ["Legacy_g", "Legacy_g"], provider="legacy",
                    label="t")
    doubled = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="ambiguous"):
        select_rows(doubled, ["Legacy_g"], provider="legacy", label="t")


def test_consumed_rows_validated() -> None:
    frame = _catalog()
    with pytest.raises(ValueError, match="non-positive error"):
        select_rows(frame, ["WISE_W3"], provider="allwise", label="t")
    select_rows(frame, ["WISE_W1"], provider="allwise", label="t")


def test_check_position() -> None:
    frame = _catalog()
    check_position(frame, RA, DEC, label="t")

    with pytest.raises(ValueError, match="arcsec"):
        check_position(frame, RA, DEC + 2.0 / 3600.0, label="t")

    # A table with no position columns is still exempt -- v1 catalog
    # tables carry none -- but the exemption announces itself, so the
    # shrinking set that goes unchecked stays visible.
    exempt = frame.drop(columns=["target_ra", "target_dec"])
    with pytest.warns(UserWarning, match="cross-check is SKIPPED"):
        check_position(exempt, RA, DEC, label="t")

    partial = frame.copy()
    partial["target_ra"] = np.nan
    check_position(partial, RA, DEC, label="t")

    garbage = frame.copy()
    garbage["target_ra"] = "oops"
    with pytest.raises((TypeError, ValueError)):
        check_position(garbage, RA, DEC, label="t")
