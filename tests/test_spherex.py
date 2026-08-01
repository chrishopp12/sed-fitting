from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sedfit.core.spherex import (
    DEFAULT_FLAGS_REJECT,
    apply_quality_cuts,
    default_split_um,
    ingest,
    load_visits,
)

FIXTURE = Path(__file__).parent / "data" / "spherex_visits.csv"
Z = 0.106


def test_default_split() -> None:
    assert default_split_um(Z) == pytest.approx(1.106 * 2.40, rel=1e-12)


def test_ingest_golden() -> None:
    spectrum, counts = ingest(FIXTURE, split_um=default_split_um(Z))
    assert counts == {"visits_total": 90, "visits_kept": 75, "channels": 41}
    assert list(spectrum["band"]) == [f"SPHEREx_{k:03d}" for k in range(41)]
    assert np.all(np.diff(spectrum["wave_um"].to_numpy()) > 0)

    first = spectrum.iloc[0]
    assert first["flux_uJy"] == pytest.approx(2412.3768, rel=1e-12)
    assert first["flux_err_uJy"] == pytest.approx(72.371304, rel=1e-12)
    assert first["scatter_uJy"] == first["flux_err_uJy"]
    assert first["wave_um"] == pytest.approx(0.752, rel=1e-12)
    assert first["n_exp"] == 1

    last = spectrum.iloc[40]
    assert last["flux_uJy"] == pytest.approx(1227.1816436214738, rel=1e-12)
    assert last["flux_err_uJy"] == pytest.approx(18.408860760783682, rel=1e-12)
    assert last["scatter_uJy"] == pytest.approx(15.760650466774242, rel=1e-12)
    assert last["wave_um"] == pytest.approx(4.962192869069101, rel=1e-12)
    assert last["bandwidth_um"] == pytest.approx(0.04, rel=1e-12)
    assert last["n_exp"] == 4


def test_flag_cut_off_keeps_all() -> None:
    spectrum, counts = ingest(FIXTURE, split_um=default_split_um(Z),
                              flags_reject=0)
    assert counts["visits_kept"] == counts["visits_total"] == 90
    assert counts["channels"] > 41


def test_strict_column_contract(tmp_path) -> None:
    visits = pd.read_csv(FIXTURE)
    for column in ("lambda", "lambda_width", "flux", "flux_err", "flags"):
        broken = tmp_path / f"missing_{column}.csv"
        visits.drop(columns=[column]).to_csv(broken, index=False)
        with pytest.raises(ValueError, match="missing required"):
            load_visits(broken)


def test_requested_cut_needs_column() -> None:
    visits = pd.read_csv(FIXTURE)
    without_ql = visits.drop(columns=["fit_ql"])
    with pytest.raises(ValueError, match="fit_ql"):
        apply_quality_cuts(without_ql, fit_ql_max=1.0)
    without_bkg = visits.drop(columns=["local_bkg_flg"])
    with pytest.raises(ValueError, match="local_bkg"):
        apply_quality_cuts(without_bkg, drop_local_bkg=True)
    apply_quality_cuts(without_ql)


def test_cuts_removing_everything_error(tmp_path) -> None:
    everything = (1 << 40) - 1
    with pytest.raises(ValueError, match="removed every visit"):
        ingest(FIXTURE, split_um=default_split_um(Z), flags_reject=everything)


def test_default_mask_value() -> None:
    assert DEFAULT_FLAGS_REJECT == ((1 << 40) - 1) & ~(1 << 21)
