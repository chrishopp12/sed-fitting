from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sedfit.core.registry import load_registry
from sedfit.core.table import (
    COLUMNS,
    read_sed_table,
    validate_sed_table,
    write_sed_table,
)

REG = load_registry()


def _frame() -> pd.DataFrame:
    rows = [
        ("CFHT_u", 10.0, 0.5, np.nan, np.nan, np.nan, np.nan),
        ("WISE_W1", 900.0, 9.0, np.nan, np.nan, np.nan, np.nan),
        ("SPHEREx_000", 400.0, 4.0, 6.0, 0.75, 0.02, 3.0),
        ("SPHEREx_001", 410.0, 4.1, 5.5, 0.77, 0.02, 2.0),
    ]
    frame = pd.DataFrame(rows, columns=list(COLUMNS[:-1]))
    frame["qa_flags"] = np.full(len(frame), np.nan, dtype=object)
    return frame


def test_round_trip(tmp_path) -> None:
    path = tmp_path / "sed.csv"
    write_sed_table(_frame(), path, REG)
    back = read_sed_table(path, REG)
    assert list(back["band"]) == list(_frame()["band"])
    assert np.isnan(back.loc[0, "wave_um"])
    assert float(back.loc[2, "wave_um"]) == 0.75
    lines = path.read_text().splitlines()
    assert lines[1].endswith(",,,,,")


def _expect(frame: pd.DataFrame, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        validate_sed_table(frame, REG, label="t")


def test_violations() -> None:
    _expect(_frame()[list(COLUMNS[::-1])], "columns")

    extra = _frame()
    extra["extra"] = 1
    _expect(extra, "columns")

    f = _frame()
    f.loc[0, "band"] = "NOPE_x"
    _expect(f, "unknown bands")

    f = _frame()
    f.loc[1, "flux_err_uJy"] = 0.0
    _expect(f, "non-positive")

    f = _frame()
    f.loc[1, "flux_uJy"] = np.nan
    _expect(f, "non-finite flux")

    f = _frame()
    f.loc[0, "wave_um"] = 0.36
    _expect(f, "broadband rows must leave wave_um")

    f = _frame()
    f.loc[2, "wave_um"] = np.nan
    _expect(f, "positive wave_um")

    f = _frame()
    f.loc[2, "n_exp"] = 1.5
    _expect(f, "integer n_exp")

    f = _frame()
    f.loc[3, "band"] = "SPHEREx_000"
    _expect(f, "duplicate bands")

    _expect(_frame().iloc[:0], "empty")


def test_qa_flags_round_trip_and_rules(tmp_path) -> None:
    f = _frame()
    f.loc[0, "qa_flags"] = "cov=1.000;maskfrac=0.031;conv=-1;refit=1.78"
    path = tmp_path / "sed.csv"
    write_sed_table(f, path, REG)
    back = read_sed_table(path, REG)
    assert back.loc[0, "qa_flags"] == "cov=1.000;maskfrac=0.031;conv=-1;refit=1.78"
    assert pd.isna(back.loc[1, "qa_flags"])
    assert pd.isna(back.loc[2, "qa_flags"])

    f = _frame()
    f.loc[2, "qa_flags"] = "cov=1.0"
    _expect(f, "SPHEREx rows must leave qa_flags")

    f = _frame()
    f.loc[0, "qa_flags"] = "not a token string"
    _expect(f, "unparseable qa_flags")

    f = _frame()
    f.loc[0, "qa_flags"] = "=1.0"
    _expect(f, "unparseable qa_flags")


def test_write_validates(tmp_path) -> None:
    f = _frame()
    f.loc[0, "flux_err_uJy"] = -1.0
    with pytest.raises(ValueError):
        write_sed_table(f, tmp_path / "x.csv", REG)
    assert not (tmp_path / "x.csv").exists()
