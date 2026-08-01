from __future__ import annotations

import numpy as np
import pytest

from sedfit.core.registry import load_registry
from sedfit.core.stitch import measure_scale

REG = load_registry()


def _flat_spec(lo_um: float, hi_um: float, level: float = 100.0):
    wave = np.linspace(lo_um, hi_um, 2000)
    return wave, np.full_like(wave, level)


def test_scale_on_flat_spectrum() -> None:
    wave, fnu = _flat_spec(0.6, 6.0)
    scale, info = measure_scale(
        ["WISE_W1", "WISE_W2"], [50.0, 50.0],
        spec_wave_um=wave, spec_fnu_uJy=fnu, registry=REG)
    assert scale == pytest.approx(2.0, rel=1e-10)
    assert info["anchor"] == "WISE_W2"
    for row in info["bands"]:
        assert row["coverage"] == pytest.approx(1.0, rel=1e-12)
        assert row["synth"] == pytest.approx(100.0, rel=1e-10)
        assert row["ratio"] == pytest.approx(2.0, rel=1e-10)


def test_anchor_respects_coverage_gate() -> None:
    wave, fnu = _flat_spec(0.6, 4.5)
    scale, info = measure_scale(
        ["WISE_W1", "WISE_W2"], [50.0, 50.0],
        spec_wave_um=wave, spec_fnu_uJy=fnu, registry=REG)
    assert info["anchor"] == "WISE_W1"
    assert scale == pytest.approx(2.0, rel=1e-10)
    w2 = next(r for r in info["bands"] if r["band"] == "WISE_W2")
    assert w2["coverage"] < 0.98


def test_no_overlap_floats() -> None:
    wave, fnu = _flat_spec(2.0, 5.0)
    scale, info = measure_scale(
        ["GALEX_NUV"], [10.0],
        spec_wave_um=wave, spec_fnu_uJy=fnu, registry=REG)
    assert scale is None
    assert info["anchor"] is None
    assert info["bands"][0]["coverage"] == 0.0


def test_parallel_input_lengths() -> None:
    wave, fnu = _flat_spec(0.6, 6.0)
    with pytest.raises(ValueError):
        measure_scale(["WISE_W1", "WISE_W2"], [50.0],
                      spec_wave_um=wave, spec_fnu_uJy=fnu, registry=REG)
