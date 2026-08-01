from __future__ import annotations

import numpy as np
import pytest

from sedfit.core.synth import make_spherex_tophat, pivot_wave_AA, synth_fnu


def test_tophat_shape() -> None:
    wave, thru = make_spherex_tophat(1.234, 0.025)
    assert wave.size == thru.size == 27
    assert thru[0] == thru[-1] == 0.0
    assert np.all(thru[1:-1] == 1.0)
    center, width = 1.234e4, 0.025e4
    lo, hi = center - width / 2.0, center + width / 2.0
    pad = max(width * 1e-3, 1e-3)
    assert wave[0] == pytest.approx(lo - pad, rel=1e-12)
    assert wave[1] == pytest.approx(lo, rel=1e-12)
    assert wave[-2] == pytest.approx(hi, rel=1e-12)
    assert wave[-1] == pytest.approx(hi + pad, rel=1e-12)
    assert np.all(np.diff(wave) > 0)


def test_synth_flat_spectrum() -> None:
    wave, thru = make_spherex_tophat(1.0, 0.02)
    spec_wave = np.linspace(0.9, 1.1, 200)
    spec_fnu = np.full_like(spec_wave, 5.0)
    fnu, coverage = synth_fnu(wave, thru, spec_wave, spec_fnu)
    assert fnu == pytest.approx(5.0, rel=1e-12)
    assert coverage == pytest.approx(1.0, rel=1e-12)


def test_synth_uncovered() -> None:
    wave, thru = make_spherex_tophat(3.0, 0.02)
    spec_wave = np.linspace(0.9, 1.1, 50)
    fnu, coverage = synth_fnu(wave, thru, spec_wave, np.ones(50))
    assert np.isnan(fnu)
    assert coverage == 0.0


def test_synth_partial_coverage() -> None:
    wave, thru = make_spherex_tophat(1.0, 0.10)
    spec_wave = np.linspace(0.90, 1.0, 100)
    fnu, coverage = synth_fnu(wave, thru, spec_wave, np.full(100, 2.0))
    assert np.isfinite(fnu)
    assert 0.0 < coverage < 1.0


def test_pivot_of_symmetric_tophat() -> None:
    wave, thru = make_spherex_tophat(2.0, 0.02)
    assert pivot_wave_AA(wave, thru) == pytest.approx(2.0e4, rel=1e-4)
