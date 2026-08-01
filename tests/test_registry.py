from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sedfit.core.registry import Registry, load_registry

EXPECTED_INSTRUMENTS = {"CFHT", "GALEX", "JPLUS", "Legacy", "PS1", "SDSS",
                        "SPHEREx", "WISE"}


def test_load_packaged() -> None:
    reg = load_registry()
    assert set(reg.instruments) == EXPECTED_INSTRUMENTS
    assert len(reg.bands) == 37
    assert "CFHT_u" in reg
    assert "JPLUS_J0430" in reg


def test_instrument_identity() -> None:
    reg = load_registry()
    assert reg.instrument_of("CFHT_u") == "CFHT"
    assert reg.instrument_of("WISE_W3") == "WISE"
    assert reg.instrument_of("SPHEREx_042") == "SPHEREx"
    assert reg.is_spherex("SPHEREx_000")
    with pytest.raises(KeyError):
        reg.instrument_of("CHFT_u")
    with pytest.raises(ValueError):
        reg.require_known(["CFHT_u", "NOPE_x"])


def test_bandpass_both_sources() -> None:
    reg = load_registry()
    for band in ("WISE_W1", "JPLUS_J0430"):
        wave, thru = reg.get_bandpass(band)
        assert wave.size > 2 and wave.shape == thru.shape
        assert np.all(np.diff(wave) >= 0)
        assert (thru > 0).any()
    with pytest.raises(KeyError):
        reg.get_bandpass("SPHEREx_000")


def test_pivot_ordering() -> None:
    reg = load_registry()
    cfht = [reg.pivot_wave_AA(f"CFHT_{b}") for b in "ugriz"]
    assert cfht == sorted(cfht)
    wise = [reg.pivot_wave_AA(f"WISE_W{n}") for n in (1, 2, 3, 4)]
    assert wise == sorted(wise)
    assert reg.pivot_wave_AA("GALEX_FUV") < reg.pivot_wave_AA("GALEX_NUV")
    jplus = [reg.pivot_wave_AA(b)
             for b in ("JPLUS_J0430", "JPLUS_J0515", "JPLUS_J0660")]
    assert jplus == sorted(jplus)


def test_hashes_stable() -> None:
    per_band = load_registry().per_band_hashes()
    assert len(per_band) == 37
    assert all(len(h) == 64 for h in per_band.values())
    assert load_registry().bandpass_hash() == load_registry().bandpass_hash()


def _minimal(tmp_path: Path) -> dict:
    (tmp_path / "flat.dat").write_text("1000.0 0.5\n2000.0 0.5\n")
    return {
        "schema_version": 1,
        "instruments": ["X", "SPHEREx"],
        "spherex": {"prefix": "SPHEREx_", "instrument": "SPHEREx"},
        "bands": {"X_a": {"instrument": "X", "curve": "flat.dat"}},
    }


def _load(tmp_path: Path, registry: dict) -> Registry:
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(registry))
    return Registry(path)


def test_minimal_fixture_loads(tmp_path: Path) -> None:
    reg = _load(tmp_path, _minimal(tmp_path))
    wave, thru = reg.get_bandpass("X_a")
    assert wave.size == 2
    assert 1000.0 < reg.pivot_wave_AA("X_a") < 2000.0


def test_strict_loading(tmp_path: Path) -> None:
    bad = _minimal(tmp_path)
    bad["extra"] = 1
    with pytest.raises(ValueError, match="unknown keys"):
        _load(tmp_path, bad)

    bad = _minimal(tmp_path)
    bad["schema_version"] = 99
    with pytest.raises(ValueError, match="schema_version"):
        _load(tmp_path, bad)

    bad = _minimal(tmp_path)
    bad["bands"]["X_a"] = {"instrument": "X", "curve": "flat.dat", "sedpy": "y"}
    with pytest.raises(ValueError, match="exactly one"):
        _load(tmp_path, bad)

    bad = _minimal(tmp_path)
    bad["bands"]["X_a"] = {"instrument": "X"}
    with pytest.raises(ValueError, match="exactly one"):
        _load(tmp_path, bad)

    bad = _minimal(tmp_path)
    bad["bands"]["X_a"] = {"instrument": "NOPE", "curve": "flat.dat"}
    with pytest.raises(ValueError, match="unknown instrument"):
        _load(tmp_path, bad)

    bad = _minimal(tmp_path)
    bad["bands"]["X_a"] = {"instrument": "X", "curve": "missing.dat"}
    with pytest.raises(ValueError, match="not found"):
        _load(tmp_path, bad)

    bad = _minimal(tmp_path)
    bad["bands"]["SPHEREx_bad"] = {"instrument": "X", "curve": "flat.dat"}
    with pytest.raises(ValueError, match="prefix"):
        _load(tmp_path, bad)

    bad = _minimal(tmp_path)
    bad["bands"]["X_a"] = {"instrument": "X", "curve": "flat.dat",
                           "wave_eff_AA": 1500.0}
    with pytest.raises(ValueError, match="unknown keys"):
        _load(tmp_path, bad)


def test_acquisition_band_universe_is_ingestable() -> None:
    # Every band label the acquisition layer can emit (HST F###W
    # excluded: camera-ambiguous curves) must resolve to a loadable
    # bandpass; a measured band must never drop for lack of a registry
    # entry.
    emittable = [
        "GALEX_FUV", "GALEX_NUV",
        "SDSS_u", "SDSS_g", "SDSS_r", "SDSS_i", "SDSS_z",
        "CFHT_u", "CFHT_g", "CFHT_r", "CFHT_i", "CFHT_z",
        "Legacy_g", "Legacy_r", "Legacy_i", "Legacy_z",
        "PS1_g", "PS1_r", "PS1_i", "PS1_z", "PS1_y",
        "WISE_W1", "WISE_W2", "WISE_W3", "WISE_W4",
        "JPLUS_uJAVA", "JPLUS_J0378", "JPLUS_J0395", "JPLUS_J0410",
        "JPLUS_J0430", "JPLUS_gSDSS", "JPLUS_J0515", "JPLUS_rSDSS",
        "JPLUS_J0660", "JPLUS_iSDSS", "JPLUS_J0861", "JPLUS_zSDSS",
    ]
    reg = load_registry()
    missing = [b for b in emittable if b not in reg]
    assert missing == []
    for band in emittable:
        wave, thru = reg.get_bandpass(band)
        assert len(wave) > 10 and thru.max() > 0
