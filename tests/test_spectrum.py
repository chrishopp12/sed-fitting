from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from sedfit.core.fitconfig import (
    hash_projection,
    load_fit_config,
    parse_fit_config,
    resolve_config,
)
from sedfit.core.provenance import canonical_json, run_id
from sedfit.core.spectrum import (
    SPECTRUM_COLUMNS,
    apply_spectrum_policy,
    read_spectrum,
    n_air,
    sidecar_path,
    vac_to_air,
)

N_CHANNELS = 40

# bcg_template_fit.py's n_air, transcribed verbatim from the RM J0019 MUSE
# workspace. The linear backend's air Chebyshev is bit-identical to that
# reference only while the two relations agree, so the reference is the
# fixture rather than a rounded table of expected values.
_REFERENCE_N_AIR_TERMS = (8.34254e-5, 2.406147e-2, 130.0, 1.5998e-4, 38.9)


def _reference_n_air(lam_vac):
    a, b1, c1, b2, c2 = _REFERENCE_N_AIR_TERMS
    s2 = (1e4 / np.asarray(lam_vac, float)) ** 2
    return 1.0 + a + b1 / (c1 - s2) + b2 / (c2 - s2)


def test_n_air_matches_the_reference_exactly() -> None:
    lam = np.linspace(4750.0, 9350.2, 4601)
    assert np.abs(n_air(lam) - _reference_n_air(lam)).max() == 0.0


def test_vac_to_air_offset() -> None:
    """The offset the Chebyshev coordinate sees, and its known landmark."""
    lam = np.array([4750.0, 9000.0, 9155.0, 9350.2])
    offset = lam - vac_to_air(lam)
    assert offset[0] == pytest.approx(1.328, abs=1e-3)
    assert offset[1] == pytest.approx(2.470, abs=1e-3)
    assert offset[3] == pytest.approx(2.565, abs=1e-3)
    # the +82 km/s at 9155 A the MUSE work quotes as its rule of thumb
    assert (offset[2] / lam[2] * 299792.458) == pytest.approx(82.26, abs=0.01)
    # strictly increasing with wavelength, and air is always the shorter
    assert (offset > 0).all() and (np.diff(offset) > 0).all()


def _write_spectrum(tmp_path, *, name="spec.csv", sidecar=None, **columns):
    wave = np.linspace(9000.0, 9390.0, N_CHANNELS)
    frame = pd.DataFrame({
        "wave_A": columns.get("wave_A", wave),
        "flux_uJy": columns.get("flux_uJy", np.full(N_CHANNELS, 5.0)),
        "flux_err_uJy": columns.get("flux_err_uJy",
                                    np.full(N_CHANNELS, 0.5)),
        "mask": columns.get("mask", np.ones(N_CHANNELS, dtype=int)),
    })
    path = tmp_path / name
    frame.to_csv(path, index=False)
    meta = {"wave_frame": "vacuum", "flux_unit": "uJy"}
    meta.update(sidecar or {})
    sidecar_path(path).write_text(json.dumps(meta), encoding="utf-8")
    return path


def _prospector_raw(spectrum=None, **top):
    prospector = {"stellar_library": "miles", "fit_redshift": False}
    if spectrum is not None:
        prospector["spectrum"] = spectrum
    raw = {"schema_version": 2, "backend": "prospector", "name": "t",
           "prospector": prospector}
    raw.update(top)
    return raw


# ------------------------------------
# Reading
# ------------------------------------

def test_read_round_trip(tmp_path) -> None:
    path = _write_spectrum(tmp_path, sidecar={"prepared_by": "test"})
    spectrum = read_spectrum(path)
    assert tuple(spectrum.wave_A[:2]) == pytest.approx((9000.0, 9010.0))
    assert spectrum.mask.dtype == bool and spectrum.mask.all()
    assert spectrum.provenance["prepared_by"] == "test"
    assert spectrum.sha256 == read_spectrum(path).sha256
    assert spectrum.csv_bytes == path.read_bytes()


def test_read_rejects_contract_violations(tmp_path) -> None:
    bad_cols = tmp_path / "cols.csv"
    pd.DataFrame({"wave_A": [1.0], "flux_uJy": [1.0]}).to_csv(bad_cols,
                                                              index=False)
    sidecar_path(bad_cols).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="columns"):
        read_spectrum(bad_cols)

    wave = np.linspace(9000.0, 9390.0, N_CHANNELS)
    wave[5] = wave[4]
    with pytest.raises(ValueError, match="strictly increasing"):
        read_spectrum(_write_spectrum(tmp_path, name="wave.csv",
                                      wave_A=wave))

    with pytest.raises(ValueError, match="mask must be 0 or 1"):
        read_spectrum(_write_spectrum(tmp_path, name="mask.csv",
                                      mask=np.full(N_CHANNELS, 2)))

    with pytest.raises(ValueError, match="every channel is masked"):
        read_spectrum(_write_spectrum(tmp_path, name="all.csv",
                                      mask=np.zeros(N_CHANNELS, dtype=int)))

    err = np.full(N_CHANNELS, 0.5)
    err[3] = 0.0
    with pytest.raises(ValueError, match="non-positive flux_err"):
        read_spectrum(_write_spectrum(tmp_path, name="err.csv",
                                      flux_err_uJy=err))


def test_masked_channels_may_be_non_finite(tmp_path) -> None:
    flux = np.full(N_CHANNELS, 5.0)
    mask = np.ones(N_CHANNELS, dtype=int)
    flux[7] = np.nan
    mask[7] = 0
    spectrum = read_spectrum(_write_spectrum(tmp_path, flux_uJy=flux,
                                             mask=mask))
    assert not spectrum.mask[7]


def test_sidecar_required_and_checked(tmp_path) -> None:
    path = _write_spectrum(tmp_path, name="naked.csv")
    sidecar_path(path).unlink()
    with pytest.raises(FileNotFoundError, match="sidecar"):
        read_spectrum(path)

    air = _write_spectrum(tmp_path, name="air.csv",
                          sidecar={"wave_frame": "air"})
    with pytest.raises(ValueError, match="wave_frame"):
        read_spectrum(air)

    cgs = _write_spectrum(tmp_path, name="cgs.csv",
                          sidecar={"flux_unit": "erg/s/cm2/A"})
    with pytest.raises(ValueError, match="flux_unit"):
        read_spectrum(cgs)


# ------------------------------------
# Policy
# ------------------------------------

def test_policy_transforms(tmp_path) -> None:
    spectrum = read_spectrum(_write_spectrum(tmp_path))
    flux, err, mask = apply_spectrum_policy(spectrum, mu_lensing=2.0,
                                            err_floor=0.1)
    assert flux[0] == pytest.approx(2.5)
    assert err[0] == pytest.approx(np.hypot(0.25, 0.25))
    assert mask.all()

    _, _, windowed = apply_spectrum_policy(
        spectrum, mu_lensing=1.0, mask_windows=[[9000.0, 9095.0]])
    cut = spectrum.wave_A <= 9095.0
    assert not windowed[cut].any() and windowed[~cut].all()

    with pytest.raises(ValueError, match="no unmasked channels"):
        apply_spectrum_policy(spectrum,
                              mask_windows=[[8000.0, 10000.0]])


# ------------------------------------
# Config
# ------------------------------------

def test_spectrum_block_defaults_and_errors(tmp_path) -> None:
    cfg = parse_fit_config(_prospector_raw({"file": "s.csv"}))
    block = cfg["prospector"]["spectrum"]
    assert block["polyorder"] == 12
    assert block["smooth_sigma_prior"] == [10.0, 400.0]
    assert block["jitter_prior"] is None
    assert block["outlier_prior"] is None
    assert block["eline_sigma_kms"] is None

    with pytest.raises(ValueError, match="eline_sigma_kms"):
        parse_fit_config(_prospector_raw({"file": "s.csv",
                                          "eline_sigma_kms": -5.0}))

    with pytest.raises(ValueError, match="missing required keys"):
        parse_fit_config(_prospector_raw({}))
    with pytest.raises(ValueError, match="polyorder"):
        parse_fit_config(_prospector_raw({"file": "s.csv",
                                          "polyorder": -1}))
    with pytest.raises(ValueError, match="meaningless"):
        parse_fit_config(_prospector_raw(
            {"file": "s.csv", "smooth_sigma_fixed": 60.0,
             "smooth_sigma_prior": [10.0, 100.0]}))
    with pytest.raises(ValueError, match="jitter_init"):
        parse_fit_config(_prospector_raw(
            {"file": "s.csv", "jitter_prior": [0.5, 2.0],
             "jitter_init": 3.0}))
    with pytest.raises(ValueError, match="outlier_prior"):
        parse_fit_config(_prospector_raw(
            {"file": "s.csv", "outlier_prior": [0.0, 1.5]}))
    with pytest.raises(ValueError, match="outlier_nsigma"):
        parse_fit_config(_prospector_raw(
            {"file": "s.csv", "outlier_prior": None,
             "outlier_nsigma": 50.0}))
    with pytest.raises(ValueError, match="mask_windows"):
        parse_fit_config(_prospector_raw({"file": "s.csv",
                                          "mask_windows": []}))

    jitter = parse_fit_config(_prospector_raw(
        {"file": "s.csv", "jitter_prior": [0.5, 3.0]}))
    assert jitter["prospector"]["spectrum"]["jitter_init"] == 1.0
    outlier = parse_fit_config(_prospector_raw(
        {"file": "s.csv", "outlier_prior": [1.0e-5, 0.1]}))
    assert outlier["prospector"]["spectrum"]["outlier_nsigma"] == 50.0


def test_relative_file_resolves_against_config_dir(tmp_path) -> None:
    config_path = tmp_path / "configs" / "joint.json"
    config_path.parent.mkdir()
    config_path.write_text(json.dumps(_prospector_raw(
        {"file": "../east_spectrum.csv"})), encoding="utf-8")
    cfg = load_fit_config(config_path)
    resolved_file = cfg["prospector"]["spectrum"]["file"]
    assert resolved_file == str((tmp_path / "east_spectrum.csv").resolve())


# ------------------------------------
# Identity
# ------------------------------------

def test_null_spectrum_preserves_identity() -> None:
    resolved = resolve_config(parse_fit_config(_prospector_raw()),
                              target_z_ref=0.106,
                              seed_source=lambda: 42)
    assert resolved["prospector"]["spectrum"] is None
    projected = canonical_json(hash_projection(resolved))
    assert "spectrum" not in projected


def test_spectrum_path_is_execution_only() -> None:
    def resolved_for(path):
        cfg = parse_fit_config(_prospector_raw(
            {"file": path, "jitter_prior": [0.5, 3.0]}))
        return resolve_config(cfg, target_z_ref=0.106,
                              seed_source=lambda: 42)

    digests = {"spectrum": "a" * 64}
    here = hash_projection(resolved_for("/data/a/spec.csv"),
                           digests=digests)
    there = hash_projection(resolved_for("/data/b/spec.csv"),
                            digests=digests)
    assert "file" not in json.dumps(here)
    assert (run_id(here, "p" * 64, "q" * 64)
            == run_id(there, "p" * 64, "q" * 64))

    other = hash_projection(resolved_for("/data/a/spec.csv"),
                            digests={"spectrum": "b" * 64})
    assert (run_id(here, "p" * 64, "q" * 64)
            != run_id(other, "p" * 64, "q" * 64))
