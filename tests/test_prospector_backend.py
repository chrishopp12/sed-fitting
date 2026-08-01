from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("prospect")

from sedfit.backends.prospector.exact_filter import ExactFilter
from sedfit.backends.prospector.model import (
    assert_stellar_library,
    attach_norm_masks,
    build_model,
    build_sps,
)
from sedfit.backends.prospector.obs import UJY_PER_MAGGIE, build_obs
from sedfit.core.fitconfig import parse_fit_config, resolve_config
from sedfit.core.policy import apply_policy
from sedfit.core.registry import load_registry
from sedfit.core.synth import make_spherex_tophat
from sedfit.core.table import COLUMNS

REG = load_registry()
MU = 2.0
FLOOR = 0.05


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


def _cfg(**overrides) -> dict:
    prospector = {"stellar_library": "miles",
                  "zred": {"prior": "normal", "sigma": 0.004,
                           "bounds": [0.05, 0.16]}}
    prospector.update(overrides)
    raw = {"schema_version": 2, "backend": "prospector", "name": "t",
           "mu_lensing": MU, "err_floor": FLOOR,
           "prospector": prospector}
    cfg = parse_fit_config(raw)
    return resolve_config(cfg, target_z_ref=0.106, reference_redshift=0.106,
                          seed_source=lambda: 42)


def _policy(frame, **overrides):
    kwargs = dict(registry=REG, min_valid_bands=4, min_snr_broadband=0.0,
                  err_floor=FLOOR, mu_lensing=MU)
    kwargs.update(overrides)
    return apply_policy(frame, **kwargs)


def test_mu_applied_exactly_once() -> None:
    frame = _frame()
    obs = build_obs(_cfg(), frame, _policy(frame), registry=REG)
    expected = frame["flux_uJy"].to_numpy(float) / MU / UJY_PER_MAGGIE
    assert np.array_equal(obs["maggies"], expected)


def test_error_vector_byte_for_byte() -> None:
    frame = _frame()
    obs = build_obs(_cfg(), frame, _policy(frame), registry=REG)
    flux = frame["flux_uJy"].to_numpy(float) / MU
    err_stat = frame["flux_err_uJy"].to_numpy(float) / MU
    expected = (np.sqrt(err_stat**2 + (FLOOR * np.maximum(flux, 0.0))**2)
                / UJY_PER_MAGGIE)
    assert np.array_equal(obs["maggies_unc"], expected)
    # negative flux: the clamped floor leaves the statistical error alone
    w1 = list(frame["band"]).index("WISE_W1")
    assert obs["maggies_unc"][w1] == (20.0 / MU) / UJY_PER_MAGGIE


def test_excluded_bands_omitted() -> None:
    frame = _frame()
    policy = _policy(frame, bands_include=["CFHT"], min_valid_bands=1)
    obs = build_obs(_cfg(), frame, policy, registry=REG)
    assert [f.name for f in obs["filters"]] == ["CFHT_u", "CFHT_g"]
    assert len(obs["maggies"]) == 2
    assert all(isinstance(f, ExactFilter) for f in obs["filters"])
    assert obs["redshift"] == pytest.approx(0.106)


def test_spherex_tophats_from_core_synth() -> None:
    frame = _frame()
    obs = build_obs(_cfg(), frame, _policy(frame), registry=REG)
    tophat = next(f for f in obs["filters"] if f.name == "SPHEREx_000")
    wave, thru = make_spherex_tophat(0.75, 0.02)
    assert tophat.wavelength.size == wave.size == 27
    assert tophat.wavelength[0] == pytest.approx(wave[0], rel=1e-12)
    assert tophat.wavelength[-1] == pytest.approx(wave[-1], rel=1e-12)


def test_model_continuity() -> None:
    model = build_model(_cfg())
    free = set(model.free_params)
    assert {"zred", "logmass", "logsfr_ratios", "dust2", "logzsol"} <= free
    assert model.config_dict["agebins"]["depends_on"] is not None
    prior = model.config_dict["zred"]["prior"]
    assert type(prior).__name__ == "ClippedNormal"
    assert float(prior.params["mean"]) == pytest.approx(0.106)


def test_model_parametric_variants() -> None:
    dynesty = build_model(_cfg(sfh="delayed-tau"))
    assert {"mass", "tau"} <= set(dynesty.free_params)

    emcee = build_model(_cfg(sfh="delayed-tau", sampler="emcee",
                             dynesty=None, emcee={}))
    free = set(emcee.free_params)
    assert {"logmass", "logtau"} <= free
    assert "mass" not in free
    assert emcee.config_dict["mass"]["depends_on"] is not None


def test_fixed_redshift() -> None:
    model = build_model(_cfg(fit_redshift=False, zred=None))
    assert "zred" not in set(model.free_params)
    assert float(np.atleast_1d(model.config_dict["zred"]["init"])[0]) \
        == pytest.approx(0.106)


def test_free_norm_binding() -> None:
    frame = _frame()
    cfg = _cfg(free_norm_instruments=["WISE"])
    model = build_model(cfg)
    assert "WISE_norm" in set(model.free_params)

    obs = build_obs(cfg, frame, _policy(frame), registry=REG)
    attach_norm_masks(model, obs, cfg, REG)
    mask = model.norm_groups["WISE_norm"]
    assert mask.sum() == 1

    galex = _cfg(free_norm_instruments=["GALEX"])
    galex_model = build_model(galex)
    with pytest.raises(ValueError, match="matches no band"):
        attach_norm_masks(galex_model, obs, galex, REG)


def test_sp_prior_uniform_shape() -> None:
    model = build_model(_cfg(dust2_prior_type="uniform",
                             dust2_prior=[0.0, 0.4],
                             logzsol_prior_type="uniform",
                             logzsol_prior=[-1.0, 0.5],
                             mass_range=[1.0e10, 1.0e12]))
    dust2 = model.config_dict["dust2"]["prior"]
    assert type(dust2).__name__ == "TopHat"
    assert float(dust2.params["maxi"]) == pytest.approx(0.4)
    logzsol = model.config_dict["logzsol"]["prior"]
    assert type(logzsol).__name__ == "TopHat"
    logmass = model.config_dict["logmass"]["prior"]
    assert float(logmass.params["maxi"]) == pytest.approx(12.0)


def test_model_picklable() -> None:
    # dynesty checkpointing pickles the sampler, and the model with it
    import pickle

    model = build_model(_cfg())
    rebuilt = pickle.loads(pickle.dumps(model))
    assert set(rebuilt.free_params) == set(model.free_params)


def test_stellar_library_assert() -> None:
    sps = build_sps(_cfg())
    assert type(sps).__name__ == "FastStepBasis"
    assert_stellar_library(sps, "miles")
    with pytest.raises(RuntimeError, match="stellar_library mismatch"):
        assert_stellar_library(sps, "c3k")
