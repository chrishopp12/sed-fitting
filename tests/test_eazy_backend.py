from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from synthetic import (BANDS, PACKAGED_TEMPLATES as _PACKAGED, REG, Z_TRUE,
                       fake_templates as _fake_templates,
                       synth_frame as _synth_frame)

pytest.importorskip("eazy")

from sedfit.backends.eazy.fitting import run_official
from sedfit.backends.eazy.quick import run_quick
from sedfit.backends.eazy.results import load_run
from sedfit.backends.eazy.templates import (content_digests, template_summary,
                                            write_templates_param)
from sedfit.core.fitconfig import parse_fit_config, resolve_config
from sedfit.core.policy import MISSING_SENTINEL, apply_policy


def _config(tmp_path, **eazy_overrides) -> dict:
    eazy = {"engine": "eazy-py", "z_min": 0.05, "z_max": 0.15,
            "z_step": 0.005, "z_step_type": "linear", "tef": False,
            "templates": str(_fake_templates(tmp_path)), "n_proc": 0}
    eazy.update(eazy_overrides)
    cfg = parse_fit_config({"schema_version": 2, "backend": "eazy",
                            "name": "closure", "err_floor": 0.0,
                            "min_snr_broadband": 0.0, "eazy": eazy})
    return resolve_config(cfg, target_z_ref=Z_TRUE, reference_redshift=Z_TRUE)


def test_official_closure(tmp_path) -> None:
    frame = _synth_frame()
    cfg = _config(tmp_path, z_fixed=Z_TRUE)
    policy = apply_policy(frame, registry=REG, min_valid_bands=5,
                          min_snr_broadband=0.0, err_floor=0.0)
    result = run_official(cfg, frame, policy, tmp_path / "run", registry=REG)

    assert int(result.nusefilt[0]) == len(BANDS)
    assert abs(float(result.z_chi2[0]) - Z_TRUE) <= 0.011
    assert float(result.chi2_best[0]) < 5.0
    assert result.template_names == ["t0_spec.dat", "t1_spec.dat",
                                     "t2_spec.dat"]
    assert result.chi2_fixed is not None

    back = load_run(tmp_path / "run")
    assert back.z_chi2[0] == result.z_chi2[0]
    assert back.config["eazy"]["z_step"] == 0.005


def test_sentinel_and_nusefilt(tmp_path) -> None:
    frame = _synth_frame()
    cfg = _config(tmp_path)
    policy = apply_policy(frame, registry=REG, bands_include=["CFHT"],
                          min_valid_bands=5, min_snr_broadband=0.0,
                          err_floor=0.0)
    result = run_official(cfg, frame, policy, tmp_path / "run", registry=REG)

    assert int(result.nusefilt[0]) == 5
    catalog = pd.read_csv(tmp_path / "run" / "catalog.csv")
    assert catalog.loc[0, "f_WISE_W1"] == MISSING_SENTINEL
    assert catalog.loc[0, "e_WISE_W1"] == MISSING_SENTINEL
    assert catalog.loc[0, "f_CFHT_u"] != MISSING_SENTINEL


def test_filter_res_contents(tmp_path) -> None:
    frame = _synth_frame()
    cfg = _config(tmp_path)
    policy = apply_policy(frame, registry=REG, min_valid_bands=5,
                          min_snr_broadband=0.0, err_floor=0.0)
    run_official(cfg, frame, policy, tmp_path / "run", registry=REG)

    info = (tmp_path / "run" / "FILTER.RES.info").read_text().splitlines()
    assert len(info) == len(BANDS)
    assert info[0].split()[1] == "CFHT_u"
    translate = (tmp_path / "run" / "zphot.translate").read_text()
    assert "f_WISE_W1 F6" in translate and "e_WISE_W1 E6" in translate


def test_quick_vs_official_agreement(tmp_path) -> None:
    frame = _synth_frame()
    cfg = _config(tmp_path, z_fixed=Z_TRUE)
    policy = apply_policy(frame, registry=REG, min_valid_bands=5,
                          min_snr_broadband=0.0, err_floor=0.0)
    official = run_official(cfg, frame, policy, tmp_path / "official",
                            registry=REG)
    quick = run_quick(cfg, frame, policy, tmp_path / "quick", registry=REG)

    assert int(quick.nusefilt[0]) == int(official.nusefilt[0])
    assert quick.template_names == official.template_names
    # The discontinuous synthetic template maximizes the interpolation
    # deviation between the engines; smooth spectra agree to ~1e-5 in
    # z500, so the bound here is deliberately looser.
    assert abs(float(quick.z_ml[0]) - float(official.z_ml[0])) <= 5e-5
    assert abs(float(quick.z_percentiles[0, 2])
               - float(official.z_percentiles[0, 2])) <= 5e-5
    # The discontinuous template magnifies the quadrature deviation in
    # chi2; smooth spectra agree to ~4e-4 relative.
    assert (abs(float(quick.chi2_best[0]) - float(official.chi2_best[0]))
            <= 1e-2 * max(float(official.chi2_best[0]), 0.1))
    assert (abs(float(quick.chi2_fixed[0]) - float(official.chi2_fixed[0]))
            <= 1e-2 * float(official.chi2_fixed[0]))

    back = load_run(tmp_path / "quick")
    assert back.photz is None
    assert back.z_ml[0] == quick.z_ml[0]


def test_quick_envelope_guards(tmp_path) -> None:
    frame = _synth_frame()
    cfg = _config(tmp_path, fitter="bvls")
    policy = apply_policy(frame, registry=REG, min_valid_bands=5,
                          min_snr_broadband=0.0, err_floor=0.0)
    with pytest.raises(ValueError, match="nnls"):
        run_quick(cfg, frame, policy, tmp_path / "run", registry=REG)


def test_packaged_template_digests() -> None:
    cfg = parse_fit_config({"schema_version": 2, "backend": "eazy",
                            "name": "x", "eazy": {"templates": _PACKAGED}})
    digests = content_digests(cfg["eazy"])
    assert len(digests["templates"]) == 129
    assert len(digests["tef"]) == 1
    assert all(len(h) == 16 for h in digests["templates"].values())


def test_param_mode_digests_hash_spectra(tmp_path) -> None:
    specs = sorted(_fake_templates(tmp_path).glob("*_spec.dat"))
    param = write_templates_param(specs, tmp_path / "set.param")
    cfg = parse_fit_config({"schema_version": 2, "backend": "eazy", "name": "x",
                            "eazy": {"templates": str(param), "tef": False}})
    digests = content_digests(cfg["eazy"])
    assert set(digests["templates"]) == {p.name for p in specs}

    before = dict(digests["templates"])
    specs[0].write_text(specs[0].read_text() + "\n123.0 1.0\n")
    after = content_digests(cfg["eazy"])["templates"]
    assert after[specs[0].name] != before[specs[0].name]


def test_template_summary_fingerprint(tmp_path) -> None:
    default_cfg = parse_fit_config({"schema_version": 2, "backend": "eazy",
                                    "name": "x",
                                    "eazy": {"templates": _PACKAGED}})
    default_sum = template_summary(default_cfg["eazy"],
                                   content_digests(default_cfg["eazy"]))
    assert default_sum["n"] == 129
    assert len(default_sum["set_sha256_16"]) == 16
    assert default_sum["source"] == _PACKAGED

    tdir = _fake_templates(tmp_path)
    other_cfg = parse_fit_config({"schema_version": 2, "backend": "eazy",
                                  "name": "y",
                                  "eazy": {"templates": str(tdir)}})
    other_sum = template_summary(other_cfg["eazy"],
                                 content_digests(other_cfg["eazy"]))
    assert other_sum["n"] == 3
    assert other_sum["source"] == str(tdir)
    assert other_sum["set_sha256_16"] != default_sum["set_sha256_16"]


# ------------------------------------
# Template set resolution
# ------------------------------------

def test_a_packaged_set_resolves_by_name() -> None:
    from sedfit.backends.eazy.templates import (packaged_template_sets,
                                                resolve_template_source)
    names = packaged_template_sets()
    assert "brown14_vac_cosmos160" in names
    for name in names:
        assert resolve_template_source(name).is_dir()


def test_a_path_wins_over_a_packaged_name(tmp_path) -> None:
    from sedfit.backends.eazy.templates import resolve_template_source
    local = tmp_path / "brown14"
    local.mkdir()
    (local / "one_spec.dat").write_text("1000 1.0\n", encoding="utf-8")
    assert resolve_template_source(local) == local


def test_an_unknown_set_names_the_packaged_ones() -> None:
    from sedfit.backends.eazy.templates import resolve_template_source
    with pytest.raises(ValueError, match="packaged sets are"):
        resolve_template_source("no_such_basis")


def test_naming_a_set_matches_naming_its_path() -> None:
    from sedfit.backends.eazy.templates import content_digests
    by_name = content_digests({"templates": "brown14",
                               "template_pattern": "*.dat", "tef": False})
    by_path = content_digests({"templates": str(_PACKAGED),
                               "template_pattern": "*.dat", "tef": False})
    assert by_name == by_path
