from __future__ import annotations

from pathlib import Path

import pytest

from sedfit.core.fitconfig import (
    hash_projection,
    parse_fit_config,
    resolve_config,
)
from sedfit.core.provenance import run_id

# Re-minted when templates/template_pattern/tef_file left the
# projection: identity follows template CONTENT, not its path.
GOLDEN_PROJECTED_RUN_ID = "d5bcde62db23"


# templates is required (no silent default), so every eazy fixture names
# the packaged atlas; the projection strips the path, so which directory
# it is cannot affect any identity assertion below.
PACKAGED = str(Path(__file__).resolve().parents[1]
               / "sedfit" / "data" / "templates" / "brown14")


def _eazy(**overrides) -> dict:
    raw = {"schema_version": 2, "backend": "eazy", "name": "x"}
    eazy = dict(overrides.pop("eazy", {}))
    eazy.setdefault("templates", PACKAGED)
    raw.update(overrides)
    raw["eazy"] = eazy
    return raw


def _prospector(**block) -> dict:
    prospector = {"stellar_library": "miles",
                  "zred": {"prior": "normal", "sigma": 0.004}}
    prospector.update(block)
    return {"schema_version": 2, "backend": "prospector", "name": "x",
            "prospector": prospector}


def test_eazy_defaults() -> None:
    cfg = parse_fit_config(_eazy())
    assert cfg["min_valid_bands"] == 5
    assert cfg["err_floor"] == 0.05
    assert cfg["eazy"]["engine"] == "quick"
    assert cfg["eazy"]["tef"] and cfg["eazy"]["tef_scale"] == 1.0
    assert cfg["eazy"]["z_step_type"] == "linear"


def test_prospector_defaults() -> None:
    cfg = parse_fit_config(_prospector())
    block = cfg["prospector"]
    assert block["sfh"] == "continuity" and block["n_agebins"] == 7
    assert block["tau_range"] is None
    assert block["gas_logu_free"] is None
    assert block["sampler"] == "dynesty" and block["emcee"] is None
    assert block["zred"]["bounds"] == [0.0, 1.0]


def _fails(raw: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        parse_fit_config(raw)


def test_strict_keys_and_discriminants() -> None:
    _fails(_eazy(sys_err=0.05), "unknown keys")
    _fails(_eazy(eazy={"noise_floor": 0.05}), "unknown keys")
    _fails(_eazy(prospector={"stellar_library": "miles"}), "illegal under")
    _fails({"schema_version": 2, "backend": "prospector", "name": "x"},
           "stellar_library is required")
    _fails(_prospector(stellar_library="basel"), "not in")


def test_eazy_conditionals() -> None:
    _fails(_eazy(eazy={"tef": False, "tef_file": "f.dat"}), "tef is off")
    _fails(_eazy(eazy={"tef": False, "tef_scale": 1.0}), "meaningless")
    off = parse_fit_config(_eazy(eazy={"tef": False}))
    assert off["eazy"]["tef_scale"] is None

    _fails(_eazy(eazy={"prior": True}), "requires prior_file")
    _fails(_eazy(eazy={"prior_file": "p.dat"}), "prior is off")
    _fails(_eazy(eazy={"z_fixed": 0.5}), "outside the grid")


def test_quick_envelope() -> None:
    _fails(_eazy(eazy={"prior": True, "prior_file": "p", "prior_filter": "f"}),
           "eazy-py")
    _fails(_eazy(eazy={"fitter": "bvls"}), "NNLS-only")
    _fails(_eazy(eazy={"extra_params": {"IGM_SCALE": 1.0}}), "eazy-py")
    official = parse_fit_config(_eazy(eazy={"engine": "eazy-py",
                                            "extra_params": {"X": 1}}))
    assert official["eazy"]["extra_params"] == {"X": 1}


def test_prospector_conditionals() -> None:
    _fails(_prospector(zred=None), "requires a zred block")
    _fails(_prospector(fit_redshift=False), "fit_redshift is off")
    _fails(_prospector(zred={"prior": "uniform", "mean": 0.1}),
           "prior is uniform")
    _fails(_prospector(zred={"prior": "normal"}), "sigma")
    _fails(_prospector(tau_range=[0.1, 30.0]), "continuity")
    _fails(_prospector(sfh="tau", n_agebins=7), "parametric")
    _fails(_prospector(emcee={"nwalkers": 64}), "sampler is dynesty")
    _fails(_prospector(nebular=False, gas_logu_free=True), "nebular is off")

    tau = parse_fit_config(_prospector(sfh="tau"))
    assert tau["prospector"]["tau_range"] == [0.1, 30.0]
    assert tau["prospector"]["tie_tage_to_tuniv"] is True

    nebular = parse_fit_config(_prospector(nebular=True))
    assert nebular["prospector"]["gas_logu_free"] is False


def test_sp_prior_shapes() -> None:
    uniform = parse_fit_config(_prospector(dust2_prior_type="uniform",
                                           dust2_prior=[0.0, 0.4]))
    assert uniform["prospector"]["dust2_prior"] == [0.0, 0.4]
    assert uniform["prospector"]["logzsol_prior_type"] == "clipped_normal"

    _fails(_prospector(dust2_prior_type="uniform"),
           "uniform prior takes 2 values")
    _fails(_prospector(logzsol_prior=[0.0, 0.4]),
           "clipped_normal prior takes 4 values")
    _fails(_prospector(dust2_prior_type="tophat"), "not in")


def test_resolution() -> None:
    cfg = parse_fit_config(_prospector())
    resolved = resolve_config(cfg, target_z_ref=0.1062,
                              reference_redshift=0.106,
                              seed_source=lambda: 902113)
    assert resolved["z_ref"] == 0.1062
    assert resolved["prospector"]["zred"]["mean"] == 0.106
    assert resolved["prospector"]["seed"] == 902113

    mismatched = _prospector()
    mismatched["z_ref"] = 0.2
    with pytest.raises(ValueError, match="disagrees with the roster"):
        resolve_config(parse_fit_config(mismatched),
                       target_z_ref=0.1062, reference_redshift=0.106)

    seeded = parse_fit_config(_prospector(seed=7))
    resolved = resolve_config(seeded, target_z_ref=0.1062,
                              reference_redshift=0.106)
    assert resolved["prospector"]["seed"] == 7


def test_projection_golden() -> None:
    cfg = parse_fit_config(_eazy(name="golden", eazy={"z_fixed": 0.106}))
    resolved = resolve_config(cfg, target_z_ref=0.106,
                              reference_redshift=0.106)
    projected = hash_projection(resolved, digests={"tef": "abc123"})
    assert "name" not in projected
    assert "n_proc" not in projected["eazy"]
    rid = run_id(projected, "p" * 64, "q" * 64)
    assert rid == GOLDEN_PROJECTED_RUN_ID

    renamed = dict(resolved, name="other")
    renamed["eazy"] = dict(resolved["eazy"], n_proc=16)
    same = run_id(hash_projection(renamed, digests={"tef": "abc123"}),
                  "p" * 64, "q" * 64)
    assert same == rid

    drifted = run_id(hash_projection(resolved, digests={"tef": "def456"}),
                     "p" * 64, "q" * 64)
    assert drifted != rid


def test_identity_follows_template_content_not_its_path(tmp_path) -> None:
    # The same curves under a different directory are the same science,
    # so they must give the same run identity.
    import shutil

    moved = tmp_path / "elsewhere"
    shutil.copytree(PACKAGED, moved)
    here = resolve_config(parse_fit_config(_eazy(name="a")),
                          target_z_ref=0.106, reference_redshift=0.106)
    there = resolve_config(
        parse_fit_config(_eazy(name="b", eazy={"templates": str(moved)})),
        target_z_ref=0.106, reference_redshift=0.106)
    digests = {"templates": {"NGC_3379_spec.dat": "0" * 16}}
    assert (run_id(hash_projection(here, digests=digests), "p" * 64, "q" * 64)
            == run_id(hash_projection(there, digests=digests),
                      "p" * 64, "q" * 64))


def test_an_eazy_config_must_name_its_template_set() -> None:
    bare = {"schema_version": 2, "backend": "eazy", "name": "x"}
    _fails(bare, "templates is required")
    _fails({**bare, "eazy": {"templates": "  "}}, "templates is required")


def test_qa_gates_config() -> None:
    assert parse_fit_config(_eazy())["qa_gates"] is None
    cfg = parse_fit_config(_eazy(qa_gates={"maskfrac": {"max": 0.5},
                                           "conv": {"min": 0}}))
    assert cfg["qa_gates"] == {"maskfrac": {"max": 0.5}, "conv": {"min": 0}}
    _fails(_eazy(qa_gates={"CHFT": {"max": 1}}), "unrecognized tokens")
    _fails(_eazy(qa_gates={}), "non-empty")
    _fails(_eazy(qa_gates={"cov": {"lo": 1}}), "unrecognized keys")


def test_qa_gates_null_preserves_identity() -> None:
    resolved = resolve_config(parse_fit_config(_eazy(name="golden",
                                                     eazy={"z_fixed": 0.106})),
                              target_z_ref=0.106, reference_redshift=0.106)
    assert resolved["qa_gates"] is None
    projected = hash_projection(resolved, digests={"tef": "abc123"})
    assert run_id(projected, "p" * 64, "q" * 64) == GOLDEN_PROJECTED_RUN_ID
