from __future__ import annotations

import json

import numpy as np
import pytest
from test_build import _tree
from synthetic import fake_templates as _fake_templates

from sedfit.core.build import build_target, write_build
from sedfit.core.fitconfig import parse_fit_config
from sedfit.core.registry import load_registry
from sedfit.core.roster import load_roster
from sedfit.core.runs import read_manifest
from sedfit.jobs import phot_path_for, plan_job, run_job

REG = load_registry()


def _raw_cfg(tmp_path, **extra) -> dict:
    raw = {
        "schema_version": 2, "backend": "eazy", "name": "jobtest",
        "err_floor": 0.0, "min_snr_broadband": 0.0,
        "eazy": {"z_min": 0.05, "z_max": 0.15, "z_step": 0.005,
                 "z_step_type": "linear", "tef": False,
                 "templates": str(tmp_path / "templates")},
    }
    raw.update(extra)
    return raw


def _setup(tmp_path):
    roster = load_roster(_tree(tmp_path), REG)
    result = build_target(roster, "A", "tilt2", registry=REG)
    write_build(result, REG)
    _fake_templates(tmp_path)
    return roster, parse_fit_config(_raw_cfg(tmp_path))


def test_an_undispatchable_backend_is_refused_before_anything_is_written(
        tmp_path) -> None:
    """A backend jobs.py cannot run is refused by name, early.

    The four branches this guards used to end in a bare `else:` meaning
    "prospector". Measured 2026-08-25: a linear config sent through the old
    code died in apply_policy on `TypeError: '<' not supported between
    instances of NoneType and int`, naming neither the backend nor the
    reason. It failed before stage_run only because the photometry-only
    fields are nulled under a spectrum-only backend -- a backend that got
    past the policy gates would have reached stage_run first, which is the
    failure DESIGN.md 16.6 describes. The run-directory assertion below
    pins that property for whatever backend comes next.
    """
    roster, _ = _setup(tmp_path)
    linear = parse_fit_config({
        "schema_version": 2, "backend": "linear", "name": "jobtest",
        "linear": {"templates": str(tmp_path / "templates"),
                   "template_wave_range": [3600.0, 7400.0],
                   "z_min": 0.05, "z_max": 0.15, "poly_wave_frame": "air",
                   "spectrum": {"file": str(tmp_path / "spec.csv")}}})

    for verb in (plan_job, run_job):
        with pytest.raises(NotImplementedError, match="'linear'") as excinfo:
            verb(roster, "A", "tilt2", linear, registry=REG)
        assert "no photometry table" in str(excinfo.value)

    assert not (roster.targets["A"].dir / "SED" / "tilt2" / "linear").exists()
    assert not roster.manifest_path.exists()


def test_job_end_to_end(tmp_path) -> None:
    roster, cfg = _setup(tmp_path)
    row = run_job(roster, "A", "tilt2", cfg, registry=REG)

    assert row["status"] == "ok"
    assert row["engine"] == "quick"
    assert len(row["run_id"]) == 12
    assert row["templates"]["n"] == 3
    assert len(row["templates"]["set_sha256_16"]) == 16
    assert row["templates"]["source"].endswith("templates")
    assert 0.05 < row["estimates"]["zred_p50"] < 0.15
    assert row["n_bands_fit"] == row["n_bands_table"]

    run_dir = roster.data_root / row["path"]
    for product in ("config.json", "phot.csv", "phot.provenance.json",
                    "manifest.json", "summary.csv", "arrays.npz"):
        assert (run_dir / product).exists()
    assert (run_dir / "phot.csv").read_bytes() \
        == phot_path_for(roster.targets["A"], "tilt2").read_bytes()

    rows, problems = read_manifest(roster.manifest_path)
    assert problems == []
    assert rows[-1]["run_id"] == row["run_id"]

    plot_files = list((run_dir / "plots").glob("*.png"))
    assert any(p.name.startswith("zscan") for p in plot_files)

    echoed = json.loads((run_dir / "config.json").read_text())
    assert echoed["z_ref"] == pytest.approx(0.106)


def test_rerun_replaces_never_siblings(tmp_path) -> None:
    roster, cfg = _setup(tmp_path)
    first = run_job(roster, "A", "tilt2", cfg, registry=REG)
    second = run_job(roster, "A", "tilt2", cfg, registry=REG)
    assert first["run_id"] == second["run_id"]

    backend_dir = roster.data_root / "gal_a" / "SED" / "tilt2" / "eazy"
    run_dirs = [d for d in backend_dir.iterdir() if d.is_dir()]
    assert len(run_dirs) == 1

    rows, _ = read_manifest(roster.manifest_path)
    assert len(rows) == 2


def test_sidecar_cross_checks(tmp_path) -> None:
    roster, cfg = _setup(tmp_path)
    phot_path = phot_path_for(roster.targets["A"], "tilt2")
    sidecar_path = phot_path.parent / (phot_path.stem + ".provenance.json")
    sidecar = json.loads(sidecar_path.read_text())

    tampered = dict(sidecar, recipe="other_recipe")
    sidecar_path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="sidecar recipe"):
        run_job(roster, "A", "tilt2", cfg, registry=REG)

    tampered = dict(sidecar, z_ref=0.2)
    sidecar_path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="z_ref"):
        run_job(roster, "A", "tilt2", cfg, registry=REG)

    drifted = json.loads(json.dumps(sidecar))
    band = next(iter(drifted["registry"]["bands"]))
    drifted["registry"]["bands"][band] = "0" * 16
    sidecar_path.write_text(json.dumps(drifted))
    with pytest.raises(ValueError, match="bandpass hashes drifted"):
        run_job(roster, "A", "tilt2", cfg, registry=REG)

    row = run_job(roster, "A", "tilt2", cfg, registry=REG,
                  allow_registry_drift=True)
    assert any("drifted" in w for w in row["warnings"])


def test_identity_separates_configs(tmp_path) -> None:
    roster, cfg = _setup(tmp_path)
    first = run_job(roster, "A", "tilt2", cfg, registry=REG)

    variant = parse_fit_config(_raw_cfg(
        tmp_path, bands_include=["WISE", "SPHEREx", "Legacy"],
        name="subset"))
    second = run_job(roster, "A", "tilt2", variant, registry=REG)
    assert second["run_id"] != first["run_id"]
    assert second["n_excluded"] > 0


def test_plan_job_validates_without_writing(tmp_path) -> None:
    roster, cfg = _setup(tmp_path)
    plan = plan_job(roster, "A", "tilt2", cfg, registry=REG)
    assert len(plan["run_id"]) == 12
    assert plan["policy"].n_fit > 0
    assert not (roster.data_root / "gal_a" / "SED").exists()
    assert not (roster.manifest_path).exists()

    row = run_job(roster, "A", "tilt2", cfg, registry=REG)
    assert row["run_id"] == plan["run_id"]
    assert row["n_qa_rejected"] == 0


# ------------------------------------
# The machinery stamp
# ------------------------------------

def test_the_quick_engine_stamps_no_eazy_version() -> None:
    from sedfit.jobs import _backend_versions
    assert _backend_versions("eazy", "quick") == {}
    assert set(_backend_versions("eazy", "eazy-py")) == {"eazy"}


def test_an_absent_distribution_stamps_none(monkeypatch) -> None:
    import importlib.metadata as md
    from sedfit.jobs import _backend_versions

    real = md.version

    def absent(dist):
        if dist == "eazy":
            raise md.PackageNotFoundError(dist)
        return real(dist)

    monkeypatch.setattr(md, "version", absent)
    assert _backend_versions("eazy", "eazy-py") == {"eazy": None}
