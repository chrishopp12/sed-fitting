from __future__ import annotations

import csv
import json

import pytest
from test_build import _tree
from test_eazy_backend import _fake_templates
from test_jobs import _raw_cfg

from sedfit.batch import (
    STATUS_FAILED,
    STATUS_OK,
    STATUS_SKIPPED,
    expand_jobs,
    read_target_list,
    run_batch,
)
from sedfit.core.build import build_target, write_build
from sedfit.core.registry import load_registry
from sedfit.core.roster import load_roster
from sedfit.core.runs import read_manifest
from sedfit.jobs import run_job

REG = load_registry()

# none_ref declares "spherex": null, so it applies to the SPHEREx-carrying
# target and the one without alike; no_block omits the block entirely and
# so applies only to the target that declares no SPHEREx table.
BOTH = "none_ref"
B_ONLY = "no_block"


def _setup(tmp_path, recipe=BOTH):
    """A roster with both targets built under one recipe applying to both."""
    roster_path = _tree(tmp_path)
    roster = load_roster(roster_path, REG)
    for target in ("A", "B"):
        result = build_target(roster, target, recipe, registry=REG)
        write_build(result, REG)
    _fake_templates(tmp_path)
    config_path = tmp_path / "cfg.json"
    config_path.write_text(
        json.dumps(_raw_cfg(tmp_path, min_valid_bands=2)))
    return roster_path, config_path, roster


def _sidecar_path(roster, name, recipe=BOTH):
    target = roster.targets[name]
    return (target.dir / "Photometry"
            / f"{target.prefix}_sed_{recipe}.provenance.json")


def _report(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _cfg(config_path):
    from sedfit.core.fitconfig import load_fit_config

    return load_fit_config(config_path)


def test_batch_matches_serial_run_ids(tmp_path) -> None:
    roster_path, config_path, roster = _setup(tmp_path)
    summary = run_batch(roster_path, config_path, recipes=[BOTH],
                        workers=2, progress=lambda line: None)

    assert summary["n_ok"] == 2
    assert summary["n_failed"] == 0
    rows = _report(summary["report"])
    assert [r["target"] for r in rows] == ["A", "B"]
    assert all(r["status"] == STATUS_OK for r in rows)
    assert all(len(r["run_id"]) == 12 for r in rows)

    manifest, problems = read_manifest(roster.manifest_path)
    assert problems == []
    assert len(manifest) == 2

    # The same jobs run serially land on the same identities.
    serial = {name: run_job(roster, name, BOTH, _cfg(config_path),
                            registry=REG, force=True)
              for name in ("A", "B")}
    for row in rows:
        assert row["run_id"] == serial[row["target"]]["run_id"]
        assert float(row["zred_p50"]) == pytest.approx(
            serial[row["target"]]["estimates"]["zred_p50"])


def test_batch_collects_failures_without_aborting(tmp_path) -> None:
    roster_path, config_path, roster = _setup(tmp_path)
    path = _sidecar_path(roster, "A")
    sidecar = json.loads(path.read_text())
    path.write_text(json.dumps(dict(sidecar, recipe="wrong_recipe")))

    summary = run_batch(roster_path, config_path, recipes=[BOTH],
                        workers=1, progress=lambda line: None)

    assert summary["n_ok"] == 1
    assert summary["n_failed"] == 1
    rows = {r["target"]: r for r in _report(summary["report"])}
    assert rows["A"]["status"] == STATUS_FAILED
    assert rows["A"]["stage"] == "plan"
    assert "sidecar recipe" in rows["A"]["error"]
    assert rows["B"]["status"] == STATUS_OK


def test_resume_skips_completed_runs(tmp_path) -> None:
    roster_path, config_path, roster = _setup(tmp_path)
    first = run_batch(roster_path, config_path, recipes=[BOTH],
                      workers=1, progress=lambda line: None)
    assert first["n_ok"] == 2

    second = run_batch(roster_path, config_path, recipes=[BOTH],
                       workers=1, progress=lambda line: None)
    assert second["n_ok"] == 0
    assert second["n_skipped"] == 2
    assert all(r["status"] == STATUS_SKIPPED
               for r in _report(second["report"]))

    manifest, _ = read_manifest(roster.manifest_path)
    assert len(manifest) == 2

    third = run_batch(roster_path, config_path, recipes=[BOTH],
                      workers=1, resume=False, progress=lambda line: None)
    assert third["n_ok"] == 2


def test_stop_after_failures_abandons_the_batch(tmp_path) -> None:
    roster_path, config_path, roster = _setup(tmp_path)
    for name in ("A", "B"):
        path = _sidecar_path(roster, name)
        sidecar = json.loads(path.read_text())
        path.write_text(json.dumps(dict(sidecar, recipe="wrong_recipe")))

    summary = run_batch(roster_path, config_path, recipes=[BOTH],
                        workers=1, stop_after_failures=1,
                        progress=lambda line: None)
    assert summary["aborted"]
    assert summary["n_failed"] == 1
    assert len(_report(summary["report"])) == 1


def test_a_dereddening_build_writes_the_table_the_fit_then_reads(tmp_path) -> None:
    # The build and the fit disagreed silently: the build wrote the
    # as-measured table while plan_job went looking for the _dered one,
    # so a stale _dered table on disk would have been fitted and reported
    # as a fresh build. An explicit mw_ebv keeps this off the network.
    roster_path, config_path, roster = _setup(tmp_path)
    summary = run_batch(roster_path, config_path, recipes=[BOTH],
                        build=True, dered=True, mw_ebv=0.05, workers=1,
                        progress=lambda line: None)
    assert summary["n_ok"] == 2, summary

    for name in ("A", "B"):
        target = roster.targets[name]
        stem = target.dir / "Photometry" / f"{target.prefix}_sed_{BOTH}_dered"
        assert stem.with_suffix(".csv").is_file()
        sidecar = json.loads(
            stem.with_suffix(".provenance.json").read_text())
        assert sidecar["dered"]["ebv"] == 0.05
        assert sidecar["dered"]["broadband_A_mag"]

    # The fitted bytes must be the dereddened table, not its as-measured
    # sibling -- the whole point of the flag.
    for row in _report(summary["report"]):
        phot = (roster.data_root / row["path"] / "phot.csv").read_bytes()
        target = roster.targets[row["target"]]
        stem = target.dir / "Photometry" / f"{target.prefix}_sed_{BOTH}"
        assert phot == (stem.parent / f"{stem.name}_dered.csv").read_bytes()
        assert phot != stem.with_suffix(".csv").read_bytes()


def test_mw_ebv_without_a_dereddening_build_is_refused(tmp_path) -> None:
    roster_path, config_path, _ = _setup(tmp_path)
    with pytest.raises(ValueError, match="only reaches a dereddening build"):
        run_batch(roster_path, config_path, recipes=[BOTH], mw_ebv=0.05,
                  workers=1, progress=lambda line: None)


def test_expansion_records_inapplicable_pairs(tmp_path) -> None:
    roster_path, config_path, roster = _setup(tmp_path)
    jobs, skipped = expand_jobs(roster, config_path, recipes=[B_ONLY])

    # Target A declares a SPHEREx table; no_block has no spherex block.
    assert [job.target for job in jobs] == ["B"]
    assert [row["target"] for row in skipped] == ["A"]
    assert skipped[0]["status"] == STATUS_SKIPPED
    assert "spherex" in skipped[0]["error"].lower()

    # Naming no roster target at all is the wrong-file signature, and
    # stays fatal before any job runs.
    with pytest.raises(ValueError, match="in the roster"):
        expand_jobs(roster, config_path, targets=["nope"])


def test_a_target_the_roster_omits_is_reported_not_fatal(tmp_path) -> None:
    # One catalog drives acquisition and fitting, and generation
    # legitimately omits a target whose sources did not survive. A single
    # such name must not cost the other 65 their sweep.
    roster_path, config_path, roster = _setup(tmp_path)
    jobs, skipped = expand_jobs(roster, config_path, recipes=[BOTH],
                                targets=["A", "dropped_by_generation", "B"])

    assert [job.target for job in jobs] == ["A", "B"]
    omitted = [row for row in skipped if row["target"] == "dropped_by_generation"]
    assert len(omitted) == 1
    assert omitted[0]["status"] == STATUS_SKIPPED
    assert omitted[0]["stage"] == "roster"
    assert "not in the roster" in omitted[0]["error"]


def test_target_list_reads_the_sample_catalog(tmp_path) -> None:
    path = tmp_path / "sample.csv"
    path.write_text("name,ra_deg,dec_deg,z_ref_kind\n"
                    "B,10.0,20.0,cluster\n"
                    "A,10.0,20.0,cluster\n"
                    " ,,,\n")
    assert read_target_list(path) == ["B", "A"]

    alias = tmp_path / "alias.csv"
    alias.write_text("target,note\nA,x\n")
    assert read_target_list(alias) == ["A"]

    bad = tmp_path / "bad.csv"
    bad.write_text("ra_deg\n10.0\n")
    with pytest.raises(ValueError, match="no name or target column"):
        read_target_list(bad)

    # Read before any job runs, so a duplicate costs nothing to refuse
    # here -- and deduping silently is how a generated list hides a bug.
    duplicate = tmp_path / "duplicate.csv"
    duplicate.write_text("name\nB\nA\nB\n")
    with pytest.raises(ValueError, match="duplicate target"):
        read_target_list(duplicate)


def test_report_is_written_even_when_the_batch_is_empty(tmp_path) -> None:
    roster_path, config_path, roster = _setup(tmp_path)
    report = tmp_path / "out" / "report.csv"
    summary = run_batch(roster_path, config_path, targets=["A"],
                        recipes=[B_ONLY], workers=1, report_path=report,
                        progress=lambda line: None)

    assert summary["n_jobs"] == 0
    assert report.exists()
    assert not report.with_suffix(".csv.partial").exists()
    assert _report(report)[0]["status"] == STATUS_SKIPPED
