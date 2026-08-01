from __future__ import annotations

import copy
import json
import subprocess
import sys

from test_generate import CAMPAIGN, _catalog, _row, _tree

# These exercise the quick engine, which needs no eazy-py, so they run on
# a core install.

NO_SPHEREX_RECIPE = {
    "plain": {"reference": "none",
              "sources": [{"source": "wise", "role": "float"},
                          {"source": "legacy", "role": "float"}],
              "spherex": None},
}


def _sedfit(*args) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "sedfit", *args],
                          capture_output=True, text=True)


def _campaign_without_spherex(tmp_path, monkeypatch):
    """The generate fixture tree, with one recipe needing no spectrum."""
    import test_generate

    campaign = copy.deepcopy(CAMPAIGN)
    campaign["recipes"] = NO_SPHEREX_RECIPE
    monkeypatch.setattr(test_generate, "CAMPAIGN", campaign)
    return _tree(tmp_path)


def _config(tmp_path) -> str:
    path = tmp_path / "quick.json"
    path.write_text(json.dumps({
        "schema_version": 2, "backend": "eazy", "name": "clitest",
        "err_floor": 0.05, "min_valid_bands": 3,
        "eazy": {"engine": "quick", "z_min": 0.05, "z_max": 0.16,
                 "z_step": 0.005, "templates": "brown14",
                 "template_pattern": "*.dat"}}), encoding="utf-8")
    return str(path)


# ------------------------------------
# Help output
# ------------------------------------

def test_every_verb_and_option_is_described() -> None:
    import argparse

    from sedfit.__main__ import build_parser

    parser = build_parser()
    subparsers = [action for action in parser._actions
                  if isinstance(action, argparse._SubParsersAction)]
    assert len(subparsers) == 1
    verbs = subparsers[0].choices
    assert set(verbs) == {"roster", "build", "fit", "batch", "run", "plot",
                          "manifest"}

    assert parser.description
    for name, subparser in verbs.items():
        assert subparser.description, f"{name}: no description"
        for action in subparser._actions:
            if action.dest == "help":
                continue
            assert action.help, f"{name}: {action.dest} has no help text"


# ------------------------------------
# The run verb
# ------------------------------------

def test_run_chains_roster_build_and_fit(tmp_path, monkeypatch) -> None:
    campaign = _campaign_without_spherex(tmp_path, monkeypatch)
    catalog = _catalog(tmp_path, [_row("A", "gal_a"), _row("B", "gal_b")])

    result = _sedfit("run", "--catalog", str(catalog),
                     "--campaign", str(campaign),
                     "--config", _config(tmp_path), "--workers", "1")
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "roster.json").is_file()
    assert "2 ok, 0 failed" in result.stdout
    for stage in ("roster", "batch"):
        assert f"  {stage:8s} ok" in result.stdout


def test_run_reports_a_failed_stage_and_exits_nonzero(
        tmp_path, monkeypatch) -> None:
    campaign = _campaign_without_spherex(tmp_path, monkeypatch)
    catalog = _catalog(tmp_path, [_row("A", "gal_a")])
    config = tmp_path / "impossible.json"
    config.write_text(json.dumps({
        "schema_version": 2, "backend": "eazy", "name": "clitest",
        "min_valid_bands": 99,
        "eazy": {"engine": "quick", "z_min": 0.05, "z_max": 0.16,
                 "z_step": 0.005, "templates": "brown14",
                 "template_pattern": "*.dat"}}), encoding="utf-8")

    result = _sedfit("run", "--catalog", str(catalog),
                     "--campaign", str(campaign),
                     "--config", str(config), "--workers", "1")
    assert result.returncode == 1
    assert "roster   ok" in result.stdout
    assert "batch    FAILED" in result.stdout


# ------------------------------------
# Actionable errors
# ------------------------------------

def test_a_missing_roster_is_named(tmp_path) -> None:
    result = _sedfit("build", "--roster", str(tmp_path / "nope.json"))
    assert result.returncode != 0
    assert "roster not found" in (result.stdout + result.stderr)


def test_an_unknown_target_lists_the_declared_ones(
        tmp_path, monkeypatch) -> None:
    campaign = _campaign_without_spherex(tmp_path, monkeypatch)
    catalog = _catalog(tmp_path, [_row("A", "gal_a")])
    assert _sedfit("roster", "--catalog", str(catalog),
                   "--campaign", str(campaign)).returncode == 0

    result = _sedfit("build", "--roster", str(tmp_path / "roster.json"),
                     "--target", "nonexistent")
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "unknown target 'nonexistent'" in combined
    assert "the roster declares: A" in combined


def test_mw_ebv_without_deredden_is_refused(tmp_path) -> None:
    result = _sedfit("build", "--roster", str(tmp_path / "nope.json"),
                     "--mw-ebv", "0.05")
    assert result.returncode != 0
    assert "--mw-ebv applies only with --deredden" in (result.stdout
                                                       + result.stderr)


def test_a_run_dir_without_a_config_is_named(tmp_path) -> None:
    result = _sedfit("plot", "--run-dir", str(tmp_path))
    assert result.returncode != 0
    assert "is not a run directory" in (result.stdout + result.stderr)
