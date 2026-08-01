from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from sedfit.core.registry import load_registry
from sedfit.core.roster import load_roster

REG = load_registry()
SPHEREX_FIXTURE = Path(__file__).parent / "data" / "spherex_visits.csv"

CAT_ROWS = [
    ("Legacy_g", 120.0, 1.2, "Legacy_DR9"),
    ("Legacy_r", 250.0, 2.0, "Legacy_DR9"),
    ("WISE_W1", 900.0, 9.0, "unWISE_Legacy_DR9"),
    ("WISE_W1", 780.0, 30.0, "AllWISE"),
]


def _roster_dict() -> dict:
    return {
        "schema_version": 2,
        "cluster": "TestCluster",
        "data_root": "data",
        "cluster_redshift": 0.106,
        "targets": {
            "A": {
                "position": {"ra_deg": 10.0, "dec_deg": 20.0,
                             "frame": "icrs", "authority": "test"},
                "z_ref_kind": "cluster",
                "dir": "gal_a",
                "prefix": "galA",
                "spherex": {"table": "Photometry/SPHEREx/visits.csv",
                            "model": "sersic", "provenance": "test"},
                "sources": {
                    "legacy": {"path": "Photometry/cat.csv",
                               "bands": ["Legacy_g", "Legacy_r"],
                               "kind": "catalog",
                               "provider": "legacy: test fixture"},
                    "wise": {"path": "Photometry/cat.csv",
                             "bands": ["WISE_W1"],
                             "kind": "catalog",
                             "provider": "unwise"},
                },
            },
            "B": {
                "position": {"ra_deg": 11.0, "dec_deg": 21.0,
                             "frame": "icrs", "authority": "test"},
                "z_ref": 0.108,
                "z_ref_kind": "spec",
                "dir": "gal_b",
                "prefix": "galB",
                "sources": {
                    "legacy": {"path": "Photometry/cat.csv",
                               "bands": ["Legacy_g"],
                               "kind": "catalog",
                               "provider": "legacy"},
                },
            },
        },
        "recipes": {
            "tilt": {"reference": "anchors",
                     "sources": [{"source": "legacy", "role": "anchor"},
                                 {"source": "wise", "role": "anchor"}],
                     "spherex": {"cuts": {}, "binning": {"split_um": None}}},
            "plain": {"reference": "none",
                      "sources": [{"source": "legacy", "role": "float"}],
                      "spherex": None},
        },
    }


def _build_tree(tmp_path: Path, roster: dict) -> Path:
    for gal in ("gal_a", "gal_b"):
        phot = tmp_path / "data" / gal / "Photometry"
        phot.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(CAT_ROWS, columns=["band", "flux_uJy",
                                                "flux_err_uJy", "source"])
        frame.to_csv(phot / "cat.csv", index=False)
    spherex_dir = tmp_path / "data" / "gal_a" / "Photometry" / "SPHEREx"
    spherex_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(SPHEREX_FIXTURE, spherex_dir / "visits.csv")
    path = tmp_path / "roster.json"
    path.write_text(json.dumps(roster))
    return path


def test_happy_load(tmp_path) -> None:
    roster = load_roster(_build_tree(tmp_path, _roster_dict()), REG)
    assert roster.sample == "TestCluster"
    assert roster.data_root == (tmp_path / "data").resolve()

    a = roster.targets["A"]
    assert a.z_ref == pytest.approx(0.106)
    # The fixture declares the deprecated 'cluster' spelling; the loader
    # normalizes it to the canonical kind.
    assert a.z_ref_kind == "reference"
    assert a.dir.is_dir() and a.dir.name == "gal_a"
    assert a.spherex.table.is_file()
    assert a.sources["legacy"].path.is_file()
    assert a.sources["wise"].provider == "unwise"

    b = roster.targets["B"]
    assert b.z_ref == pytest.approx(0.108)
    assert b.spherex is None

    assert set(roster.recipes) == {"tilt", "plain"}
    assert roster.recipes["plain"].spherex_declared


def _expect(tmp_path: Path, roster: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        load_roster(_build_tree(tmp_path, roster), REG)


def test_z_ref_single_home(tmp_path) -> None:
    bad = _roster_dict()
    bad["targets"]["A"]["z_ref"] = 0.106
    _expect(tmp_path, bad, "derives z_ref")

    bad = _roster_dict()
    del bad["targets"]["B"]["z_ref"]
    _expect(tmp_path, bad, "requires a numeric z_ref")


def test_strict_schema(tmp_path) -> None:
    bad = _roster_dict()
    bad["schema_version"] = 1
    _expect(tmp_path, bad, "schema_version")

    bad = _roster_dict()
    bad["data_root_env"] = "SOME_VAR"
    _expect(tmp_path, bad, "unknown keys")

    bad = _roster_dict()
    bad["targets"]["A"]["truth"] = False
    _expect(tmp_path, bad, "unknown keys")

    bad = _roster_dict()
    bad["targets"]["A"]["position"]["frame"] = "fk5"
    _expect(tmp_path, bad, "not in")

    bad = _roster_dict()
    bad["targets"]["A"]["spherex"]["model"] = "gaussian"
    _expect(tmp_path, bad, "not in")

    bad = _roster_dict()
    bad["targets"]["A"]["sources"]["legacy"]["provider"] = "legacy_tractor: x"
    _expect(tmp_path, bad, "unknown provider token")

    bad = _roster_dict()
    bad["targets"]["A"]["sources"]["legacy"]["bands"] = ["Legacy_g", "NOPE_x"]
    _expect(tmp_path, bad, "not in the registry")


def test_declarations_verified_against_files(tmp_path) -> None:
    bad = _roster_dict()
    bad["targets"]["A"]["sources"]["legacy"]["path"] = "Photometry/nope.csv"
    _expect(tmp_path, bad, "does not exist")

    bad = _roster_dict()
    bad["targets"]["A"]["sources"]["legacy"]["bands"] = ["Legacy_g", "Legacy_z"]
    _expect(tmp_path, bad, "matching no row")

    bad = _roster_dict()
    bad["targets"]["A"]["spherex"]["table"] = "Photometry/SPHEREx/nope.csv"
    _expect(tmp_path, bad, "does not exist")


def test_ambiguous_declaration(tmp_path) -> None:
    roster = _roster_dict()
    path = _build_tree(tmp_path, roster)
    cat = tmp_path / "data" / "gal_a" / "Photometry" / "cat.csv"
    frame = pd.read_csv(cat)
    pd.concat([frame, frame.iloc[[0]]], ignore_index=True).to_csv(cat,
                                                                  index=False)
    with pytest.raises(ValueError, match="more than one row"):
        load_roster(path, REG)


def test_recipe_cross_checks(tmp_path) -> None:
    bad = _roster_dict()
    bad["recipes"]["tilt"]["sources"][1]["source"] = "nonexistent"
    _expect(tmp_path, bad, "exists on no target")

    bad = _roster_dict()
    bad["recipes"]["tilt"]["sources"][1]["bands"] = ["WISE_W2"]
    _expect(tmp_path, bad, "not declared by source")


# ------------------------------------
# Sample naming and the manifest path
# ------------------------------------

def _canonical_roster_dict() -> dict:
    """The same roster under the canonical key spellings."""
    raw = _roster_dict()
    raw["sample"] = raw.pop("cluster")
    raw["reference_redshift"] = raw.pop("cluster_redshift")
    raw["targets"]["A"]["z_ref_kind"] = "reference"
    return raw


def test_canonical_and_deprecated_spellings_agree(tmp_path) -> None:
    old = load_roster(_build_tree(tmp_path / "old", _roster_dict()), REG)
    new = load_roster(
        _build_tree(tmp_path / "new", _canonical_roster_dict()), REG)
    assert old.sample == new.sample == "TestCluster"
    assert old.reference_redshift == new.reference_redshift == 0.106
    assert old.targets["A"].z_ref_kind == new.targets["A"].z_ref_kind
    assert old.targets["A"].z_ref == new.targets["A"].z_ref


def test_both_spellings_of_one_field_is_an_error(tmp_path) -> None:
    raw = _canonical_roster_dict()
    raw["cluster"] = "TestCluster"
    _expect(tmp_path, raw, "declare only")


def test_reference_redshift_is_required_only_by_reference_targets(
        tmp_path) -> None:
    # Every target carries its own redshift, so the sample needs none.
    raw = _canonical_roster_dict()
    del raw["reference_redshift"]
    raw["targets"]["A"]["z_ref_kind"] = "spec"
    raw["targets"]["A"]["z_ref"] = 0.104
    roster = load_roster(_build_tree(tmp_path, raw), REG)
    assert roster.reference_redshift is None
    assert roster.targets["A"].z_ref == 0.104


def test_a_reference_target_without_a_reference_redshift_is_an_error(
        tmp_path) -> None:
    raw = _canonical_roster_dict()
    del raw["reference_redshift"]
    _expect(tmp_path, raw, "requires the roster to declare")


def test_manifest_path_defaults_under_data_root(tmp_path) -> None:
    roster = load_roster(_build_tree(tmp_path, _canonical_roster_dict()), REG)
    assert roster.manifest_path == roster.data_root / "sed_fitting/runs.jsonl"


def test_manifest_path_is_declarable(tmp_path) -> None:
    raw = _canonical_roster_dict()
    raw["manifest_path"] = "Galaxies/sed_fitting/runs.jsonl"
    roster = load_roster(_build_tree(tmp_path, raw), REG)
    assert roster.manifest_path == (roster.data_root
                                    / "Galaxies/sed_fitting/runs.jsonl")


def test_an_absolute_manifest_path_is_refused(tmp_path) -> None:
    raw = _canonical_roster_dict()
    raw["manifest_path"] = str(tmp_path / "elsewhere.jsonl")
    _expect(tmp_path, raw, "must be relative to data_root")


def test_a_missing_roster_names_the_path(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="roster not found"):
        load_roster(tmp_path / "nope.json", REG)
