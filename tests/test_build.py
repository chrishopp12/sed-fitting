from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sedfit.core.build import build_target, write_build
from sedfit.core.registry import load_registry
from sedfit.core.roster import load_roster
from sedfit.core.synth import synth_fnu
from sedfit.core.table import read_sed_table

REG = load_registry()
RA, DEC = 10.0, 20.0

CAT_ROWS = [
    ("WISE_W1", 100.0, 1.0, "unWISE_Legacy_DR9"),
    ("WISE_W2", 200.0, 2.0, "unWISE_Legacy_DR9"),
    ("Legacy_z", 25.0, 0.5, "Legacy_DR9"),
    ("GALEX_NUV", 5.0, 0.5, "GALEX_GUVcat_AIS"),
]

SOURCES = {
    "wise1": {"path": "Photometry/cat.csv", "bands": ["WISE_W1"],
              "kind": "catalog", "provider": "unwise"},
    "wise2": {"path": "Photometry/cat.csv", "bands": ["WISE_W2"],
              "kind": "catalog", "provider": "unwise"},
    "wise_dup": {"path": "Photometry/cat.csv", "bands": ["WISE_W1"],
                 "kind": "catalog", "provider": "unwise"},
    "legacy": {"path": "Photometry/cat.csv", "bands": ["Legacy_z"],
               "kind": "catalog", "provider": "legacy"},
    "galex": {"path": "Photometry/cat.csv", "bands": ["GALEX_NUV"],
              "kind": "catalog", "provider": "galex"},
}

RECIPES = {
    "tilt2": {"reference": "anchors",
              "sources": [{"source": "wise1", "role": "anchor"},
                          {"source": "wise2", "role": "anchor"},
                          {"source": "legacy", "role": "stitch"},
                          {"source": "galex", "role": "float"}],
              "spherex": {}},
    "one_anchor": {"reference": "anchors",
                   "sources": [{"source": "wise2", "role": "anchor"},
                               {"source": "galex", "role": "float"}],
                   "spherex": {}},
    "spherex_ref": {"reference": "spherex",
                    "sources": [{"source": "legacy", "role": "stitch"},
                                {"source": "wise1", "role": "stitch"},
                                {"source": "galex", "role": "float"}],
                    "spherex": {}},
    "none_ref": {"reference": "none",
                 "sources": [{"source": "legacy", "role": "float"},
                             {"source": "galex", "role": "float"}],
                 "spherex": None},
    "no_block": {"reference": "none",
                 "sources": [{"source": "galex", "role": "float"}]},
    "dup": {"reference": "anchors",
            "sources": [{"source": "wise1", "role": "anchor"},
                        {"source": "wise_dup", "role": "anchor"}],
            "spherex": {}},
    "stitch_galex": {"reference": "spherex",
                     "sources": [{"source": "galex", "role": "stitch"}],
                     "spherex": {}},
    "raw_spherex": {"reference": "spherex", "sources": [], "spherex": {}},
}


def _visits() -> pd.DataFrame:
    lam = np.arange(0.80, 6.001, 0.05)
    return pd.DataFrame({"lambda": lam, "lambda_width": 0.02,
                         "flux": 100.0, "flux_err": 1.0, "flags": 1 << 21})


def _tree(tmp_path: Path) -> Path:
    for gal in ("gal_a", "gal_b"):
        phot = tmp_path / "data" / gal / "Photometry"
        phot.mkdir(parents=True, exist_ok=True)
        cat = pd.DataFrame(CAT_ROWS, columns=["band", "flux_uJy",
                                              "flux_err_uJy", "source"])
        cat["target_ra"] = RA
        cat["target_dec"] = DEC
        cat.to_csv(phot / "cat.csv", index=False)
    spherex_dir = tmp_path / "data" / "gal_a" / "Photometry" / "SPHEREx"
    spherex_dir.mkdir(parents=True)
    _visits().to_csv(spherex_dir / "visits.csv", index=False)

    roster = {
        "schema_version": 2,
        "cluster": "TestCluster",
        "data_root": "data",
        "cluster_redshift": 0.106,
        "targets": {
            "A": {"position": {"ra_deg": RA, "dec_deg": DEC, "frame": "icrs",
                               "authority": "test"},
                  "z_ref_kind": "cluster", "dir": "gal_a", "prefix": "galA",
                  "spherex": {"table": "Photometry/SPHEREx/visits.csv",
                              "model": "psf", "provenance": "synthetic"},
                  "sources": SOURCES},
            "B": {"position": {"ra_deg": RA, "dec_deg": DEC, "frame": "icrs",
                               "authority": "test"},
                  "z_ref": 0.108, "z_ref_kind": "spec",
                  "dir": "gal_b", "prefix": "galB",
                  "sources": SOURCES},
        },
        "recipes": RECIPES,
    }
    path = tmp_path / "roster.json"
    path.write_text(json.dumps(roster))
    return path


def _synth_band(band: str, frame: pd.DataFrame) -> float:
    spherex = frame[[REG.is_spherex(b) for b in frame["band"]]]
    wave, thru = REG.get_bandpass(band)
    fnu, _ = synth_fnu(wave, thru, spherex["wave_um"].to_numpy(),
                       spherex["flux_uJy"].to_numpy())
    return fnu


def test_tilt_build(tmp_path) -> None:
    roster = load_roster(_tree(tmp_path), REG)
    result = build_target(roster, "A", "tilt2", registry=REG)
    frame = result.frame

    by_band = frame.set_index("band")
    assert by_band.loc["WISE_W1", "flux_uJy"] == 100.0
    assert by_band.loc["WISE_W2", "flux_uJy"] == 200.0
    assert by_band.loc["GALEX_NUV", "flux_uJy"] == 5.0

    spherex = frame[[REG.is_spherex(b) for b in frame["band"]]]
    ratio = (spherex["flux_err_uJy"] / spherex["flux_uJy"]).to_numpy()
    assert ratio == pytest.approx(np.full_like(ratio, 0.01), rel=1e-9)

    assert _synth_band("WISE_W1", frame) == pytest.approx(100.0, rel=0.02)
    assert _synth_band("WISE_W2", frame) == pytest.approx(200.0, rel=0.02)

    tilt = result.sidecar["reference"]["tilt"]
    assert tilt["blue"] == "WISE_W1" and tilt["red"] == "WISE_W2"
    assert tilt["factor_min"] < 1.0 < tilt["factor_max"]

    stitch = result.sidecar["stitch"]
    assert len(stitch) == 1 and stitch[0]["instrument"] == "Legacy"
    scale = stitch[0]["scale"]
    assert by_band.loc["Legacy_z", "flux_uJy"] == pytest.approx(25.0 * scale)
    assert by_band.loc["Legacy_z", "flux_err_uJy"] == pytest.approx(0.5 * scale)

    out = write_build(result, REG)
    assert out.name == "galA_sed_tilt2.csv"
    back = read_sed_table(out, REG)
    assert len(back) == len(frame)
    sidecar = json.loads((out.parent / "galA_sed_tilt2.provenance.json")
                         .read_text())
    assert sidecar["z_ref"] == pytest.approx(0.106)
    assert set(sidecar["registry"]["bands"]) == {"WISE_W1", "WISE_W2",
                                                 "Legacy_z", "GALEX_NUV"}
    assert sidecar["spherex"]["counts"]["visits_kept"] == 105


def test_one_anchor_rescale(tmp_path) -> None:
    roster = load_roster(_tree(tmp_path), REG)
    result = build_target(roster, "A", "one_anchor", registry=REG)
    spherex = result.frame[[REG.is_spherex(b) for b in result.frame["band"]]]
    assert spherex["flux_uJy"].to_numpy() == pytest.approx(
        np.full(len(spherex), 200.0), rel=1e-9)
    assert spherex["flux_err_uJy"].to_numpy() == pytest.approx(
        np.full(len(spherex), 2.0), rel=1e-9)
    assert spherex["scatter_uJy"].to_numpy() == pytest.approx(
        np.full(len(spherex), 2.0), rel=1e-9)
    assert result.sidecar["reference"]["anchors"][0]["c"] == pytest.approx(2.0)


def test_spherex_reference_scales_errors(tmp_path) -> None:
    roster = load_roster(_tree(tmp_path), REG)
    result = build_target(roster, "A", "spherex_ref", registry=REG)
    by_band = result.frame.set_index("band")
    assert by_band.loc["Legacy_z", "flux_uJy"] == pytest.approx(100.0, rel=1e-6)
    assert by_band.loc["Legacy_z", "flux_err_uJy"] == pytest.approx(2.0, rel=1e-6)
    assert by_band.loc["WISE_W1", "flux_uJy"] == pytest.approx(100.0, rel=1e-6)
    spherex = result.frame[[REG.is_spherex(b) for b in result.frame["band"]]]
    assert spherex["flux_uJy"].to_numpy() == pytest.approx(
        np.full(len(spherex), 100.0))


def test_raw_spherex_only(tmp_path) -> None:
    roster = load_roster(_tree(tmp_path), REG)
    result = build_target(roster, "A", "raw_spherex", registry=REG)
    frame = result.frame

    assert all(REG.is_spherex(b) for b in frame["band"])
    assert result.sidecar["n_broadband"] == 0
    assert result.sidecar["sources"] == []
    assert result.sidecar["reference"] == {"kind": "spherex"}
    assert result.sidecar["registry"]["bands"] == {}

    # Raw frame: the ingested spectrum passes through untouched.
    ratio = (frame["flux_err_uJy"] / frame["flux_uJy"]).to_numpy()
    assert ratio == pytest.approx(np.full_like(ratio, 0.01), rel=1e-9)
    assert frame["flux_uJy"].to_numpy() == pytest.approx(
        np.full(len(frame), 100.0), rel=1e-9)

    out = write_build(result, REG)
    assert out.name == "galA_sed_raw_spherex.csv"
    back = read_sed_table(out, REG)
    assert len(back) == len(frame)


def test_none_reference_broadband_only(tmp_path) -> None:
    roster = load_roster(_tree(tmp_path), REG)
    result = build_target(roster, "B", "none_ref", registry=REG)
    assert list(result.frame["band"]) == ["Legacy_z", "GALEX_NUV"]
    assert result.frame["flux_uJy"].to_numpy() == pytest.approx([25.0, 5.0])
    assert result.sidecar["spherex"] is None


def test_applicability_errors(tmp_path) -> None:
    roster = load_roster(_tree(tmp_path), REG)
    with pytest.raises(ValueError, match="no spherex block"):
        build_target(roster, "A", "no_block", registry=REG)
    with pytest.raises(ValueError, match="declares no SPHEREx table"):
        build_target(roster, "B", "tilt2", registry=REG)


def test_duplicate_band_across_entries(tmp_path) -> None:
    roster = load_roster(_tree(tmp_path), REG)
    with pytest.raises(ValueError, match="consumed by both"):
        build_target(roster, "A", "dup", registry=REG)


def test_stitch_without_coverage(tmp_path) -> None:
    roster = load_roster(_tree(tmp_path), REG)
    with pytest.raises(ValueError, match="declare it float"):
        build_target(roster, "A", "stitch_galex", registry=REG)


def test_position_cross_check(tmp_path) -> None:
    path = _tree(tmp_path)
    roster = load_roster(path, REG)
    cat = tmp_path / "data" / "gal_a" / "Photometry" / "cat.csv"
    frame = pd.read_csv(cat)
    frame["target_ra"] = RA + 1.0 / 60.0
    frame.to_csv(cat, index=False)
    with pytest.raises(ValueError, match="arcsec"):
        build_target(roster, "A", "tilt2", registry=REG)


# ------------------------------------
# qa_flags carry + extinction cross-check
# ------------------------------------
def _qa_tree(tmp_path: Path, *, dered_states=None) -> Path:
    phot = tmp_path / "data" / "gal_q" / "Photometry"
    phot.mkdir(parents=True)
    cat = pd.DataFrame(
        [("CFHT_r", 50.0, 0.5, "sedphot_aperture_scene_ivm",
          "cov=1.000;maskfrac=0.031;conv=-1;atbound=2"),
         ("CFHT_i", 60.0, 0.6, "sedphot_aperture_scene_ivm",
          "cov=1.000;maskfrac=0.010;conv=25"),
         ("Legacy_z", 25.0, 0.5, "Legacy_DR9", "3")],
        columns=["band", "flux_uJy", "flux_err_uJy", "source", "flags"])
    cat["target_ra"] = RA
    cat["target_dec"] = DEC
    if dered_states is not None:
        cat["dered_applied"] = dered_states
    cat.to_csv(phot / "cat.csv", index=False)

    roster = {
        "schema_version": 2,
        "cluster": "TestCluster",
        "data_root": "data",
        "cluster_redshift": 0.106,
        "targets": {
            "Q": {"position": {"ra_deg": RA, "dec_deg": DEC,
                               "frame": "icrs", "authority": "test"},
                  "z_ref_kind": "cluster", "dir": "gal_q", "prefix": "galQ",
                  "sources": {
                      "measured": {"path": "Photometry/cat.csv",
                                   "bands": ["CFHT_r", "CFHT_i"],
                                   "kind": "measured",
                                   "provider": "aperture: scene r=12"},
                      "legacy": {"path": "Photometry/cat.csv",
                                 "bands": ["Legacy_z"],
                                 "kind": "catalog", "provider": "legacy"},
                  }},
        },
        "recipes": {"floats": {"reference": "none",
                               "sources": [{"source": "measured",
                                            "role": "float"},
                                           {"source": "legacy",
                                            "role": "float"}],
                               "spherex": None}},
    }
    path = tmp_path / "roster.json"
    path.write_text(json.dumps(roster))
    return path


def test_qa_flags_carried_for_measured_providers_only(tmp_path) -> None:
    roster = load_roster(_qa_tree(tmp_path), REG)
    frame = build_target(roster, "Q", "floats", registry=REG).frame
    by_band = frame.set_index("band")
    assert by_band.loc["CFHT_r", "qa_flags"] \
        == "cov=1.000;maskfrac=0.031;conv=-1;atbound=2"
    assert by_band.loc["CFHT_i", "qa_flags"] == "cov=1.000;maskfrac=0.010;conv=25"
    # catalog-provider rows never carry archive flags
    assert pd.isna(by_band.loc["Legacy_z", "qa_flags"])


def test_qa_flags_absent_column_tolerated(tmp_path) -> None:
    roster_path = _qa_tree(tmp_path)
    cat_path = tmp_path / "data" / "gal_q" / "Photometry" / "cat.csv"
    frame = pd.read_csv(cat_path).drop(columns=["flags"])
    frame.to_csv(cat_path, index=False)
    roster = load_roster(roster_path, REG)
    built = build_target(roster, "Q", "floats", registry=REG).frame
    assert built["qa_flags"].isna().all()


def test_dered_cross_checks(tmp_path) -> None:
    # double correction: dereddening build over already-dereddened rows
    roster = load_roster(_qa_tree(tmp_path, dered_states=[True] * 3), REG)
    with pytest.raises(ValueError, match="already-dereddened"):
        build_target(roster, "Q", "floats", registry=REG, mw_ebv=0.02)

    # mixed extinction states in one table
    roster = load_roster(
        _qa_tree(tmp_path / "b", dered_states=[False, False, True]), REG)
    with pytest.raises(ValueError, match="common extinction scale"):
        build_target(roster, "Q", "floats", registry=REG)

    # uniform as-measured states build fine
    roster = load_roster(
        _qa_tree(tmp_path / "c", dered_states=[False] * 3), REG)
    frame = build_target(roster, "Q", "floats", registry=REG).frame
    assert len(frame) == 3
