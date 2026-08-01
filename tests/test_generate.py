from __future__ import annotations

import csv
import json

import pandas as pd
import pytest
from test_build import CAT_ROWS, RA, DEC

from sedfit.core.generate import (
    generate_roster,
    load_campaign,
    read_sample_catalog,
    write_roster,
)
from sedfit.core.registry import load_registry
from sedfit.core.roster import load_roster

REG = load_registry()

CAMPAIGN = {
    "schema_version": 1,
    "description": "test campaign",
    "cluster": "TestCluster",
    "data_root": "data",
    "cluster_redshift": 0.106,
    "position_authority": "test catalog",
    "sources": {
        "wise": {"path": "Photometry/{label}_cat.csv",
                 "bands": ["WISE_W1", "WISE_W2"], "kind": "catalog",
                 "provider": "unwise"},
        "legacy": {"path": "Photometry/{label}_cat.csv",
                   "bands": ["Legacy_z"], "kind": "catalog",
                   "provider": "legacy"},
        "galex": {"path": "Photometry/{prefix}_galex.csv",
                  "bands": ["GALEX_NUV"], "kind": "catalog",
                  "provider": "galex"},
    },
    "recipes": {
        "wise_only": {"reference": "anchors",
                      "sources": [{"source": "wise", "role": "anchor"},
                                  {"source": "legacy", "role": "stitch"}],
                      "spherex": None},
    },
}


def _write_cat(path, rows=CAT_ROWS):
    frame = pd.DataFrame(rows, columns=["band", "flux_uJy", "flux_err_uJy",
                                        "source"])
    frame["target_ra"] = RA
    frame["target_dec"] = DEC
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _tree(tmp_path, *, galaxies=(("gal_a", "A"), ("gal_b", "B"))):
    """A campaign dir beside a data tree, as the CLI resolves them.

    Each entry is (directory, label). The two differ so the fixture
    exercises the rule that the file stem follows the label, which the
    reader derives from the target name rather than the directory.
    """
    for gal, label in galaxies:
        _write_cat(tmp_path / "data" / gal / "Photometry" / f"{label}_cat.csv")
    campaign_path = tmp_path / "campaign.json"
    campaign_path.write_text(json.dumps(CAMPAIGN))
    return campaign_path


def _catalog(tmp_path, rows):
    path = tmp_path / "sample.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _row(name, dirname, kind="cluster", z_ref="", **extra):
    return {"name": name, "ra_deg": RA, "dec_deg": DEC, "z_ref_kind": kind,
            "z_ref": z_ref, "dir": dirname, "label": "", "notes": "",
            **extra}


# ------------------------------------
# The sample catalog
# ------------------------------------

def test_the_catalog_carries_only_per_target_facts(tmp_path) -> None:
    path = _catalog(tmp_path, [_row("A", "gal_a"),
                               _row("B", "gal_b", kind="spec", z_ref="0.108")])
    targets, _ = read_sample_catalog(path)
    assert [t["name"] for t in targets] == ["A", "B"]
    assert targets[0]["z_ref"] is None            # derived from the campaign
    assert targets[1]["z_ref"] == 0.108
    # One stem, defaulted off the name rather than the directory.
    assert targets[0]["label"] == "A"
    assert targets[0]["prefix"] == targets[0]["label"]


def test_the_label_default_follows_the_name(tmp_path) -> None:
    # dir unlike the name: the stem must follow the name, which is the
    # column the product filenames are derived from.
    path = _catalog(tmp_path, [_row("M 87", "Galaxies/m87_field"),
                               _row("B", "gal_b", label="named_stem")])
    targets, _ = read_sample_catalog(path)
    assert targets[0]["label"] == "M_87"          # sanitized, not "m87_field"
    assert targets[0]["dir"] == "Galaxies/m87_field"
    assert targets[1]["label"] == "named_stem"    # an explicit label wins


def test_the_catalog_accepts_either_position_spelling(tmp_path) -> None:
    path = tmp_path / "short.csv"
    path.write_text("name,ra,dec,z_ref_kind\nA,10.0,20.0,cluster\n",
                    encoding="utf-8")
    target = read_sample_catalog(path)[0][0]
    assert (target["ra_deg"], target["dec_deg"]) == (10.0, 20.0)


def test_the_catalog_names_the_columns_it_ignored(tmp_path) -> None:
    # An unrecognized column rides along, so naming it is what makes a
    # typo'd optional column visible next to the default it took.
    path = _catalog(tmp_path, [_row("A", "gal_a", priority="5", notes="x")])
    _, line = read_sample_catalog(path)
    assert "using name, ra_deg, dec_deg, z_ref_kind, z_ref, dir, label" in line
    assert "ignoring notes, priority" in line


def test_a_cluster_target_may_not_restate_its_redshift(tmp_path) -> None:
    path = _catalog(tmp_path, [_row("A", "gal_a", z_ref="0.106")])
    with pytest.raises(ValueError, match="leave z_ref blank"):
        read_sample_catalog(path)


def test_a_spec_target_must_supply_one(tmp_path) -> None:
    path = _catalog(tmp_path, [_row("A", "gal_a", kind="spec")])
    with pytest.raises(ValueError, match="requires a numeric z_ref"):
        read_sample_catalog(path)


def test_the_catalog_refuses_what_it_cannot_use(tmp_path) -> None:
    path = _catalog(tmp_path, [_row("A", "gal_a"), _row("A", "gal_b")])
    with pytest.raises(ValueError, match="duplicate target"):
        read_sample_catalog(path)

    half_position = tmp_path / "half.csv"
    half_position.write_text("name,ra_deg\nA,1.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no ra_deg/dec_deg"):
        read_sample_catalog(half_position)

    no_kind = tmp_path / "no_kind.csv"
    no_kind.write_text("name,ra_deg,dec_deg\nA,1.0,2.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required column"):
        read_sample_catalog(no_kind)

    kind = _catalog(tmp_path, [_row("A", "gal_a", kind="photometric")])
    with pytest.raises(ValueError, match="z_ref_kind"):
        read_sample_catalog(kind)


# ------------------------------------
# The campaign config
# ------------------------------------

def test_the_campaign_is_validated_up_front(tmp_path) -> None:
    path = tmp_path / "c.json"
    path.write_text(json.dumps(dict(CAMPAIGN, sources={
        "x": {"path": "p.csv", "bands": ["Legacy_z"], "kind": "catalog",
              "provider": "nonsense"}})))
    with pytest.raises(ValueError, match="unknown provider token"):
        load_campaign(path)

    path.write_text(json.dumps(dict(CAMPAIGN, schema_version=99)))
    with pytest.raises(ValueError, match="schema_version"):
        load_campaign(path)

    stray = dict(CAMPAIGN)
    stray["recipies"] = {}
    path.write_text(json.dumps(stray))
    with pytest.raises(ValueError):
        load_campaign(path)


# ------------------------------------
# Generation
# ------------------------------------

def _generate(tmp_path, rows, campaign=None):
    campaign_path = _tree(tmp_path)
    if campaign is not None:
        campaign_path.write_text(json.dumps(campaign))
    return generate_roster(read_sample_catalog(_catalog(tmp_path, rows))[0],
                           load_campaign(campaign_path), registry=REG,
                           campaign_dir=tmp_path)


def test_the_generated_roster_loads(tmp_path) -> None:
    roster, report = _generate(tmp_path, [_row("A", "gal_a"),
                                          _row("B", "gal_b")])
    out, report_path = write_roster(roster, report, tmp_path / "roster.json")

    loaded = load_roster(out, REG)
    assert sorted(loaded.targets) == ["A", "B"]
    assert loaded.targets["A"].z_ref == 0.106      # from reference_redshift
    assert loaded.sample == "TestCluster"
    assert report_path.exists()


def test_an_absent_source_is_omitted_not_declared(tmp_path) -> None:
    # The galex file is never written, so declaring it would produce a
    # roster that cannot load.
    roster, report = _generate(tmp_path, [_row("A", "gal_a")])
    assert sorted(roster["targets"]["A"]["sources"]) == ["legacy", "wise"]

    dropped = [r for r in report if r["status"] == "dropped"]
    assert any(r["source"] == "galex" for r in dropped)
    assert all("absent" in r["detail"] for r in dropped
               if r["source"] == "galex")
    write_roster(roster, report, tmp_path / "roster.json")
    assert load_roster(tmp_path / "roster.json", REG)


def test_a_band_no_row_supplies_is_dropped_from_its_source(tmp_path) -> None:
    campaign_path = _tree(tmp_path)
    # W2 is present for gal_a but not gal_b, so the two targets must end
    # up declaring different bands under the same source name.
    _write_cat(tmp_path / "data" / "gal_b" / "Photometry" / "B_cat.csv",
               rows=[r for r in CAT_ROWS if r[0] != "WISE_W2"])
    roster, report = generate_roster(
        read_sample_catalog(_catalog(tmp_path, [_row("A", "gal_a"),
                                                _row("B", "gal_b")]))[0],
        load_campaign(campaign_path), registry=REG, campaign_dir=tmp_path)

    assert roster["targets"]["A"]["sources"]["wise"]["bands"] == \
        ["WISE_W1", "WISE_W2"]
    assert roster["targets"]["B"]["sources"]["wise"]["bands"] == ["WISE_W1"]
    partial = [r for r in report if r["status"] == "partial"]
    assert any(r["target"] == "B" and "WISE_W2" in r["detail"]
               for r in partial)

    write_roster(roster, report, tmp_path / "roster.json")
    assert load_roster(tmp_path / "roster.json", REG)


def test_a_target_with_no_directory_is_reported_and_skipped(tmp_path) -> None:
    roster, report = _generate(tmp_path, [_row("A", "gal_a"),
                                          _row("GONE", "gal_missing")])
    assert list(roster["targets"]) == ["A"]
    assert any(r["target"] == "GONE" and "no directory" in r["detail"]
               for r in report)


def test_generation_fails_when_nothing_survives(tmp_path) -> None:
    with pytest.raises(ValueError, match="no target survived"):
        _generate(tmp_path, [_row("GONE", "gal_missing")])


def test_a_campaign_band_outside_the_registry_is_an_error(tmp_path) -> None:
    campaign = json.loads(json.dumps(CAMPAIGN))
    campaign["sources"]["wise"]["bands"] = ["WISE_W1", "NOT_A_BAND"]
    with pytest.raises(ValueError, match="not in the registry"):
        _generate(tmp_path, [_row("A", "gal_a")], campaign=campaign)


def test_path_templates_resolve_per_target(tmp_path) -> None:
    campaign = json.loads(json.dumps(CAMPAIGN))
    campaign["sources"] = {"wise": dict(CAMPAIGN["sources"]["wise"],
                                        path="Photometry/{name}_cat.csv")}
    campaign["recipes"] = {"wise_only": {
        "reference": "anchors",
        "sources": [{"source": "wise", "role": "anchor"}],
        "spherex": None}}
    # The file on disk is named for the LABEL ("A"), so a {name} template
    # on a target whose name differs must miss rather than silently match
    # something: the two fields are distinct and resolve independently.
    with pytest.raises(ValueError, match="no target survived"):
        _generate(tmp_path, [_row("A_renamed", "gal_a", label="A")],
                  campaign=campaign)


def test_an_unknown_template_field_is_named(tmp_path) -> None:
    campaign = json.loads(json.dumps(CAMPAIGN))
    campaign["sources"]["wise"]["path"] = "Photometry/{galaxy}_cat.csv"
    with pytest.raises(ValueError, match="unknown field"):
        _generate(tmp_path, [_row("A", "gal_a")], campaign=campaign)


def test_spherex_is_declared_only_where_its_table_exists(tmp_path) -> None:
    campaign = json.loads(json.dumps(CAMPAIGN))
    campaign["spherex"] = {"table": "Photometry/SPHEREx/{label}_sx.csv",
                           "model": "psf", "provenance": "synthetic"}
    campaign_path = _tree(tmp_path)
    campaign_path.write_text(json.dumps(campaign))
    sx = tmp_path / "data" / "gal_a" / "Photometry" / "SPHEREx"
    sx.mkdir(parents=True)
    (sx / "A_sx.csv").write_text("wave_um\n1.0\n")

    roster, report = generate_roster(
        read_sample_catalog(_catalog(tmp_path, [_row("A", "gal_a"),
                                                _row("B", "gal_b")]))[0],
        load_campaign(campaign_path), registry=REG, campaign_dir=tmp_path)

    assert "spherex" in roster["targets"]["A"]
    assert "spherex" not in roster["targets"]["B"]
    assert any(r["source"] == "spherex" and r["target"] == "B"
               and r["status"] == "dropped" for r in report)


def test_a_source_may_declare_several_filename_variants(tmp_path) -> None:
    # A tree assembled over time carries naming variants; losing a whole
    # source to one is worse than naming them.
    campaign = json.loads(json.dumps(CAMPAIGN))
    campaign["sources"] = {"wise": dict(
        CAMPAIGN["sources"]["wise"],
        path=["Photometry/{label}_missing.csv", "Photometry/{label}_cat.csv"])}
    campaign["recipes"] = {"wise_only": {
        "reference": "anchors",
        "sources": [{"source": "wise", "role": "anchor"}],
        "spherex": None}}
    roster, report = _generate(tmp_path, [_row("A", "gal_a")],
                               campaign=campaign)
    assert roster["targets"]["A"]["sources"]["wise"]["path"] == \
        "Photometry/A_cat.csv"
    write_roster(roster, report, tmp_path / "roster.json")
    assert load_roster(tmp_path / "roster.json", REG)


def test_every_variant_is_named_when_none_exists(tmp_path) -> None:
    campaign = json.loads(json.dumps(CAMPAIGN))
    campaign["sources"]["galex"]["path"] = ["Photometry/{label}_x.csv",
                                            "Photometry/{label}_y.csv"]
    _, report = _generate(tmp_path, [_row("A", "gal_a")], campaign=campaign)
    detail = next(r["detail"] for r in report if r["source"] == "galex")
    assert "A_x.csv" in detail and "A_y.csv" in detail


def test_a_variant_wins_by_supplying_a_band_not_by_existing(tmp_path) -> None:
    # Both files exist. The first carries no WISE row, so stopping at the
    # first that EXISTS would drop the source; the second supplies it.
    campaign = json.loads(json.dumps(CAMPAIGN))
    campaign["sources"] = {"wise": dict(
        CAMPAIGN["sources"]["wise"],
        path=["Photometry/{label}_other.csv", "Photometry/{label}_cat.csv"])}
    campaign["recipes"] = {"wise_only": {
        "reference": "anchors",
        "sources": [{"source": "wise", "role": "anchor"}],
        "spherex": None}}
    campaign_path = _tree(tmp_path)
    campaign_path.write_text(json.dumps(campaign))
    _write_cat(tmp_path / "data" / "gal_a" / "Photometry" / "A_other.csv",
               rows=[r for r in CAT_ROWS if not r[0].startswith("WISE")])

    roster, report = generate_roster(
        read_sample_catalog(_catalog(tmp_path, [_row("A", "gal_a")]))[0],
        load_campaign(campaign_path), registry=REG, campaign_dir=tmp_path)
    assert roster["targets"]["A"]["sources"]["wise"]["path"] == \
        "Photometry/A_cat.csv"
    assert next(r for r in report if r["source"] == "wise")["status"] == "kept"


def test_a_dropped_source_names_what_each_variant_failed_on(tmp_path) -> None:
    campaign = json.loads(json.dumps(CAMPAIGN))
    campaign["sources"]["galex"]["path"] = ["Photometry/{label}_nogalex.csv",
                                            "Photometry/{label}_gone.csv"]
    campaign_path = _tree(tmp_path)
    campaign_path.write_text(json.dumps(campaign))
    _write_cat(tmp_path / "data" / "gal_a" / "Photometry" / "A_nogalex.csv",
               rows=[r for r in CAT_ROWS if not r[0].startswith("GALEX")])

    _, report = generate_roster(
        read_sample_catalog(_catalog(tmp_path, [_row("A", "gal_a")]))[0],
        load_campaign(campaign_path), registry=REG, campaign_dir=tmp_path)
    detail = next(r["detail"] for r in report if r["source"] == "galex")
    # The present-but-wrong file and the absent one are distinguished, so
    # the report says which kind of failure each variant was.
    assert "nogalex.csv: no declared band" in detail
    assert "gone.csv: absent" in detail


def test_one_stem_names_a_target_everywhere(tmp_path) -> None:
    # A named label drives both the source filenames the generator looks
    # for and the roster prefix that built tables are written under.
    path = _catalog(tmp_path, [dict(_row("A", "gal_a"), label="custom")])
    target = read_sample_catalog(path)[0][0]
    assert target["label"] == "custom"
    assert target["prefix"] == "custom"


def _spherex_campaign(pattern):
    campaign = json.loads(json.dumps(CAMPAIGN))
    campaign["spherex"] = {"table": pattern, "model": "sersic",
                           "provenance": "sedphot spherex"}
    return campaign


def _sx_dir(tmp_path, gal):
    d = tmp_path / "data" / gal / "Photometry" / "SPHEREx"
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_the_spherex_table_is_matched_by_pattern(tmp_path) -> None:
    # The tag hashes the extraction config, so the name differs per
    # galaxy and cannot be written down in advance.
    campaign_path = _tree(tmp_path)
    campaign_path.write_text(json.dumps(
        _spherex_campaign("Photometry/SPHEREx/table_photometry.*.csv")))
    (_sx_dir(tmp_path, "gal_a") / "table_photometry.sersic-a1b2c3.csv"
     ).write_text("wave_um\n1.0\n")
    (_sx_dir(tmp_path, "gal_b") / "table_photometry.sersic-9f8e7d.csv"
     ).write_text("wave_um\n1.0\n")

    roster, _ = generate_roster(
        read_sample_catalog(_catalog(tmp_path, [_row("A", "gal_a"),
                                                _row("B", "gal_b")]))[0],
        load_campaign(campaign_path), registry=REG, campaign_dir=tmp_path)
    assert roster["targets"]["A"]["spherex"]["table"] == \
        "Photometry/SPHEREx/table_photometry.sersic-a1b2c3.csv"
    assert roster["targets"]["B"]["spherex"]["table"] == \
        "Photometry/SPHEREx/table_photometry.sersic-9f8e7d.csv"


def test_several_extractions_refuse_rather_than_pick(tmp_path) -> None:
    # A galaxy with more than one extraction on disk is exactly where a
    # silent choice would bury which one a flux came from.
    campaign_path = _tree(tmp_path)
    campaign_path.write_text(json.dumps(
        _spherex_campaign("Photometry/SPHEREx/table_photometry.*.csv")))
    for tag in ("psf-111111", "sersic-222222"):
        (_sx_dir(tmp_path, "gal_a") / f"table_photometry.{tag}.csv"
         ).write_text("wave_um\n1.0\n")

    with pytest.raises(ValueError, match="matches 2 tables"):
        generate_roster(
            read_sample_catalog(_catalog(tmp_path, [_row("A", "gal_a")]))[0],
            load_campaign(campaign_path), registry=REG, campaign_dir=tmp_path)


def test_patterns_are_tried_in_order(tmp_path) -> None:
    # A tree mid-migration carries both the tagged convention and the
    # older bare name; the campaign states which it prefers.
    campaign_path = _tree(tmp_path)
    campaign_path.write_text(json.dumps(_spherex_campaign([
        "Photometry/SPHEREx/table_photometry.*.csv",
        "Photometry/SPHEREx/table_photometry.csv"])))
    (_sx_dir(tmp_path, "gal_a") / "table_photometry.csv").write_text("w\n1\n")
    (_sx_dir(tmp_path, "gal_b") / "table_photometry.sersic-abc123.csv"
     ).write_text("w\n1\n")

    roster, report = generate_roster(
        read_sample_catalog(_catalog(tmp_path, [_row("A", "gal_a"),
                                                _row("B", "gal_b")]))[0],
        load_campaign(campaign_path), registry=REG, campaign_dir=tmp_path)
    assert roster["targets"]["A"]["spherex"]["table"].endswith(
        "table_photometry.csv")
    assert roster["targets"]["B"]["spherex"]["table"].endswith(
        "sersic-abc123.csv")
    assert all(r["status"] == "kept" for r in report if r["source"] == "spherex")


def test_a_target_with_no_extraction_is_reported(tmp_path) -> None:
    campaign_path = _tree(tmp_path)
    campaign_path.write_text(json.dumps(
        _spherex_campaign("Photometry/SPHEREx/table_photometry.*.csv")))
    _, report = generate_roster(
        read_sample_catalog(_catalog(tmp_path, [_row("A", "gal_a")]))[0],
        load_campaign(campaign_path), registry=REG, campaign_dir=tmp_path)
    row = next(r for r in report if r["source"] == "spherex")
    assert row["status"] == "dropped" and "no match" in row["detail"]


def test_a_wholly_failed_generation_names_the_reason(tmp_path) -> None:
    # The report dies with the exception, so the message has to carry
    # why -- a data_root pointing somewhere real but wrong looks exactly
    # like a tree with no photometry in it.
    campaign_path = _tree(tmp_path)
    with pytest.raises(ValueError, match="no directory"):
        _generate(tmp_path, [_row("GONE", "nowhere"), _row("ALSO", "neither")])
    try:
        _generate(tmp_path, [_row("GONE", "nowhere")])
    except ValueError as err:
        assert str(tmp_path / "data") in str(err)     # the data_root tried
