from __future__ import annotations

import pytest

from sedfit.core.recipe import DEFAULT_MIN_COVERAGE, parse_recipe
from sedfit.core.spherex import DEFAULT_FLAGS_REJECT


def _tilt() -> dict:
    return {
        "reference": "anchors",
        "sources": [
            {"source": "legacy", "role": "anchor"},
            {"source": "wise", "role": "anchor"},
            {"source": "jplus", "role": "stitch", "bands": ["JPLUS_gSDSS"]},
        ],
        "spherex": {"cuts": {}, "binning": {"split_um": None}},
    }


def test_parse_defaults() -> None:
    recipe = parse_recipe("tilt", _tilt())
    assert recipe.reference == "anchors"
    assert [e.role for e in recipe.sources] == ["anchor", "anchor", "stitch"]
    assert recipe.spherex_declared
    assert recipe.spherex.cuts.flags_reject == DEFAULT_FLAGS_REJECT
    assert recipe.spherex.cuts.fit_ql_max is None
    assert not recipe.spherex.cuts.drop_local_bkg
    assert recipe.spherex.binning.split_um is None
    assert recipe.spherex.binning.blue_merge_dlam_um == 0.001
    assert recipe.spherex.binning.red_bin_resolution == 1.0
    assert recipe.min_coverage == DEFAULT_MIN_COVERAGE


def test_spherex_tristate() -> None:
    explicit_null = _tilt()
    explicit_null["spherex"] = None
    recipe = parse_recipe("r", explicit_null)
    assert recipe.spherex_declared and recipe.spherex is None

    absent = _tilt()
    del absent["spherex"]
    recipe = parse_recipe("r", absent)
    assert not recipe.spherex_declared and recipe.spherex is None


def _expect(raw: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        parse_recipe("r", raw)


def test_role_legality() -> None:
    three = _tilt()
    three["sources"].append({"source": "extra", "role": "anchor"})
    _expect(three, "one or two anchor")

    none_anchor = _tilt()
    none_anchor["sources"] = [{"source": "legacy", "role": "stitch"}]
    _expect(none_anchor, "one or two anchor")

    spherex_anchor = _tilt()
    spherex_anchor["reference"] = "spherex"
    _expect(spherex_anchor, "only legal under")

    none_stitch = {"reference": "none",
                   "sources": [{"source": "legacy", "role": "stitch"}]}
    _expect(none_stitch, "illegal under reference 'none'")

    float_anywhere = {"reference": "none",
                      "sources": [{"source": "legacy", "role": "float"}]}
    parse_recipe("r", float_anywhere)


def test_empty_sources() -> None:
    raw = {"reference": "spherex", "sources": [], "spherex": {}}
    recipe = parse_recipe("raw", raw)
    assert recipe.sources == ()
    assert recipe.reference == "spherex"
    assert recipe.spherex_declared

    _expect({"reference": "anchors", "sources": []},
            "only legal under reference 'spherex'")
    _expect({"reference": "none", "sources": []},
            "only legal under reference 'spherex'")


def test_strict_keys() -> None:
    bad = _tilt()
    bad["error_policy"] = {"default": 0.05}
    _expect(bad, "unknown keys")

    bad = _tilt()
    bad["sources"][0]["truth"] = True
    _expect(bad, "unknown keys")

    bad = _tilt()
    bad["spherex"]["cuts"] = {"flags": 0}
    _expect(bad, "unknown keys")

    bad = _tilt()
    bad["spherex"]["binning"] = {"split": 2.4}
    _expect(bad, "unknown keys")


def test_value_rules() -> None:
    bad = _tilt()
    bad["sources"].append({"source": "legacy", "role": "float"})
    _expect(bad, "more than once")

    bad = _tilt()
    bad["sources"][2]["bands"] = ["JPLUS_gSDSS"]
    bad["sources"][0]["bands"] = ["JPLUS_gSDSS"]
    _expect(bad, "more than one source entry")

    bad = _tilt()
    bad["min_coverage"] = 1.2
    _expect(bad, "min_coverage")

    bad = _tilt()
    bad["min_coverage"] = 0.0
    _expect(bad, "min_coverage")

    bad = _tilt()
    bad["spherex"]["cuts"] = {"flags_reject": -1}
    _expect(bad, "flags_reject")

    bad = _tilt()
    bad["spherex"]["binning"] = {"split_um": -2.4}
    _expect(bad, "split_um")

    bad = _tilt()
    bad["sources"][2]["bands"] = []
    _expect(bad, "non-empty list")

    bad = _tilt()
    bad["reference"] = "tilt"
    _expect(bad, "not in")
