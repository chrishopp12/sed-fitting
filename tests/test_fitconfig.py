from __future__ import annotations

import json
from pathlib import Path

import pytest

from sedfit.core import fitconfig
from sedfit.core.fitconfig import (
    hash_projection,
    parse_fit_config,
    resolve_config,
)
from sedfit.core.provenance import canonical_json, run_id

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


SSP = str(Path(__file__).resolve().parents[1]
          / "sedfit" / "data" / "templates" / "ssp_miles")


def _linear_block(**overrides) -> dict:
    """A minimally valid linear block: every LINEAR_REQUIRED field, no more."""
    block = {"templates": SSP, "template_wave_range": [3600.0, 7400.0],
             "z_min": 0.264, "z_max": 0.274, "poly_wave_frame": "air",
             "spectrum": {"file": "spec.csv"}}
    block.update(overrides)
    return block


def _linear(**overrides) -> dict:
    linear = _linear_block(**overrides.pop("linear", {}))
    raw = {"schema_version": 2, "backend": "linear", "name": "x"}
    raw.update(overrides)
    raw["linear"] = linear
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


def test_backend_exclusion_scales_past_two() -> None:
    """Every non-selected backend block is rejected, not just one.

    With two backends the old "the other one" formulation was correct, so
    this pins the property rather than today's behavior: a third name must
    not slip a stray block through to be silently stripped.
    """
    _fails(_eazy(linear=_linear_block()), "illegal under")
    with pytest.raises(ValueError, match="illegal under") as excinfo:
        parse_fit_config(_eazy(prospector={"stellar_library": "miles"},
                               linear=_linear_block()))
    assert "'prospector'" in str(excinfo.value)
    assert "'linear'" in str(excinfo.value)


def test_backend_without_a_parser_raises(monkeypatch) -> None:
    """A backend in BACKENDS with no parser errors instead of misrouting.

    The dispatch used to end in a bare `else` that meant prospector, so an
    unparsed backend was told it needed a prospector block.
    """
    monkeypatch.setattr(fitconfig, "BACKENDS",
                        fitconfig.BACKENDS + ("nonesuch",))
    monkeypatch.setitem(fitconfig.USES_PHOTOMETRY, "nonesuch", True)
    _fails({"schema_version": 2, "backend": "nonesuch", "name": "x"},
           "has no parser")


def test_backend_without_a_photometry_declaration_raises(monkeypatch) -> None:
    """A backend that does not say whether it reads photometry errors.

    Without the declaration the shared photometry-only fields would
    materialize to their defaults and enter the run identity unremarked.
    """
    monkeypatch.setattr(fitconfig, "BACKENDS",
                        fitconfig.BACKENDS + ("nonesuch",))
    _fails({"schema_version": 2, "backend": "nonesuch", "name": "x"},
           "does not declare USES_PHOTOMETRY")


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


def test_linear_defaults() -> None:
    block = parse_fit_config(_linear())["linear"]
    assert block["template_pattern"] == "*.dat"
    assert block["template_dv_kms"] == 15.0
    assert block["z_step"] == 4.0e-4
    assert (block["sigma_min"], block["sigma_max"]) == (50.0, 400.0)
    assert block["poly_order"] == 8 and block["n_poly_iter"] == 4
    assert block["clip_sigma"] == 4.0
    assert block["error_method"] == "hessian"
    # null means "decide from what is loaded", not "off"
    assert block["normalize_range"] is None and block["poly_domain"] is None
    assert block["spectrum"]["err_floor"] == 0.0
    assert block["spectrum"]["mask_windows"] is None


def test_linear_requires_the_fields_a_wrong_default_would_hide() -> None:
    """Each of these is silent when wrong, so none of them may default."""
    for field in ("templates", "template_wave_range", "z_min", "z_max",
                  "poly_wave_frame", "spectrum"):
        bare = _linear_block()
        del bare[field]
        _fails({"schema_version": 2, "backend": "linear", "name": "x",
                "linear": bare}, f"missing required keys.*{field}")
    _fails({"schema_version": 2, "backend": "linear", "name": "x"},
           "requires a linear block")


def test_linear_conditionals() -> None:
    _fails(_linear(linear={"templates": "  "}), "templates is required")
    _fails(_linear(linear={"template_pattern": " "}), "non-empty glob")
    _fails(_linear(linear={"template_wave_range": [7400.0, 3600.0]}),
           "must be below")
    _fails(_linear(linear={"z_min": 0.3}), "z_min 0.3 >= z_max")
    _fails(_linear(linear={"sigma_min": 500.0}), "sigma_min 500.0 >= sigma_max")
    _fails(_linear(linear={"poly_order": -1}), "integer >= 0")
    _fails(_linear(linear={"n_poly_iter": 0}), "integer >= 1")
    _fails(_linear(linear={"poly_wave_frame": "observed"}), "not in")
    _fails(_linear(linear={"error_method": "delta_chi2"}), "not in")
    _fails(_linear(linear={"spectrum": {"file": "s.csv", "polyorder": 4}}),
           "unknown keys")
    # normalize_range must lie inside the loaded range, or the basis would
    # only find that out mid-fit
    _fails(_linear(linear={"normalize_range": [3500.0, 7350.0]}),
           "must lie inside template_wave_range")
    inside = parse_fit_config(
        _linear(linear={"normalize_range": [3750.0, 7350.0]}))
    assert inside["linear"]["normalize_range"] == [3750.0, 7350.0]


def test_a_spectrum_only_backend_refuses_photometry_fields() -> None:
    """Writing one is an error; not writing one leaves no trace in identity."""
    for field, value in (("bands_include", ["W1"]), ("min_valid_bands", 5),
                         ("min_snr_broadband", 2.0), ("err_floor", 0.05),
                         ("qa_gates", {"maskfrac": {"max": 0.5}})):
        _fails(_linear(**{field: value}), "never reads")

    cfg = parse_fit_config(_linear())
    assert all(cfg[f] is None for f in fitconfig.PHOTOMETRY_ONLY_TOP)
    # mu_lensing is not photometry-only: it divides a spectrum's flux too
    assert cfg["mu_lensing"] == 1.0
    resolved = resolve_config(cfg, target_z_ref=0.269)
    projected = json.loads(canonical_json(hash_projection(resolved)))
    assert not (set(projected) & set(fitconfig.PHOTOMETRY_ONLY_TOP))
    # the nested one is a different field and stays
    assert projected["linear"]["spectrum"]["err_floor"] == 0.0


def test_linear_identity_drops_machine_paths(tmp_path) -> None:
    """The basis directory and the spectrum path leave the identity hash."""
    resolved = resolve_config(parse_fit_config(_linear()), target_z_ref=0.269)
    projected = hash_projection(resolved, digests={"templates": {"a": "0" * 16}})
    assert "templates" not in projected["linear"]
    assert "template_pattern" not in projected["linear"]
    assert "file" not in projected["linear"]["spectrum"]

    moved = resolve_config(
        parse_fit_config(_linear(linear={"templates": str(tmp_path)})),
        target_z_ref=0.269)
    digests = {"templates": {"a": "0" * 16}}
    assert (run_id(hash_projection(moved, digests=digests), "p" * 64, "q" * 64)
            == run_id(hash_projection(resolved, digests=digests),
                      "p" * 64, "q" * 64))


def test_a_relative_linear_spectrum_resolves_against_the_config(
        tmp_path) -> None:
    config = tmp_path / "fit.json"
    config.write_text(json.dumps(_linear()))
    loaded = fitconfig.load_fit_config(config)
    assert (loaded["linear"]["spectrum"]["file"]
            == str((tmp_path / "spec.csv").resolve()))


def test_every_backend_declares_whether_it_uses_photometry() -> None:
    """The declaration gates the photometry-only fields, so it cannot lapse."""
    missing = sorted(set(fitconfig.BACKENDS) - set(fitconfig.USES_PHOTOMETRY))
    assert not missing, (
        f"{missing} have no USES_PHOTOMETRY entry: without it the shared "
        f"photometry-only fields materialize to their defaults and enter the "
        f"run identity of a fit that never reads a band.")


def test_every_backend_declares_its_execution_only_fields() -> None:
    """Adding a backend must not silently leave its paths in the identity.

    A backend whose block genuinely holds no execution-only field declares an
    empty tuple; the point is that the entry is a deliberate act rather than
    an omission nobody notices until a run_id forks.
    """
    declared = {path[0] for path in fitconfig.EXECUTION_ONLY if path}
    missing = sorted(set(fitconfig.BACKENDS) - declared)
    assert not missing, (
        f"{missing} have no EXECUTION_ONLY entry: a path-valued field would "
        f"hash as a literal string, so moving the repo forks every run_id. "
        f"Declare an empty tuple if there is genuinely nothing to strip.")


def test_execution_only_table_reaches_any_depth_and_tolerates_gaps(
        monkeypatch) -> None:
    """The projection follows the table down, and skips what is not there.

    The branch-per-backend form it replaced could only reach the two blocks
    it named by hand.
    """
    monkeypatch.setitem(fitconfig.EXECUTION_ONLY, ("linear", "spectrum"),
                        ("file",))
    monkeypatch.setitem(fitconfig.EXECUTION_ONLY, ("absent",), ("gone",))
    resolved = {"name": "x", "prospector": None,
                "linear": {"templates": "/machine/path",
                           "spectrum": {"file": "/machine/s.csv",
                                        "err_floor": 0.01}}}
    projected = hash_projection(resolved)

    assert "name" not in projected
    assert "file" not in projected["linear"]["spectrum"]
    assert projected["linear"]["spectrum"]["err_floor"] == 0.01
    # A declared path whose block is absent, and one whose block is null,
    # are both no-ops rather than errors.
    assert "absent" not in projected
    assert projected["prospector"] is None


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


RESOLUTION = {"file": "ssp_c3k_a", "unit": "sigma_kms",
              "wave_frame": "vacuum"}
LSF = {"constant": 3000.0, "unit": "R", "on_undersampled": "ignore"}


def test_the_two_resolution_curves_share_one_shape() -> None:
    block = parse_fit_config(_linear(linear={
        "lsf": dict(LSF), "template_resolution": dict(RESOLUTION)}))["linear"]
    assert block["lsf"]["on_undersampled"] == "ignore"
    assert block["lsf"]["file"] is None and block["lsf"]["wave_frame"] is None
    assert block["template_resolution"]["constant"] is None


@pytest.mark.parametrize("linear,message", [
    ({"lsf": dict(LSF)}, "template_resolution is required"),
    ({"lsf": {"constant": 3000.0, "unit": "R"},
      "template_resolution": dict(RESOLUTION)}, "on_undersampled"),
    ({"lsf": {"unit": "R", "on_undersampled": "raise"},
      "template_resolution": dict(RESOLUTION)}, "exactly one of constant"),
    ({"lsf": {"constant": 1.0, "file": "l.txt", "unit": "R",
              "wave_frame": "air", "on_undersampled": "raise"},
      "template_resolution": dict(RESOLUTION)}, "exactly one of constant"),
    ({"lsf": {"file": "l.txt", "unit": "fwhm_A", "on_undersampled": "raise"},
      "template_resolution": dict(RESOLUTION)}, "wave_frame is required"),
    ({"lsf": {"constant": 3000.0, "unit": "R", "wave_frame": "air",
              "on_undersampled": "raise"},
      "template_resolution": dict(RESOLUTION)}, "means nothing without"),
    ({"lsf": {"constant": 3000.0, "unit": "parsecs",
              "on_undersampled": "raise"},
      "template_resolution": dict(RESOLUTION)}, "unit"),
    ({"lsf": {"constant": 3000.0, "unit": "R", "on_undersampled": "clamp"},
      "template_resolution": dict(RESOLUTION)}, "on_undersampled"),
])
def test_a_resolution_block_that_could_be_silent_is_refused(linear, message
                                                            ) -> None:
    _fails(_linear(linear=linear), message)


def test_the_scan_block_defaults_its_step_to_the_basis() -> None:
    scan = parse_fit_config(_linear(linear={"scan": {}}))["linear"]["scan"]
    assert scan["z_step_coarse"] is None
    assert scan["n_poly_iter_coarse"] == 1
    assert scan["window_steps"] == 10
    assert scan["minima_dv_kms"] == 1000.0


def test_the_new_blocks_are_null_by_default() -> None:
    """Null drops out of canonical JSON, so the frozen goldens do not move."""
    block = parse_fit_config(_linear())["linear"]
    for key in ("gas", "lsf", "template_resolution", "scan", "transmission"):
        assert block[key] is None
