"""The linear backend's run entry point: identity, staging, and the fit.

Not a jobs.py path. jobs.py is photometry-first at five gates and its run_id
takes the photometry and bandpass hashes; a linear fit has neither, so it
writes its own run directory. DESIGN.md section 16.3.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sedfit.backends.linear import runner
from sedfit.core import fitconfig
from sedfit.core.fitconfig import (
    hash_projection,
    parse_fit_config,
    resolve_config,
)
from sedfit.core.provenance import run_id
from sedfit.core.spectrum import read_spectrum, vac_to_air
from test_linear_backend import DV_KMS, NORMALIZE, WAVE_RANGE, _write_templates

TRUTH_Z, TRUTH_SIGMA = 0.2690, 250.0
WEIGHTS = np.array([0.5, 0.0, 0.3, 0.2])


def _config(templates, spectrum_file, **overrides) -> dict:
    block = {"templates": str(templates),
             "template_pattern": "ssp_*.dat",
             "template_wave_range": list(WAVE_RANGE),
             "template_dv_kms": DV_KMS,
             "normalize_range": list(NORMALIZE),
             "z_min": 0.264, "z_max": 0.2741, "z_step": 4e-4,
             "sigma_min": 150.0, "sigma_max": 400.0, "sigma_step": 50.0,
             "poly_order": 8, "poly_domain": [4750.0, 9350.2],
             "poly_wave_frame": "air", "clip_sigma": None,
             "spectrum": {"file": str(spectrum_file)}}
    block.update(overrides)
    return resolve_config(
        parse_fit_config({"schema_version": 2, "backend": "linear",
                          "name": "bcg", "linear": block}),
        target_z_ref=TRUTH_Z)


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    """A template directory and a synthetic spectrum built from that basis.

    The spectrum is written in the package's own file contract -- observed
    vacuum wavelengths, f_nu in microjanskys -- so the runner's f_nu to
    f_lambda conversion is exercised rather than bypassed.
    """
    root = tmp_path_factory.mktemp("linear_runner")
    templates = root / "templates"
    templates.mkdir()
    paths = _write_templates(templates)

    from sedfit.backends.linear import TemplateBasis
    basis = TemplateBasis(paths, wave_range=WAVE_RANGE, dv_kms=DV_KMS,
                          normalize_range=NORMALIZE)
    wave_vac = np.arange(4800.0, 9300.0, 1.25)
    flam = WEIGHTS @ basis.design(wave_vac, TRUTH_Z, TRUTH_SIGMA)
    # invert the runner's own conversion so the file is honest microjanskys
    flux_uJy = flam * wave_vac ** 2 / runner.FLAM_PER_UJY
    err_uJy = np.full_like(flux_uJy, flux_uJy.mean() * 1e-3)

    spectrum_file = root / "spec.csv"
    import pandas as pd
    pd.DataFrame({"wave_A": wave_vac, "flux_uJy": flux_uJy,
                  "flux_err_uJy": err_uJy,
                  "mask": np.ones_like(wave_vac, int)}).to_csv(
        spectrum_file, index=False)
    (root / "spec.provenance.json").write_text(json.dumps(
        {"wave_frame": "vacuum", "flux_unit": "uJy", "source": "synthetic"}))
    return root, templates, spectrum_file


def test_flam_conversion_round_trips(workspace) -> None:
    wave = np.array([4750.0, 9155.0])
    uJy = np.array([1.0, 2.5])
    flam = runner.to_flam(wave, uJy)
    assert np.allclose(flam * wave ** 2 / runner.FLAM_PER_UJY, uJy)
    # 1 uJy at 9155 A is 3.577e-19 cgs, i.e. 0.3577 in FLAM_UNIT
    assert flam[1] / 2.5 == pytest.approx(0.35769, rel=1e-4)


def test_run_recovers_the_truth_and_stages_a_directory(workspace) -> None:
    root, templates, spectrum_file = workspace
    resolved = _config(templates, spectrum_file)
    spectrum = read_spectrum(spectrum_file)
    row = runner.run(resolved, spectrum, root / "runs")

    assert row["status"] == "ok"
    assert len(row["run_id"]) == 12
    estimates = row["estimates"]
    assert estimates["redshift"] == pytest.approx(TRUTH_Z, abs=2e-5)
    assert estimates["sigma_kms"] == pytest.approx(TRUTH_SIGMA, abs=2.0)
    assert estimates["redshift_err"] > 0 and estimates["sigma_err"] > 0
    # the fractions are the truth weights back out, not merely normalized
    names = sorted(p.name for p in Path(templates).glob("ssp_*.dat"))
    recovered = np.array([estimates["light_fractions"].get(n, 0.0)
                          for n in names])
    assert recovered.sum() == pytest.approx(1.0)
    assert np.allclose(recovered, WEIGHTS, atol=5e-3)

    run_dir = Path(row["path"])
    assert run_dir.name.startswith(row["run_id"]) and run_dir.name.endswith(
        "-bcg")
    for product in ("config.json", "spectrum.csv",
                    "spectrum.provenance.json", "manifest.json", "fit.json",
                    "model.npz"):
        assert (run_dir / product).exists(), product
    # a spectrum-only run stages no photometry
    assert not (run_dir / "phot.csv").exists()
    assert (run_dir / "spectrum.csv").read_bytes() \
        == spectrum_file.read_bytes()

    # the jsonl accumulates beside the run directories, not inside one
    assert not (run_dir / runner.MANIFEST_NAME).exists()
    manifest = json.loads(
        (root / "runs" / runner.MANIFEST_NAME).read_text().splitlines()[-1])
    assert manifest["run_id"] == row["run_id"]

    fit = json.loads((run_dir / "fit.json").read_text())
    assert fit["poly_wave_frame"] == "air"
    assert fit["error_method"] == "hessian"
    assert fit["flux_unit"] == runner.FLAM_UNIT
    assert set(fit["digests"]) == {"templates", "spectrum"}
    assert fit["digests"]["spectrum"] == spectrum.sha256


def test_identity_ignores_paths_but_follows_content(workspace, tmp_path
                                                    ) -> None:
    """Moving the basis must not fork the run_id; changing it must."""
    _, templates, spectrum_file = workspace
    spectrum = read_spectrum(spectrum_file)
    here = runner.plan(_config(templates, spectrum_file), spectrum)

    import shutil
    moved = tmp_path / "elsewhere"
    shutil.copytree(templates, moved)
    there = runner.plan(_config(moved, spectrum_file), spectrum)
    assert there["run_id"] == here["run_id"]

    edited = tmp_path / "edited"
    shutil.copytree(templates, edited)
    victim = sorted(edited.glob("ssp_*.dat"))[0]
    table = np.loadtxt(victim)
    table[:, 1] *= 1.01
    np.savetxt(victim, table, fmt="%14.4f %.6e")
    assert runner.plan(_config(edited, spectrum_file),
                       spectrum)["run_id"] != here["run_id"]

    # and a science field still moves it
    assert runner.plan(_config(templates, spectrum_file, poly_order=6),
                       spectrum)["run_id"] != here["run_id"]


def test_a_null_hash_drops_out_of_the_identity(workspace) -> None:
    """What lets a photometry-less run_id exist at all.

    canonical_json omits null-valued keys, so run_id(projection) hashes
    {"config": ...} alone. If that ever changes, every identity in the repo
    moves -- the two frozen goldens would catch it, and so does this.
    """
    _, templates, spectrum_file = workspace
    spectrum = read_spectrum(spectrum_file)
    resolved = _config(templates, spectrum_file)
    projected = hash_projection(
        resolved, digests=runner.content_digests(
            runner.resolve_templates(resolved["linear"]), spectrum))
    assert run_id(projected) == run_id(projected, None, None)
    assert run_id(projected) != run_id(projected, "p" * 64, "q" * 64)


def test_the_polynomial_coordinate_is_the_one_the_config_names(workspace
                                                               ) -> None:
    _, templates, spectrum_file = workspace
    spectrum = read_spectrum(spectrum_file)
    air = runner.plan(_config(templates, spectrum_file), spectrum)
    vacuum = runner.plan(
        _config(templates, spectrum_file, poly_wave_frame="vacuum"), spectrum)

    assert np.allclose(air["poly_wave"], vac_to_air(spectrum.wave_A))
    assert np.array_equal(vacuum["poly_wave"], spectrum.wave_A)
    # a free choice of coordinate, but not a free choice of identity
    assert air["run_id"] != vacuum["run_id"]


def test_a_null_poly_domain_spans_the_fitted_pixels(workspace) -> None:
    _, templates, spectrum_file = workspace
    spectrum = read_spectrum(spectrum_file)
    prepared = runner.plan(
        _config(templates, spectrum_file, poly_domain=None), spectrum)
    fitted_wave = prepared["poly_wave"][prepared["fitted"]]
    assert prepared["poly_domain"] == (fitted_wave.min(), fitted_wave.max())


def test_mask_windows_shrink_the_fitted_set(workspace) -> None:
    _, templates, spectrum_file = workspace
    spectrum = read_spectrum(spectrum_file)
    full = runner.plan(_config(templates, spectrum_file), spectrum)
    masked = runner.plan(
        _config(templates, spectrum_file,
                spectrum={"file": str(spectrum_file),
                          "mask_windows": [[9029.0, 9275.0]]}), spectrum)
    assert masked["fitted"].sum() < full["fitted"].sum()
    assert masked["run_id"] != full["run_id"]


def _write_transmission(path, *, wave, factor) -> Path:
    np.savetxt(path, np.c_[wave, factor], fmt="%12.4f %.8f")
    return path


def test_transmission_multiplies_the_model(workspace, tmp_path) -> None:
    """T is a term of the advertised model, so a config must be able to set it.

    `fitting.py` advertises Cheb * T * sum_k a_k B_k. Before 2026-08-25 T was
    the one term no config could reach, and it silently defaulted to ones.
    """
    _, templates, spectrum_file = workspace
    spectrum = read_spectrum(spectrum_file)
    wave = np.linspace(4700.0, 9400.0, 400)
    # a 20% absorption trough, the shape a telluric band has
    factor = 1.0 - 0.2 * np.exp(-0.5 * ((wave - 7600.0) / 40.0) ** 2)
    curve = _write_transmission(tmp_path / "telluric.dat", wave=wave,
                                factor=factor)

    flat = runner.plan(_config(templates, spectrum_file), spectrum)
    assert np.array_equal(flat["transmission"],
                          np.ones_like(spectrum.wave_A))

    absorbed = runner.plan(
        _config(templates, spectrum_file,
                transmission={"file": str(curve), "wave_frame": "vacuum"}),
        spectrum)
    trough = absorbed["transmission"]
    assert trough.min() == pytest.approx(0.8, abs=0.02)
    assert trough.max() == pytest.approx(1.0, abs=1e-6)
    # and it changes the identity, because it changes the model
    assert absorbed["run_id"] != flat["run_id"]
    assert "transmission" in absorbed["digests"]
    assert "transmission" not in flat["digests"]


def test_transmission_reaches_the_fitted_model(workspace, tmp_path) -> None:
    """Not just the plan dict -- `run` must hand T to fit_spectrum.

    A spectrum carrying a 25% telluric trough the model cannot represent
    absorbs it into the residuals. Measured 2026-08-25: chi2/dof 0.00 with T
    against 205 without, redshift off by 0.9 km/s, light fractions off by 9%.
    """
    _, templates, _ = workspace
    from sedfit.backends.linear import TemplateBasis
    basis = TemplateBasis(sorted(Path(templates).glob("ssp_*.dat")),
                          wave_range=WAVE_RANGE, dv_kms=DV_KMS,
                          normalize_range=NORMALIZE)
    wave = np.arange(4800.0, 9300.0, 1.25)
    curve_w = np.linspace(4700.0, 9400.0, 4000)
    curve_f = 1.0 - 0.25 * np.exp(-0.5 * ((curve_w - 7600.0) / 30.0) ** 2)
    curve = _write_transmission(tmp_path / "tell.dat", wave=curve_w,
                                factor=curve_f)

    # imprint the trough on the data, so only a fit that models T can match
    flam = (WEIGHTS @ basis.design(wave, TRUTH_Z, TRUTH_SIGMA)) * np.interp(
        wave, curve_w, curve_f)
    uJy = flam * wave ** 2 / runner.FLAM_PER_UJY
    spectrum_file = tmp_path / "absorbed.csv"
    import pandas as pd
    pd.DataFrame({"wave_A": wave, "flux_uJy": uJy,
                  "flux_err_uJy": np.full_like(uJy, uJy.mean() * 2e-3),
                  "mask": np.ones_like(wave, int)}).to_csv(spectrum_file,
                                                           index=False)
    (tmp_path / "absorbed.provenance.json").write_text(json.dumps(
        {"wave_frame": "vacuum", "flux_unit": "uJy"}))
    spectrum = read_spectrum(spectrum_file)

    names = sorted(p.name for p in Path(templates).glob("ssp_*.dat"))
    scores = {}
    for tag, block in (("with", {"file": str(curve),
                                 "wave_frame": "vacuum"}), ("without", None)):
        row = runner.run(
            _config(templates, spectrum_file, transmission=block), spectrum,
            tmp_path / f"runs_{tag}")
        estimates = row["estimates"]
        fractions = np.array([estimates["light_fractions"].get(n, 0.0)
                              for n in names])
        offset_kms = (abs(estimates["redshift"] - TRUTH_Z) / (1 + TRUTH_Z)
                      * 299792.458)
        scores[tag] = (estimates["chi2_per_dof"], offset_kms,
                       np.abs(fractions - WEIGHTS).max())

    assert scores["with"][0] < 1.0 < scores["without"][0]
    assert scores["with"][1] < 0.1 < scores["without"][1]
    assert scores["with"][2] < 1e-4 < scores["without"][2]


def test_transmission_frame_is_the_one_the_config_names(workspace, tmp_path
                                                        ) -> None:
    """Reading an air-frame curve as vacuum misplaces it by ~2.5 A."""
    _, templates, spectrum_file = workspace
    spectrum = read_spectrum(spectrum_file)
    wave = np.linspace(4700.0, 9400.0, 4000)
    factor = 1.0 - 0.2 * np.exp(-0.5 * ((wave - 9155.0) / 3.0) ** 2)
    curve = _write_transmission(tmp_path / "sharp.dat", wave=wave,
                                factor=factor)

    frames = {}
    for frame in ("vacuum", "air"):
        frames[frame] = runner.plan(
            _config(templates, spectrum_file,
                    transmission={"file": str(curve), "wave_frame": frame}),
            spectrum)
    shift = np.abs(frames["air"]["transmission"]
                   - frames["vacuum"]["transmission"]).max()
    assert shift > 0.05, "a 2.5 A frame error on a 3 A feature must show"
    assert frames["air"]["run_id"] != frames["vacuum"]["run_id"]


def test_transmission_file_is_validated(workspace, tmp_path) -> None:
    _, templates, spectrum_file = workspace
    spectrum = read_spectrum(spectrum_file)

    def _plan(name, wave, factor):
        curve = _write_transmission(tmp_path / name, wave=wave, factor=factor)
        return runner.plan(
            _config(templates, spectrum_file,
                    transmission={"file": str(curve),
                                  "wave_frame": "vacuum"}), spectrum)

    good = np.linspace(4700.0, 9400.0, 50)
    with pytest.raises(ValueError, match="must increase strictly"):
        _plan("backwards.dat", good[::-1], np.ones(50))
    with pytest.raises(ValueError, match="finite and positive"):
        _plan("zeroed.dat", good, np.zeros(50))
    with pytest.raises(ValueError, match="finite and positive"):
        _plan("nan.dat", good, np.where(np.arange(50) == 7, np.nan, 1.0))


def test_transmission_config_is_strict(workspace, tmp_path) -> None:
    from sedfit.core.fitconfig import parse_fit_config as parse
    _, templates, spectrum_file = workspace
    base = {"schema_version": 2, "backend": "linear", "name": "x"}

    def _block(**over):
        block = {"templates": str(templates),
                 "template_wave_range": [3600.0, 7400.0],
                 "z_min": 0.264, "z_max": 0.274, "poly_wave_frame": "air",
                 "spectrum": {"file": str(spectrum_file)}}
        block.update(over)
        return {**base, "linear": block}

    # the frame is required, exactly as poly_wave_frame is
    with pytest.raises(ValueError, match="missing required keys.*wave_frame"):
        parse(_block(transmission={"file": "t.dat"}))
    with pytest.raises(ValueError, match="not in"):
        parse(_block(transmission={"file": "t.dat", "wave_frame": "observed"}))
    with pytest.raises(ValueError, match="unknown keys"):
        parse(_block(transmission={"file": "t.dat", "wave_frame": "air",
                                   "pwv": 1.7}))
    assert parse(_block())["linear"]["transmission"] is None


def test_a_relative_transmission_resolves_against_the_config(workspace,
                                                             tmp_path) -> None:
    _, templates, spectrum_file = workspace
    config = tmp_path / "fit.json"
    config.write_text(json.dumps({
        "schema_version": 2, "backend": "linear", "name": "x",
        "linear": {"templates": str(templates),
                   "template_wave_range": [3600.0, 7400.0],
                   "z_min": 0.264, "z_max": 0.274, "poly_wave_frame": "air",
                   "transmission": {"file": "telluric.dat",
                                    "wave_frame": "air"},
                   "spectrum": {"file": str(spectrum_file)}}}))
    loaded = fitconfig.load_fit_config(config)
    assert (loaded["linear"]["transmission"]["file"]
            == str((tmp_path / "telluric.dat").resolve()))


def test_fit_json_records_the_bands_it_actually_used(workspace) -> None:
    """A null in the config must not become a null in the record."""
    root, templates, spectrum_file = workspace
    spectrum = read_spectrum(spectrum_file)
    resolved = _config(templates, spectrum_file, normalize_range=None,
                       poly_domain=None)
    assert resolved["linear"]["normalize_range"] is None
    assert resolved["linear"]["poly_domain"] is None

    row = runner.run(resolved, spectrum, root / "runs_resolved")
    fit = json.loads((Path(row["path"]) / "fit.json").read_text())
    assert fit["normalize_range"] == list(WAVE_RANGE)
    assert fit["poly_domain"] is not None and len(fit["poly_domain"]) == 2
    assert fit["transmission"] is None


def test_a_non_linear_config_is_refused(workspace) -> None:
    _, _, spectrum_file = workspace
    spectrum = read_spectrum(spectrum_file)
    eazy = resolve_config(parse_fit_config({
        "schema_version": 2, "backend": "eazy", "name": "x",
        "eazy": {"templates": str(Path(__file__).resolve().parents[1]
                                  / "sedfit" / "data" / "templates"
                                  / "brown14")}}), target_z_ref=0.106)
    with pytest.raises(ValueError, match="not a linear config"):
        runner.plan(eazy, spectrum)


def test_the_run_records_the_grid_it_scanned(workspace, tmp_path) -> None:
    """A blind redshift is a minimum-selection problem, so keep the scan."""
    _, templates, spectrum_file = workspace
    spectrum = read_spectrum(spectrum_file)
    row = runner.run(_config(templates, spectrum_file), spectrum,
                     tmp_path / "runs")

    fit_json = json.loads((Path(row["path"]) / "fit.json").read_text())
    assert fit_json["grid_n_poly_iter"] == 4
    assert fit_json["minima"][0]["delta_chi2"] == 0.0
    assert all(m["delta_chi2"] >= 0.0 for m in fit_json["minima"])
    assert row["estimates"]["n_minima"] == len(fit_json["minima"])
    assert row["estimates"]["delta_chi2"] == fit_json["delta_chi2"]

    model = np.load(Path(row["path"]) / "model.npz")
    assert model["chi2_grid"].shape == (model["sigma_grid"].size,
                                        model["redshift_grid"].size)
    # the winner of the grid, not the polished fit -- they differ by design
    assert model["chi2_grid"].min() == pytest.approx(
        fit_json["minima"][0]["chi2"])


def test_gas_enters_the_identity_by_content(workspace, tmp_path) -> None:
    """The line list is a scientific choice, so it is hashed like the basis."""
    _, templates, spectrum_file = workspace
    spectrum = read_spectrum(spectrum_file)
    without = runner.plan(_config(templates, spectrum_file), spectrum)
    with_gas = runner.plan(
        _config(templates, spectrum_file, gas={"lines": "optical"}), spectrum)
    assert without["run_id"] != with_gas["run_id"]
    assert "lines" in with_gas["digests"]
    assert without["gas"] is None

    short = tmp_path / "short.txt"
    short.write_text("Hbeta 4862.683\nHalpha 6564.608\n")
    edited = runner.plan(
        _config(templates, spectrum_file, gas={"lines": str(short)}), spectrum)
    assert edited["run_id"] != with_gas["run_id"]
    # a path is not part of the identity; its content is
    moved = tmp_path / "moved.txt"
    moved.write_text(short.read_text())
    assert runner.plan(_config(templates, spectrum_file,
                               gas={"lines": str(moved)}),
                       spectrum)["run_id"] == edited["run_id"]


def test_a_gas_run_reports_its_lines_apart_from_its_templates(workspace,
                                                              tmp_path) -> None:
    _, templates, spectrum_file = workspace
    spectrum = read_spectrum(spectrum_file)
    row = runner.run(_config(templates, spectrum_file,
                             gas={"lines": "optical", "sigma_kms": 120.0}),
                     spectrum, tmp_path / "gasruns")

    assert row["gas"]["sigma_kms"] == 120.0
    assert row["gas"]["n_columns"] == 18
    fit_json = json.loads((Path(row["path"]) / "fit.json").read_text())
    # amplitudes are the stellar basis; line fluxes are their own field
    assert set(fit_json["amplitudes"]) == {p.name for p in
                                           runner.resolve_templates(
                                               _config(templates,
                                                       spectrum_file)["linear"])}
    assert len(fit_json["gas_fluxes"]) == 18
    assert fit_json["gas"]["ratio_locked"][0]["ratios"] == [1.0, 2.98]


RESOLUTION = {"file": "ssp_c3k_a", "unit": "sigma_kms", "wave_frame": "vacuum"}
LSF = {"constant": 3000.0, "unit": "R", "on_undersampled": "ignore"}


def test_a_scan_block_runs_the_two_stage_scan(workspace, tmp_path) -> None:
    _, templates, spectrum_file = workspace
    spectrum = read_spectrum(spectrum_file)
    row = runner.run(_config(templates, spectrum_file, scan={}), spectrum,
                     tmp_path / "scanruns")

    assert row["estimates"]["redshift"] == pytest.approx(TRUTH_Z, abs=1e-3)
    assert row["estimates"]["scan_z_step"] == pytest.approx(1.5e-3)
    fit_json = json.loads((Path(row["path"]) / "fit.json").read_text())
    # the coarse pass ran at 1 and the fit at 4, and they are recorded apart
    assert fit_json["scan"]["n_poly_iter"] == 1
    assert fit_json["grid_n_poly_iter"] == 4
    model = np.load(Path(row["path"]) / "model.npz")
    assert model["coarse_chi2"].size == model["coarse_redshift_grid"].size
    assert model["coarse_redshift_grid"].size < model["redshift_grid"].size


def test_no_scan_block_means_no_scan(workspace, tmp_path) -> None:
    _, templates, spectrum_file = workspace
    spectrum = read_spectrum(spectrum_file)
    row = runner.run(_config(templates, spectrum_file), spectrum,
                     tmp_path / "plainruns")
    assert row["estimates"]["scan_delta_chi2"] is None
    fit_json = json.loads((Path(row["path"]) / "fit.json").read_text())
    assert fit_json["scan"] is None
    assert "coarse_chi2" not in np.load(Path(row["path"]) / "model.npz")


def test_the_resolution_curves_enter_the_identity(workspace) -> None:
    _, templates, spectrum_file = workspace
    spectrum = read_spectrum(spectrum_file)
    plain = runner.plan(_config(templates, spectrum_file), spectrum)
    with_lsf = runner.plan(
        _config(templates, spectrum_file, lsf=dict(LSF),
                template_resolution=dict(RESOLUTION)), spectrum)

    assert plain["lsf"] is None and with_lsf["lsf"] is not None
    assert plain["run_id"] != with_lsf["run_id"]
    # the packaged curve is hashed by content, not named by path
    assert "template_resolution" in with_lsf["digests"]

    coarser = runner.plan(
        _config(templates, spectrum_file,
                lsf={**LSF, "constant": 2000.0},
                template_resolution=dict(RESOLUTION)), spectrum)
    assert coarser["run_id"] != with_lsf["run_id"]


def test_an_lsf_run_records_both_curves(workspace, tmp_path) -> None:
    _, templates, spectrum_file = workspace
    spectrum = read_spectrum(spectrum_file)
    row = runner.run(
        _config(templates, spectrum_file, lsf=dict(LSF),
                template_resolution=dict(RESOLUTION)), spectrum,
        tmp_path / "lsfruns")
    fit_json = json.loads((Path(row["path"]) / "fit.json").read_text())
    assert fit_json["lsf"]["on_undersampled"] == "ignore"
    assert fit_json["template_resolution"]["file"] == "ssp_c3k_a"


def test_the_declared_dispersion_range_reaches_the_polish(workspace,
                                                          tmp_path) -> None:
    """sigma_min/sigma_max are config fields; the simplex now sees them."""
    _, templates, spectrum_file = workspace
    spectrum = read_spectrum(spectrum_file)
    row = runner.run(_config(templates, spectrum_file, sigma_min=150.0,
                             sigma_max=200.0, sigma_step=25.0), spectrum,
                     tmp_path / "boundruns")
    assert row["estimates"]["sigma_kms"] <= 200.0 + 1e-9
    assert row["estimates"]["sigma_pinned"] is True
    fit_json = json.loads((Path(row["path"]) / "fit.json").read_text())
    assert fit_json["sigma_pinned"] is True

    wide = runner.run(_config(templates, spectrum_file), spectrum,
                      tmp_path / "wideruns")
    assert wide["estimates"]["sigma_pinned"] is False
