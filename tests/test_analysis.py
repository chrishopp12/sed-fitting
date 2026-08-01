from __future__ import annotations

import numpy as np
import pytest
from test_jobs import _setup

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sedfit.analysis.ew import compute_equivalent_width
from sedfit.analysis.lines import (
    LINE_CATALOGS,
    load_emission_lines,
    mark_catalogs,
    mark_lines,
)
from sedfit.analysis.plots import SEDPlotData, sed_figure
from sedfit.backends.eazy.plots import generate_plots
from sedfit.backends.eazy.results import load_run
from sedfit.core.registry import load_registry
from sedfit.jobs import run_job

REG = load_registry()


def test_ew_gaussian_notch() -> None:
    z = 0.1
    center_rest = 5000.0
    depth, sigma_AA = 0.4, 6.0
    wave = np.arange(4000.0, 7000.0, 1.0) * (1 + z)
    center_obs = center_rest * (1 + z)
    flux = 1.0 - depth * np.exp(-0.5 * ((wave - center_obs)
                                        / (sigma_AA * (1 + z)))**2)

    # a smooth notch never crosses the continuum, so the automatic
    # boundary walk correctly reports non-convergence; the analytic check
    # uses fixed bounds
    auto = compute_equivalent_width(wave, flux, center_rest, z)
    assert not auto["converged"]

    half_width = 4 * sigma_AA * (1 + z)
    result = compute_equivalent_width(
        wave, flux, center_rest, z,
        fixed_bounds=(center_obs - half_width, center_obs + half_width))
    assert result["converged"]
    expected_obs = depth * sigma_AA * (1 + z) * np.sqrt(2 * np.pi)
    assert result["ew_obs"] == pytest.approx(expected_obs, rel=0.05)
    assert result["ew_rest"] == pytest.approx(expected_obs / (1 + z),
                                              rel=0.05)
    assert result["ew_obs"] > 0          # absorption is positive


def test_lines_annotation() -> None:
    lines = load_emission_lines()
    assert len(lines) > 10
    assert all(w1 <= w2 for (_, w1), (_, w2) in zip(lines, lines[1:]))

    fig, ax = plt.subplots()
    ax.set_xlim(3000, 9000)
    ax.set_ylim(0, 1)
    mark_lines(ax, lines, redshift=0.1)
    mark_catalogs(ax, ["curated_emission", "deimos_absorption"],
                  redshift=0.1, emission_lines=lines)
    assert len(ax.lines) > 5
    plt.close(fig)
    assert set(LINE_CATALOGS) >= {"curated_emission", "deimos_absorption",
                                  "feii_star", "blue_side"}


def test_sed_figure(tmp_path) -> None:
    data = SEDPlotData(
        title="synthetic",
        bands=["CFHT_u", "WISE_W1", "SPHEREx_000"],
        wave_AA=np.array([3800.0, 33500.0, 7500.0]),
        fobs=np.array([10.0, 900.0, 400.0]),
        efobs=np.array([0.5, 9.0, 4.0]),
        model_phot=np.array([10.5, 890.0, 405.0]),
        instruments=["CFHT", "WISE", "SPHEREx"],
        curves=[(np.linspace(3000, 60000, 500),
                 np.full(500, 300.0), "model")],
    )
    out = sed_figure(data, save_path=tmp_path / "sed.png")
    assert out.exists() and out.stat().st_size > 10000


def test_eazy_run_dir_plots(tmp_path) -> None:
    roster, cfg = _setup(tmp_path)
    row = run_job(roster, "A", "tilt2", cfg, registry=REG)
    run_dir = roster.data_root / row["path"]

    result = load_run(run_dir)
    written = generate_plots(result, registry=REG, z_ref=0.106)
    # the flat-spectrum fixture pins z_ml to the grid edge, so the SED
    # figure may be legitimately absent; the z-scan always renders
    assert any(p.name.startswith("zscan") for p in written)
    for path in written:
        assert path.exists()
        assert path.parent == run_dir / "plots"


def test_eazy_sed_figure_from_closure(tmp_path) -> None:
    from test_eazy_backend import Z_TRUE, _config, _synth_frame

    from sedfit.backends.eazy.quick import run_quick
    from sedfit.core.policy import apply_policy

    frame = _synth_frame()
    cfg = _config(tmp_path, z_fixed=Z_TRUE)
    policy = apply_policy(frame, registry=REG, min_valid_bands=5,
                          min_snr_broadband=0.0, err_floor=0.0)
    result = run_quick(cfg, frame, policy, tmp_path / "run", registry=REG)
    written = generate_plots(result, registry=REG, z_ref=Z_TRUE)
    names = {p.name for p in written}
    assert any(n.startswith("sed_") for n in names)
    assert any(n.startswith("sed_fixed") for n in names)
    assert any(n.startswith("zscan") for n in names)
