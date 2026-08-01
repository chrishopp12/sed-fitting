"""
plots.py

Prospector Diagnostic Figures
---------------------------------------------------------

Corner and trace figures from a normalized FitResult, and the unified
SED figure at the MAP -- rebuilt self-containedly from a run directory
(config.json + phot.csv per the run-directory contract; no external
state).

Requirements:
  - numpy, matplotlib, corner; prospect + fsps for the MAP SED
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from sedfit.analysis.plots import SEDPlotData, sed_figure
from sedfit.backends.prospector.results import FitResult

if TYPE_CHECKING:
    from sedfit.core.registry import Registry


def _display_chain(result: FitResult) -> tuple[np.ndarray, list[str]]:
    """The chain with logify applied to log-displayed parameters."""
    chain = np.array(result.chain, dtype=float, copy=True)
    labels = list(result.theta_labels)
    for i, label in enumerate(labels):
        root = label.split("_")[0]
        if root in result.logify:
            chain[:, i] = np.log10(chain[:, i])
            labels[i] = f"log10({label})"
    return chain, labels


def plot_corner(result: FitResult, save_path: str | Path) -> Path:
    """Weighted corner plot with 16/50/84 titles."""
    import corner as corner_module

    chain, labels = _display_chain(result)
    fig = corner_module.corner(chain, weights=result.weights, labels=labels,
                               quantiles=[0.16, 0.5, 0.84],
                               show_titles=True, title_fmt=".4f",
                               label_kwargs={"fontsize": 9},
                               title_kwargs={"fontsize": 8})
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return save_path


def plot_trace(result: FitResult, save_path: str | Path) -> Path:
    """Per-parameter sample traces (dynesty: colored by log weight;
    emcee: per-walker lines from the raw 3-D chain)."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if result.sampler == "emcee" and result.raw_chain_3d is not None:
        chain = result.raw_chain_3d
        nwalkers, niter, ndim = chain.shape
        fig, axes = plt.subplots(ndim, 1, figsize=(9, 1.6 * ndim),
                                 sharex=True)
        for i, ax in enumerate(np.atleast_1d(axes)):
            ax.plot(chain[:, :, i].T, color="k", alpha=0.1, lw=0.5)
            ax.axvline(result.burn_cut, color="firebrick", ls=":")
            ax.set_ylabel(result.theta_labels[i], fontsize=8)
        np.atleast_1d(axes)[-1].set_xlabel("iteration")
    else:
        chain, labels = _display_chain(result)
        iteration = np.arange(chain.shape[0])
        color = np.log10(np.maximum(result.weights, 1e-300))
        fig, axes = plt.subplots(chain.shape[1], 1,
                                 figsize=(9, 1.6 * chain.shape[1]),
                                 sharex=True)
        for i, ax in enumerate(np.atleast_1d(axes)):
            ax.scatter(iteration, chain[:, i], c=color, s=1, cmap="viridis",
                       rasterized=True)
            ax.set_ylabel(labels[i], fontsize=8)
        np.atleast_1d(axes)[-1].set_xlabel("sample (color: log10 weight)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return save_path


def map_sed_plot_data(cfg: dict, obs: dict, model, sps, map_theta, *,
                      registry: Registry) -> SEDPlotData:
    """The unified figure's contents at the MAP, from live fit objects."""
    from sedfit.backends.prospector.obs import UJY_PER_MAGGIE

    spec, phot, _ = model.predict(map_theta, obs=obs, sps=sps)
    wave_model = np.asarray(sps.wavelengths, float)
    labels = list(model.theta_labels())
    z_map = (float(map_theta[labels.index("zred")]) if "zred" in labels
             else float(cfg["z_ref"]))
    curve = (wave_model * (1.0 + z_map),
             np.asarray(spec, float) * UJY_PER_MAGGIE, "MAP spectrum")

    bands = [f.name for f in obs["filters"]]
    fobs = np.asarray(obs["maggies"], float) * UJY_PER_MAGGIE
    efobs = np.asarray(obs["maggies_unc"], float) * UJY_PER_MAGGIE
    model_phot = np.asarray(phot, float) * UJY_PER_MAGGIE
    chi2 = float(np.sum(((fobs - model_phot) / efobs) ** 2))
    ndof = max(1, len(bands) - len(labels))

    return SEDPlotData(
        title=(f"{cfg['name']} -- prospector MAP  "
               f"(z = {z_map:.4f}, $\\chi^2_\\nu$ = {chi2 / ndof:.1f})"),
        bands=bands,
        wave_AA=np.array([f.wave_effective for f in obs["filters"]]),
        fobs=fobs, efobs=efobs, model_phot=model_phot,
        instruments=[registry.instrument_of(b) for b in bands],
        curves=[curve],
    )


def plot_run(run_dir: str | Path, result: FitResult, cfg: dict, obs: dict,
             model, sps, *, registry: Registry) -> list[Path]:
    """Corner, trace, and MAP-SED figures from live fit objects."""
    plots = Path(run_dir) / "plots"
    return [
        plot_corner(result, plots / "corner.png"),
        plot_trace(result, plots / "trace.png"),
        sed_figure(map_sed_plot_data(cfg, obs, model, sps,
                                     result.map_theta, registry=registry),
                   save_path=plots / "map_sed.png"),
    ]


def generate_plots(run_dir: str | Path, *, registry: Registry,
                   burn_frac: float = 0.25) -> list[Path]:
    """Corner, trace, and MAP-SED figures, rebuilt from a run directory."""
    import pandas as pd

    from sedfit.backends.prospector.model import (
        attach_norm_masks,
        build_model,
        build_sps,
    )
    from sedfit.backends.prospector.obs import build_obs
    from sedfit.backends.prospector.results import load_results
    from sedfit.core.policy import apply_policy
    from sedfit.core.table import validate_sed_table

    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(run_dir / "phot.csv")
    validate_sed_table(frame, registry, label=str(run_dir / "phot.csv"))
    policy = apply_policy(frame, registry=registry,
                          bands_include=cfg["bands_include"],
                          min_valid_bands=cfg["min_valid_bands"],
                          min_snr_broadband=cfg["min_snr_broadband"],
                          err_floor=cfg["err_floor"],
                          mu_lensing=cfg["mu_lensing"],
                          qa_gates=cfg.get("qa_gates"))
    obs = build_obs(cfg, frame, policy, registry=registry)
    model = build_model(cfg)
    attach_norm_masks(model, obs, cfg, registry)
    sps = build_sps(cfg)
    result = load_results(str(run_dir / "result.h5"), burn_frac=burn_frac,
                          cfg=cfg)
    return plot_run(run_dir, result, cfg, obs, model, sps,
                    registry=registry)
