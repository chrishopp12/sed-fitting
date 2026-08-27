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


def _phot_view(obs: dict) -> dict:
    """A shallow photometry-only view of an obs dictionary, so predict
    returns the full-grid model curve with no calibration vector."""
    view = dict(obs)
    view.update(spectrum=None, wavelength=None, unc=None, mask=None)
    return view


def map_sed_plot_data(cfg: dict, obs: dict, model, sps, map_theta, *,
                      registry: Registry) -> SEDPlotData:
    """The unified figure's contents at the MAP, from live fit objects."""
    from sedfit.backends.prospector.obs import UJY_PER_MAGGIE

    joint = obs.get("spectrum") is not None
    spec, phot, _ = model.predict(map_theta, obs=_phot_view(obs), sps=sps)
    wave_model = np.asarray(sps.wavelengths, float)
    labels = list(model.theta_labels())
    z_map = (float(map_theta[labels.index("zred")]) if "zred" in labels
             else float(cfg["z_ref"]))
    curves = [(wave_model * (1.0 + z_map),
               np.asarray(spec, float) * UJY_PER_MAGGIE, "MAP spectrum",
               {"color": "crimson", "lw": 0.9, "zorder": 3})]

    bands = [f.name for f in obs["filters"]]
    fobs = np.asarray(obs["maggies"], float) * UJY_PER_MAGGIE
    efobs = np.asarray(obs["maggies_unc"], float) * UJY_PER_MAGGIE
    model_phot = np.asarray(phot, float) * UJY_PER_MAGGIE
    chi2 = float(np.sum(((fobs - model_phot) / efobs) ** 2))
    if joint:
        # photometry alone has no meaningful ndof in a joint fit; the
        # spectrum-fit figure carries the per-channel statistic
        title = (f"{cfg['name']} -- prospector MAP  (z = {z_map:.4f}, "
                 f"phot $\\chi^2$ = {chi2:.1f} over {len(bands)} bands)")
    else:
        ndof = max(1, len(bands) - len(labels))
        title = (f"{cfg['name']} -- prospector MAP  "
                 f"(z = {z_map:.4f}, $\\chi^2_\\nu$ = {chi2 / ndof:.1f})")

    if joint:
        # observed spectrum divided by the MAP calibration vector, so it
        # overlays the total-flux model curve
        model.predict(map_theta, obs=obs, sps=sps)
        speccal = np.asarray(model._speccal, float)
        mask = np.asarray(obs["mask"], bool)
        decal = np.asarray(obs["spectrum"], float) * UJY_PER_MAGGIE / speccal
        decal[~mask] = np.nan
        curves.append((np.asarray(obs["wavelength"], float), decal,
                       "observed spectrum / calibration",
                       {"color": "0.35", "lw": 0.5, "alpha": 0.9,
                        "zorder": 2}))

        # with prospect drawing the nebular lines (eline_sigma), the
        # photometry-view predict above returns a line-free continuum;
        # draw every line onto the wide curve at its cached width (the
        # caches are populated by the real-obs predict just done)
        inspec = bool(np.atleast_1d(
            model.params.get("nebemlineinspec", [True]))[0])
        if not inspec:
            wave_full, flux_full = curves[0][0], curves[0][1]
            espec = model.predict_eline_spec(wave=wave_full)
            curves[0] = (wave_full,
                         flux_full + espec.sum(axis=1) * UJY_PER_MAGGIE,
                         *curves[0][2:])

    return SEDPlotData(
        title=title,
        bands=bands,
        wave_AA=np.array([f.wave_effective for f in obs["filters"]]),
        fobs=fobs, efobs=efobs, model_phot=model_phot,
        instruments=[registry.instrument_of(b) for b in bands],
        curves=curves,
    )


def plot_spectrum_fit(cfg: dict, obs: dict, model, sps, map_theta,
                      save_path: str | Path) -> Path:
    """Observed spectrum against the calibrated MAP model.

    Three panels: the spectrum with the calibrated model over it (masked
    channels shaded), the chi residuals of the unmasked channels, and
    the maximum-likelihood calibration vector.
    """
    from sedfit.backends.prospector.obs import UJY_PER_MAGGIE

    spec_model, _, _ = model.predict(map_theta, obs=obs, sps=sps)
    speccal = np.asarray(model._speccal, float)
    wave = np.asarray(obs["wavelength"], float)
    mask = np.asarray(obs["mask"], bool)
    fobs = np.asarray(obs["spectrum"], float) * UJY_PER_MAGGIE
    eobs = np.asarray(obs["unc"], float) * UJY_PER_MAGGIE
    fmod = np.asarray(spec_model, float) * UJY_PER_MAGGIE

    chi = np.where(mask, (fobs - fmod) / eobs, np.nan)
    chi2 = float(np.nansum(chi**2))
    n_fit = int(mask.sum())

    fig, axes = plt.subplots(
        3, 1, figsize=(11, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1, 1], "hspace": 0.06})
    shown = np.where(mask, fobs, np.nan)
    axes[0].plot(wave, shown, color="0.35", lw=0.6, label="observed")
    axes[0].plot(wave, np.where(mask, fmod, np.nan), color="crimson",
                 lw=0.9, label="calibrated MAP model")
    axes[0].set_ylabel("flux density [uJy]")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[0].set_title(f"{cfg['name']} -- spectrum fit  "
                      f"($\\chi^2/n$ = {chi2 / max(1, n_fit):.2f} over "
                      f"{n_fit} channels)", fontsize=10)

    axes[1].axhline(0.0, color="0.6", lw=0.6)
    axes[1].plot(wave, chi, color="0.2", lw=0.5)
    axes[1].set_ylim(-6, 6)
    axes[1].set_ylabel("$\\chi$")

    axes[2].plot(wave, speccal, color="steelblue", lw=0.9)
    axes[2].set_ylabel("calibration")
    axes[2].set_xlabel("observed wavelength [Angstrom, vacuum]")

    edges = np.flatnonzero(np.diff(mask.astype(int)))
    starts = [0] if not mask[0] else []
    starts += [int(i) + 1 for i in edges if not mask[int(i) + 1]]
    stops = [int(i) + 1 for i in edges if mask[int(i) + 1]]
    if not mask[-1]:
        stops.append(mask.size)
    for ax in axes:
        for lo, hi in zip(starts, stops):
            ax.axvspan(wave[lo], wave[min(hi, mask.size - 1)],
                       color="0.85", zorder=0)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return save_path


def plot_run(run_dir: str | Path, result: FitResult, cfg: dict, obs: dict,
             model, sps, *, registry: Registry) -> list[Path]:
    """Corner, trace, and MAP-SED figures from live fit objects, plus
    the spectrum-fit figure on joint fits."""
    plots = Path(run_dir) / "plots"
    written = [
        plot_corner(result, plots / "corner.png"),
        plot_trace(result, plots / "trace.png"),
        sed_figure(map_sed_plot_data(cfg, obs, model, sps,
                                     result.map_theta, registry=registry),
                   save_path=plots / "map_sed.png"),
    ]
    if obs.get("spectrum") is not None:
        written.append(plot_spectrum_fit(cfg, obs, model, sps,
                                         result.map_theta,
                                         plots / "spectrum_fit.png"))
    return written


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
    spectrum = None
    if cfg["prospector"].get("spectrum") is not None:
        from sedfit.core.spectrum import read_spectrum

        # the staged copy, per the run-directory contract
        spectrum = read_spectrum(run_dir / "spectrum.csv")
    obs = build_obs(cfg, frame, policy, registry=registry,
                    spectrum=spectrum)
    model = build_model(cfg)
    attach_norm_masks(model, obs, cfg, registry)
    sps = build_sps(cfg)
    result = load_results(str(run_dir / "result.h5"), burn_frac=burn_frac,
                          cfg=cfg)
    return plot_run(run_dir, result, cfg, obs, model, sps,
                    registry=registry)
