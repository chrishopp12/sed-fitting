"""
plots.py

eazy Diagnostic Figures
---------------------------------------------------------

Produces the unified SED figure's data from a FitResult (live or
rehydrated -- no eazy import), plus the backend-specific redshift-scan
figure (delta-chi2 and P(z), with per-template curves in single mode).

Requirements:
  - numpy, matplotlib
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from sedfit.analysis.plots import SEDPlotData, sed_figure
from sedfit.backends.eazy.results import FitResult

if TYPE_CHECKING:
    from sedfit.core.registry import Registry


def _plots_dir(result: FitResult, save_dir) -> Path:
    if save_dir is not None:
        return Path(save_dir)
    run_dir = Path(result.run_dir)
    plots = run_dir / "plots"
    return plots if plots.is_dir() else run_dir


def sed_plot_data(result: FitResult, *, registry: Registry, iobj: int = 0,
                  fixed: bool = False,
                  z_ref: float | None = None) -> SEDPlotData | None:
    """The unified figure's contents for one object; None without a SED."""
    sed = (result.seds_fixed if fixed else result.seds)[iobj]
    if sed is None:
        print(f"no stored SED for object index {iobj} "
              f"({'fixed' if fixed else 'photo-z'})")
        return None

    valid = np.asarray(result.ok_data[iobj], bool)
    bands = [b for b, ok in zip(result.bands, valid) if ok]
    fobs = np.asarray(sed["fobs"], float)[valid]
    efobs = np.asarray(sed["efobs"], float)[valid]
    model = np.asarray(sed["model"], float)[valid]

    coeffs = (result.coeffs_fixed if fixed else result.coeffs_best)[iobj]
    n_active = int((coeffs > 0).sum())
    ndof = max(1, int(result.nusefilt[iobj]) - n_active - 1)
    redchi2 = float(np.sum(((fobs - model) / efobs) ** 2)) / ndof
    ref = f", ref = {z_ref:.4f}" if z_ref is not None else ""
    tag = "fixed-z" if fixed else "photo-z"

    return SEDPlotData(
        title=(f"{result.ids[iobj]} -- eazy {tag} solution  "
               f"(z = {sed['z']:.4f}, $\\chi^2_\\nu$ = {redchi2:.1f}{ref})"),
        bands=bands,
        wave_AA=np.asarray(result.pivot, float)[valid],
        fobs=fobs, efobs=efobs, model_phot=model,
        instruments=[registry.instrument_of(b) for b in bands],
        curves=[(np.asarray(sed["templz"], float),
                 np.asarray(sed["templf"], float), "Best-fit spectrum")],
    )


def plot_sed(result: FitResult, *, registry: Registry, iobj: int = 0,
             fixed: bool = False, z_ref: float | None = None,
             save_dir=None) -> Path | None:
    """Render the unified SED figure for one object."""
    data = sed_plot_data(result, registry=registry, iobj=iobj, fixed=fixed,
                         z_ref=z_ref)
    if data is None:
        return None
    suffix = "_fixed" if fixed else ""
    out = _plots_dir(result, save_dir) / f"sed{suffix}_{result.ids[iobj]}.png"
    return sed_figure(data, save_path=out)


def plot_zscan(result: FitResult, iobj: int = 0, *,
               z_ref: float | None = None, save_dir=None,
               n_singles: int = 8) -> Path:
    """Delta-chi2 and P(z) panels for one object.

    In single mode the chi2 panel overlays the n_singles best
    per-template curves, each referenced to its own minimum (a lone
    template's absolute chi2 sits far above the combo's; the label
    carries the best single's absolute penalty).
    """
    oid = result.ids[iobj]
    zgrid = np.asarray(result.zgrid, float)
    chi2 = np.asarray(result.chi2_fit[iobj], float)
    dchi2 = chi2 - np.nanmin(chi2)

    if result.lnp is not None:
        lnp = np.asarray(result.lnp[iobj], float)
    else:
        lnp = -0.5 * dchi2
    pz = np.exp(lnp - np.nanmax(lnp))
    norm = np.trapezoid(pz, zgrid)
    if norm > 0:
        pz = pz / norm

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    ax = axes[0]
    ax.plot(zgrid, dchi2, "-", color="k", lw=1.4, label="combo", zorder=10)
    if result.singles_chi2 is not None:
        singles = np.asarray(result.singles_chi2[:, iobj, :], float)
        order = np.argsort(singles.min(axis=1))[:n_singles]
        gap = np.nanmin(singles[order]) - np.nanmin(chi2)
        for rank, t in enumerate(order):
            finite = np.isfinite(singles[t])
            if not finite.any():
                continue
            best = rank == 0
            name = result.template_names[t]
            ax.plot(zgrid, singles[t] - singles[t][finite].min(), "-",
                    color="firebrick" if best else "0.75",
                    lw=1.2 if best else 0.7, zorder=5 if best else 4,
                    label=(f"best single: {name} "
                           f"(min $\\chi^2$ +{gap:.0f} vs combo)"
                           if best else None))
    ax.axhline(1, ls=":", color="0.6")
    ax.set_ylim(0, 30)
    ax.set_xlabel("redshift")
    ax.set_ylabel(r"$\Delta\chi^2$")

    ax = axes[1]
    ax.plot(zgrid, pz, "-", color="k", lw=1.4)
    ax.set_xlabel("redshift")
    ax.set_ylabel(r"$P(z)$")

    for ax in axes:
        if result.z_ml[iobj] > 0:
            ax.axvline(result.z_ml[iobj], color="C1", lw=1.2,
                       label=f"$z_{{\\rm ml}}$ = {result.z_ml[iobj]:.4f}")
        if result.z_fixed is not None:
            ax.axvline(result.z_fixed, color="C0", lw=1.0, ls="-.",
                       label=f"$z_{{\\rm fixed}}$ = {result.z_fixed:.4f}")
        if z_ref is not None:
            ax.axvline(z_ref, ls="--", color="k", lw=1.0,
                       label=f"ref = {z_ref:.4f}")
        ax.legend(fontsize=8)

    mode = result.config["eazy"]["mode"]
    fig.suptitle(f"{oid}: eazy redshift scan "
                 f"({len(result.template_names)} templates, {mode})")
    fig.tight_layout()

    out = _plots_dir(result, save_dir) / f"zscan_{oid}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def generate_plots(result: FitResult, *, registry: Registry,
                   z_ref: float | None = None) -> list[Path]:
    """All figures for every object in a run."""
    written = []
    for iobj in range(len(result.ids)):
        variants = (False, True) if result.z_fixed is not None else (False,)
        for fixed in variants:
            path = plot_sed(result, registry=registry, iobj=iobj,
                            fixed=fixed, z_ref=z_ref)
            if path:
                written.append(path)
        written.append(plot_zscan(result, iobj, z_ref=z_ref))
    return written
