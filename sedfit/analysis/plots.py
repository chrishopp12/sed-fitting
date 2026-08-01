"""
plots.py

The Unified SED Figure
---------------------------------------------------------

One SED + chi-residual figure rendered from a backend-neutral
SEDPlotData (observed photometry grouped by instrument, model
photometry, optional model curves, residuals in sigma), plus the shared
style constants every backend figure imports. Producers live in the
backends; this module never touches fit objects.

Requirements:
  - numpy, matplotlib
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ------------------------------------
# Shared style
# ------------------------------------

INSTRUMENT_COLOR = {
    "GALEX": "darkviolet", "SDSS": "seagreen", "Legacy": "olivedrab",
    "CFHT": "steelblue", "JPLUS": "teal", "PS1": "darkgoldenrod",
    "WISE": "darkorange", "SPHEREx": "0.55",
}
INSTRUMENT_LABEL = {"JPLUS": "J-PLUS"}
SED_XTICKS = [1e3, 2e3, 3e3, 5e3, 1e4, 2e4, 3e4, 5e4, 1e5, 2e5]


# ------------------------------------
# Data contract
# ------------------------------------

@dataclass
class SEDPlotData:
    """Everything the unified figure needs, in plain arrays.

    curves are (wave_AA, fnu_uJy, label) model spectra; instruments are
    per-band registry instruments (the producer resolves them).
    """
    title: str
    bands: list[str]
    wave_AA: np.ndarray
    fobs: np.ndarray
    efobs: np.ndarray
    model_phot: np.ndarray
    instruments: list[str]
    curves: list[tuple] = field(default_factory=list)


# ------------------------------------
# Axis helpers
# ------------------------------------

def apply_sed_xaxis(ax, wave) -> None:
    """Log wavelength axis with the shared tick set [observed Angstrom]."""
    lo, hi = float(np.min(wave)) * 0.8, float(np.max(wave)) * 1.25
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ticks = [t for t in SED_XTICKS if lo <= t <= hi]
    if ticks:
        ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x:.0f}" if x < 1e4 else f"{x / 1e3:.0f}k"))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel(r"$\lambda_{\rm obs}$ ($\AA$)")


def _apply_yaxis(ax, flux, model, curves, pad=3.0) -> None:
    lo, hi = ax.get_xlim()
    values = [np.asarray(flux, float), np.asarray(model, float)]
    for wave, fnu, _ in curves:
        inview = (np.asarray(wave) >= lo) & (np.asarray(wave) <= hi)
        spec = np.asarray(fnu, float)[inview]
        spec = spec[np.isfinite(spec) & (spec > 0)]
        if spec.size:
            values.append(np.percentile(spec, [1, 99]))
    stacked = np.concatenate([np.atleast_1d(v) for v in values])
    stacked = stacked[np.isfinite(stacked) & (stacked > 0)]
    ax.set_yscale("log")
    if stacked.size:
        ax.set_ylim(stacked.min() / pad, stacked.max() * pad)


# ------------------------------------
# The figure
# ------------------------------------

def sed_figure(data: SEDPlotData, *, save_path: str | Path) -> Path:
    """Render the SED + chi-residual figure.

    Parameters
    ----------
    data : SEDPlotData
        The producer-supplied contents.
    save_path : str or Path
        Output PNG path.

    Returns
    -------
    png_path : Path
    """
    wave = np.asarray(data.wave_AA, float)
    fobs = np.asarray(data.fobs, float)
    efobs = np.asarray(data.efobs, float)
    model = np.asarray(data.model_phot, float)
    instruments = np.asarray(data.instruments)
    chi = (fobs - model) / efobs

    fig, axes = plt.subplots(2, 1, figsize=(11, 7),
                             gridspec_kw=dict(height_ratios=[1, 4]),
                             sharex=True)

    ax = axes[1]
    for curve_wave, curve_fnu, label in data.curves:
        ax.plot(curve_wave, curve_fnu, color="firebrick", lw=0.8, alpha=0.8,
                label=label)
    for inst in dict.fromkeys(instruments):
        m = instruments == inst
        is_sx = inst == "SPHEREx"
        ax.errorbar(wave[m], fobs[m], efobs[m], linestyle="", marker="o",
                    color=INSTRUMENT_COLOR.get(inst, "slategray"),
                    ms=3 if is_sx else 7, mec="none" if is_sx else "k",
                    mew=0.5, elinewidth=0.6, alpha=0.55 if is_sx else 0.95,
                    zorder=8 if is_sx else 10,
                    label=INSTRUMENT_LABEL.get(inst, inst))
    sx = instruments == "SPHEREx"
    if (~sx).any():
        ax.plot(wave[~sx], model[~sx], linestyle="", marker="x", color="k",
                ms=5, mew=1.0, zorder=11, label="Model phot")
    if sx.any():
        ax.plot(wave[sx], model[sx], linestyle="", marker="x", color="k",
                ms=2, mew=0.5, alpha=0.7, zorder=9)
    ax.set_ylabel(r"$f_\nu$ ($\mu$Jy)")
    apply_sed_xaxis(ax, wave)
    _apply_yaxis(ax, fobs, model, data.curves)
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.set_title(data.title)

    ax = axes[0]
    for inst in dict.fromkeys(instruments):
        m = instruments == inst
        is_sx = inst == "SPHEREx"
        ax.plot(wave[m], chi[m], linestyle="", marker="o",
                color=INSTRUMENT_COLOR.get(inst, "slategray"),
                ms=3 if is_sx else 5, alpha=0.55 if is_sx else 0.95)
    ax.axhline(0, color="k", linestyle=":")
    ax.set_ylabel(r"$\chi$")
    ax.set_ylim(-5, 5)

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path
