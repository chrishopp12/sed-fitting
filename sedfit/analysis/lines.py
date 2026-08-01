"""
lines.py

Spectral Line Catalogs and Plot Annotation
---------------------------------------------------------

Lines are (label, rest_wavelength_AA) pairs grouped into named catalogs
(LINE_CATALOGS). mark_catalogs overlays any selection on a matplotlib
axis; which catalogs a target draws is the caller's choice.
load_emission_lines refines the curated emission wavelengths against
the packaged FSPS emlines_info.dat list.

Requirements:
  - numpy
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_EMLINES_DAT = (Path(__file__).resolve().parents[1] / "data"
                       / "emlines_info.dat")


# ------------------------------------
# Catalog data
# ------------------------------------
# Every catalog is (label, vacuum rest wavelength in Angstrom), ordered
# by wavelength.

CURATED_EMISSION_LINES = [
    (r"Ly$\alpha$",    1216),
    ("C II]",          2326),
    ("Blnd",           2424),
    ("[O II]",         2471),
    ("[Al II]",        2660),
    ("[Al II]",        2670),
    ("Mg II]",         2796.352),
    ("Mg II]",         2803.531),
    ("Ar IV",          2854),
    ("[Ar III]",       3109),
    ("[Ne III]",       3343),
    ("[Ne V]",         3426),
    ("[O II]",         3727.09),
    ("H10",            3798.98),
    ("H9",             3836.48),
    ("[Ne III]",       3869.86),
    ("H8",             3890.15),
    (r"H$\epsilon$",   3971.20),
    ("[S II]",         4069.75),
    ("[S II]",         4077.50),
    (r"H$\delta$",     4102.89),
    (r"H$\gamma$",     4341.68),
    ("[O III]",        4364.44),
    (r"H$\beta$",      4862.68),
    ("[O III]",        4960.29),
    ("[O III]",        5008.24),
    ("[N II]",         6549.86),
    (r"H$\alpha$",     6564.61),
    ("[N II]",         6585.27),
    ("[S II]",         6718.29),
    ("[S II]",         6732.67),
]

DEIMOS_ABSORPTION_LINES = [
    ("Fe II",  2344.21),
    ("Fe II",  2374.46),
    ("Fe II",  2382.76),
    ("Fe II",  2586.65),
    ("Fe II",  2600.17),
]

FEII_STAR_LINES = [
    ("Fe II*", 2365.55),
    ("Fe II*", 2396.36),
    ("Fe II*", 2612.65),
    ("Fe II*", 2626.45),
    ("Fe II*", 2632.11),
]

BLUE_SIDE_LINES = [
    ("N IV]",    1486.50),
    ("C IV",     1548.20),
    ("C IV",     1550.77),
    ("He II",    1640.42),
    ("O III]",   1660.81),
    ("O III]",   1666.15),
    ("[Ne III]", 1814.56),
    ("[Al III]", 1854.72),
    ("[Al III]", 1862.79),
    ("[C III]",  1906.68),
    ("C III]",   1908.73),
    ("N II]",    2141.7057),
]


# -- Catalog registry -------------------------------------------------

@dataclass(frozen=True)
class LineCatalog:
    """A named set of lines with a default marker style.

    Attributes
    ----------
    lines : sequence of (str, float)
        ``(label, rest_wavelength_AA)`` pairs.
    color, linestyle, lw, alpha : matplotlib style for the vertical markers.
    highlight_wls : tuple of float
        Rest wavelengths drawn in ``highlight_color`` instead of ``color``
        (matched within 5 A).
    """
    lines: tuple
    color: str = "gray"
    linestyle: str = "-"
    lw: float = 0.7
    alpha: float = 0.6
    highlight_wls: tuple = ()
    highlight_color: str = "steelblue"


LINE_CATALOGS = {
    "curated_emission": LineCatalog(
        tuple(CURATED_EMISSION_LINES),
        color="gray", linestyle="-",
        highlight_wls=(3727.0,), highlight_color="steelblue"),
    "deimos_absorption": LineCatalog(
        tuple(DEIMOS_ABSORPTION_LINES),
        color="darkorange", linestyle="--", alpha=0.7),
    "feii_star": LineCatalog(
        tuple(FEII_STAR_LINES),
        color="mediumvioletred", linestyle="--", alpha=0.7),
    "blue_side": LineCatalog(
        tuple(BLUE_SIDE_LINES),
        color="dodgerblue", linestyle="--", alpha=0.7),
}


# -- Line loading -----------------------------------------------------

def _parse_emlines_dat(dat_path):
    """Parse the FSPS ``emlines_info.dat`` file.

    Expected format: ``wavelength,label`` per line, with ``#`` comments.

    Returns
    -------
    lines : list of (float, str)
        (wavelength, label) pairs.
    """
    lines = []
    with open(dat_path, encoding="utf-8") as f:
        for row in f:
            row = row.strip()
            if not row or row.startswith("#"):
                continue
            parts = row.split(",", 1)
            if len(parts) >= 2:
                try:
                    lines.append((float(parts[0]), parts[1].strip()))
                except ValueError:
                    continue
    return lines


def load_emission_lines(dat_path: str | Path | None = None
                        ) -> list[tuple[str, float]]:
    """Curated emission lines, refined to FSPS wavelengths when available.

    Both lists are vacuum, so the refinement moves each line by less than
    the rounding of its curated value. When *dat_path* points to an
    existing file, each curated wavelength is snapped to the nearest
    unclaimed FSPS line within 4 A, and a curated line with no match in
    that window is dropped. Otherwise the curated wavelengths are
    returned unchanged.

    Parameters
    ----------
    dat_path : str or Path or None
        Path to ``emlines_info.dat``. ``None`` uses the packaged copy.

    Returns
    -------
    lines : list of (str, float)
        ``(label, rest_wavelength)`` sorted by wavelength.
    """
    if dat_path is None:
        dat_path = DEFAULT_EMLINES_DAT
    else:
        dat_path = Path(dat_path)

    if not dat_path.is_file():
        print("  WARNING: emlines_info.dat not found - using curated "
              "wavelengths directly")
        return list(CURATED_EMISSION_LINES)

    all_dat = _parse_emlines_dat(dat_path)
    matched = []
    used = set()
    for label, target_wl in CURATED_EMISSION_LINES:
        cands = [(abs(dat_wl - target_wl), i)
                 for i, (dat_wl, _) in enumerate(all_dat)
                 if abs(dat_wl - target_wl) <= 4 and i not in used]
        if cands:
            _, best = min(cands)
            matched.append((label, all_dat[best][0]))
            used.add(best)
    matched.sort(key=lambda x: x[1])
    return matched


# -- Plot annotation --------------------------------------------------

def mark_lines(ax, lines, redshift, *, color="gray", linestyle="-",
               lw=0.7, alpha=0.6, xlim=None, label_lines=True,
               min_label_spacing_ang=80, highlight_wls=(),
               highlight_color="steelblue"):
    """Draw vertical markers for ``(label, rest_wl)`` lines on *ax*.

    Lines are redshifted to the observed frame; only those inside *xlim*
    are drawn.  Labels closer than *min_label_spacing_ang* to the previous
    one are dropped to avoid clutter.

    Parameters
    ----------
    ax : matplotlib Axes
    lines : sequence of (str, float)
        ``(label, rest_wavelength)`` pairs.
    redshift : float
    xlim : (float, float) or None
        Observed-frame draw range; defaults to the current axis limits.
    highlight_wls : tuple of float
        Rest wavelengths drawn in *highlight_color* (matched within 5 A).
    """
    a = 1.0 + redshift
    if xlim is None:
        xlim = ax.get_xlim()
    obs_lines = sorted(((wl * a, label, wl) for label, wl in lines),
                       key=lambda t: t[0])
    last_pos = -np.inf
    for obs_wl, label, rest_wl in obs_lines:
        if obs_wl < xlim[0] or obs_wl > xlim[1]:
            continue
        c = color
        if any(abs(rest_wl - h) < 5 for h in highlight_wls):
            c = highlight_color
        ax.axvline(obs_wl, color=c, ls=linestyle, lw=lw, alpha=alpha, zorder=1)
        if not label_lines or (obs_wl - last_pos) < min_label_spacing_ang:
            continue
        ax.text(obs_wl, ax.get_ylim()[1], f" {label}",
                rotation=90, va="top", ha="right", fontsize=6,
                color=c, alpha=0.8)
        last_pos = obs_wl


def mark_catalogs(ax, names, redshift, *, catalogs=None, emission_lines=None,
                  xlim=None, min_label_spacing_ang=80):
    """Overlay each catalog in *names* using its registered style.

    Parameters
    ----------
    ax : matplotlib Axes
    names : sequence of str
        Catalog keys into *catalogs* (default ``LINE_CATALOGS``).
    redshift : float
    catalogs : dict or None
        Name -> ``LineCatalog`` mapping; defaults to ``LINE_CATALOGS``.
    emission_lines : sequence of (str, float) or None
        FSPS-refined curated emission lines; used in place of the static
        ``curated_emission`` catalog when provided.
    xlim : (float, float) or None
        Observed-frame draw range.
    """
    if catalogs is None:
        catalogs = LINE_CATALOGS
    for name in names:
        cat = catalogs.get(name)
        if cat is None:
            print(f"  WARNING: unknown line catalog '{name}' - skipping")
            continue
        lines = cat.lines
        if name == "curated_emission" and emission_lines is not None:
            lines = emission_lines
        mark_lines(ax, lines, redshift, color=cat.color,
                   linestyle=cat.linestyle, lw=cat.lw, alpha=cat.alpha,
                   xlim=xlim, min_label_spacing_ang=min_label_spacing_ang,
                   highlight_wls=cat.highlight_wls,
                   highlight_color=cat.highlight_color)
