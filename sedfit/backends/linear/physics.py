"""
physics.py

Physical Coherence of a Fitted Emission-Line Set
---------------------------------------------------------

The gas basis constrains almost nothing: NNLS enforces non-negativity and the
ratio-locked groups fix two same-upper-level doublets. Everything else is a
free amplitude, so a set of line fluxes can come back internally impossible --
Hgamma brighter than Hbeta, a density doublet outside its own limits, an
auroral line rivalling its nebular partner.

This module REPORTS those, it does not prevent them. Nothing here changes a
fitted value, so a fit's numbers are identical whether or not it is called.

Requirements:
    numpy

Notes:
    Rationale in DESIGN.md section 17.
"""
from __future__ import annotations

from dataclasses import dataclass

C_KMS = 299792.458

# ------------------------------------
# Bounds
# ------------------------------------
# Case B recombination at 1e4 K. Dust reddening only ever INCREASES each of
# these, because the numerator is the redder line -- so each is a one-sided
# LOWER bound that holds for any non-negative extinction, with no dust law
# assumed and nothing tuned. This is what makes the Balmer series checkable
# without knowing the reddening: the decrement is a measurement, its
# ORDERING is not.
BALMER_MINIMUM = (
    ("Halpha", "Hbeta", 2.86),
    ("Hbeta", "Hgamma", 2.13),
    ("Hgamma", "Hdelta", 1.81),
)

# Two-sided, because these are density diagnostics bounded by their own low-
# and high-density limits.
#
# The limits carry a MARGIN and it is not cosmetic. A genuinely diffuse
# source sits AT the low-density asymptote, so measurement noise scatters it
# across the boundary about half the time -- a real RM J0019 arc fits to
# [OII]3730/[OII]3727 = 1.511 against a theoretical limit of 1.50, and a
# hard bound there would flag correct physics on the one confirmed-good
# detection we have. The margin is 10% of the limit, chosen to clear that
# scatter while a grossly broken ratio (3.0, an inverted doublet) still
# fires. Tighten it only with flux errors in hand, which would let the test
# ask the right question: is the ratio outside the limit by more than its
# own uncertainty?
_DENSITY_MARGIN = 0.10
DENSITY_DOUBLETS = (
    ("[SII]6716", "[SII]6731", 0.44, 1.45),
    ("[OII]3730", "[OII]3727", 0.35, 1.50),
)

# Auroral over nebular, a temperature diagnostic. Real photoionized gas sits
# near 0.01-0.02 and reaches ~0.05 only at extreme temperature; the bound is
# set well above that so it flags a broken fit rather than a hot one.
AURORAL_MAXIMUM = (
    ("[OIII]4363", "[OIII]5007", 0.20),
)

# Rest-wavelength uncertainty from NIST level data, verified 2026-08-27. It
# is a floor on any velocity derived from these lines and no S/N removes it,
# so a joint fit weighting a 2.9 km/s species equally with a 0.2 km/s one is
# letting the worst-known lines set the answer.
REST_FLOOR_KMS = {
    "[OII]3727": 0.2, "[OII]3730": 0.2, "[NeIII]3869": 2.3,
    "Hdelta": 0.2, "Hgamma": 0.2, "[OIII]4363": 0.2, "HeII4686": 0.2,
    "Hbeta": 0.2, "[OIII]4959": 0.2, "[OIII]5007": 0.2,
    "[OI]6300": 0.2, "[OI]6364": 0.2, "[NII]6548": 2.9, "Halpha": 0.4,
    "[NII]6584": 2.9, "[SII]6716": 1.0, "[SII]6731": 1.0,
    "[ArIII]7136": 1.7, "[SIII]9069": 2.9, "[SIII]9531": 2.9,
}

# Below this a line is not measured well enough for a ratio to mean anything.
_MIN_FLUX_FRACTION = 1e-3


@dataclass(frozen=True)
class Violation:
    """One physical bound a fitted line set does not respect."""

    kind: str
    lines: tuple[str, ...]
    observed: float | None
    bound: float
    detail: str


def _usable(fluxes: dict, *names: str) -> bool:
    scale = max(fluxes.values()) if fluxes else 0.0
    return all(name in fluxes and fluxes[name] > _MIN_FLUX_FRACTION * scale
               for name in names)


def check_ratios(fluxes: dict) -> list[Violation]:
    """Bounds a fitted line set violates. Empty means nothing was checkable.

    Parameters
    ----------
    fluxes : dict
        Component name to fitted line flux, e.g. `LinearFit.gas_fluxes`.
        Names absent or at negligible flux are skipped rather than failed:
        a line that was not measured cannot violate anything.

    Returns
    -------
    violations : list of Violation
    """
    out: list[Violation] = []
    for red, blue, floor in BALMER_MINIMUM:
        if not _usable(fluxes, red, blue):
            continue
        ratio = fluxes[red] / fluxes[blue]
        if ratio < floor:
            out.append(Violation(
                "balmer", (red, blue), ratio, floor,
                f"{red}/{blue} = {ratio:.3f} is below the case B minimum "
                f"{floor}; dust can only raise this ratio, so no reddening "
                f"explains it"))
    for a, b, low, high in DENSITY_DOUBLETS:
        if not _usable(fluxes, a, b):
            continue
        ratio = fluxes[a] / fluxes[b]
        lo, hi = low * (1 - _DENSITY_MARGIN), high * (1 + _DENSITY_MARGIN)
        if not lo <= ratio <= hi:
            out.append(Violation(
                "density", (a, b), ratio, lo if ratio < lo else hi,
                f"{a}/{b} = {ratio:.3f} is outside the density-diagnostic "
                f"range [{low}, {high}] by more than the {_DENSITY_MARGIN:.0%} "
                f"margin a source sitting at a limit needs"))
    for auroral, nebular, ceiling in AURORAL_MAXIMUM:
        if not _usable(fluxes, auroral, nebular):
            continue
        ratio = fluxes[auroral] / fluxes[nebular]
        if ratio > ceiling:
            out.append(Violation(
                "temperature", (auroral, nebular), ratio, ceiling,
                f"{auroral}/{nebular} = {ratio:.3f} exceeds {ceiling}, which "
                f"no photoionized temperature reaches"))
    return out


def check_presence(present, in_band) -> list[Violation]:
    """Balmer lines detected without a brighter partner that was reachable.

    Flux-free, and that is the point: every ratio in `BALMER_MINIMUM` is a
    one-sided lower bound, so the redder line is ALWAYS the brighter one. If
    the fainter member was detected and the brighter was in band and was not,
    the set is incoherent whatever the fluxes were.

    Parameters
    ----------
    present : iterable of str
        Component names identified in this solution.
    in_band : iterable of str
        Component names whose wavelength falls in the observed range.

    Returns
    -------
    violations : list of Violation
    """
    present, in_band = set(present), set(in_band)
    out: list[Violation] = []
    for brighter, fainter, floor in BALMER_MINIMUM:
        if fainter in present and brighter in in_band \
                and brighter not in present:
            out.append(Violation(
                "absence", (brighter, fainter), None, floor,
                f"{fainter} is identified but {brighter} is in band and is "
                f"not, and {brighter} is at least {floor}x brighter"))
    return out


def velocity_floor_kms(names) -> float | None:
    """The worst rest-wavelength floor among these lines, or None.

    A velocity from a joint fit cannot be better than its worst-known line
    unless the fit is weighted, which this backend does not do.
    """
    floors = [REST_FLOOR_KMS[n] for n in names if n in REST_FLOOR_KMS]
    return max(floors) if floors else None
