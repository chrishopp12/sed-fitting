#!/usr/bin/env python3
"""
make_line_lists.py

Regenerate the Shipped Emission-Line Lists
---------------------------------------------------------

Writes the `linear` backend's gas line lists from one declared table, in the
format `sedfit/data/lines/` ships. Every wavelength carries its source, and
`--audit` reports where the sources disagree.

Data products:
    sedfit/data/lines/<set>.txt
        two columns, component name and rest VACUUM wavelength [A]

Requirements:
    numpy (audit only), $SPS_HOME is NOT needed

Notes:
    `--check` compares against the shipped lists and writes nothing.
    `--audit` compares every wavelength against sedfit/data/emlines_info.dat.
    Rationale in DESIGN.md.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "sedfit" / "data"
LINES = DATA / "lines"
EMLINES_DAT = DATA / "emlines_info.dat"

NAME_WIDTH = 25
# Beyond this separation the nearest reference line is a different transition.
MATCH_TOLERANCE_A = 2.0
FMT = "{name:<%d}{wave:.3f}" % NAME_WIDTH
C_KMS = 299792.458

# ------------------------------------
# Sources
# ------------------------------------
# STANDARD  NIST ASD vacuum (verified 2026-08-27 against ASD with show_av=3,
#           the all-vacuum switch -- the default returns AIR above 2000 A).
#           Every optical value agrees to better than 1 km/s, and MaNGA DAP's
#           elpmpl11.par matches all 20 digit-for-digit.
#           The LIMIT IS THE ATOMIC DATA, NOT THE TRANSCRIPTION: propagating
#           NIST's level uncertainties leaves a rest-wavelength floor of
#           ~0.2 km/s on [OI]/[OII]/[OIII]/Hbeta-gamma-delta, 0.4 on Halpha,
#           1.0 on [SII], 1.7 on [ArIII], 2.3 on [NeIII] and 2.7-2.9 on
#           [NII] and [SIII]. No S/N removes it; it belongs in a velocity
#           error budget, and it argues for weighting oxygen and Balmer above
#           [NII]/[SIII] rather than fitting them equally.
#           The Balmer values are NIST's intensity-weighted CENTRE OF GRAVITY,
#           correct for an unresolved profile. Some tables instead use the
#           Ritz wavelength of g-weighted mean levels, ~1.2 km/s higher; that
#           is a different convention, not a correction.
#           [NeIII]3869 disagrees with the Atomic Line List by 23 km/s. ALL
#           carries a stale 1D2 level; NIST and a refereed RR Tel measurement
#           both support the value here.
# MORTON    Morton vacuum, via analysis/lines.py's blue_side catalog, verified
#           against Cloudy at N II] 2141.7057.
#
# emlines_info.dat is deliberately NOT a source. It is FSPS's nebular
# bookkeeping list and sits +4.4 to +5.2 km/s high across the optical, which
# is larger than the whole error budget of a velocity fit. `--audit` measures
# that rather than trusting this comment.
STANDARD = "standard vacuum"
NIST = "NIST ASD vacuum, verified 2026-08-27"
# Where NIST's Ritz value rests on a poorly determined level, a direct
# measurement wins. Carry the stated uncertainty; these are NOT known to
# the 0.01 A the other entries are.
MEASURED = "direct measurement, see notes"

# (name, wave_vac_A, source). Ordered by wavelength within each block.
OPTICAL = [
    ("[OII]3727", 3727.092, STANDARD),
    ("[OII]3730", 3729.875, STANDARD),
    ("[NeIII]3869", 3869.856, STANDARD),
    ("Hdelta", 4102.892, STANDARD),
    ("Hgamma", 4341.684, STANDARD),
    ("[OIII]4363", 4364.436, STANDARD),
    ("HeII4686", 4687.015, STANDARD),
    ("Hbeta", 4862.683, STANDARD),
    ("[OIII]4959", 4960.295, STANDARD),
    ("[OIII]5007", 5008.240, STANDARD),
    ("[OI]6300", 6302.046, STANDARD),
    ("[OI]6364", 6365.535, STANDARD),
    ("[NII]6548", 6549.860, STANDARD),
    ("Halpha", 6564.608, STANDARD),
    ("[NII]6584", 6585.271, STANDARD),
    ("[SII]6716", 6718.295, STANDARD),
    ("[SII]6731", 6732.674, STANDARD),
    ("[ArIII]7136", 7137.757, STANDARD),
    ("[SIII]9069", 9071.100, STANDARD),
    ("[SIII]9531", 9533.200, STANDARD),
]

# Rest-UV, for redshifts past the optical list's reach. Every entry is given
# to two decimals or better at its source; the integer-valued entries of
# analysis/lines.py's curated set are deliberately EXCLUDED, because half an
# Angstrom is invisible on a plot and is 44-133 km/s in a fit.
ULTRAVIOLET = [
    ("Lyalpha", 1215.670, NIST),
    ("NIV]1486", 1486.500, NIST),
    ("CIV1548", 1548.200, NIST),
    # 1550.781, not 1550.770: Griesmann & Kling (2000) FTS, 1550.7812(2).
    ("CIV1550", 1550.781, NIST),
    # A 7-component blend spanning 1640.332-1640.533 = 36.6 km/s. This is the
    # gA-weighted centroid and it MOVES with the level populations; it cannot
    # be trusted below a few km/s however many decimals it carries.
    ("HeII1640", 1640.420, NIST),
    ("OIII]1661", 1660.809, NIST),
    ("OIII]1666", 1666.150, NIST),
    # NIST's Ritz 1814.559 rests on a 1S0 level ~2.5 cm-1 high; its Ne III
    # levels reproduce the optical lines to 0.003 A but this one does not.
    # Young+2011 (STIS) 1814.645 +- 0.037 and Bowen (1960) 1814.65 +- 0.01
    # agree. +-0.037 A is +-6 km/s: do not treat this line as 0.01 A known.
    ("[NeIII]1815", 1814.645, MEASURED),
    # Allowed E1 resonance doublet (3s 2S - 3p 2P), NOT forbidden. The
    # bracket notation these carried in analysis/lines.py implies the wrong
    # physics; they are Al III the way C IV and Mg II are.
    ("AlIII1855", 1854.718, NIST),
    ("AlIII1863", 1862.791, NIST),
    ("[CIII]1907", 1906.683, NIST),
    ("CIII]1909", 1908.734, NIST),
    # NOT 2141.706. N II has no transition there: the 3P-5S multiplet is
    # 2139.683 / 2143.450 vacuum, and 2141.7 is the AIR-frame A-weighted
    # blend centroid of the two -- wrong frame and not a transition. This is
    # the strong member. Dropping it entirely is equally defensible: it is a
    # weak intercombination line and every line is false-positive surface.
    ("NII]2143", 2143.450, NIST),
    ("MgII2796", 2796.352, NIST),
    ("MgII2803", 2803.531, NIST),
]

# Interstellar and wind absorption. NOT emission: these belong in a mask, and
# no scaled-solar basis reproduces them (DESIGN 17.7's fifth mask family).
INTERSTELLAR_UV = [
    ("FeII2344", 2344.213, NIST),
    ("FeII2374", 2374.460, NIST),
    ("FeII2383", 2382.764, NIST),
    ("FeII2587", 2586.649, NIST),
    ("FeII2600", 2600.172, NIST),
]

SETS = {
    "optical": (OPTICAL, "optical emission-line list"),
    "uv_optical": (sorted(ULTRAVIOLET + OPTICAL, key=lambda r: r[1]),
                   "rest-UV + optical emission-line list"),
    "interstellar_uv": (INTERSTELLAR_UV,
                        "rest-UV interstellar ABSORPTION list, for masking"),
}


# ------------------------------------
# Helpers
# ------------------------------------
def _render(rows: list, description: str) -> str:
    head = (f"# sedfit {description}; rest VACUUM wavelength [A]\n"
            f"# component            wave_vac_A\n")
    return head + "".join(FMT.format(name=n, wave=w) + "\n" for n, w, _ in rows)


def _first_difference(path: Path, built: str) -> str:
    shipped = path.read_text().splitlines()
    lines = built.splitlines()
    if len(shipped) != len(lines):
        return f"{len(shipped)} lines shipped, {len(lines)} built"
    for number, (a, b) in enumerate(zip(shipped, lines), 1):
        if a != b:
            return f"line {number}: {a!r} shipped, {b!r} built"
    return "identical"


def _read_emlines() -> list:
    rows = []
    for line in EMLINES_DAT.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        head, _, tail = line.partition(",")
        rows.append((float(head), tail.strip()))
    return rows


# ------------------------------------
# Commands
# ------------------------------------
def audit() -> int:
    """Report every wavelength against FSPS's list, worst offender last."""
    reference = _read_emlines()
    print(f"{len(reference)} lines in {EMLINES_DAT.name}; "
          f"comparing each shipped wavelength to its nearest")
    for name, (rows, _) in SETS.items():
        deltas, absent = [], []
        for component, wave, source in rows:
            near, label = min(reference, key=lambda r: abs(r[0] - wave))
            # Beyond this the "nearest" line is a different transition and the
            # velocity is meaningless. Absorption lists have no counterpart at
            # all, and reporting thousands of km/s for them is how an audit
            # teaches people to ignore it.
            if abs(near - wave) > MATCH_TOLERANCE_A:
                absent.append(component)
                continue
            deltas.append(((near - wave) / wave * C_KMS, component, label))
        if not deltas:
            print(f"  {name:16s} n={len(rows):3d}  no counterpart in "
                  f"{EMLINES_DAT.name} -- nothing to compare")
            continue
        deltas.sort(key=lambda d: abs(d[0]))
        worst = deltas[-1]
        median = deltas[len(deltas) // 2]
        note = f"   ({len(absent)} with no counterpart)" if absent else ""
        print(f"  {name:16s} n={len(rows):3d}  median |dv| {abs(median[0]):6.2f} "
              f"km/s   worst {worst[0]:+7.2f} km/s at {worst[1]}{note}")
    print("\nA non-zero median is expected and is not a defect in these lists: "
          "emlines_info.dat is FSPS's nebular bookkeeping and runs high in the "
          "optical. It is a drift detector, not an authority.")
    return 0


def generate(check: bool) -> int:
    mismatched = []
    for name, (rows, description) in SETS.items():
        waves = [w for _, w, _ in rows]
        if sorted(waves) != waves:
            raise ValueError(f"{name}: wavelengths are not increasing")
        if len({n for n, _, _ in rows}) != len(rows):
            raise ValueError(f"{name}: duplicate component names")
        built = _render(rows, description)
        path = LINES / f"{name}.txt"
        if check:
            if not path.exists():
                mismatched.append(f"{name}.txt: absent")
            elif path.read_text() != built:
                mismatched.append(f"{name}.txt: {_first_difference(path, built)}")
        else:
            LINES.mkdir(parents=True, exist_ok=True)
            path.write_text(built)

    if check:
        if mismatched:
            print(f"{len(mismatched)}/{len(SETS)} differ from the shipped lists:")
            for line in mismatched:
                print(f"  {line}")
            return 1
        print(f"all {len(SETS)} reproduce the shipped lists exactly")
        return 0
    for name, (rows, _) in SETS.items():
        print(f"wrote {name}.txt ({len(rows)} lines)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[4])
    parser.add_argument("--check", action="store_true",
                        help="compare against the shipped lists, write nothing")
    parser.add_argument("--audit", action="store_true",
                        help="report every wavelength against emlines_info.dat")
    args = parser.parse_args(argv)
    if args.audit:
        return audit()
    return generate(args.check)


if __name__ == "__main__":
    sys.exit(main())
