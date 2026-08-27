#!/usr/bin/env python3
"""
make_ssp_templates.py

Regenerate a Shipped FSPS SSP Template Grid
---------------------------------------------------------

Writes the age x metallicity SSP grid for whichever spectral library the
active python-fsps build carries, in the format `sedfit/data/templates/`
ships. Run in `sedfit_miles` for `ssp_miles`, `sedfit_c3k` for `ssp_c3k_a`.

Data products:
    sedfit/data/templates/ssp_<library>/ssp_<isochrones>_<library>_t<age>_z<logzsol>.dat
        two columns, rest VACUUM wavelength [A] and f_lambda [L_sun/A]
    sedfit/data/templates/ssp_<library>/resolution.txt
        two columns, rest VACUUM wavelength [A] and sigma [km/s]

Requirements:
    numpy, python-fsps (compiled against the target spectral library),
    $SPS_HOME

Notes:
    `--check` compares against the shipped grid and writes nothing.
    Rationale in DESIGN.md.
"""
from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path

import numpy as np

# ------------------------------------
# Grid
# ------------------------------------

AGES_GYR = [1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 13.7]
LOGZSOL = [-0.5, -0.25, 0.0, 0.25, 0.5]

# FSPS runs 91 A to 1e8 A. The shipped grids are cut to this.
WAVE_RANGE = (900.0, 120000.0)

FMT = "%14.4f %.6e"

DATA = Path(__file__).resolve().parent.parent / "sedfit" / "data" / "templates"


# ------------------------------------
# Resolution
# ------------------------------------

C_KMS = 299792.458
FWHM_PER_SIGMA = 2.3548

# Not "resolution.dat": the linear backend's default template_pattern is
# "*.dat", so a .dat here loads as one more template.
RESOLUTION_NAME = "resolution.txt"
RESOLUTION_FMT = "%14.4f %12.4f"

# The .res files hold |sigma| in km/s; FSPS's sign marks whether the
# underlying specification was constant in velocity (+) or in wavelength (-).
RES_SOURCES = {"c3k_a": ("C3K", "c3k_afe+0.0")}

# MILES ships no .res. FSPS splices it into the BaSeL grid, and the window it
# actually uses is narrower than MILES's published 3525-7500 A, so the window
# is read off the grid instead of declared.
MILES_FWHM_A = 2.51
MILES_FINE_STEP_A = 2.0


# ------------------------------------
# Helpers
# ------------------------------------
def _population():
    """An SSP StellarPopulation and its (isochrones, spectra) libraries."""
    import fsps
    pop = fsps.StellarPopulation(zcontinuous=1, sfh=0, add_neb_emission=False)
    libraries = tuple(name.decode() for name in pop.libraries[:2])
    return pop, libraries, fsps.__version__


def _header(version: str, libraries: tuple[str, str], imf_type: int,
            logzsol: float, age: float) -> str:
    return (f"FSPS {version} SSP; isochrones/spectra {libraries}; "
            f"imf_type {imf_type}; logzsol {logzsol:+.2f}; tage {age} Gyr; "
            f"sfh 0; no nebular; rest wavelength [A] VACUUM as shipped by "
            f"FSPS; f_lambda [L_sun/A], arbitrary normalization")


def _filename(libraries: tuple[str, str], age: float, logzsol: float) -> str:
    return f"ssp_{libraries[0]}_{libraries[1]}_t{age:04.1f}_z{logzsol:+.2f}.dat"


def _render(table: np.ndarray, header: str, fmt: str = FMT) -> str:
    """The file text `np.savetxt` would write for this table and header."""
    buffer = io.StringIO()
    np.savetxt(buffer, table, fmt=fmt, header=header)
    return buffer.getvalue()


def _first_difference(path: Path, built: str) -> str:
    shipped_lines = path.read_text().splitlines()
    built_lines = built.splitlines()
    if len(shipped_lines) != len(built_lines):
        return f"{len(shipped_lines)} lines shipped, {len(built_lines)} built"
    for number, (shipped, line) in enumerate(zip(shipped_lines, built_lines), 1):
        if shipped != line:
            return f"line {number}: {shipped!r} shipped, {line!r} built"
    return "identical"


def _spectra_dir() -> Path:
    home = os.environ.get("SPS_HOME")
    if not home:
        raise RuntimeError("$SPS_HOME is unset; the resolution curves are read "
                           "from $SPS_HOME/SPECTRA")
    return Path(home) / "SPECTRA"


def _res_curve(directory: str, stem: str) -> tuple[np.ndarray, np.ndarray]:
    """One library's (wavelength, sigma_kms), decoded from its .res file."""
    base = _spectra_dir() / directory / stem
    wave = np.loadtxt(f"{base}.lambda")
    res = np.loadtxt(f"{base}.res")
    if wave.shape != res.shape:
        raise ValueError(f"{stem}: {wave.size} wavelengths against {res.size} "
                         f"resolutions")
    return wave, np.abs(res)


def _longest_run(flags: np.ndarray) -> tuple[int, int]:
    """First and last index of the longest contiguous True run."""
    best = (0, 0, 0)
    index = 0
    while index < len(flags):
        if flags[index]:
            stop = index
            while stop + 1 < len(flags) and flags[stop + 1]:
                stop += 1
            if stop - index + 1 > best[0]:
                best = (stop - index + 1, index, stop)
            index = stop + 1
        else:
            index += 1
    if not best[0]:
        raise ValueError("no finely sampled run found")
    return best[1], best[2]


def _miles_curve() -> tuple[np.ndarray, np.ndarray, str]:
    wave = np.loadtxt(_spectra_dir() / "MILES" / "miles.lambda")
    basel_wave, basel_sigma = _res_curve("BaSeL3.1", "basel")
    sigma = np.interp(wave, basel_wave, basel_sigma)
    start, stop = _longest_run(np.diff(wave) < MILES_FINE_STEP_A)
    low, high = float(wave[start]), float(wave[stop + 1])
    inside = (wave >= low) & (wave <= high)
    sigma[inside] = C_KMS * (MILES_FWHM_A / FWHM_PER_SIGMA) / wave[inside]
    return wave, sigma, (f"MILES {MILES_FWHM_A} A FWHM over {low:.2f}-{high:.2f} "
                         f"A, SPECTRA/BaSeL3.1/basel.{{lambda,res}} outside it")


def _resolution(libraries: tuple[str, str],
                wave: np.ndarray) -> tuple[np.ndarray, str]:
    """sigma_kms at each template wavelength, and where it came from."""
    library = libraries[1]
    if library in RES_SOURCES:
        directory, stem = RES_SOURCES[library]
        curve_wave, curve_sigma = _res_curve(directory, stem)
        source = f"SPECTRA/{directory}/{stem}.{{lambda,res}}"
    elif library == "miles":
        curve_wave, curve_sigma, source = _miles_curve()
    else:
        raise ValueError(f"no resolution source known for spectral library "
                         f"{library!r}; add one to RES_SOURCES")
    if wave.min() < curve_wave.min() or wave.max() > curve_wave.max():
        raise ValueError(f"{library}: the template grid "
                         f"{wave.min():.1f}-{wave.max():.1f} A leaves the "
                         f"resolution curve {curve_wave.min():.1f}-"
                         f"{curve_wave.max():.1f} A")
    return np.interp(wave, curve_wave, curve_sigma), source


def _resolution_header(version: str, libraries: tuple[str, str],
                       source: str) -> str:
    return (f"FSPS {version} SSP line-spread width; isochrones/spectra "
            f"{libraries}; rest wavelength [A] VACUUM as shipped by FSPS; "
            f"sigma [km/s] of the spectral library at that wavelength, "
            f"decoded as |value| from {source}")


def _spectrum(pop, age: float) -> np.ndarray:
    wave, flam = pop.get_spectrum(tage=age, peraa=True)
    keep = (wave >= WAVE_RANGE[0]) & (wave <= WAVE_RANGE[1])
    return np.c_[wave[keep], flam[keep]]


# ------------------------------------
# Commands
# ------------------------------------
def generate(pop, libraries: tuple[str, str], version: str,
             out_dir: Path, check: bool) -> int:
    imf_type = int(pop.params["imf_type"])
    print(f"fsps {version}, libraries {libraries}, imf_type {imf_type}")
    if not check:
        out_dir.mkdir(parents=True, exist_ok=True)

    mismatched = []
    grid_wave = None
    for logzsol in LOGZSOL:
        pop.params["logzsol"] = logzsol
        for age in AGES_GYR:
            table = _spectrum(pop, age)
            grid_wave = table[:, 0]
            name = _filename(libraries, age, logzsol)
            header = _header(version, libraries, imf_type, logzsol, age)
            path = out_dir / name
            if check:
                if not path.exists():
                    mismatched.append(f"{name}: absent")
                    continue
                # Compare what WOULD be written, not the in-memory floats:
                # FMT keeps 7 significant digits, so a shipped value read back
                # never equals its float64 source bit for bit.
                built = _render(table, header)
                if path.read_text() != built:
                    mismatched.append(f"{name}: {_first_difference(path, built)}")
            else:
                path.write_text(_render(table, header))

    sigma, source = _resolution(libraries, grid_wave)
    resolution = _render(np.c_[grid_wave, sigma],
                         _resolution_header(version, libraries, source),
                         fmt=RESOLUTION_FMT)
    resolution_path = out_dir / RESOLUTION_NAME
    if check:
        if not resolution_path.exists():
            mismatched.append(f"{RESOLUTION_NAME}: absent")
        elif resolution_path.read_text() != resolution:
            mismatched.append(f"{RESOLUTION_NAME}: "
                              f"{_first_difference(resolution_path, resolution)}")
    else:
        resolution_path.write_text(resolution)

    count = len(AGES_GYR) * len(LOGZSOL) + 1
    if check:
        if mismatched:
            print(f"{len(mismatched)}/{count} differ from the shipped grid:")
            for line in mismatched:
                print(f"  {line}")
            return 1
        print(f"all {count} reproduce the shipped grid exactly")
        return 0
    print(f"wrote {count - 1} templates and {RESOLUTION_NAME} to {out_dir}")
    print(f"  resolution from {source}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[4])
    parser.add_argument("--check", action="store_true",
                        help="compare against the shipped grid, write nothing")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="override the destination [default: the shipped "
                             "directory for the active library]")
    args = parser.parse_args(argv)

    pop, libraries, version = _population()
    out_dir = args.out_dir or DATA / f"ssp_{libraries[1]}"
    return generate(pop, libraries, version, out_dir, args.check)


if __name__ == "__main__":
    sys.exit(main())
