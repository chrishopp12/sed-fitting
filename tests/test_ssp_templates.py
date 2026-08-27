"""Shipped SSP grids agree with the generator's constants.

Does not run FSPS; `tools/make_ssp_templates.py --check` does that, in an env
with the matching compiled library.
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
TOOL = REPO / "tools" / "make_ssp_templates.py"
GRIDS = ("ssp_miles", "ssp_c3k_a")

HEADER = re.compile(
    r"^# FSPS \S+ SSP; isochrones/spectra \('(?P<iso>\w+)', '(?P<lib>\w+)'\); "
    r"imf_type \d+; logzsol (?P<z>[+-]\d\.\d\d); tage (?P<age>[\d.]+) Gyr; "
    r"sfh 0; no nebular; rest wavelength \[A\] VACUUM as shipped by FSPS; "
    r"f_lambda \[L_sun/A\], arbitrary normalization$")

RESOLUTION_HEADER = re.compile(
    r"^# FSPS \S+ SSP line-spread width; isochrones/spectra "
    r"\('(?P<iso>\w+)', '(?P<lib>\w+)'\); rest wavelength \[A\] VACUUM as "
    r"shipped by FSPS; sigma \[km/s\] of the spectral library at that "
    r"wavelength, decoded as \|value\| from (?P<source>.+)$")

# The window each library is itself over, safely inside its boundaries.
C3K_FLAT_RANGE = (3000.0, 9000.0)
C3K_R = 3000.0
MILES_FLAT_RANGE = (4000.0, 7000.0)


def _resolution(tool, grid: str):
    table = np.loadtxt(tool.DATA / grid / tool.RESOLUTION_NAME)
    return table[:, 0], table[:, 1]


def _tool():
    spec = importlib.util.spec_from_file_location("make_ssp_templates", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tool():
    return _tool()


@pytest.mark.parametrize("grid", GRIDS)
def test_grid_is_complete(tool, grid: str) -> None:
    directory = tool.DATA / grid
    files = sorted(directory.glob("*.dat"))
    assert len(files) == len(tool.AGES_GYR) * len(tool.LOGZSOL)
    library = grid[len("ssp_"):]
    expected = {f"ssp_mist_{library}_t{age:04.1f}_z{logzsol:+.2f}.dat"
                for age in tool.AGES_GYR for logzsol in tool.LOGZSOL}
    assert {path.name for path in files} == expected


@pytest.mark.parametrize("grid", GRIDS)
def test_header_matches_filename(tool, grid: str) -> None:
    for path in sorted((tool.DATA / grid).glob("*.dat")):
        match = HEADER.match(path.read_text().splitlines()[0])
        assert match, f"{path.name}: header does not match the written form"
        stem = f"ssp_{match['iso']}_{match['lib']}"
        assert path.name.startswith(stem), f"{path.name}: header library differs"
        assert f"_t{float(match['age']):04.1f}_" in path.name
        assert f"_z{float(match['z']):+.2f}." in path.name


@pytest.mark.parametrize("grid", GRIDS)
def test_wavelength_grid_is_the_documented_cut(tool, grid: str) -> None:
    low, high = tool.WAVE_RANGE
    spans, counts = set(), set()
    for path in sorted((tool.DATA / grid).glob("*.dat")):
        wave = np.loadtxt(path)[:, 0]
        assert wave[0] >= low and wave[-1] <= high
        assert np.all(np.diff(wave) > 0)
        spans.add((wave[0], wave[-1]))
        counts.add(len(wave))
    assert len(spans) == 1, f"{grid}: files disagree on the wavelength span"
    assert len(counts) == 1, f"{grid}: files disagree on the row count"


@pytest.mark.parametrize("grid", GRIDS)
def test_rows_are_written_in_the_declared_format(tool, grid: str) -> None:
    path = next(iter(sorted((tool.DATA / grid).glob("*.dat"))))
    for line in path.read_text().splitlines()[1:]:
        wave, flux = (float(value) for value in line.split())
        assert line == tool.FMT % (wave, flux)


@pytest.mark.parametrize("grid", GRIDS)
def test_resolution_shares_the_template_grid(tool, grid: str) -> None:
    wave, sigma = _resolution(tool, grid)
    for path in sorted((tool.DATA / grid).glob("*.dat")):
        assert np.array_equal(np.loadtxt(path)[:, 0], wave)
    assert np.all(np.isfinite(sigma)) and np.all(sigma > 0)


@pytest.mark.parametrize("grid", GRIDS)
def test_resolution_is_not_swept_up_as_a_template(tool, grid: str) -> None:
    # The linear backend's default template_pattern is "*.dat", so the
    # resolution curve must not carry that extension.
    assert not tool.RESOLUTION_NAME.endswith(".dat")
    assert (tool.DATA / grid / tool.RESOLUTION_NAME) not in set(
        (tool.DATA / grid).glob("*.dat"))


@pytest.mark.parametrize("grid", GRIDS)
def test_resolution_header_names_its_library(tool, grid: str) -> None:
    path = tool.DATA / grid / tool.RESOLUTION_NAME
    match = RESOLUTION_HEADER.match(path.read_text().splitlines()[0])
    assert match, f"{grid}: resolution header does not match the written form"
    assert grid == f"ssp_{match['lib']}"
    assert match["source"].strip()


@pytest.mark.parametrize("grid", GRIDS)
def test_resolution_rows_are_written_in_the_declared_format(
        tool, grid: str) -> None:
    path = tool.DATA / grid / tool.RESOLUTION_NAME
    for line in path.read_text().splitlines()[1:]:
        wave, sigma = (float(value) for value in line.split())
        assert line == tool.RESOLUTION_FMT % (wave, sigma)


def test_c3k_is_flat_in_velocity(tool) -> None:
    # Constant sigma_kms is what makes a library's width z-invariant: a
    # redshift is a uniform stretch in ln lambda.
    wave, sigma = _resolution(tool, "ssp_c3k_a")
    inside = (wave >= C3K_FLAT_RANGE[0]) & (wave <= C3K_FLAT_RANGE[1])
    assert np.allclose(sigma[inside], sigma[inside][0])
    resolving = tool.C_KMS / (sigma[inside][0] * tool.FWHM_PER_SIGMA)
    assert resolving == pytest.approx(C3K_R, abs=1.0)


def test_miles_is_flat_in_wavelength(tool) -> None:
    wave, sigma = _resolution(tool, "ssp_miles")
    inside = (wave >= MILES_FLAT_RANGE[0]) & (wave <= MILES_FLAT_RANGE[1])
    fwhm = sigma[inside] * tool.FWHM_PER_SIGMA * wave[inside] / tool.C_KMS
    assert np.allclose(fwhm, tool.MILES_FWHM_A)
    # ... and therefore NOT flat in velocity, unlike C3K
    assert not np.allclose(sigma[inside], sigma[inside][0])
