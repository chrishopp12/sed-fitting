"""
synthetic.py

Shared Test Fixtures
---------------------------------------------------------

Synthetic templates and photometry used by more than one test module.
Nothing here imports a backend stack, so a module that needs only these
helpers runs on a core install.

Requirements:
  - numpy, pandas
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sedfit.core.registry import load_registry
from sedfit.core.synth import synth_fnu
from sedfit.core.table import COLUMNS

PACKAGED_TEMPLATES = str(Path(__file__).resolve().parents[1]
                         / "sedfit" / "data" / "templates" / "brown14")

REG = load_registry()
BANDS = ("CFHT_u", "CFHT_g", "CFHT_r", "CFHT_i", "CFHT_z", "WISE_W1")
Z_TRUE = 0.10
NORM_UJY = 100.0


def fake_templates(tmp_path) -> Path:
    """Three break templates: flat fnu with a drop blueward of the break."""
    tdir = tmp_path / "templates"
    tdir.mkdir()
    wave = np.linspace(1000.0, 2.0e5, 8000)
    for i, (break_AA, drop) in enumerate([(5000.0, 0.2), (4000.0, 0.5),
                                          (7000.0, 0.1)]):
        fnu_shape = np.where(wave < break_AA, drop, 1.0)
        flam = fnu_shape / wave**2
        np.savetxt(tdir / f"t{i}_spec.dat", np.column_stack([wave, flam]))
    return tdir


def synth_frame() -> pd.DataFrame:
    """Photometry synthesized from template 0 redshifted to Z_TRUE."""
    spec_wave_um = np.linspace(0.1, 20.0, 20000)
    break_obs_AA = 5000.0 * (1 + Z_TRUE)
    fnu = np.where(spec_wave_um * 1e4 < break_obs_AA, 0.2, 1.0) * NORM_UJY
    rows = []
    for band in BANDS:
        wave, thru = REG.get_bandpass(band)
        flux, _ = synth_fnu(wave, thru, spec_wave_um, fnu)
        rows.append((band, flux, 0.01 * abs(flux),
                     np.nan, np.nan, np.nan, np.nan))
    frame = pd.DataFrame(rows, columns=list(COLUMNS[:-1]))
    frame["qa_flags"] = np.full(len(frame), np.nan, dtype=object)
    return frame
