"""
filters_io.py

FILTER.RES Generation
---------------------------------------------------------

Writes the run-local FILTER.RES from registry bandpasses and per-object
SPHEREx tophats, numbered in fit order, plus the translate file mapping
catalog columns to filter numbers. FILTER.RES is regenerated per run and
read only by that run's eazy instance.

Requirements:
  - numpy; eazy (lazy import)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from sedfit.core.synth import make_spherex_tophat

if TYPE_CHECKING:
    import pandas as pd

    from sedfit.core.registry import Registry


def build_filter_res(frame: pd.DataFrame, registry: Registry,
                     res_path: str | Path) -> dict[str, int]:
    """Write FILTER.RES (+ .info) for a table's bands, in table order.

    Parameters
    ----------
    frame : pd.DataFrame
        The SED table; SPHEREx rows supply their tophat geometry.
    registry : Registry
        Bandpass authority for the broadband rows.
    res_path : str or Path
        Output path; the .info sidecar is written next to it.

    Returns
    -------
    band_numbers : dict[str, int]
        Band name -> 1-based filter number (eazy translate convention).
    """
    from eazy.filters import FilterDefinition

    blocks = []
    info_lines = []
    band_numbers: dict[str, int] = {}
    for number, row in enumerate(frame.itertuples(index=False), start=1):
        band = str(row.band)
        if registry.is_spherex(band):
            wave, thru = make_spherex_tophat(float(row.wave_um),
                                             float(row.bandwidth_um))
        else:
            wave, thru = registry.get_bandpass(band)
        fdef = FilterDefinition(name=band, wave=np.asarray(wave, float),
                                throughput=np.asarray(thru, float))
        blocks.append(fdef.for_filter_file())
        info_lines.append(f"{number:4d} {band:16s} lambda_c= {fdef.pivot:.4e}")
        band_numbers[band] = number

    res_path = Path(res_path)
    # A stale FILTER.RES.npy sidecar would silently override the text file.
    stale = Path(f"{res_path}.npy")
    if stale.exists():
        stale.unlink()
    res_path.write_text("\n".join(blocks) + "\n", encoding="utf-8")
    Path(f"{res_path}.info").write_text("\n".join(info_lines) + "\n", encoding="utf-8")
    return band_numbers


def write_translate(band_numbers: dict[str, int],
                    translate_path: str | Path) -> None:
    """Write the translate file: one f_<band>/e_<band> pair per filter.

    Each catalog column is mapped to its eazy filter number, flux
    columns as F<n> and error columns as E<n>. eazy-py requires a real
    file here rather than None.
    """
    lines = []
    for band, number in band_numbers.items():
        lines.append(f"f_{band} F{number}")
        lines.append(f"e_{band} E{number}")
    Path(translate_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
