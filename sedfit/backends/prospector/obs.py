"""
obs.py

Observation-Dictionary Construction
---------------------------------------------------------

Builds the Prospector obs dictionary from a policy-applied SED table:
converts the policy's uJy flux and error vectors to maggies and assembles
the matching sedpy filter list. Magnification and the error floor are
applied upstream, in core policy. Prospector has no missing-band concept,
so non-fittable rows are omitted from obs; the accounting records them.

Requirements:
  - numpy, pandas, sedpy, prospect (lazy)

Notes:
  - Filters are constructed as Filter(band, data=...) with the curve
    supplied directly from the registry, and SPHEREx channels as
    core.synth tophats -- the one tophat implementation.
  - make_exact is applied before AND after fix_obs, which may rebuild
    the filter list.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from sedfit.backends.prospector.exact_filter import make_exact
from sedfit.core.policy import PolicyResult
from sedfit.core.spectrum import Spectrum, apply_spectrum_policy
from sedfit.core.synth import make_spherex_tophat

if TYPE_CHECKING:
    import pandas as pd

    from sedfit.core.registry import Registry

UJY_PER_MAGGIE = 3.631e9    # 0 AB mag = 1 maggie = 3.631e9 uJy


def build_obs(
        cfg: dict,
        frame: pd.DataFrame,
        policy: PolicyResult,
        *,
        registry: Registry,
        spectrum: Spectrum | None = None,
    ) -> dict:
    """The Prospector obs dictionary for one policy-applied table.

    Parameters
    ----------
    cfg : dict
        Resolved fit config (backend "prospector").
    frame : pd.DataFrame
        The validated SED table (supplies SPHEREx channel geometry).
    policy : PolicyResult
        core.policy output; its flux/err vectors are consumed verbatim.
    registry : Registry
        Bandpass authority.
    spectrum : Spectrum or None
        core.spectrum.read_spectrum output for a joint fit; the config's
        spectrum block supplies its policy (err_floor, mask_windows) and
        the shared mu_lensing is applied to it here, exactly as core
        policy applies it to the photometry. [default: None]

    Returns
    -------
    obs : dict
        prospect-ready (fix_obs applied); fittable bands only.
    """
    from sedpy.observate import Filter
    from prospect.utils.obsutils import fix_obs

    fittable = np.asarray(policy.fittable, bool)
    bands = [b for b, ok in zip(policy.bands, fittable) if ok]
    geometry = {str(row.band): (row.wave_um, row.bandwidth_um)
                for row in frame.itertuples(index=False)}

    filters = []
    for band in bands:
        if registry.is_spherex(band):
            wave_um, bandwidth_um = geometry[band]
            data = make_spherex_tophat(float(wave_um), float(bandwidth_um))
        else:
            data = registry.get_bandpass(band)
        filters.append(Filter(band, data=data))

    exact = cfg["prospector"]["exact_filters"]
    if exact:
        make_exact(filters)

    obs = {
        "filters": filters,
        "maggies": policy.flux_uJy[fittable] / UJY_PER_MAGGIE,
        "maggies_unc": policy.flux_err_uJy[fittable] / UJY_PER_MAGGIE,
        "spectrum": None,
        "wavelength": None,
        "unc": None,
        "redshift": cfg["z_ref"],
    }
    if spectrum is not None:
        spec_cfg = cfg["prospector"]["spectrum"]
        flux, err, mask = apply_spectrum_policy(
            spectrum,
            mu_lensing=cfg["mu_lensing"],
            err_floor=spec_cfg["err_floor"],
            mask_windows=spec_cfg["mask_windows"])
        if int(mask.sum()) <= spec_cfg["polyorder"]:
            raise ValueError(
                f"{int(mask.sum())} unmasked channels cannot constrain a "
                f"polyorder-{spec_cfg['polyorder']} calibration polynomial")
        obs["wavelength"] = spectrum.wave_A.copy()
        obs["spectrum"] = flux / UJY_PER_MAGGIE
        obs["unc"] = err / UJY_PER_MAGGIE
        obs["mask"] = mask
    obs = fix_obs(obs)
    if exact:                       # fix_obs may rebuild the filter list
        make_exact(obs["filters"])
    return obs
