"""
fitting.py

Official eazy-py Engine
---------------------------------------------------------

Builds a self-contained run directory (catalog, translate, FILTER.RES,
templates.param, TEF curve, config echo), constructs eazy.photoz.PhotoZ,
runs the official fit (all-template photo-z, optional single-template
mode, optional fixed-z evaluation), and captures the state into a
FitResult. Fixed-z evaluation goes through fit_at_zbest with FIX_ZSPEC
off; the redshift must lie strictly inside the realized grid, which this
module enforces (eazy skips edge values silently).

Data products (per run directory):
  catalog.csv  zphot.translate  FILTER.RES(.info)
  templates.param (directory-mode template sets only)
  template_error.dat  config.json  zphot.param.echo
  summary.csv / arrays.npz [/ singles.csv]

Requirements:
  - numpy, pandas, astropy; eazy (lazy import)

Notes:
  - The floor is applied once, in core policy: SYS_ERR is pinned to 0.
  - Missing sentinel -1e30 pairs with NOT_OBS_THRESHOLD = -9e29; eazy's
    default threshold is -90 in catalog units (uJy), inside the range of
    real negative fluxes.
  - ARRAY_NBITS = 64: float32 underflow makes single-template fits NaN
    (eazy-py 0.8.6).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from sedfit.backends.eazy.filters_io import build_filter_res, write_translate
from sedfit.backends.eazy.results import (
    FitResult,
    Z_PERCENTILES,
    extract_sed,
    percentiles_from_lnp,
    save_outputs,
)
from sedfit.backends.eazy.templates import prepare_templates_param, resolve_tef
from sedfit.core.policy import MISSING_SENTINEL, PolicyResult

if TYPE_CHECKING:
    from sedfit.core.registry import Registry

# ------------------------------------
# Constants
# ------------------------------------

BASE_EAZY_PARAMS = {
    "PRIOR_ABZP": 23.9,             # catalog flux of 1 == AB 23.9 == 1 uJy
    "CAT_HAS_EXTCORR": "y",
    "MW_EBV": 0.0,
    "FIX_ZSPEC": "n",
    "APPLY_PRIOR": "n",
    "VERBOSITY": 1,
    "ARRAY_NBITS": 64,
    "NOT_OBS_THRESHOLD": -9e29,
    "SYS_ERR": 0.0,
}


# ------------------------------------
# Run inputs
# ------------------------------------

def write_catalog(cfg: dict, policy: PolicyResult,
                  catalog_path: str | Path) -> None:
    """Write the one-object wide catalog: f_<band>/e_<band> in uJy.

    Columns follow table order; non-fittable rows carry the missing
    sentinel in both flux and error.

    Parameters
    ----------
    cfg : dict
        Resolved fit config; supplies the object id.
    policy : PolicyResult
        core.policy output; supplies the bands, fluxes, errors, and the
        fittable mask.
    catalog_path : str or Path
        Destination CSV path.
    """
    columns = {"id": [cfg["name"]]}
    for band, flux, err, ok in zip(policy.bands, policy.flux_uJy,
                                   policy.flux_err_uJy, policy.fittable):
        columns[f"f_{band}"] = [float(flux) if ok else MISSING_SENTINEL]
        columns[f"e_{band}"] = [float(err) if ok else MISSING_SENTINEL]
    pd.DataFrame(columns).to_csv(catalog_path, index=False)


def build_eazy_params(cfg: dict, run_dir: Path, *, catalog: Path,
                      filter_res: Path, templates_param: Path,
                      tef_file: Path) -> dict:
    """The eazy parameter overrides for this run (keys uppercase).

    Parameters
    ----------
    cfg : dict
        Resolved fit config, backend "eazy".
    run_dir : Path
        Run directory; roots MAIN_OUTPUT_FILE.
    catalog : Path
        Catalog file (CATALOG_FILE).
    filter_res : Path
        Run-local FILTER.RES (FILTERS_RES).
    templates_param : Path
        Templates .param file (TEMPLATES_FILE).
    tef_file : Path
        Template-error curve (TEMP_ERR_FILE).

    Returns
    -------
    params : dict
        BASE_EAZY_PARAMS updated with the run's grid, fitter, template
        and prior settings, then with the config's extra_params.
    """
    eazy_cfg = cfg["eazy"]
    linear = eazy_cfg["z_step_type"] == "linear"
    params = dict(BASE_EAZY_PARAMS)
    params.update({
        "CATALOG_FILE": str(catalog),
        "FILTERS_RES": str(filter_res),
        "TEMPLATES_FILE": str(templates_param),
        "TEMP_ERR_FILE": str(tef_file),
        "TEMP_ERR_A2": (eazy_cfg["tef_scale"] if eazy_cfg["tef"] else 0.0),
        "Z_MIN": eazy_cfg["z_min"],
        # np.arange excludes the endpoint on linear grids; pad half a step.
        "Z_MAX": eazy_cfg["z_max"] + (eazy_cfg["z_step"] / 2.0 if linear
                                      else 0.0),
        "Z_STEP": eazy_cfg["z_step"],
        "Z_STEP_TYPE": 0 if linear else 1,
        "FITTER": eazy_cfg["fitter"],
        "MAIN_OUTPUT_FILE": str(run_dir / cfg["name"]),
    })
    if eazy_cfg["prior"]:
        params["APPLY_PRIOR"] = "y"
        params["PRIOR_FILE"] = str(Path(eazy_cfg["prior_file"])
                                   .expanduser().resolve())
        params["PRIOR_FILTER"] = eazy_cfg["prior_filter"]
    for key, value in eazy_cfg["extra_params"].items():
        params[str(key).upper()] = value
    return params


# ------------------------------------
# Fit execution
# ------------------------------------

def run_official(
        cfg: dict,
        frame: pd.DataFrame,
        policy: PolicyResult,
        run_dir: str | Path,
        *,
        registry: Registry,
        phot_sha256: str | None = None,
        grid_from: FitResult | None = None,
    ) -> FitResult:
    """Run the official eazy-py fit for one policy-applied table.

    Parameters
    ----------
    cfg : dict
        Resolved fit config (core.fitconfig), backend "eazy".
    frame : pd.DataFrame
        The validated SED table (band order defines fit order).
    policy : PolicyResult
        core.policy output for this table and config.
    run_dir : str or Path
        Destination run directory.
    registry : Registry
        Bandpass authority.
    phot_sha256 : str or None
        Photometry byte hash; required to offer this result for grid
        reuse. [default: None]
    grid_from : FitResult or None
        Same-session result whose template grid is reused; legal only
        with identical photometry bytes, z grid, templates, and band
        sequence. [default: None]
    """
    eazy_cfg = cfg["eazy"]
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    band_numbers = build_filter_res(frame, registry, run_dir / "FILTER.RES")
    write_translate(band_numbers, run_dir / "zphot.translate")
    write_catalog(cfg, policy, run_dir / "catalog.csv")
    tef_dst = run_dir / "template_error.dat"
    shutil.copyfile(resolve_tef(eazy_cfg), tef_dst)
    templates_param = prepare_templates_param(eazy_cfg, run_dir)
    (run_dir / "config.json").write_text(
        json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    params = build_eazy_params(cfg, run_dir,
                               catalog=run_dir / "catalog.csv",
                               filter_res=run_dir / "FILTER.RES",
                               templates_param=templates_param,
                               tef_file=tef_dst)

    tempfilt = None
    if grid_from is not None:
        tempfilt = _reusable_tempfilt(cfg, grid_from, list(policy.bands),
                                      phot_sha256)

    # eazy imports matplotlib.pyplot inside its fit paths; force a
    # non-interactive backend before the first eazy import.
    import matplotlib
    matplotlib.use("Agg")
    from eazy.photoz import PhotoZ

    photz = PhotoZ(param_file=None,
                   translate_file=str(run_dir / "zphot.translate"),
                   zeropoint_file=None,
                   params=params,
                   load_prior=eazy_cfg["prior"],
                   load_products=False,
                   n_proc=eazy_cfg["n_proc"],
                   compute_tef_lnp=bool(eazy_cfg["tef_lnp"]),
                   tempfilt=tempfilt)

    if photz.NOBJ != 1 or photz.NFILT != len(policy.bands):
        raise RuntimeError(f"catalog mismatch: eazy sees NOBJ={photz.NOBJ}, "
                           f"NFILT={photz.NFILT}; expected 1 object, "
                           f"{len(policy.bands)} filters")
    nusefilt = int(np.asarray(photz.nusefilt)[0])
    if nusefilt != policy.n_fit:
        raise RuntimeError(f"inclusion mismatch: eazy fits {nusefilt} bands, "
                           f"core policy included {policy.n_fit}")

    photz.fit_catalog(n_proc=eazy_cfg["n_proc"], prior=eazy_cfg["prior"],
                      beta_prior=False)

    # Capture the photo-z state before any fixed-z refit overwrites it.
    zgrid = np.asarray(photz.zgrid, float)
    z_ml = np.asarray(photz.zml, float).copy()
    chi2_fit = np.asarray(photz.chi2_fit, float).copy()
    lnp = (np.asarray(photz.lnp, float).copy()
           if hasattr(photz, "lnp") else None)
    z_chi2 = zgrid[np.argmin(chi2_fit, axis=1)]
    warn_grid_edges(cfg["name"], zgrid, z_ml, z_chi2)

    if lnp is not None:
        z_percentiles = percentiles_from_lnp(zgrid, lnp)
    else:
        print("WARNING: no lnp available; percentiles set to NaN")
        z_percentiles = np.full((1, len(Z_PERCENTILES)), np.nan)

    seds = [extract_sed(photz, 0) if z_ml[0] > 0 else None]

    result = FitResult(
        config=cfg,
        run_dir=run_dir,
        ids=[cfg["name"]],
        bands=list(policy.bands),
        template_names=[t.name for t in photz.templates],
        pivot=np.asarray(photz.pivot, float),
        zgrid=zgrid,
        fnu=np.asarray(photz.fnu, float).copy(),
        efnu=np.asarray(photz.efnu, float).copy(),
        ok_data=np.asarray(photz.ok_data, bool).copy(),
        nusefilt=np.asarray(photz.nusefilt, int).copy(),
        chi2_fit=chi2_fit,
        lnp=lnp,
        z_ml=z_ml,
        z_chi2=z_chi2,
        z_percentiles=z_percentiles,
        chi2_best=np.asarray(photz.chi2_best, float).copy(),
        coeffs_best=np.asarray(photz.coeffs_best, float).copy(),
        fmodel=np.asarray(photz.fmodel, float).copy(),
        seds=seds,
        phot_sha256=phot_sha256,
        photz=photz,
    )

    if eazy_cfg["mode"] == "single":
        # Called with no arguments so it reuses the PhotoZ template grid,
        # which makes the bandpass integrals and IGM treatment match the
        # combo fit exactly.
        _, ampl, singles_chi2, _ = photz.fit_single_templates(verbose=False)
        result.singles_chi2 = np.asarray(singles_chi2, float)
        result.singles_ampl = np.asarray(ampl, float)

    if eazy_cfg["save_zcoeffs"] and hasattr(photz, "fit_coeffs"):
        result.fit_coeffs = np.asarray(photz.fit_coeffs, float).copy()

    if eazy_cfg["z_fixed"] is not None:
        z_fixed = float(eazy_cfg["z_fixed"])
        if not (zgrid[0] < z_fixed < zgrid[-1]):
            raise ValueError(f"z_fixed={z_fixed} is not strictly inside the "
                             f"realized grid ({zgrid[0]:.4f}, "
                             f"{zgrid[-1]:.4f}); eazy would silently skip it")
        photz.fit_at_zbest(zbest=np.full(1, z_fixed))
        result.z_fixed = z_fixed
        result.chi2_fixed = np.asarray(photz.chi2_best, float).copy()
        result.coeffs_fixed = np.asarray(photz.coeffs_best, float).copy()
        result.fmodel_fixed = np.asarray(photz.fmodel, float).copy()
        result.seds_fixed = [extract_sed(photz, 0, z=z_fixed)]

    try:
        photz.param.write(str(run_dir / "zphot.param.echo"))
    except Exception as err:
        print(f"WARNING: could not write parameter echo: {err}")

    save_outputs(result)
    return result


def _reusable_tempfilt(cfg: dict, grid_from: FitResult, bands: list[str],
                       phot_sha256: str | None):
    """Validate a previous run's template grid for reuse.

    Reuse is legal only between fits of identical photometry bytes:
    SPHEREx tophats are per-object, so cross-target reuse would integrate
    templates through another galaxy's channel geometry.
    """
    if grid_from.photz is None:
        raise ValueError("grid_from carries no live PhotoZ handle "
                         "(rehydrated run?)")
    if phot_sha256 is None or grid_from.phot_sha256 != phot_sha256:
        raise ValueError("grid_from is not reusable: photometry bytes differ "
                         "(or hashes were not supplied)")
    ours, theirs = cfg["eazy"], grid_from.config["eazy"]
    same_setup = all(ours[key] == theirs[key]
                     for key in ("z_min", "z_max", "z_step", "z_step_type",
                                 "templates", "template_pattern"))
    if not same_setup or list(grid_from.bands) != list(bands):
        raise ValueError("grid_from is not reusable: band sequence, redshift "
                         "grid, or template settings differ")
    print(f"reusing template grid from run {grid_from.config['name']!r}")
    return grid_from.photz.tempfilt


def warn_grid_edges(name: str, zgrid: np.ndarray, z_ml: np.ndarray,
                    z_chi2: np.ndarray) -> None:
    """Flag failed or edge-pinned solutions (eazy's -1 sentinel is silent)."""
    if z_ml[0] <= 0:
        print(f"WARNING: {name!r}: z_ml sentinel {z_ml[0]:.1f} -- ln P(z) "
              f"peaks at the first grid point; widen the grid")
    elif np.isclose(z_chi2[0], zgrid[0]) or np.isclose(z_chi2[0], zgrid[-1]):
        print(f"WARNING: {name!r}: chi2 minimum on the grid edge "
              f"(z={z_chi2[0]:.4f}); widen the grid")
