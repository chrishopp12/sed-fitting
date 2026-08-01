"""
results.py

Results Container and Output Products
---------------------------------------------------------

Collects everything an eazy fit produced into a FitResult and writes the
backend's own summary, singles and array products. eazy-py's
standard_output is unsuitable here: its rest-frame filter indices (UBVJ,
absolute magnitudes) are hardcoded and select the wrong filters out of a
run-local FILTER.RES, and its stellar-population columns assume a
physically scaled rather than a shape-normalized template atlas.

Redshift estimators per object:
  z_ml     maximum of ln P(z) = -chi2/2 + tef_lnp (parabola-refined)
  z_chi2   the raw grid argmin of chi2(z)
  z500     the P(z) median (with z025/z160/z840/z975)

Data products (per run directory):
  summary.csv    one row per object (redshifts, chi2, template content)
  singles.csv    single mode: per object x template best-z and chi2
  arrays.npz     zgrid, chi2(z), ln P(z), coefficients, model photometry,
                 best-fit SED curves, and the fixed-z products when present

Requirements:
  - numpy, astropy
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from astropy.table import Table

Z_PERCENTILES = (2.5, 16.0, 50.0, 84.0, 97.5)
PERCENTILE_LABELS = ("z025", "z160", "z500", "z840", "z975")

SED_KEYS = ("templz", "templf", "model", "fobs", "efobs")


@dataclass
class FitResult:
    """In-memory record of one eazy-backend run.

    Arrays are copies of the engine state at capture time; photz holds
    the live eazy-py object for the current session only (None for the
    quick engine and for runs rehydrated from disk; after a fixed-z
    refit it reflects the fixed-z state, not the photo-z state).
    """
    config: dict
    run_dir: Path
    ids: list[str]
    bands: list[str]
    template_names: list[str]
    pivot: np.ndarray            # (NFILT,) Angstrom
    zgrid: np.ndarray            # (NZ,)
    fnu: np.ndarray              # (NOBJ, NFILT) catalog fluxes, uJy
    efnu: np.ndarray             # (NOBJ, NFILT) fit errors, uJy
    ok_data: np.ndarray          # (NOBJ, NFILT) bool
    nusefilt: np.ndarray         # (NOBJ,) valid-band counts
    chi2_fit: np.ndarray         # (NOBJ, NZ)
    lnp: np.ndarray | None       # (NOBJ, NZ) ln P(z); None if unavailable
    z_ml: np.ndarray             # (NOBJ,) max-lnp redshift (-1 = failed)
    z_chi2: np.ndarray           # (NOBJ,) grid argmin of chi2
    z_percentiles: np.ndarray    # (NOBJ, 5) at Z_PERCENTILES
    chi2_best: np.ndarray        # (NOBJ,) chi2 at z_ml
    coeffs_best: np.ndarray      # (NOBJ, NTEMP) at z_ml
    fmodel: np.ndarray           # (NOBJ, NFILT) model photometry at z_ml
    seds: list = field(default_factory=list)
    z_fixed: float | None = None
    chi2_fixed: np.ndarray | None = None
    coeffs_fixed: np.ndarray | None = None
    fmodel_fixed: np.ndarray | None = None
    seds_fixed: list = field(default_factory=list)
    singles_chi2: np.ndarray | None = None            # (NTEMP, NOBJ, NZ)
    singles_ampl: np.ndarray | None = None            # (NTEMP, NOBJ, NZ)
    fit_coeffs: np.ndarray | None = None              # (NOBJ, NZ, NTEMP)
    phot_sha256: str | None = None
    photz: object = field(default=None, repr=False, compare=False)


def percentiles_from_lnp(zgrid: np.ndarray, lnp: np.ndarray, *,
                         percentiles: Sequence[float] = Z_PERCENTILES
                         ) -> np.ndarray:
    """P(z) percentiles by direct trapezoidal CDF on the fit grid.

    Replaces PhotoZ.pz_percentiles, which resamples ln P(z) with an
    Akima spline onto its own log-spaced zoom grid; when a zoom endpoint
    lands one float ULP outside the fit grid, the spline returns NaN
    there, a leading NaN poisons the cumulative integral, and every
    percentile collapses to the grid start (eazy-py 0.8.6).

    Parameters
    ----------
    zgrid : np.ndarray
        Fit redshift grid (NZ).
    lnp : np.ndarray
        ln P(z), shape (NOBJ, NZ); non-finite entries are excluded.
    percentiles : sequence of float
        Percentiles in percent. [default: Z_PERCENTILES]

    Returns
    -------
    zlimits : np.ndarray
        Shape (NOBJ, len(percentiles)); NaN rows where P(z) is unusable.
    """
    lnp = np.atleast_2d(np.asarray(lnp, float))
    zgrid = np.asarray(zgrid, float)
    fractions = np.asarray(percentiles, float) / 100.0
    zlimits = np.full((lnp.shape[0], len(fractions)), np.nan)
    for i, row in enumerate(lnp):
        good = np.isfinite(row)
        if good.sum() < 3:
            continue
        z = zgrid[good]
        pz = np.exp(row[good] - row[good].max())
        cdf = np.concatenate(
            [[0.0], np.cumsum(0.5 * (pz[1:] + pz[:-1]) * np.diff(z))])
        if cdf[-1] <= 0:
            continue
        zlimits[i] = np.interp(fractions, cdf / cdf[-1], z)
    return zlimits


def extract_sed(photz: object, iobj: int, *,
                z: float | None = None) -> dict | None:
    """Best-fit SED pieces for one object via eazy's show_fit.

    show_fit(..., get_spec=True) returns its payload before drawing
    anything, re-fitting the coefficients at z (or the stored zbest)
    with the configured solver -- an exact, side-effect-free extraction.
    Returns None when the evaluation fails (e.g. the -1 failure
    sentinel).
    """
    try:
        payload = photz.show_fit(iobj, id_is_idx=True, zshow=z,
                                 get_spec=True, show_fnu=1)
    except Exception as err:
        print(f"WARNING: SED extraction failed for object index {iobj}: "
              f"{err}")
        return None
    sed = {key: np.asarray(payload[key], float) for key in SED_KEYS}
    sed["z"] = float(np.squeeze(payload["z"]))
    sed["chi2"] = float(np.squeeze(payload["chi2"]))
    return sed


# ------------------------------------
# Summaries
# ------------------------------------

def summarize(result: FitResult) -> Table:
    """Per-object summary table (the contents of summary.csv).

    redchi2 uses dof = n_bands - n_active - 1, counting the active
    template coefficients plus the redshift as fitted parameters.
    """
    rows = []
    for i, oid in enumerate(result.ids):
        n_bands = int(result.nusefilt[i])
        n_active = int((result.coeffs_best[i] > 0).sum())
        row = {
            "id": oid,
            "n_bands": n_bands,
            "z_ml": float(result.z_ml[i]),
            "z_chi2": float(result.z_chi2[i]),
            "chi2_best": float(result.chi2_best[i]),
            "n_active": n_active,
            "redchi2": float(result.chi2_best[i])
                       / max(1, n_bands - n_active - 1),
        }
        for label, value in zip(PERCENTILE_LABELS, result.z_percentiles[i]):
            row[label] = float(value)
        if result.z_fixed is not None:
            n_active_fixed = int((result.coeffs_fixed[i] > 0).sum())
            row.update({
                "z_fixed": float(result.z_fixed),
                "chi2_fixed": float(result.chi2_fixed[i]),
                "n_active_fixed": n_active_fixed,
                "redchi2_fixed": float(result.chi2_fixed[i])
                                 / max(1, n_bands - n_active_fixed - 1),
            })
        if result.singles_chi2 is not None:
            row.update(_best_single(result, i))
        rows.append(row)
    return Table(rows=rows)


def singles_table(result: FitResult) -> Table:
    """Single mode: per object x template best redshift and chi2."""
    rows = []
    for i, oid in enumerate(result.ids):
        chi2 = result.singles_chi2[:, i, :]
        ampl = result.singles_ampl[:, i, :]
        iz = np.argmin(chi2, axis=1)
        for t, name in enumerate(result.template_names):
            rows.append({
                "id": oid,
                "template": name,
                "z_best": float(result.zgrid[iz[t]]),
                "chi2_min": float(chi2[t, iz[t]]),
                "ampl": float(ampl[t, iz[t]]),
            })
    return Table(rows=rows)


def _best_single(result: FitResult, iobj: int) -> dict:
    """Best single template for one object (positive amplitude required).

    fit_single_templates solves an unconstrained analytic amplitude, so a
    template can formally fit best with negative flux; those are excluded
    as unphysical.
    """
    chi2 = result.singles_chi2[:, iobj, :]
    ampl = result.singles_ampl[:, iobj, :]
    iz = np.argmin(chi2, axis=1)
    ntemp = chi2.shape[0]
    chi2_min = chi2[np.arange(ntemp), iz]
    ampl_min = ampl[np.arange(ntemp), iz]
    usable = ampl_min > 0
    if not usable.any():
        print(f"WARNING: object {result.ids[iobj]!r}: no single template "
              f"with positive amplitude; reporting the raw chi2 minimum")
        usable = np.ones(ntemp, bool)
    best = int(np.flatnonzero(usable)[np.argmin(chi2_min[usable])])
    return {
        "single_template": result.template_names[best],
        "z_single": float(result.zgrid[iz[best]]),
        "chi2_single": float(chi2_min[best]),
    }


# ------------------------------------
# Disk round-trip
# ------------------------------------

def save_outputs(result: FitResult) -> None:
    """Write summary.csv (+ singles.csv) and arrays.npz into the run dir."""
    run_dir = Path(result.run_dir)
    summarize(result).write(run_dir / "summary.csv", format="ascii.csv",
                            overwrite=True)
    if result.singles_chi2 is not None:
        singles_table(result).write(run_dir / "singles.csv",
                                    format="ascii.csv", overwrite=True)
    np.savez_compressed(run_dir / "arrays.npz", **_arrays_payload(result))


def _arrays_payload(result: FitResult) -> dict:
    payload = {
        "ids": np.array(result.ids, dtype=str),
        "bands": np.array(result.bands, dtype=str),
        "template_names": np.array(result.template_names, dtype=str),
        "pivot": result.pivot,
        "zgrid": result.zgrid,
        "fnu": result.fnu,
        "efnu": result.efnu,
        "ok_data": result.ok_data,
        "nusefilt": result.nusefilt,
        "chi2_fit": result.chi2_fit,
        "z_ml": result.z_ml,
        "z_chi2": result.z_chi2,
        "z_percentiles": result.z_percentiles,
        "chi2_best": result.chi2_best,
        "coeffs_best": result.coeffs_best,
        "fmodel": result.fmodel,
    }
    if result.lnp is not None:
        payload["lnp"] = result.lnp
    if result.z_fixed is not None:
        payload["z_fixed"] = np.array([result.z_fixed])
        payload["chi2_fixed"] = result.chi2_fixed
        payload["coeffs_fixed"] = result.coeffs_fixed
        payload["fmodel_fixed"] = result.fmodel_fixed
    if result.singles_chi2 is not None:
        payload["singles_chi2"] = result.singles_chi2
        payload["singles_ampl"] = result.singles_ampl
    if result.fit_coeffs is not None:
        payload["fit_coeffs"] = result.fit_coeffs
    for prefix, seds in (("sed", result.seds),
                         ("sed_fixed", result.seds_fixed)):
        for i, sed in enumerate(seds):
            if sed is None:
                continue
            for key in SED_KEYS:
                payload[f"{prefix}{i}_{key}"] = sed[key]
            payload[f"{prefix}{i}_z"] = np.array([sed["z"]])
    return payload


def load_run(run_dir: str | Path) -> FitResult:
    """Rehydrate a FitResult from a run directory (no eazy import).

    The live photz handle is not recoverable; everything the summary and
    plotting layers need comes from config.json and arrays.npz.

    Parameters
    ----------
    run_dir : str or Path
        Run directory written by either engine.

    Returns
    -------
    result : FitResult
        The stored run, with photz None.
    """
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    with np.load(run_dir / "arrays.npz") as npz:
        arrays = {key: npz[key] for key in npz.files}

    ids = [str(x) for x in arrays["ids"]]
    seds, seds_fixed = [], []
    for prefix, target in (("sed", seds), ("sed_fixed", seds_fixed)):
        for i in range(len(ids)):
            if f"{prefix}{i}_templz" in arrays:
                sed = {key: arrays[f"{prefix}{i}_{key}"] for key in SED_KEYS}
                sed["z"] = float(arrays[f"{prefix}{i}_z"][0])
                target.append(sed)
            else:
                target.append(None)

    return FitResult(
        config=config,
        run_dir=run_dir,
        ids=ids,
        bands=[str(x) for x in arrays["bands"]],
        template_names=[str(x) for x in arrays["template_names"]],
        pivot=arrays["pivot"],
        zgrid=arrays["zgrid"],
        fnu=arrays["fnu"],
        efnu=arrays["efnu"],
        ok_data=arrays["ok_data"],
        nusefilt=arrays["nusefilt"],
        chi2_fit=arrays["chi2_fit"],
        lnp=arrays.get("lnp"),
        z_ml=arrays["z_ml"],
        z_chi2=arrays["z_chi2"],
        z_percentiles=arrays["z_percentiles"],
        chi2_best=arrays["chi2_best"],
        coeffs_best=arrays["coeffs_best"],
        fmodel=arrays["fmodel"],
        seds=seds,
        z_fixed=(float(arrays["z_fixed"][0]) if "z_fixed" in arrays
                 else None),
        chi2_fixed=arrays.get("chi2_fixed"),
        coeffs_fixed=arrays.get("coeffs_fixed"),
        fmodel_fixed=arrays.get("fmodel_fixed"),
        seds_fixed=seds_fixed,
        singles_chi2=arrays.get("singles_chi2"),
        singles_ampl=arrays.get("singles_ampl"),
        fit_coeffs=arrays.get("fit_coeffs"),
    )
