"""
jobs.py

Fit-Job Orchestration
---------------------------------------------------------

Runs one fit job end to end: locate the built photometry by the fixed
output convention, read its bytes exactly once, cross-check the build
sidecar against the run context, apply the shared data policy, compute
the run identity, stage the run directory, dispatch the backend, and
finalize the central manifest row. Backends are imported lazily so the
light stack can import this module.

Data products (per job):
  <target.dir>/SED/<recipe>/<backend>/<run_id[-label]>/   (runs.stage_run)
  <data_root>/Galaxies/sed_fitting/runs.jsonl             (one row appended)

Requirements:
  - numpy, pandas (light stack); the backend stacks at dispatch time

Notes:
  - Sidecar cross-checks (all hard errors): sidecar recipe == the job's
    recipe; sidecar per-band bandpass hashes == the live registry
    (allow_registry_drift overrides, recorded in the row); sidecar build
    z_ref == the fit's resolved z_ref.
"""

from __future__ import annotations

import io
import json
import platform
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

import sedfit
from sedfit.core.build import SED_TABLE_SUBDIR
from sedfit.core.fitconfig import hash_projection, resolve_config
from sedfit.core.policy import apply_policy
from sedfit.core.provenance import git_state, run_id as make_run_id, sha256_bytes
from sedfit.core.runs import finalize_run, now_iso, stage_run
from sedfit.core.table import validate_sed_table

if TYPE_CHECKING:
    from sedfit.core.registry import Registry
    from sedfit.core.roster import Roster, Target

# ------------------------------------
# Job context
# ------------------------------------

def phot_path_for(target: Target, recipe_name: str, *,
                  dered: bool = False) -> Path:
    """The built SED table's path for this target and recipe.

    Parameters
    ----------
    target : Target
        The roster target, whose dir and prefix locate the file.
    recipe_name : str
        Recipe whose table is wanted.
    dered : bool
        Select the dereddened table. [default: False]

    Returns
    -------
    path : Path
        The table path, which need not exist yet.
    """
    suffix = "_dered" if dered else ""
    return (target.dir / SED_TABLE_SUBDIR
            / f"{target.prefix}_sed_{recipe_name}{suffix}.csv")


def _check_sidecar(sidecar: dict, *, recipe_name: str, z_ref: float,
                   bands: list[str], registry: Registry,
                   allow_registry_drift: bool) -> list[str]:
    """Stale-table cross-checks against the build sidecar.

    Returns
    -------
    warnings : list of str
        Non-fatal findings for the run's manifest row.
    """
    warnings = []
    if sidecar.get("recipe") != recipe_name:
        raise ValueError(f"sidecar recipe {sidecar.get('recipe')!r} != job "
                         f"recipe {recipe_name!r}; rebuild the table")
    sidecar_z = sidecar.get("z_ref")
    if sidecar_z is None or abs(float(sidecar_z) - z_ref) > 0:
        raise ValueError(f"sidecar build z_ref {sidecar_z!r} != resolved "
                         f"fit z_ref {z_ref}; rebuild the table")

    live = {b: h[:16] for b, h in registry.per_band_hashes().items()}
    stored = sidecar.get("registry", {}).get("bands", {})
    drifted = sorted(b for b in bands
                     if b in stored and stored[b] != live.get(b))
    if drifted:
        message = (f"bandpass hashes drifted since the build: {drifted}; "
                   f"rebuild the table")
        if not allow_registry_drift:
            raise ValueError(message)
        warnings.append(message)
    return warnings


def _backend_versions(backend: str, engine: str | None = None) -> dict:
    """Versions of the packages a run's numbers actually depend on.

    The quick eazy engine is implemented here in numpy and scipy, so an
    eazy-py version is stamped only for the official engine. A package
    that is absent stamps None rather than failing the run.

    Parameters
    ----------
    backend : str
        'eazy' or 'prospector'.
    engine : str or None
        The eazy engine name; ignored for other backends.

    Returns
    -------
    versions : dict
        Distribution name -> version string, or None when not installed.
    """
    from importlib.metadata import PackageNotFoundError, version

    if backend == "eazy":
        distributions = {"eazy": "eazy"} if engine == "eazy-py" else {}
    else:
        distributions = {"prospector": "astro-prospector", "fsps": "fsps",
                         "dynesty": "dynesty"}

    versions = {}
    for name, dist in distributions.items():
        try:
            versions[name] = version(dist)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def _machinery(backend: str, fsps_libraries: list[str] | None = None, *,
               engine: str | None = None) -> dict:
    import astropy

    versions = {"numpy": np.__version__, "pandas": pd.__version__,
                "astropy": astropy.__version__,
                "python": platform.python_version(),
                **_backend_versions(backend, engine)}
    state = git_state(Path(sedfit.__file__).resolve().parents[1])
    return {"package_version": sedfit.__version__,
            "git_rev": state["rev"], "git_dirty": state["dirty"],
            "fsps_libraries": fsps_libraries, "versions": versions}


# ------------------------------------
# Backend dispatch
# ------------------------------------

def _run_eazy(resolved, frame, policy, run_dir, registry, phot_sha256,
              plots):
    from sedfit.backends.eazy.fitting import run_official
    from sedfit.backends.eazy.quick import run_quick
    from sedfit.backends.eazy.results import summarize

    engine = resolved["eazy"]["engine"]
    runner = run_quick if engine == "quick" else run_official
    result = runner(resolved, frame, policy, run_dir, registry=registry,
                    phot_sha256=phot_sha256)
    if plots:
        try:
            from sedfit.backends.eazy.plots import generate_plots

            generate_plots(result, registry=registry,
                           z_ref=resolved["z_ref"])
        except Exception as err:
            print(f"WARNING: plot generation failed: {err}")
    row = summarize(result)[0]
    estimates = {"zred_p50": float(row["z500"]),
                 "zred_p16": float(row["z160"]),
                 "zred_p84": float(row["z840"]),
                 "z_ml": float(row["z_ml"]),
                 "z_chi2": float(row["z_chi2"]),
                 "chi2_best": float(row["chi2_best"])}
    return estimates


def _prepare_sps(resolved):
    from sedfit.backends.prospector.model import (
        assert_stellar_library,
        build_sps,
        spectral_library,
    )

    sps = build_sps(resolved)
    assert_stellar_library(sps, resolved["prospector"]["stellar_library"])
    return sps, list(spectral_library(sps))


def _run_prospector(resolved, frame, policy, run_dir, registry, sps,
                    plots):
    from sedfit.backends.prospector.fitting import run_sampler, save_results
    from sedfit.backends.prospector.model import attach_norm_masks, build_model
    from sedfit.backends.prospector.obs import build_obs

    obs = build_obs(resolved, frame, policy, registry=registry)
    model = build_model(resolved)
    attach_norm_masks(model, obs, resolved, registry)

    output = run_sampler(resolved, obs, model, sps,
                         checkpoint=str(Path(run_dir) / "checkpoint.save"))
    save_results(Path(run_dir) / "result.h5", resolved, model, obs, output,
                 sps)
    if plots:
        try:
            from sedfit.backends.prospector.plots import plot_run
            from sedfit.backends.prospector.results import load_results

            loaded = load_results(str(Path(run_dir) / "result.h5"),
                                  cfg=resolved)
            plot_run(run_dir, loaded, resolved, obs, model, sps,
                     registry=registry)
        except Exception as err:
            print(f"WARNING: plot generation failed: {err}")

    chain = np.atleast_2d(output["sampling"][0]["samples"])
    weights = np.exp(output["sampling"][0]["logwt"]
                     - output["sampling"][0]["logz"][-1])
    labels = model.theta_labels()
    estimates = {}
    for key in ("zred", "logmass"):
        if key in labels:
            values = chain[:, labels.index(key)]
            order = np.argsort(values)
            cdf = np.cumsum(weights[order])
            cdf = cdf / cdf[-1]
            p16, p50, p84 = np.interp([0.16, 0.5, 0.84], cdf, values[order])
            estimates[f"{key}_p16"] = float(p16)
            estimates[f"{key}_p50"] = float(p50)
            estimates[f"{key}_p84"] = float(p84)
    return estimates


# ------------------------------------
# The job
# ------------------------------------

def plan_job(
        roster: Roster,
        target_name: str,
        recipe_name: str,
        cfg: dict,
        *,
        registry: Registry,
        allow_registry_drift: bool = False,
        phot_csv: str | Path | None = None,
        dered: bool = False,
    ) -> dict:
    """Resolve and validate one job without writing anything.

    The fit verb's full validation prefix -- config resolution,
    photometry read and schema check, sidecar cross-checks, data
    policy, and run identity -- shared by run_job and fit --dry-run.

    Returns
    -------
    plan : dict
        target, resolved, backend, phot_path, phot_bytes, phot_sha,
        frame, sidecar, warnings, policy, digests, template_prov,
        run_id.
    """
    if target_name not in roster.targets:
        raise ValueError(f"unknown target {target_name!r}; the roster "
                         f"declares {sorted(roster.targets)}")
    if recipe_name not in roster.recipes:
        raise ValueError(f"unknown recipe {recipe_name!r}; the roster "
                         f"declares {sorted(roster.recipes)}")
    target = roster.targets[target_name]
    resolved = resolve_config(cfg, target_z_ref=target.z_ref,
                              reference_redshift=roster.reference_redshift)
    backend = resolved["backend"]

    phot_path = (Path(phot_csv) if phot_csv is not None
                 else phot_path_for(target, recipe_name, dered=dered))
    if not phot_path.is_file():
        raise FileNotFoundError(
            f"{phot_path} has not been built; run: sedfit build --roster "
            f"{roster.path} --target {target_name} --recipe {recipe_name}"
            + (" --deredden" if dered else ""))
    phot_bytes = phot_path.read_bytes()          # read once
    phot_sha = sha256_bytes(phot_bytes)
    frame = pd.read_csv(io.BytesIO(phot_bytes))
    validate_sed_table(frame, registry, label=str(phot_path))

    sidecar_path = phot_path.parent / (phot_path.stem + ".provenance.json")
    if not sidecar_path.is_file():
        raise FileNotFoundError(
            f"{phot_path.name} has no provenance sidecar at "
            f"{sidecar_path.name}; rebuild the table")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    warnings = _check_sidecar(sidecar, recipe_name=recipe_name,
                              z_ref=resolved["z_ref"],
                              bands=[str(b) for b in frame["band"]],
                              registry=registry,
                              allow_registry_drift=allow_registry_drift)

    policy = apply_policy(frame, registry=registry,
                          bands_include=resolved["bands_include"],
                          min_valid_bands=resolved["min_valid_bands"],
                          min_snr_broadband=resolved["min_snr_broadband"],
                          err_floor=resolved["err_floor"],
                          mu_lensing=resolved["mu_lensing"],
                          qa_gates=resolved.get("qa_gates"))

    digests = None
    template_prov = None
    if backend == "eazy":
        from sedfit.backends.eazy.templates import (content_digests,
                                                    template_summary)

        digests = content_digests(resolved["eazy"])
        template_prov = template_summary(resolved["eazy"], digests)
    rid = make_run_id(hash_projection(resolved, digests=digests), phot_sha,
                      registry.bandpass_hash())
    return {"target": target, "resolved": resolved, "backend": backend,
            "phot_path": phot_path, "phot_bytes": phot_bytes,
            "phot_sha": phot_sha, "frame": frame, "sidecar": sidecar,
            "warnings": warnings, "policy": policy, "digests": digests,
            "template_prov": template_prov, "run_id": rid}


def run_job(
        roster: Roster,
        target_name: str,
        recipe_name: str,
        cfg: dict,
        *,
        registry: Registry,
        label: str | None = None,
        force: bool = False,
        allow_registry_drift: bool = False,
        phot_csv: str | Path | None = None,
        dered: bool = False,
        plots: bool = True,
    ) -> dict:
    """Run one fit job; returns the appended manifest row.

    Parameters
    ----------
    cfg : dict
        Parsed (unresolved) fit config from core.fitconfig.
    label : str or None
        Run-directory label; None uses the config name. [default: None]
    force : bool
        Allow replacing an existing run whose machinery stamps differ.
    allow_registry_drift : bool
        Demote the sidecar bandpass-hash mismatch to a recorded warning.
    phot_csv : path or None
        Explicit photometry path (roster-less use); None locates it by
        the fixed output convention. [default: None]
    plots : bool
        Render the run's figures into its plots/ directory after a
        successful fit (a plot failure warns, never fails the run).
        [default: True]
    """
    plan = plan_job(roster, target_name, recipe_name, cfg,
                    registry=registry,
                    allow_registry_drift=allow_registry_drift,
                    phot_csv=phot_csv, dered=dered)
    target = plan["target"]
    resolved = plan["resolved"]
    backend = plan["backend"]
    frame = plan["frame"]
    policy = plan["policy"]
    phot_sha = plan["phot_sha"]
    warnings = plan["warnings"]
    template_prov = plan["template_prov"]
    rid = plan["run_id"]

    sps = None
    fsps_libraries = None
    if backend == "prospector":
        sps, fsps_libraries = _prepare_sps(resolved)
    machinery = _machinery(backend, fsps_libraries,
                           engine=resolved.get("eazy", {}).get("engine"))
    backend_dir = target.dir / "SED" / recipe_name / backend
    run_dir = stage_run(backend_dir, rid, config=resolved,
                        phot_bytes=plan["phot_bytes"], phot_sha256=phot_sha,
                        sidecar=plan["sidecar"], machinery=machinery,
                        label=label or resolved["name"], force=force)

    row = {
        "run_id": rid,
        "path": str(run_dir.relative_to(roster.data_root)),
        "written": now_iso(),
        "target": target_name, "recipe": recipe_name, "backend": backend,
        "engine": None, "sampler": None,
        "seed": (resolved.get("prospector") or {}).get("seed"),
        **machinery,
        "phot_sha256_16": phot_sha[:16],
        "config_sha256_16": sha256_bytes(
            json.dumps(resolved, sort_keys=True).encode())[:16],
        "bandpass_sha256_16": registry.bandpass_hash()[:16],
        "bands_include": resolved["bands_include"],
        "err_floor": resolved["err_floor"],
        "mu_lensing": resolved["mu_lensing"],
        "z_ref": resolved["z_ref"],
        "status": "failed", "estimates": None,
    }
    accounting = policy.accounting()
    accounting["warnings"] = warnings + accounting["warnings"]
    row.update(accounting)
    if backend == "eazy":
        row["engine"] = resolved["eazy"]["engine"]
        row["templates"] = template_prov
    else:
        row["sampler"] = resolved["prospector"]["sampler"]

    manifest_path = roster.manifest_path
    try:
        if backend == "eazy":
            row["estimates"] = _run_eazy(resolved, frame, policy, run_dir,
                                         registry, phot_sha, plots)
        else:
            row["estimates"] = _run_prospector(resolved, frame, policy,
                                               run_dir, registry, sps,
                                               plots)
        row["status"] = "ok"
    except Exception as err:
        row["error"] = f"{type(err).__name__}: {err}"
        finalize_run(run_dir, manifest_path, row)
        raise
    finalize_run(run_dir, manifest_path, row)
    return row


def run_jobs(roster: Roster, jobs: list[dict], *, registry: Registry,
             dered: bool = False, **kwargs) -> list[dict]:
    """Run a jobs list ({target, recipe, config, label?, dered?}) sequentially.

    ``dered`` sets the default; a per-job ``dered`` key overrides it.
    """
    from sedfit.core.fitconfig import load_fit_config

    rows = []
    for job in jobs:
        cfg = load_fit_config(job["config"])
        rows.append(run_job(roster, job["target"], job["recipe"], cfg,
                            registry=registry, label=job.get("label"),
                            dered=job.get("dered", dered), **kwargs))
    return rows
