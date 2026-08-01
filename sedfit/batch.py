"""
batch.py

Batch Fit Orchestration
---------------------------------------------------------

Runs many fit jobs over one roster. Expands the target x recipe grid
against the applicability rules, dispatches the jobs to a process pool
whose workers load the registry and roster exactly once, records every
outcome instead of raising on the first failure, and writes a sorted
per-job report. Job identity is the run_id, so a resumed batch skips
exactly the jobs already complete on disk.

Data products:
  <report>            one row per planned job, sorted by target/recipe
  <report>.partial    the same rows in completion order, removed on exit
  <log_dir>/*.log     per-job stdout

Requirements:
  - numpy, pandas (via sedfit.core); the backend stacks load inside
    each worker

Notes:
  - Workers are spawned on every platform, so a worker's state is
    exactly what its initializer builds.
  - BLAS thread caps are set before the pool starts so children inherit
    them; a value already in the environment wins.
  - Building is opt-in: a rebuilt table differs from the one on disk
    whenever its measured inputs changed after it was written, which
    moves run_id and defeats a resume.
"""

from __future__ import annotations

import csv
import json
import multiprocessing
import os
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sedfit.core.roster import Roster

# ------------------------------------
# Constants
# ------------------------------------

BLAS_ENV_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
MAX_WORKERS_BY_BACKEND = {"eazy": 8, "prospector": 1}
DEFAULT_STOP_AFTER_FAILURES = 20
TARGET_COLUMN_ALIASES = ("name", "target")

REPORT_COLUMNS = ("target", "recipe", "backend", "status", "stage", "run_id",
                  "path", "seconds", "zred_p50", "zred_p16", "zred_p84",
                  "n_bands_fit", "n_bands_table", "error")

STATUS_OK = "ok"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"

_WORKER: dict = {}


# ------------------------------------
# Data model
# ------------------------------------

@dataclass(frozen=True)
class BatchJob:
    """One planned fit: a target, a recipe, a config path, and a label."""

    target: str
    recipe: str
    config: str
    label: str | None = None


# ------------------------------------
# Job expansion
# ------------------------------------

def read_target_list(path: str | Path) -> list[str]:
    """Target names from a CSV, in file order.

    Parameters
    ----------
    path : str or Path
        CSV with a ``name`` (or ``target``) column; other columns are
        ignored, so the shared sample catalog can be passed directly.

    Returns
    -------
    names : list of str
        Names in file order, blanks dropped, duplicates removed.
    """
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        column = next((c for c in TARGET_COLUMN_ALIASES if c in fields), None)
        if column is None:
            raise ValueError(f"{path}: no {' or '.join(TARGET_COLUMN_ALIASES)} "
                             f"column (found: {fields})")
        names = []
        for row in reader:
            name = (row.get(column) or "").strip()
            if not name:
                continue
            if name in names:
                raise ValueError(f"{path}: duplicate target {name!r} under "
                                 f"column {column!r}")
            names.append(name)
    if not names:
        raise ValueError(f"{path}: no target names under column {column!r}")
    return names


def expand_jobs(
        roster: Roster,
        config_path: str | Path,
        *,
        targets: list[str] | None = None,
        recipes: list[str] | None = None,
        label: str | None = None,
    ) -> tuple[list[BatchJob], list[dict]]:
    """The applicable target x recipe grid, plus the pairs it excluded.

    Parameters
    ----------
    targets, recipes : list of str, optional
        Restrict the grid; None takes every roster entry in declaration
        order. [default: None]
    label : str or None
        Run-directory label passed to every job. [default: None]

    Returns
    -------
    jobs : list of BatchJob
        Every pair that passes check_applicability.
    skipped : list of dict
        Report rows for the pairs that did not, carrying the reason,
        plus one row per requested target the roster does not declare.
        A request naming no roster target at all is a hard error; an
        unknown recipe always is.
    """
    from sedfit.core.build import check_applicability

    config_path = str(Path(config_path).resolve())
    target_names = list(targets) if targets else list(roster.targets)
    recipe_names = list(recipes) if recipes else list(roster.recipes)

    # A requested name with no roster entry is reported, not fatal, since
    # generation omits a target whose sources did not survive. Zero
    # matches means the wrong file and is fatal before any job runs.
    missing = [t for t in target_names if t not in roster.targets]
    if missing and len(missing) == len(target_names):
        raise ValueError(f"no requested target is in the roster: {missing}")
    target_names = [t for t in target_names if t not in set(missing)]

    unknown = [r for r in recipe_names if r not in roster.recipes]
    if unknown:
        raise ValueError(f"recipes not in the roster: {unknown}")

    jobs, skipped = [], []
    for target_name in missing:
        skipped.append(_row(target_name, "-", status=STATUS_SKIPPED,
                            stage="roster",
                            error="not in the roster (see the generation "
                                  "report for why it was omitted)"))
    for target_name in target_names:
        for recipe_name in recipe_names:
            try:
                check_applicability(roster.recipes[recipe_name],
                                    roster.targets[target_name])
            except ValueError as err:
                skipped.append(_row(target_name, recipe_name,
                                    status=STATUS_SKIPPED, stage="applicable",
                                    error=str(err)))
                continue
            jobs.append(BatchJob(target=target_name, recipe=recipe_name,
                                 config=config_path, label=label))
    return jobs, skipped


# ------------------------------------
# Report rows
# ------------------------------------

def _row(target: str, recipe: str, **fields) -> dict:
    """One report row with every column present."""
    row = {column: None for column in REPORT_COLUMNS}
    row["target"] = target
    row["recipe"] = recipe
    row.update({k: v for k, v in fields.items() if k in row})
    return row


def _row_from_manifest(job: BatchJob, row: dict, *, status: str,
                       seconds: float | None = None) -> dict:
    """A report row projected from a finalized manifest row."""
    estimates = row.get("estimates") or {}
    return _row(job.target, job.recipe, backend=row.get("backend"),
                status=status, run_id=row.get("run_id"), path=row.get("path"),
                seconds=None if seconds is None else round(seconds, 1),
                zred_p50=estimates.get("zred_p50"),
                zred_p16=estimates.get("zred_p16"),
                zred_p84=estimates.get("zred_p84"),
                n_bands_fit=row.get("n_bands_fit"),
                n_bands_table=row.get("n_bands_table"),
                error=row.get("error"))


# ------------------------------------
# Worker
# ------------------------------------

def _init_worker(roster_path: str, options: dict) -> None:
    """Load the registry and roster once per worker process."""
    import matplotlib

    from sedfit.core.registry import load_registry
    from sedfit.core.roster import load_roster

    # A spawned worker starts from a bare interpreter, so it selects the
    # file-writing backend for itself.
    matplotlib.use("Agg")
    registry = load_registry()
    _WORKER["registry"] = registry
    _WORKER["roster"] = load_roster(roster_path, registry)
    _WORKER["options"] = options
    _WORKER["configs"] = {}


def _config(path: str) -> dict:
    """The parsed fit config for a path, cached per worker."""
    from sedfit.core.fitconfig import load_fit_config

    if path not in _WORKER["configs"]:
        _WORKER["configs"][path] = load_fit_config(path)
    return _WORKER["configs"][path]


def _completed_run(roster: Roster, job: BatchJob, run_id: str,
                   backend: str) -> dict | None:
    """The manifest row of an existing completed run, when there is one."""
    from sedfit.core.runs import locate_run

    target = roster.targets[job.target]
    backend_dir = target.dir / "SED" / job.recipe / backend
    if not backend_dir.is_dir():
        return None
    run_dir = locate_run(backend_dir, run_id)
    if run_dir is None:
        return None
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    row = json.loads(manifest_path.read_text(encoding="utf-8"))
    return row if row.get("status") == STATUS_OK else None


def _run_one(job: BatchJob) -> dict:
    """Plan, resume-check, and run one job; never raises."""
    from sedfit.core.build import build_target, write_build
    from sedfit.jobs import plan_job, run_job

    roster = _WORKER["roster"]
    registry = _WORKER["registry"]
    options = _WORKER["options"]
    started = time.monotonic()
    stage = "build" if options["build"] else "plan"
    try:
        if options["build"]:
            # mw_ebv is resolved once per target in the parent
            # (run_batch) and is empty unless this is a dereddening
            # build, so the table built here matches the one the fit
            # below looks for.
            result = build_target(roster, job.target, job.recipe,
                                  registry=registry,
                                  mw_ebv=options["mw_ebv"].get(job.target))
            write_build(result, registry)

        stage = "plan"
        cfg = _config(job.config)
        plan = plan_job(roster, job.target, job.recipe, cfg,
                        registry=registry,
                        allow_registry_drift=options["allow_registry_drift"],
                        dered=options["dered"])
        backend = plan["backend"]

        if options["resume"]:
            done = _completed_run(roster, job, plan["run_id"], backend)
            if done is not None:
                return _row_from_manifest(job, done, status=STATUS_SKIPPED)

        stage = "fit"
        row = run_job(roster, job.target, job.recipe, cfg, registry=registry,
                      label=job.label, force=options["force"],
                      allow_registry_drift=options["allow_registry_drift"],
                      dered=options["dered"], plots=options["plots"])
        return _row_from_manifest(job, row, status=STATUS_OK,
                                  seconds=time.monotonic() - started)
    except Exception as err:
        return _row(job.target, job.recipe, status=STATUS_FAILED, stage=stage,
                    seconds=round(time.monotonic() - started, 1),
                    error=f"{type(err).__name__}: {err}")


def _run_one_logged(job: BatchJob) -> dict:
    """_run_one with the job's stdout captured to its own log file."""
    log_dir = _WORKER["options"]["log_dir"]
    if log_dir is None:
        return _run_one(job)
    path = Path(log_dir) / f"{job.target}__{job.recipe}.log"
    with path.open("w", encoding="utf-8") as handle:
        with redirect_stdout(handle):
            return _run_one(job)


# ------------------------------------
# The batch
# ------------------------------------

def _cap_blas_threads() -> None:
    """Pin each worker to one BLAS thread unless the environment says so."""
    for name in BLAS_ENV_VARS:
        os.environ.setdefault(name, "1")


def default_workers(backend: str) -> int:
    """Worker count for a backend, bounded by the machine and by memory."""
    ceiling = MAX_WORKERS_BY_BACKEND.get(backend, 1)
    return max(1, min(ceiling, (os.cpu_count() or 2) - 1))


def _write_report(path: Path, rows: list[dict]) -> None:
    """Write the report sorted by target and recipe."""
    ordered = sorted(rows, key=lambda r: (r["target"], r["recipe"]))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REPORT_COLUMNS))
        writer.writeheader()
        writer.writerows(ordered)


def run_batch(
        roster_path: str | Path,
        config_path: str | Path,
        *,
        targets: list[str] | None = None,
        recipes: list[str] | None = None,
        label: str | None = None,
        workers: int | None = None,
        resume: bool = True,
        build: bool = False,
        dered: bool = False,
        mw_ebv: float | None = None,
        plots: bool = True,
        force: bool = False,
        allow_registry_drift: bool = False,
        report_path: str | Path | None = None,
        log_dir: str | Path | None = None,
        stop_after_failures: int = DEFAULT_STOP_AFTER_FAILURES,
        progress: Callable[[str], None] = print,
    ) -> dict:
    """Run every applicable target x recipe pair under one fit config.

    Parameters
    ----------
    targets, recipes : list of str, optional
        Restrict the grid; None takes every roster entry. [default: None]
    workers : int or None
        Worker processes; None picks by backend (see default_workers).
        A value of 1 runs in this process, so tracebacks survive.
        [default: None]
    resume : bool
        Skip a job whose run_id already has a completed run directory.
        Ignored under force. [default: True]
    build : bool
        Rebuild each target's SED table before fitting it. Off by
        default: a rebuild that changes the table changes run_id.
        [default: False]
    mw_ebv : float or None
        One E(B-V) for every target under ``build`` and ``dered``; None
        looks each target's value up from SFD. Meaningless without both
        (a hard error), since only a dereddening build consumes it.
        [default: None]
    force : bool
        Replace existing runs whose machinery stamps differ.
        [default: False]
    report_path : str or Path, optional
        Report CSV; None writes batch_report.csv beside the roster.
        [default: None]
    log_dir : str or Path, optional
        Per-job stdout logs; None writes batch_logs/ beside the report.
        [default: None]
    stop_after_failures : int
        Abandon the batch once this many jobs have failed; 0 disables.
        [default: 20]
    progress : callable
        Receives one line per completed job. [default: print]

    Returns
    -------
    summary : dict
        n_jobs, n_ok, n_failed, n_skipped, report, aborted.
    """
    from sedfit.core.fitconfig import load_fit_config
    from sedfit.core.registry import load_registry
    from sedfit.core.roster import load_roster

    roster_path = str(Path(roster_path).resolve())
    registry = load_registry()
    roster = load_roster(roster_path, registry)
    backend = load_fit_config(config_path)["backend"]

    jobs, rows = expand_jobs(roster, config_path, targets=targets,
                             recipes=recipes, label=label)

    report_path = (Path(report_path) if report_path is not None
                   else Path(roster_path).parent / "batch_report.csv")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir = (Path(log_dir) if log_dir is not None
               else report_path.parent / "batch_logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    if mw_ebv is not None and not (build and dered):
        raise ValueError("--mw-ebv only reaches a dereddening build; pass it "
                         "with both --build and --deredden, or build the "
                         "_dered tables with `sedfit build --deredden`")

    # Resolved HERE, not in the workers: one lookup per target instead of
    # one per job, no concurrent hammering of the SFD service, and the
    # same value for every recipe a target runs under.
    ebv: dict[str, float] = {}
    if build and dered:
        from sedfit.core.dered import fetch_ebv_sfd

        for name in dict.fromkeys(job.target for job in jobs):
            position = roster.targets[name].position
            ebv[name] = (mw_ebv if mw_ebv is not None
                         else fetch_ebv_sfd(position.ra_deg, position.dec_deg))
        progress(f"E(B-V) for {len(ebv)} target(s): "
                 + (f"explicit {mw_ebv}" if mw_ebv is not None
                    else f"SFD, {min(ebv.values()):.4f}-{max(ebv.values()):.4f}"))

    if workers is None:
        workers = default_workers(backend)
    workers = max(1, min(workers, len(jobs) or 1))
    options = {"resume": resume and not force, "build": build, "dered": dered,
               "plots": plots, "force": force, "mw_ebv": ebv,
               "allow_registry_drift": allow_registry_drift,
               "log_dir": str(log_dir)}

    progress(f"{len(jobs)} jobs ({backend}), {workers} worker(s); "
             f"{len(rows)} pair(s) not applicable")
    partial_path = report_path.with_suffix(report_path.suffix + ".partial")
    counts = {STATUS_OK: 0, STATUS_FAILED: 0, STATUS_SKIPPED: len(rows)}
    aborted = False

    with partial_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REPORT_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        handle.flush()

        def record(row: dict) -> None:
            rows.append(row)
            writer.writerow(row)
            handle.flush()
            counts[row["status"]] = counts.get(row["status"], 0) + 1
            done = counts[STATUS_OK] + counts[STATUS_FAILED]
            detail = (row["error"] if row["status"] == STATUS_FAILED
                      else f"z50={row['zred_p50']:.4f}"
                      if row["zred_p50"] is not None else "")
            progress(f"[{done}/{len(jobs)}] {row['status']:7s} "
                     f"{row['target']}/{row['recipe']} {detail}")

        if workers == 1:
            _init_worker(roster_path, options)
            for job in jobs:
                record(_run_one_logged(job))
                if stop_after_failures and \
                        counts[STATUS_FAILED] >= stop_after_failures:
                    aborted = True
                    break
        else:
            _cap_blas_threads()
            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=workers, mp_context=context,
                                     initializer=_init_worker,
                                     initargs=(roster_path, options)) as pool:
                futures = {pool.submit(_run_one_logged, job): job
                           for job in jobs}
                for future in as_completed(futures):
                    record(future.result())
                    if stop_after_failures and \
                            counts[STATUS_FAILED] >= stop_after_failures:
                        aborted = True
                        for pending in futures:
                            pending.cancel()
                        break

    _write_report(report_path, rows)
    partial_path.unlink()

    if aborted:
        progress(f"ABORTED after {counts[STATUS_FAILED]} failures "
                 f"(--stop-after-failures {stop_after_failures})")
    progress(f"{counts[STATUS_OK]} ok, {counts[STATUS_FAILED]} failed, "
             f"{counts[STATUS_SKIPPED]} skipped -> {report_path}")
    return {"n_jobs": len(jobs), "n_ok": counts[STATUS_OK],
            "n_failed": counts[STATUS_FAILED],
            "n_skipped": counts[STATUS_SKIPPED],
            "report": str(report_path), "aborted": aborted}
