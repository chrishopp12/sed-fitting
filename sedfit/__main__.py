#!/usr/bin/env python3
"""
__main__.py

sedfit Command-Line Interface
---------------------------------------------------------

Six verbs, in workflow order: roster, build, fit, batch, plot, manifest.
A seventh, run, chains roster, build and batch in one call.

Requirements:
  - numpy, pandas, astropy (via sedfit.core)

Usage:
  sedfit VERB [options]      -- sedfit --help lists the verbs
  sedfit VERB --help         -- every option for one verb

Examples:
  Generate a roster from a sample catalog and a campaign config:
    sedfit roster --catalog sweep_targets.csv --campaign campaign.json

  Assemble every target's SED table:
    sedfit build --roster roster.json

  Fit one target with one recipe:
    sedfit fit --roster roster.json --target bcg_1 --recipe master_tilt \
        --config configs/quick_tilt.json

  Build and fit the whole sample in parallel:
    sedfit batch --roster roster.json --config configs/quick_tilt.json --build

  Everything from a sample catalog in one call:
    sedfit run --catalog sweep_targets.csv --campaign campaign.json \
        --config configs/quick_tilt.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

from sedfit.batch import DEFAULT_STOP_AFTER_FAILURES
from sedfit.core.registry import load_registry
from sedfit.core.roster import load_roster

# The CLI writes figures to files and never opens a window. Library
# callers keep whatever backend they have already chosen.
matplotlib.use("Agg")

DEFAULT_ROSTER_NAME = "roster.json"


# ------------------------------------
# Verb handlers
# ------------------------------------

def _registry(args: argparse.Namespace):
    """The packaged registry, or the one named by --registry."""
    return load_registry(getattr(args, "registry", None))


def _roster(args: argparse.Namespace) -> int:
    """Generate a roster from a sample catalog and a campaign config."""
    from sedfit.core.generate import (
        generate_roster,
        load_campaign,
        read_sample_catalog,
        write_roster,
    )

    registry = _registry(args)
    targets, column_report = read_sample_catalog(args.catalog)
    campaign = load_campaign(args.campaign)
    print(column_report)
    roster, report = generate_roster(
        targets, campaign, registry=registry,
        campaign_dir=Path(args.campaign).resolve().parent)

    counts: dict[str, int] = {}
    for row in report:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print(f"{len(targets)} catalog rows -> {len(roster['targets'])} targets")
    print("  " + ", ".join(f"{n} {status}"
                           for status, n in sorted(counts.items())))
    for row in report:
        if row["status"] != "kept":
            print(f"  {row['status']:8s} {row['target']}/{row['source']}: "
                  f"{row['detail']}")

    if args.dry_run:
        print("dry run: nothing written")
        return 0
    written, report_path = write_roster(roster, report, _roster_out(args))
    print(f"{written}\n{report_path}")
    return 0


def _build(args: argparse.Namespace) -> int:
    """Assemble the SED table for each requested target and recipe."""
    from sedfit.core.build import build_target, check_applicability, write_build
    from sedfit.core.dered import fetch_ebv_sfd

    if args.mw_ebv is not None and not args.deredden:
        raise SystemExit("--mw-ebv applies only with --deredden")

    registry = _registry(args)
    roster = load_roster(args.roster, registry)
    targets = [_known(args.target, roster.targets, "target")] if args.target \
        else list(roster.targets)
    recipes = [_known(args.recipe, roster.recipes, "recipe")] if args.recipe \
        else list(roster.recipes)
    ebv_cache: dict[str, float] = {}

    built = skipped = 0
    for target_name in targets:
        for recipe_name in recipes:
            target = roster.targets[target_name]
            recipe = roster.recipes[recipe_name]
            try:
                check_applicability(recipe, target)
            except ValueError as err:
                if args.target and args.recipe:
                    raise
                print(f"skip {target_name}/{recipe_name}: {err}")
                skipped += 1
                continue
            if args.dry_run:
                print(f"ok {target_name}/{recipe_name}")
                built += 1
                continue
            mw_ebv = None
            if args.deredden:
                if args.mw_ebv is not None:
                    mw_ebv = args.mw_ebv
                else:
                    if target_name not in ebv_cache:
                        ebv_cache[target_name] = fetch_ebv_sfd(
                            target.position.ra_deg, target.position.dec_deg)
                    mw_ebv = ebv_cache[target_name]
            result = build_target(roster, target_name, recipe_name,
                                  registry=registry, mw_ebv=mw_ebv)
            out = write_build(result, registry)
            print(f"built {target_name}/{recipe_name} -> {out}"
                  + (f"  [dered E(B-V)={mw_ebv:.4f}]" if mw_ebv is not None
                     else ""))
            built += 1
    if args.dry_run:
        print(f"dry run: {built} pair(s) validate, {skipped} skipped")
    return 0


def _fit(args: argparse.Namespace) -> int:
    """Fit one built SED table, or every job in a jobs file."""
    from sedfit.core.fitconfig import load_fit_config
    from sedfit.jobs import plan_job, run_job, run_jobs

    if not args.jobs and not (args.target and args.recipe and args.config):
        raise SystemExit("fit needs --target/--recipe/--config, or --jobs")
    if args.jobs and (args.phot_csv or args.label):
        raise SystemExit("--phot-csv and --label apply to a single job, "
                         "not to --jobs")

    registry = _registry(args)
    roster = load_roster(args.roster, registry)

    if args.dry_run:
        jobs = (_read_jobs(args.jobs) if args.jobs
                else [{"target": args.target, "recipe": args.recipe,
                       "config": args.config}])
        for job in jobs:
            cfg = load_fit_config(job["config"])
            unseeded = (cfg["backend"] == "prospector"
                        and cfg["prospector"]["seed"] is None)
            plan = plan_job(roster, job["target"], job["recipe"], cfg,
                            registry=registry,
                            allow_registry_drift=args.allow_registry_drift,
                            phot_csv=None if args.jobs else args.phot_csv,
                            dered=job.get("dered", args.deredden))
            print(f"ok {job['target']}/{job['recipe']}/{plan['backend']}  "
                  f"run_id={plan['run_id']}"
                  + ("  (unseeded sampler: the real run draws a fresh id)"
                     if unseeded else ""))
        print("dry run: all jobs validate")
        return 0

    common = dict(registry=registry, force=args.force,
                  allow_registry_drift=args.allow_registry_drift,
                  dered=args.deredden, plots=not args.no_plots)
    if args.jobs:
        rows = run_jobs(roster, _read_jobs(args.jobs), **common)
    else:
        cfg = load_fit_config(args.config)
        rows = [run_job(roster, args.target, args.recipe, cfg,
                        label=args.label, phot_csv=args.phot_csv, **common)]
    for row in rows:
        estimates = row.get("estimates") or {}
        z50 = estimates.get("zred_p50")
        print(f"{row['run_id']}  {row['target']}/{row['recipe']}"
              f"/{row['backend']}  status={row['status']}"
              + (f"  z50={z50:.4f}" if z50 is not None else ""))
    return 0


def _batch(args: argparse.Namespace) -> int:
    """Fit every applicable target and recipe under one config."""
    from sedfit.batch import read_target_list, run_batch

    if args.targets and args.target:
        raise SystemExit("batch takes --targets or --target, not both")
    if args.targets:
        targets = read_target_list(args.targets)
    elif args.target:
        targets = [args.target]
    else:
        targets = None

    summary = run_batch(
        args.roster, args.config, targets=targets,
        recipes=[args.recipe] if args.recipe else None, label=args.label,
        workers=args.workers, resume=not args.no_resume, build=args.build,
        dered=args.deredden, mw_ebv=args.mw_ebv,
        plots=not args.no_plots, force=args.force,
        allow_registry_drift=args.allow_registry_drift,
        report_path=args.report, log_dir=args.log_dir,
        stop_after_failures=args.stop_after_failures)
    return 1 if (summary["n_failed"] or summary["aborted"]) else 0


def _run(args: argparse.Namespace) -> int:
    """Generate the roster, then build and fit the whole sample."""
    stages: list[tuple[str, int]] = []

    args.dry_run = False
    args.out = args.out or str(
        Path(args.campaign).resolve().parent / DEFAULT_ROSTER_NAME)
    print("== roster ==")
    stages.append(("roster", _roster(args)))

    args.roster = _roster_out(args)
    args.build = True
    args.no_resume = False
    print("\n== build + fit ==")
    try:
        stages.append(("batch", _batch(args)))
    except Exception as err:
        print(f"batch failed: {type(err).__name__}: {err}")
        stages.append(("batch", 1))

    print("\n== summary ==")
    for name, code in stages:
        print(f"  {name:8s} {'ok' if code == 0 else 'FAILED'}")
    return 0 if all(code == 0 for _, code in stages) else 1


def _plot(args: argparse.Namespace) -> int:
    """Re-render the figures for an existing run directory."""
    run_dir = Path(args.run_dir)
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        raise SystemExit(f"{run_dir} is not a run directory (no config.json)")
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    registry = _registry(args)
    if cfg["backend"] == "eazy":
        from sedfit.backends.eazy.plots import generate_plots
        from sedfit.backends.eazy.results import load_run

        written = generate_plots(load_run(run_dir), registry=registry,
                                 z_ref=cfg.get("z_ref"))
    elif cfg["backend"] == "prospector":
        from sedfit.backends.prospector.plots import generate_plots

        written = generate_plots(run_dir, registry=registry)
    else:
        raise SystemExit(f"{run_dir}: unknown backend {cfg['backend']!r}")
    for path in written:
        print(path)
    return 0


def _manifest(args: argparse.Namespace) -> int:
    """Summarize the central run manifest and check its run directories."""
    from sedfit.core.runs import read_manifest

    registry = _registry(args)
    roster = load_roster(args.roster, registry)
    rows, problems = read_manifest(roster.manifest_path)
    latest: dict[str, dict] = {}
    superseded = 0
    for row in rows:
        if row.get("path") in latest:
            superseded += 1
        latest[row.get("path")] = row
    missing = [path for path in latest
               if not (roster.data_root / path).is_dir()]
    print(f"{len(rows)} rows, {len(latest)} current runs, "
          f"{superseded} superseded")
    for problem in problems:
        print(f"problem: {problem}")
    for path in missing:
        print(f"missing directory: {path}")
    for path, row in sorted(latest.items()):
        estimates = row.get("estimates") or {}
        z50 = estimates.get("zred_p50")
        print(f"  {row['run_id']}  {row['status']:6s}  {path}"
              + (f"  z50={z50:.4f}" if z50 is not None else ""))
    return 1 if (problems or missing) else 0


# ------------------------------------
# Helpers
# ------------------------------------

def _known(name: str, available: dict, kind: str) -> str:
    """Return name, or raise naming what the roster does declare."""
    if name not in available:
        raise SystemExit(f"unknown {kind} {name!r}; the roster declares: "
                         f"{', '.join(sorted(available))}")
    return name


def _roster_out(args: argparse.Namespace) -> str:
    """The roster path to write: --out, else beside the campaign config."""
    if args.out:
        return args.out
    return str(Path(args.campaign).resolve().parent / DEFAULT_ROSTER_NAME)


def _read_jobs(path: str) -> list[dict]:
    """Load a jobs file, checking each element names its three keys."""
    jobs = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(jobs, list):
        raise SystemExit(f"{path}: expected a JSON list of job objects")
    for i, job in enumerate(jobs):
        missing = [k for k in ("target", "recipe", "config")
                   if k not in (job if isinstance(job, dict) else {})]
        if missing:
            raise SystemExit(f"{path}: job {i} is missing {missing}")
    return jobs


# ------------------------------------
# Parser
# ------------------------------------

def _add_roster_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--roster", required=True,
        help="the roster JSON declaring targets, sources and recipes")


def _add_registry_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--registry", default=None,
        help="a campaign band-registry JSON instead of the packaged one; "
             "use the same registry for roster, build, fit and plot")


def _add_fit_args(parser: argparse.ArgumentParser) -> None:
    """Options shared by fit and batch, so the two cannot drift apart."""
    parser.add_argument(
        "--force", action="store_true",
        help="replace an existing run whose recorded package versions "
             "differ from the live ones")
    parser.add_argument(
        "--allow-registry-drift", action="store_true",
        help="record a warning instead of failing when a bandpass "
             "definition changed since the table was built")
    parser.add_argument(
        "--deredden", action="store_true",
        help="fit the dereddened (_dered) table")
    parser.add_argument(
        "--no-plots", action="store_true",
        help="skip the run's figures")


def build_parser() -> argparse.ArgumentParser:
    """The full argparse tree; one subparser per verb."""
    parser = argparse.ArgumentParser(
        prog="sedfit",
        description="Roster-driven SED assembly and photometric-redshift "
                    "fitting, with eazy-py and Prospector backends.")
    subparsers = parser.add_subparsers(dest="verb", required=True,
                                       metavar="VERB")

    # -- roster --
    desc = "Generate a roster from a sample catalog and a campaign config."
    roster = subparsers.add_parser("roster", help=desc, description=desc)
    roster.add_argument("--catalog", required=True,
                        help="sample catalog CSV, one row per target")
    roster.add_argument("--campaign", required=True,
                        help="campaign config JSON: sources, recipes, sample")
    roster.add_argument("--out", default=None,
                        help="roster JSON to write; the report goes beside "
                             "it [default: roster.json next to --campaign]")
    roster.add_argument("--dry-run", action="store_true",
                        help="report what would be declared, write nothing")
    _add_registry_arg(roster)
    roster.set_defaults(handler=_roster)

    # -- build --
    desc = "Assemble each target's fit-ready SED table and provenance sidecar."
    build = subparsers.add_parser("build", help=desc, description=desc)
    _add_roster_arg(build)
    build.add_argument("--target", default=None,
                       help="one target; omit to build every roster target")
    build.add_argument("--recipe", default=None,
                       help="one recipe; omit to build every applicable one")
    build.add_argument("--dry-run", action="store_true",
                       help="validate every pair, write nothing")
    build.add_argument("--deredden", action="store_true",
                       help="correct Milky Way extinction, writing the "
                            "_dered table alongside")
    build.add_argument("--mw-ebv", type=float, default=None,
                       help="one E(B-V) for every target instead of the "
                            "per-target SFD lookup; needs --deredden")
    _add_registry_arg(build)
    build.set_defaults(handler=_build)

    # -- fit --
    desc = "Fit one built SED table, or a JSON list of jobs, with one backend."
    fit = subparsers.add_parser("fit", help=desc, description=desc)
    _add_roster_arg(fit)
    fit.add_argument("--target", default=None,
                     help="target to fit, with --recipe and --config")
    fit.add_argument("--recipe", default=None,
                     help="recipe whose built SED table to fit")
    fit.add_argument("--config", default=None,
                     help="fit-config JSON: backend, template set, priors")
    fit.add_argument("--label", default=None,
                     help="suffix for the run directory "
                          "[default: the config's name]")
    fit.add_argument("--jobs", default=None,
                     help="JSON list of {target, recipe, config, label?, "
                          "dered?} objects, run in order")
    fit.add_argument("--phot-csv", default=None,
                     help="fit this photometry table instead of the built one")
    fit.add_argument("--dry-run", action="store_true",
                     help="resolve and validate every job, write nothing")
    _add_fit_args(fit)
    _add_registry_arg(fit)
    fit.set_defaults(handler=_fit)

    # -- batch --
    desc = ("Fit every applicable target and recipe under one config, "
            "in parallel.")
    batch = subparsers.add_parser("batch", help=desc, description=desc)
    _add_roster_arg(batch)
    batch.add_argument("--config", required=True,
                       help="fit-config JSON applied to every job")
    batch.add_argument("--targets", default=None,
                       help="CSV whose name column selects targets")
    batch.add_argument("--target", default=None,
                       help="a single target, instead of --targets")
    batch.add_argument("--recipe", default=None,
                       help="one recipe; omit to run every applicable one")
    batch.add_argument("--label", default=None,
                       help="suffix for every run directory "
                            "[default: the config's name]")
    batch.add_argument("--workers", type=int, default=None,
                       help="worker processes; 1 runs in this process "
                            "[default: chosen from the backend and CPU count]")
    batch.add_argument("--no-resume", action="store_true",
                       help="re-run jobs whose run_id is already complete")
    batch.add_argument("--build", action="store_true",
                       help="rebuild each SED table before fitting it")
    batch.add_argument("--report", default=None,
                       help="report CSV [default: batch_report.csv beside "
                            "the roster]")
    batch.add_argument("--log-dir", default=None,
                       help="per-job stdout logs [default: batch_logs/ "
                            "beside the report]")
    batch.add_argument("--stop-after-failures", type=int,
                       default=DEFAULT_STOP_AFTER_FAILURES,
                       help="abandon the batch after this many failures; "
                            "0 disables")
    batch.add_argument("--mw-ebv", type=float, default=None,
                       help="one E(B-V) for every target instead of the "
                            "per-target SFD lookup; needs --build --deredden")
    _add_fit_args(batch)
    batch.set_defaults(handler=_batch)

    # -- run --
    desc = "Generate the roster, then build and fit the whole sample."
    run = subparsers.add_parser("run", help=desc, description=desc)
    run.add_argument("--catalog", required=True,
                     help="sample catalog CSV, one row per target")
    run.add_argument("--campaign", required=True,
                     help="campaign config JSON: sources, recipes, sample")
    run.add_argument("--config", required=True,
                     help="fit-config JSON applied to every job")
    run.add_argument("--out", default=None,
                     help="roster JSON to write "
                          "[default: roster.json next to --campaign]")
    run.add_argument("--recipe", default=None,
                     help="one recipe; omit to run every applicable one")
    run.add_argument("--targets", default=None,
                     help="CSV whose name column selects targets")
    run.add_argument("--target", default=None,
                     help="a single target, instead of --targets")
    run.add_argument("--label", default=None,
                     help="suffix for every run directory "
                          "[default: the config's name]")
    run.add_argument("--workers", type=int, default=None,
                     help="worker processes; 1 runs in this process "
                          "[default: chosen from the backend and CPU count]")
    run.add_argument("--report", default=None,
                     help="report CSV [default: batch_report.csv beside "
                          "the roster]")
    run.add_argument("--log-dir", default=None,
                     help="per-job stdout logs [default: batch_logs/ "
                          "beside the report]")
    run.add_argument("--stop-after-failures", type=int,
                     default=DEFAULT_STOP_AFTER_FAILURES,
                     help="abandon the batch after this many failures; "
                          "0 disables")
    run.add_argument("--mw-ebv", type=float, default=None,
                     help="one E(B-V) for every target instead of the "
                          "per-target SFD lookup; needs --deredden")
    _add_fit_args(run)
    run.set_defaults(handler=_run)

    # -- plot --
    desc = "Re-render the figures for an existing run directory."
    plot = subparsers.add_parser("plot", help=desc, description=desc)
    plot.add_argument("--run-dir", required=True,
                      help="a run directory written by sedfit fit")
    _add_registry_arg(plot)
    plot.set_defaults(handler=_plot)

    # -- manifest --
    desc = "Summarize the central run manifest and check its run directories."
    manifest = subparsers.add_parser("manifest", help=desc, description=desc)
    _add_roster_arg(manifest)
    _add_registry_arg(manifest)
    manifest.set_defaults(handler=_manifest)

    return parser


def main() -> None:
    """Parse the command line and dispatch to the verb's handler."""
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(args.handler(args))


if __name__ == "__main__":
    main()
