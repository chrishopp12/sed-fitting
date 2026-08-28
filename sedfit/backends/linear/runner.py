"""
runner.py

A Linear Fit as a Recorded Run
---------------------------------------------------------

One resolved `linear` config plus one prepared spectrum in, one staged run
directory out: the config echo, the spectrum bytes and their sidecar, the
machinery stamps, and the fit itself.

Not reached through `jobs.py`, which is photometry-first at five gates. The
caller supplies the spectrum and the output directory; there is no roster, no
recipe, and no central manifest.

Requirements:
    numpy (plus astropy and pandas through core.spectrum)

Notes:
    Rationale in DESIGN.md sections 16 and 17.
"""
from __future__ import annotations

import json
import platform
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np

import sedfit
from sedfit.backends.linear.basis import TemplateBasis
from sedfit.backends.linear.fitting import fit_spectrum, scan_spectrum
from sedfit.backends.linear.lsf import (
    LineSpread,
    Resolution,
    resolve_resolution_file,
)
from sedfit.backends.linear.gas import (
    GasBasis,
    read_line_list,
    resolve_line_list,
)
from sedfit.core.fitconfig import hash_projection
from sedfit.core.provenance import git_state, run_id as make_run_id, sha256_file
from sedfit.core.runs import finalize_run, now_iso, stage_run
from sedfit.core.spectrum import Spectrum, apply_spectrum_policy, vac_to_air


# The internal fit unit, 1e-18 erg/s/cm^2/A. core.spectrum delivers f_nu in
# microjanskys while the template files are f_lambda, so the data is converted
# rather than left for the multiplicative polynomial to absorb a lambda^2.
# FLAM_PER_UJY converts at each wavelength: f_lambda = f_nu * c / lambda^2,
# with c in Angstrom/s and the 1e-29 taking microjanskys to cgs f_nu. The
# trailing 1e18 puts the result on the reference fit's scale.
#
# Measured, not assumed (DESIGN.md 16.4b): skipping the conversion changes
# almost nothing -- an 8th-order Chebyshev absorbs lambda^2 nearly exactly,
# and scaling flux and error together leaves the weighting untouched. The
# reasons to convert are that amplitudes then carry a stated unit and that
# the reference fits f_lambda, so bit-identity requires it.
_C_ANGSTROM_PER_S = 2.99792458e18
FLAM_PER_UJY = 1e-29 * _C_ANGSTROM_PER_S * 1e18
FLAM_UNIT = "1e-18 erg/s/cm2/A"

# One append-only row per run, in the output directory beside the run
# directories -- the split jobs.py has between a run's own manifest.json and
# the roster's central manifest.
MANIFEST_NAME = "manifest.jsonl"


def resolve_templates(linear_cfg: dict) -> list[Path]:
    """The basis files a linear config names, sorted."""
    directory = Path(linear_cfg["templates"])
    if not directory.is_dir():
        raise FileNotFoundError(f"{directory} is not a template directory")
    paths = sorted(directory.glob(linear_cfg["template_pattern"]))
    if not paths:
        raise FileNotFoundError(
            f"no {linear_cfg['template_pattern']!r} in {directory}")
    return paths


def content_digests(paths: list[Path], spectrum: Spectrum,
                    transmission_cfg: dict | None = None,
                    gas_cfg: dict | None = None,
                    resolution_cfg: dict | None = None) -> dict:
    """Content hashes for the projection: templates, spectrum, and every
    curve a config points at rather than states."""
    digests = {"templates": {p.name: sha256_file(p)[:16] for p in paths},
               "spectrum": spectrum.sha256}
    if transmission_cfg is not None:
        digests["transmission"] = sha256_file(transmission_cfg["file"])[:16]
    if gas_cfg is not None:
        digests["lines"] = sha256_file(
            resolve_line_list(gas_cfg["lines"]))[:16]
    for field, path in (resolution_cfg or {}).items():
        digests[field] = sha256_file(path)[:16]
    return digests


def read_transmission(transmission_cfg: dict,
                      wave_vac: np.ndarray) -> np.ndarray:
    """T(lambda) on the spectrum's own channels, 1.0 outside the curve.

    The file is two whitespace-separated columns, wavelength [Angstrom] and
    a positive dimensionless factor, read like a template file. Its frame is
    declared in the config rather than a sidecar, matching poly_wave_frame;
    the query points are converted into that frame rather than the curve out
    of it, so a curve prepared against air data is used where it was made.
    """
    path = Path(transmission_cfg["file"])
    table = np.loadtxt(path, usecols=(0, 1))
    if table.ndim != 2 or len(table) < 2:
        raise ValueError(f"{path}: need at least two (wavelength, factor) rows")
    wave, factor = table[:, 0], table[:, 1]
    if not np.all(np.diff(wave) > 0):
        raise ValueError(f"{path}: wavelengths must increase strictly")
    if not np.all(np.isfinite(factor)) or not np.all(factor > 0):
        raise ValueError(f"{path}: every factor must be finite and positive")

    query = (wave_vac if transmission_cfg["wave_frame"] == "vacuum"
             else vac_to_air(wave_vac))
    return np.interp(query, wave, factor, left=1.0, right=1.0)


def _resolution(cfg: dict, *, packaged: bool) -> Resolution:
    """One Resolution from a config block of the shared curve shape."""
    if cfg["file"] is None:
        return Resolution(unit=cfg["unit"], constant=cfg["constant"])
    path = resolve_resolution_file(cfg["file"]) if packaged else Path(
        cfg["file"])
    return Resolution.from_file(path, unit=cfg["unit"],
                                frame=cfg["wave_frame"])


def resolution_paths(linear_cfg: dict) -> dict[str, Path]:
    """The curve files a linear config names, keyed by their config field."""
    paths = {}
    if linear_cfg["lsf"] is not None and linear_cfg["lsf"]["file"]:
        paths["lsf"] = Path(linear_cfg["lsf"]["file"])
    resolution = linear_cfg["template_resolution"]
    if resolution is not None and resolution["file"]:
        paths["template_resolution"] = resolve_resolution_file(
            resolution["file"])
    return paths


def build_lsf(linear_cfg: dict, wave_vac: np.ndarray,
              fitted: np.ndarray) -> LineSpread | None:
    """The LineSpread a linear config describes, or None when lsf is off."""
    lsf_cfg = linear_cfg["lsf"]
    if lsf_cfg is None:
        return None
    return LineSpread(
        wave_vac, _resolution(lsf_cfg, packaged=False),
        _resolution(linear_cfg["template_resolution"], packaged=True),
        on_undersampled=lsf_cfg["on_undersampled"], fitted=fitted)


def build_gas(linear_cfg: dict) -> GasBasis | None:
    """The GasBasis a linear config describes, or None when gas is off."""
    gas_cfg = linear_cfg["gas"]
    if gas_cfg is None:
        return None
    return GasBasis(read_line_list(resolve_line_list(gas_cfg["lines"])),
                    sigma_kms=gas_cfg["sigma_kms"],
                    ratio_locked=gas_cfg["ratio_locked"])


def band_coverage(gas, wave_vac: np.ndarray,
                  z_min: float, z_max: float) -> dict:
    """What the spectrum's own range does to the gas basis over a scan.

    A ratio lock is not a property of the package, it is a property of a
    LINE LIST AGAINST A BAND: tying two lines constrains nothing unless both
    are observable. With one member permanently outside the range the lock
    silently becomes a free column whose amplitude is reported as a total
    doublet flux extrapolated through an assumed ratio -- a free parameter
    with a misleading name. `[SIII]` does exactly this on MUSE, where 9531
    sits past the red edge at every redshift.

    Reported, never refused: which redshifts a caller scans is theirs, and a
    group inert over part of a range is still correct over the rest.
    """
    if gas is None:
        return {}
    low, high = float(wave_vac.min()), float(wave_vac.max())
    groups = {}
    for name, component in zip(gas.names, gas.components):
        if len(component) < 2:
            continue
        spans = []
        for rest, _ in component:
            lo = max(z_min, low / rest - 1.0)
            hi = min(z_max, high / rest - 1.0)
            spans.append((lo, hi) if hi > lo else None)
        live = [s for s in spans if s is not None]
        complete = None
        if len(live) == len(component):
            both_lo = max(s[0] for s in live)
            both_hi = min(s[1] for s in live)
            if both_hi > both_lo:
                complete = (both_lo, both_hi)
        groups[name] = {
            "members": len(component),
            "members_ever_in_band": len(live),
            "both_in_band_over": (None if complete is None
                                  else [round(complete[0], 4),
                                        round(complete[1], 4)]),
            "inert": complete is None,
        }
    return groups


def build_basis(linear_cfg: dict, paths: list[Path]) -> TemplateBasis:
    """The TemplateBasis a linear config describes."""
    wave_range = tuple(linear_cfg["template_wave_range"])
    normalize = linear_cfg["normalize_range"]
    return TemplateBasis(
        paths, wave_range=wave_range,
        dv_kms=linear_cfg["template_dv_kms"],
        normalize_range=(wave_range if normalize is None
                         else tuple(normalize)))


def to_flam(wave_A: np.ndarray, flux_uJy: np.ndarray) -> np.ndarray:
    """f_nu [uJy] to f_lambda in FLAM_UNIT, at observed vacuum wavelengths."""
    return np.asarray(flux_uJy, float) * FLAM_PER_UJY / np.asarray(
        wave_A, float) ** 2


def _machinery() -> dict:
    import scipy

    state = git_state(Path(sedfit.__file__).resolve().parents[1])
    return {"package_version": sedfit.__version__,
            "git_rev": state["rev"], "git_dirty": state["dirty"],
            "fsps_libraries": None,
            "versions": {"numpy": np.__version__,
                         "scipy": scipy.__version__,
                         "python": platform.python_version()}}


def plan(resolved: dict, spectrum: Spectrum) -> dict:
    """Resolve and hash one linear fit without writing anything.

    Returns
    -------
    plan : dict
        linear, paths, basis, gas, lsf, digests, run_id, wave_vac,
        flux, error, fitted, poly_wave, poly_domain, transmission,
        normalize_range.
    """
    if resolved["backend"] != "linear":
        raise ValueError(f"not a linear config: backend is "
                         f"{resolved['backend']!r}")
    linear = resolved["linear"]
    spec_cfg = linear["spectrum"]

    flux_uJy, err_uJy, fitted = apply_spectrum_policy(
        spectrum, mu_lensing=resolved["mu_lensing"],
        err_floor=spec_cfg["err_floor"],
        mask_windows=spec_cfg["mask_windows"])
    if not fitted.any():
        raise ValueError("the spectrum policy leaves no fittable channels")

    wave_vac = np.asarray(spectrum.wave_A, float)
    flux = to_flam(wave_vac, flux_uJy)
    error = to_flam(wave_vac, err_uJy)
    # Masked channels may be non-finite by contract; the fit never reads
    # them, but a NaN in `error` would poison var**2 arithmetic anyway.
    error = np.where(fitted, error, 1.0)
    flux = np.where(np.isfinite(flux), flux, 0.0)

    poly_wave = (vac_to_air(wave_vac)
                 if linear["poly_wave_frame"] == "air" else wave_vac)
    domain = linear["poly_domain"]
    poly_domain = (tuple(domain) if domain is not None
                   else (float(poly_wave[fitted].min()),
                         float(poly_wave[fitted].max())))

    transmission_cfg = linear["transmission"]
    transmission = (np.ones_like(wave_vac) if transmission_cfg is None
                    else read_transmission(transmission_cfg, wave_vac))

    lsf = build_lsf(linear, wave_vac, fitted)
    if lsf is not None:
        lsf.assert_degradation_is_fixed(linear["z_min"], linear["z_max"])
        flux, error = lsf.degrade(flux, error, fitted, linear["z_min"])

    paths = resolve_templates(linear)
    basis = build_basis(linear, paths)
    gas = build_gas(linear)
    digests = content_digests(paths, spectrum, transmission_cfg, linear["gas"],
                              resolution_paths(linear))
    rid = make_run_id(hash_projection(resolved, digests=digests))
    coverage = band_coverage(gas, wave_vac, float(linear["z_min"]),
                             float(linear["z_max"]))
    for name, info in coverage.items():
        if info["inert"]:
            warnings.warn(
                f"ratio-locked group {name!r} never has both members in "
                f"{wave_vac.min():.1f}-{wave_vac.max():.1f} A over z "
                f"{linear['z_min']}-{linear['z_max']}: the lock constrains "
                f"nothing and its reported flux is extrapolated through an "
                f"assumed ratio. Drop it from gas.ratio_locked.",
                RuntimeWarning, stacklevel=2)
    return {"linear": linear, "paths": paths, "basis": basis, "gas": gas,
            "lsf": lsf, "band_coverage": coverage,
            "digests": digests, "run_id": rid, "wave_vac": wave_vac,
            "flux": flux, "error": error, "fitted": fitted,
            "poly_wave": poly_wave, "poly_domain": poly_domain,
            "transmission": transmission,
            "normalize_range": basis.normalize_range}


def run(
        resolved: dict,
        spectrum: Spectrum,
        out_dir: str | Path,
        *,
        label: str | None = None,
        force: bool = False,
    ) -> dict:
    """Fit one spectrum into a staged run directory; returns the estimates.

    Parameters
    ----------
    resolved : dict
        A resolved linear fit config (fitconfig.resolve_config output).
    spectrum : Spectrum
        core.spectrum.read_spectrum output; observed-frame vacuum, uJy.
    out_dir : path
        The directory runs are created under: one subdirectory per run_id
        plus an append-only MANIFEST_NAME beside them.
    label : str or None
        Run-directory label; None uses the config name. [default: None]
    force : bool
        Allow replacing an existing run staged by different machinery.
        [default: False]

    Returns
    -------
    row : dict
        The manifest row, including 'estimates' and 'status'; also written
        to the run's manifest.json and appended to out_dir/MANIFEST_NAME.
    """
    prepared = plan(resolved, spectrum)
    linear = prepared["linear"]
    rid = prepared["run_id"]
    manifest = Path(out_dir) / MANIFEST_NAME
    machinery = _machinery()
    run_dir = stage_run(out_dir, rid, config=resolved, machinery=machinery,
                        label=label or resolved["name"], force=force,
                        spectrum_bytes=spectrum.csv_bytes,
                        spectrum_sha256=spectrum.sha256,
                        spectrum_sidecar=spectrum.provenance)

    row = {
        "run_id": rid,
        "path": str(run_dir),
        "written": now_iso(),
        "backend": "linear",
        "name": resolved["name"],
        **machinery,
        "spectrum_sha256_16": spectrum.sha256[:16],
        "templates": {"n": len(prepared["paths"]),
                      "source": str(Path(linear["templates"]).resolve())},
        "n_spec_channels": int(spectrum.mask.size),
        "n_spec_fit": int(prepared["fitted"].sum()),
        "mu_lensing": resolved["mu_lensing"],
        "z_ref": resolved["z_ref"],
        "poly_wave_frame": linear["poly_wave_frame"],
        "transmission": (None if linear["transmission"] is None
                         else linear["transmission"]["wave_frame"]),
        "gas": (None if linear["gas"] is None
                else {"lines": linear["gas"]["lines"],
                      "n_columns": prepared["gas"].n_columns,
                      "sigma_kms": linear["gas"]["sigma_kms"],
                      "band_coverage": prepared["band_coverage"]}),
        # A dispersion fitted under "ignore" is absorbing an instrument term
        # it cannot model, so it is uninterpretable whether or not it reached
        # a bound. sigma_pinned catches the bound; this catches the reason.
        "lsf_on_undersampled": (None if linear["lsf"] is None
                                else linear["lsf"]["on_undersampled"]),
        "flux_unit": FLAM_UNIT,
        "status": "failed", "estimates": None,
    }

    try:
        data = (prepared["wave_vac"], prepared["flux"], prepared["error"],
                prepared["fitted"], prepared["basis"])
        common = dict(
            redshift_grid=np.arange(linear["z_min"], linear["z_max"],
                                    linear["z_step"]),
            sigma_grid=np.arange(linear["sigma_min"],
                                 linear["sigma_max"] + 0.5 * linear[
                                     "sigma_step"], linear["sigma_step"]),
            poly_order=linear["poly_order"],
            poly_domain=prepared["poly_domain"],
            poly_wave=prepared["poly_wave"],
            transmission=prepared["transmission"],
            n_poly_iter=linear["n_poly_iter"],
            clip_sigma=linear["clip_sigma"], gas=prepared["gas"],
            lsf=prepared["lsf"],
            # The config declares the range; without this the simplex is the
            # one part of the fit that never sees it.
            sigma_bounds=(linear["sigma_min"], linear["sigma_max"]))
        scan_cfg = linear["scan"]
        scan = None
        if scan_cfg is None:
            fit = fit_spectrum(*data, **common)
        else:
            scan = scan_spectrum(
                *data, z_step_coarse=scan_cfg["z_step_coarse"],
                n_poly_iter_coarse=scan_cfg["n_poly_iter_coarse"],
                sigma_coarse=scan_cfg["sigma_coarse"],
                window_steps=scan_cfg["window_steps"],
                minima_dv_kms=scan_cfg["minima_dv_kms"], **common)
            fit = scan.fit
        write_fit(run_dir, fit, prepared, scan)
        row["estimates"] = {
            "redshift": fit.redshift, "redshift_err": fit.redshift_error,
            "sigma_kms": fit.sigma_kms, "sigma_err": fit.sigma_error,
            "sigma_pinned": fit.sigma_pinned,
            "chi2": fit.chi2, "dof": fit.dof,
            "chi2_per_dof": fit.chi2 / fit.dof if fit.dof > 0 else None,
            "n_clipped": fit.n_clipped,
            "n_active": int((fit.stellar_amplitudes > 0).sum()),
            "gas_fluxes": fit.gas_fluxes,
            "delta_chi2": fit.delta_chi2,
            "n_minima": len(fit.minima),
            # The blind statistic comes from the coarse pass when there was
            # one: it is the grid that spanned the whole redshift range.
            "scan_delta_chi2": None if scan is None else scan.delta_chi2,
            "scan_z_step": None if scan is None else scan.z_step,
            "light_fractions": fit.light_fractions}
        row["status"] = "ok"
    except Exception as err:
        row["error"] = f"{type(err).__name__}: {err}"
        finalize_run(run_dir, manifest, row)
        raise
    finalize_run(run_dir, manifest, row)
    return row


def write_fit(run_dir: str | Path, fit, prepared: dict, scan=None) -> None:
    """Write fit.json and model.npz into a staged run directory."""
    run_dir = Path(run_dir)
    (run_dir / "fit.json").write_text(json.dumps({
        "redshift": fit.redshift, "redshift_error": fit.redshift_error,
        "sigma_kms": fit.sigma_kms, "sigma_error": fit.sigma_error,
        # A dispersion held at sigma_min/sigma_max is not a measurement, and
        # the Hessian error beside it describes a curvature the bound
        # overrode.
        "sigma_pinned": fit.sigma_pinned,
        "error_method": prepared["linear"]["error_method"],
        "chi2": fit.chi2, "dof": fit.dof, "n_clipped": fit.n_clipped,
        # The grid's chi2 values come from grid_n_poly_iter iterations and
        # the fit's from n_poly_iter, so the two must never be differenced.
        "delta_chi2": fit.delta_chi2,
        "grid_n_poly_iter": fit.grid_n_poly_iter,
        "minima": [asdict(m) for m in fit.minima],
        # The bands as resolved, not as configured: both may be null in the
        # config, and a reader of this file should never have to re-derive
        # which band the light fractions below were normalized over.
        "poly_domain": list(prepared["poly_domain"]),
        "poly_wave_frame": prepared["linear"]["poly_wave_frame"],
        "normalize_range": list(prepared["normalize_range"]),
        "transmission": prepared["linear"]["transmission"],
        "gas": prepared["linear"]["gas"],
        "gas_fluxes": fit.gas_fluxes,
        "lsf": prepared["linear"]["lsf"],
        "template_resolution": prepared["linear"]["template_resolution"],
        "scan": None if scan is None else {
            "z_step": scan.z_step, "sigma_kms": scan.sigma_kms,
            "n_poly_iter": scan.n_poly_iter,
            "delta_chi2": scan.delta_chi2,
            "minima": [asdict(m) for m in scan.minima]},
        "flux_unit": FLAM_UNIT,
        "sigma_pinned": fit.sigma_pinned,
        # Reported, not enforced: an empty list means nothing was CHECKABLE,
        # not that the line set is sound.
        "physics_violations": [
            {"kind": v.kind, "lines": list(v.lines), "observed": v.observed,
             "bound": v.bound, "detail": v.detail}
            for v in fit.physics_violations],
        "velocity_floor_kms": fit.velocity_floor_kms,
        "n_gas_lines_in_band": sum(
            1 for f in fit.gas_fluxes.values() if f > 0),
        "amplitudes": {name: float(a)
                       for name, a in zip(fit.names, fit.stellar_amplitudes)},
        "light_fractions": fit.light_fractions,
        "chebyshev_coefficients": [float(c)
                                   for c in fit.chebyshev_coefficients],
        "digests": prepared["digests"],
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    np.savez_compressed(run_dir / "model.npz",
                        wave_vac=prepared["wave_vac"],
                        poly_wave=prepared["poly_wave"],
                        flux=prepared["flux"], error=prepared["error"],
                        transmission=prepared["transmission"],
                        model=fit.model, fitted=fit.fitted,
                        redshift_grid=fit.redshift_grid,
                        sigma_grid=fit.sigma_grid, chi2_grid=fit.chi2_grid,
                        **({} if scan is None else {
                            "coarse_redshift_grid": scan.redshift_grid,
                            "coarse_chi2": scan.chi2_grid}))
