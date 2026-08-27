"""
quick.py

Vectorized Photometric-Redshift Engine
---------------------------------------------------------

run_quick(cfg, frame, policy, run_dir) is the scale workhorse: the same
inputs and outputs as fitting.run_official, but eazy's likelihood
re-derived with vectorized numpy/scipy and no eazy-py import. eazy-py
rebuilds its template grid -- every template through every filter at
every grid redshift -- whenever the filter set changes, which per-object
SPHEREx tophats force on every object; this engine instead flattens all
filter curves into one wavelength array and recovers the per-band
photon-counting integrals with trapezoid weights and np.add.reduceat.

Reproduces the eazy-py 0.8.6 conventions: the redshift grid (linear
arange / log_zgrid), the TEF (CubicSpline at pivot/(1+z),
endpoint-clipped, scaled by TEMP_ERR_A2; per-band variance
efnu^2 + (TEF*max(fnu,0))^2), NNLS with internal per-template
renormalization, ln P(z) = -chi2/2 + tef_lnp with the 1100 A reddest-
filter clip and trapezoidal normalization, the 3-point parabola z_ml
with the -1 first-grid-point sentinel, single mode's unconstrained
analytic amplitudes (with the flux entering the TEF variance unclamped,
as in the official routine), and the strictly-inside-grid fixed-z
evaluation.

Known deviations (use the official engine when they matter): no IGM
absorption (warned when the grid samples rest wavelengths below 1300 A);
bandpass integrals on refined filter-curve nodes rather than template
nodes (sub-grid-step agreement, not bit-identical); best-fit and fixed-z
designs are direct projections rather than spline interpolations of the
precomputed grid.

Data products (per run directory): config.json, catalog.csv,
templates.param, template_error.dat, engine.info, plus summary.csv /
arrays.npz [/ singles.csv] -- results.load_run works on either engine's
directory.

Requirements:
  - numpy, scipy, pandas
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

from sedfit.core.nnls import nnls_fit as _nnls_fit
from sedfit.backends.eazy.fitting import warn_grid_edges, write_catalog
from sedfit.backends.eazy.results import (
    FitResult,
    percentiles_from_lnp,
    save_outputs,
)
from sedfit.backends.eazy.templates import prepare_templates_param, resolve_tef
from sedfit.core.policy import MISSING_SENTINEL, PolicyResult
from sedfit.core.synth import make_spherex_tophat, pivot_wave_AA

if TYPE_CHECKING:
    from sedfit.core.registry import Registry

# eazy-py constants reproduced here (photoz.py / utils.py, 0.8.6).
CLIGHT_AA = 2.99792458e18   # speed of light [Angstrom/s]
CLIP_WAVELENGTH = 1100.0    # compute_lnp: P(z)=0 past the reddest band
LNP_FLOOR = -1e20           # compute_lnp sentinel for non-finite ln P(z)
IGM_REST_LIMIT = 1300.0     # rest wavelength where eazy's IGM departs from 1

# Quadrature node spacing target, lambda/dlambda: curves are refined to at
# least this resolution so the trapezoid weights sample template structure
# between sparse native curve nodes.
QUAD_RESOLUTION = 2000.0


# ------------------------------------
# eazy conventions: grid, TEF, quadrature
# ------------------------------------

def zgrid_from_config(eazy_cfg: dict) -> np.ndarray:
    """The redshift grid the official engine would realize (float64).

    Linear grids are arange(Z_MIN, Z_MAX + Z_STEP/2, Z_STEP) (the
    half-step pad compensates arange's exclusive endpoint); log grids are
    eazy's utils.log_zgrid.
    """
    if eazy_cfg["z_step_type"] == "linear":
        return np.arange(eazy_cfg["z_min"],
                         eazy_cfg["z_max"] + eazy_cfg["z_step"] / 2.0,
                         eazy_cfg["z_step"], dtype=float)
    return np.exp(np.arange(np.log(1.0 + eazy_cfg["z_min"]),
                            np.log(1.0 + eazy_cfg["z_max"]),
                            eazy_cfg["z_step"])) - 1.0


def _trapz_dx(x: np.ndarray) -> np.ndarray:
    """Composite-trapezoid weights: dx @ y == trapezoid(y, x)."""
    dx = np.zeros_like(x)
    diff = np.diff(x) / 2.0
    dx[:-1] += diff
    dx[1:] += diff
    return dx


class QuickTEF:
    """eazy's TemplateError: spline at rest wavelength, endpoint-clipped.

    tef(z) returns the fractional template error per band: the curve
    spline at pivot/(1+z), scaled; outside the curve's nonzero range the
    first/last nonzero values are used instead of extrapolating.
    """

    def __init__(self, curve_file: str | Path, pivot: np.ndarray,
                 scale: float):
        curve = np.loadtxt(curve_file)
        self.te_x = curve[:, 0].astype(float)
        self.te_y = curve[:, 1].astype(float)
        self.pivot = np.asarray(pivot, float)
        self.scale = float(scale)
        nonzero = self.te_y > 0
        self.min_wavelength = self.te_x[nonzero].min()
        self.max_wavelength = self.te_x[nonzero].max()
        self.clip_lo = self.te_y[nonzero][0]
        self.clip_hi = self.te_y[nonzero][-1]
        self._spline = CubicSpline(self.te_x, self.te_y)

    def __call__(self, z: float) -> np.ndarray:
        rest = self.pivot / (1.0 + z)
        tef = self._spline(rest)
        tef[rest < self.min_wavelength] = self.clip_lo
        tef[rest > self.max_wavelength] = self.clip_hi
        return tef * self.scale


# ------------------------------------
# Projection channels and templates
# ------------------------------------

@dataclass
class Channels:
    """Flattened photon-counting projection channels for all bands.

    wave concatenates every band's filter-curve nodes (observed
    Angstrom); weight holds the matching trapezoid quadrature weights
    T/lambda * dlambda, so a segment's weighted sum over a template's
    f_nu is the photon-counting bandpass average times wsum.
    """
    wave: np.ndarray      # (M,) all bands' nodes, concatenated
    weight: np.ndarray    # (M,) T/lambda * trapezoid dlambda
    offsets: np.ndarray   # (NFILT,) segment starts for np.add.reduceat
    wsum: np.ndarray      # (NFILT,) per-band weight totals
    pivot: np.ndarray     # (NFILT,) pivot wavelengths [Angstrom]
    blue_min: float       # bluest nonzero-throughput node [Angstrom]


def _refine_curve(wave: np.ndarray, thru: np.ndarray, *,
                  resolution: float = QUAD_RESOLUTION
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Subdivide curve segments to at least lambda/resolution spacing.

    A transmission curve is piecewise linear, so added nodes leave the
    filter unchanged; they only give the quadrature sample points where
    the template spectrum has structure the native sampling would skip.

    Parameters
    ----------
    wave : np.ndarray
        Curve wavelength nodes [Angstrom].
    thru : np.ndarray
        Throughput at those nodes.
    resolution : float
        Target lambda/dlambda of the refined nodes.
        [default: QUAD_RESOLUTION]

    Returns
    -------
    refined : np.ndarray
        The subdivided wavelength nodes.
    thru_refined : np.ndarray
        Throughput linearly interpolated onto refined.
    """
    pieces = [wave[:1]]
    for i in range(len(wave) - 1):
        n_sub = int(np.ceil((wave[i + 1] - wave[i]) / (wave[i] / resolution)))
        if n_sub > 1:
            pieces.append(np.linspace(wave[i], wave[i + 1], n_sub + 1)[1:])
        else:
            pieces.append(wave[i + 1:i + 2])
    refined = np.concatenate(pieces)
    return refined, np.interp(refined, wave, thru)


def build_channels(frame: pd.DataFrame, registry: Registry) -> Channels:
    """Resolve every table band to curve nodes and quadrature weights.

    Broadband curves come from the registry; SPHEREx channels are the
    same padded tophats the official engine writes into FILTER.RES.
    """
    waves, weights, wsums, pivots = [], [], [], []
    blue_min = np.inf
    for row in frame.itertuples(index=False):
        band = str(row.band)
        if registry.is_spherex(band):
            wave, thru = make_spherex_tophat(float(row.wave_um),
                                             float(row.bandwidth_um))
        else:
            wave, thru = registry.get_bandpass(band)
        wave, thru = _refine_curve(np.asarray(wave, float),
                                   np.asarray(thru, float))
        weight = thru / wave * _trapz_dx(wave)
        waves.append(wave)
        weights.append(weight)
        wsums.append(weight.sum())
        pivots.append(pivot_wave_AA(wave, thru))
        blue_min = min(blue_min, float(wave[thru > 0].min()))
    lengths = [len(w) for w in waves]
    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(int)
    return Channels(wave=np.concatenate(waves),
                    weight=np.concatenate(weights),
                    offsets=offsets, wsum=np.asarray(wsums),
                    pivot=np.asarray(pivots), blue_min=blue_min)


def load_quick_templates(eazy_cfg: dict, run_dir) -> list[tuple]:
    """The template set as (name, wave_rest_AA, fnu_rest) triples.

    Resolution goes through templates.prepare_templates_param (identical
    set and run-dir provenance to an official run); each listed spectrum
    is two-column ASCII wavelength [Angstrom] / f_lambda. Fluxes convert
    to f_nu with eazy's constant (flam * wave**2 / CLIGHT_AA) so fitted
    amplitudes match the official engine's coefficient scale.
    """
    param_path = prepare_templates_param(eazy_cfg, run_dir)
    templates = []
    for line in Path(param_path).read_text(encoding="utf-8").splitlines():
        tokens = line.split()
        if not tokens or tokens[0].startswith("#"):
            continue
        if len(tokens) < 2:
            raise ValueError(f"{param_path}: unreadable templates row "
                             f"{line!r}")
        spec_path = Path(tokens[1])
        if not spec_path.is_absolute():
            spec_path = Path(param_path).parent / spec_path
        wave_scale = float(tokens[2]) if len(tokens) > 2 else 1.0
        try:
            data = np.loadtxt(spec_path, comments="#", usecols=(0, 1))
        except Exception as err:
            raise ValueError(
                f"the quick engine reads two-column ASCII spectra only; "
                f"{spec_path} failed ({err}); use the official engine for "
                f"FITS/CSV template tables") from None
        wave = data[:, 0] * wave_scale
        flam = data[:, 1]
        order = np.argsort(wave)
        wave, flam = wave[order], flam[order]
        keep = np.concatenate(([True], np.diff(wave) > 0))
        wave, flam = wave[keep], flam[keep]
        # eazy names templates by file basename, extension included.
        templates.append((spec_path.name, wave, flam * wave**2 / CLIGHT_AA))
    return templates


def design_matrix(z: float, templates: list[tuple],
                  ch: Channels) -> np.ndarray:
    """Synthetic photometry of every template at one redshift.

    Each template's rest-frame f_nu is interpolated onto the flattened
    channel nodes at wave/(1+z) (zero outside its coverage) and reduced
    to per-band photon-counting averages. No (1+z) flux factor: the fit
    amplitude absorbs it, exactly as in eazy.
    """
    rest = ch.wave / (1.0 + z)
    rows = np.empty((len(templates), len(ch.wsum)))
    for it, (_, wave, fnu) in enumerate(templates):
        y = np.interp(rest, wave, fnu, left=0.0, right=0.0)
        rows[it] = np.add.reduceat(y * ch.weight, ch.offsets) / ch.wsum
    return rows


def design_cube(zgrid: np.ndarray, templates: list[tuple],
                ch: Channels) -> np.ndarray:
    """The full template grid, (NZ, NTEMP, NFILT) -- eazy's tempfilt."""
    cube = np.empty((len(zgrid), len(templates), len(ch.wsum)))
    for iz, z in enumerate(zgrid):
        cube[iz] = design_matrix(z, templates, ch)
    return cube


# ------------------------------------
# Fitting cores (eazy photoz.py equivalents)
# ------------------------------------

def _fit_object_scan(cube: np.ndarray, tefgrid: np.ndarray, fnu: np.ndarray,
                     efnu: np.ndarray, ok_band: np.ndarray, *,
                     want_coeffs: bool = False
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """chi2(z), tef_lnp(z), and optionally coeffs(z) for one object.

    Returns
    -------
    chi2 : np.ndarray
        (NZ,) NNLS chi2 at each grid redshift.
    tef_lnp : np.ndarray
        (NZ,) -0.5 * log-determinant of the redshift-dependent variance,
        summed over the fitted bands.
    coeffs : np.ndarray or None
        (NZ, NTEMP) template amplitudes; None unless want_coeffs.
    """
    nz = cube.shape[0]
    chi2 = np.empty(nz)
    tef_lnp = np.empty(nz)
    coeffs = np.empty((nz, cube.shape[1])) if want_coeffs else None
    fpos = np.maximum(fnu, 0.0)
    for iz in range(nz):
        var = efnu**2 + (tefgrid[iz] * fpos)**2
        chi2[iz], coeffs_iz, _ = _nnls_fit(cube[iz], fnu, var, ok_band)
        tef_lnp[iz] = -0.5 * float(np.log(var[ok_band]).sum())
        if want_coeffs:
            coeffs[iz] = coeffs_iz
    return chi2, tef_lnp, coeffs


def _fit_singles_scan(cube: np.ndarray, tefgrid: np.ndarray, fnu: np.ndarray,
                      efnu: np.ndarray, ok_band: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Per-template analytic scan, mirroring fit_single_templates.

    Amplitudes are unconstrained (results applies the physical > 0 rule
    when reporting) and the flux enters the TEF variance unclamped, as
    in the official routine; the combo path clamps it.

    Returns
    -------
    chi2 : np.ndarray
        (NTEMP, NZ) chi2 at each template's best amplitude.
    ampl : np.ndarray
        (NTEMP, NZ) the amplitudes themselves.
    """
    var = efnu[None, :]**2 + (tefgrid * fnu[None, :])**2        # (NZ, NFILT)
    ivar = np.where(ok_band[None, :], 1.0 / var, 0.0)
    num = np.einsum("zf,ztf->zt", ivar * fnu[None, :], cube)
    den = np.einsum("zf,ztf->zt", ivar, cube**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        ampl = np.where(den > 0, num / den, 0.0)
    base = (ivar * fnu[None, :]**2).sum(axis=1)
    # chi2(a) = base - 2*a*num + a^2*den is minimized at a = num/den,
    # where it equals base - a*num.
    chi2 = np.where(den > 0, base[:, None] - ampl * num, base[:, None])
    return chi2.T, ampl.T


def _lnp_from_scan(chi2_fit, tef_lnp, zgrid, lc_reddest, *, use_tef_lnp):
    """ln P(z) per object, mirroring compute_lnp (no priors)."""
    loglike = -chi2_fit / 2.0
    if use_tef_lnp:
        loglike = loglike + tef_lnp
    clip = (CLIP_WAVELENGTH * (1.0 + zgrid))[None, :] > lc_reddest[:, None]
    loglike[clip] = -np.inf
    loglike[~np.isfinite(loglike)] = LNP_FLOOR
    lnpmax = loglike.max(axis=1)
    pz = np.exp(loglike - lnpmax[:, None])
    log_norm = np.log(pz @ _trapz_dx(zgrid))
    lnp = loglike - lnpmax[:, None] - log_norm[:, None]
    lnp[~np.isfinite(lnp)] = LNP_FLOOR
    return lnp


def _zml_parabola(zgrid: np.ndarray, lnp: np.ndarray) -> np.ndarray:
    """Maximum-likelihood redshifts, mirroring get_maxlnp_redshift.

    A parabola through the three grid points around the ln P(z) peak; a
    peak on the first grid point returns the -1 failure sentinel and a
    peak on the last returns the grid value, exactly as in eazy. The
    vertex is accepted for any finite non-zero quadratic coefficient,
    including a parabola minimum and a vertex outside the three points,
    with no bracket check -- again as in eazy.
    """
    z_ml = np.empty(lnp.shape[0])
    nz = len(zgrid)
    for i, row in enumerate(lnp):
        iz = int(np.argmax(row))
        if iz == 0:
            z_ml[i] = -1.0
            continue
        if iz >= nz - 1:
            z_ml[i] = zgrid[iz]
            continue
        x, y = zgrid[iz - 1:iz + 2], row[iz - 1:iz + 2]
        dx, dx2, dy = np.diff(x), np.diff(x**2), np.diff(y)
        c2 = ((dy[1] / dx[1] - dy[0] / dx[0])
              / (dx2[1] / dx[1] - dx2[0] / dx[0]))
        refined = np.nan
        if np.isfinite(c2) and c2 != 0:
            c1 = (dy[0] - c2 * dx2[0]) / dx[0]
            refined = -c1 / (2.0 * c2)
        z_ml[i] = refined if np.isfinite(refined) else zgrid[iz]
    return z_ml


def _sed_payload(z: float, chi2: float, coeffs: np.ndarray,
                 fmodel: np.ndarray, fnu: np.ndarray, efnu: np.ndarray,
                 tempflux: np.ndarray, wave0: np.ndarray) -> dict:
    """The per-object SED block, matching results.extract_sed's keys."""
    return {
        "templz": wave0 * (1.0 + z),
        "templf": coeffs @ tempflux,
        "model": fmodel.copy(),
        "fobs": fnu.copy(),
        "efobs": efnu.copy(),
        "z": float(z),
        "chi2": float(chi2),
    }


# ------------------------------------
# Run execution
# ------------------------------------

def _validate_quick(eazy_cfg: dict) -> None:
    """Reject configurations the quick engine cannot honor."""
    if eazy_cfg["prior"]:
        raise ValueError("the quick engine does not support priors; run "
                         "the official engine")
    if eazy_cfg["fitter"] != "nnls":
        raise ValueError(f"the quick engine implements fitter='nnls' only, "
                         f"got {eazy_cfg['fitter']!r}; run the official "
                         f"engine")
    if eazy_cfg["extra_params"]:
        print("WARNING: extra_params are eazy-py overrides; the quick "
              f"engine ignores them: {sorted(eazy_cfg['extra_params'])}")


def run_quick(
        cfg: dict,
        frame: pd.DataFrame,
        policy: PolicyResult,
        run_dir: str | Path,
        *,
        registry: Registry,
        phot_sha256: str | None = None,
    ) -> FitResult:
    """Run the vectorized quick fit for one policy-applied table.

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
        Photometry byte hash, carried into the result. [default: None]

    Returns
    -------
    result : FitResult
        The same container the official engine returns, with photz None.
    """
    eazy_cfg = cfg["eazy"]
    _validate_quick(eazy_cfg)
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    write_catalog(cfg, policy, run_dir / "catalog.csv")
    tef_dst = run_dir / "template_error.dat"
    shutil.copyfile(resolve_tef(eazy_cfg), tef_dst)
    (run_dir / "config.json").write_text(
        json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / "engine.info").write_text(
        "engine: quick (vectorized eazy-faithful reimplementation; "
        "no eazy-py)\n", encoding="utf-8")

    bands = list(policy.bands)
    channels = build_channels(frame, registry)
    templates = load_quick_templates(eazy_cfg, run_dir)
    template_names = [name for name, _, _ in templates]
    zgrid = zgrid_from_config(eazy_cfg)

    z_fixed = eazy_cfg["z_fixed"]
    if z_fixed is not None and not (zgrid[0] < z_fixed < zgrid[-1]):
        raise ValueError(f"z_fixed={z_fixed} is not strictly inside the "
                         f"realized grid ({zgrid[0]:.4f}, {zgrid[-1]:.4f})")
    if zgrid[-1] > channels.blue_min / IGM_REST_LIMIT - 1.0:
        print(f"WARNING: the grid reaches z={zgrid[-1]:.2f}, where the "
              f"bluest band samples rest wavelengths below "
              f"{IGM_REST_LIMIT:.0f} A; the quick engine applies no IGM "
              f"absorption -- use the official engine")

    tef_scale = eazy_cfg["tef_scale"] if eazy_cfg["tef"] else 0.0
    tef = QuickTEF(tef_dst, channels.pivot, scale=tef_scale or 0.0)
    tefgrid = np.array([tef(z) for z in zgrid])                # (NZ, NFILT)
    cube = design_cube(zgrid, templates, channels)             # (NZ, NTEMP, NFILT)

    # Single object; masked rows carry the sentinel like an official run.
    ok = np.asarray(policy.fittable, bool)
    fnu = np.where(ok, policy.flux_uJy, MISSING_SENTINEL)[None, :]
    efnu = np.where(ok, policy.flux_err_uJy, MISSING_SENTINEL)[None, :]
    ok_data = ok[None, :]
    nusefilt = ok_data.sum(axis=1)
    lc_reddest = (ok_data * channels.pivot[None, :]).max(axis=1)
    ids = [cfg["name"]]
    nobj, nfilt, ntemp, nz = 1, len(bands), len(templates), len(zgrid)

    chi2_fit = np.empty((nobj, nz))
    tef_lnp = np.empty((nobj, nz))
    want_coeffs = eazy_cfg["save_zcoeffs"]
    chi2_fit[0], tef_lnp[0], coeffs_z = _fit_object_scan(
        cube, tefgrid, fnu[0], efnu[0], ok_data[0],
        want_coeffs=want_coeffs)

    lnp = _lnp_from_scan(chi2_fit, tef_lnp, zgrid, lc_reddest,
                         use_tef_lnp=bool(eazy_cfg["tef_lnp"]))
    z_ml = _zml_parabola(zgrid, lnp)
    z_chi2 = zgrid[np.argmin(chi2_fit, axis=1)]
    warn_grid_edges(cfg["name"], zgrid, z_ml, z_chi2)
    z_percentiles = percentiles_from_lnp(zgrid, lnp)

    # Every template resampled onto the first one's grid for SED
    # rendering (eazy show_fit does the same).
    wave0 = templates[0][1]
    tempflux = np.array([np.interp(wave0, wave, flux)
                         for _, wave, flux in templates])

    def evaluate_at(z: float):
        var = efnu[0]**2 + (tef(z) * np.maximum(fnu[0], 0.0))**2
        chi2, coeffs, fmodel = _nnls_fit(
            design_matrix(z, templates, channels), fnu[0], var, ok_data[0])
        sed = _sed_payload(z, chi2, coeffs, fmodel, fnu[0], efnu[0],
                           tempflux, wave0)
        return chi2, coeffs, fmodel, sed

    chi2_best = np.zeros(nobj)
    coeffs_best = np.zeros((nobj, ntemp))
    fmodel = np.zeros((nobj, nfilt))
    seds: list = [None]
    # eazy only refits redshifts strictly inside the grid; edge and
    # failed solutions keep zeroed best-fit products (and no SED).
    if zgrid[0] < z_ml[0] < zgrid[-1]:
        chi2_best[0], coeffs_best[0], fmodel[0], sed = evaluate_at(z_ml[0])
        seds = [sed]

    result = FitResult(
        config=cfg,
        run_dir=run_dir,
        ids=ids,
        bands=bands,
        template_names=template_names,
        pivot=channels.pivot,
        zgrid=zgrid,
        fnu=fnu,
        efnu=efnu,
        ok_data=ok_data,
        nusefilt=nusefilt,
        chi2_fit=chi2_fit,
        lnp=lnp,
        z_ml=z_ml,
        z_chi2=z_chi2,
        z_percentiles=z_percentiles,
        chi2_best=chi2_best,
        coeffs_best=coeffs_best,
        fmodel=fmodel,
        seds=seds,
        phot_sha256=phot_sha256,
        photz=None,
    )

    if eazy_cfg["mode"] == "single":
        singles_chi2 = np.empty((ntemp, nobj, nz))
        singles_ampl = np.empty((ntemp, nobj, nz))
        singles_chi2[:, 0, :], singles_ampl[:, 0, :] = _fit_singles_scan(
            cube, tefgrid, fnu[0], efnu[0], ok_data[0])
        result.singles_chi2 = singles_chi2
        result.singles_ampl = singles_ampl

    if want_coeffs:
        result.fit_coeffs = coeffs_z[None, :, :]

    if z_fixed is not None:
        result.z_fixed = float(z_fixed)
        chi2_z, coeffs_zf, fmodel_z, sed = evaluate_at(float(z_fixed))
        result.chi2_fixed = np.array([chi2_z])
        result.coeffs_fixed = coeffs_zf[None, :]
        result.fmodel_fixed = fmodel_z[None, :]
        result.seds_fixed = [sed]

    save_outputs(result)
    return result
