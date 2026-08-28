"""
fitconfig.py

Fit Configurations
---------------------------------------------------------

Strict discriminated loading of fit configs, resolution against the
roster, and the run-identity hash projection. One JSON per fit: shared
fields at top level, exactly one backend block matching the backend
discriminant. Fields conditional on a discriminant must be null or
absent when their discriminant does not select them.

Requirements:
  - (stdlib only)

Notes:
  - Resolution materializes every default, derives z_ref from the
    roster target (an explicit value that disagrees is a hard error),
    centers a normal zred prior on the reference redshift when mean is
    null, and draws a concrete seed when seed is null.
  - hash_projection strips execution-only fields (name, n_proc) and
    merges content digests; provenance.run_id consumes its output.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from pathlib import Path

from sedfit.core.policy import validate_qa_gates
from sedfit.core.validate import require_enum, require_keys

# ------------------------------------
# Vocabularies
# ------------------------------------

SCHEMA_VERSIONS = (2,)
BACKENDS = ("eazy", "prospector", "linear")
# Whether a backend fits a built photometry table. A False here means the
# job never reads one, so the photometry-only top-level fields below are
# rejected rather than materialized into the run identity as defaults.
# test_every_backend_declares_whether_it_uses_photometry pins the entry.
USES_PHOTOMETRY = {"eazy": True, "prospector": True, "linear": False}
Z_STEP_TYPES = ("linear", "log")
EAZY_ENGINES = ("quick", "eazy-py")
EAZY_MODES = ("combo", "single")
EAZY_FITTERS = ("nnls", "bvls", "lstsq")
SFH_FAMILIES = ("delayed-tau", "tau", "continuity")
ZRED_PRIORS = ("uniform", "normal")
SAMPLERS = ("dynesty", "emcee")
STELLAR_LIBRARIES = ("miles", "c3k")
POLY_WAVE_FRAMES = ("vacuum", "air")
# "hessian" names what backends/linear actually does -- it inverts a
# finite-difference curvature, so its errors are symmetric by construction.
# "delta_chi2" is deliberately absent until something profiles chi2 for real.
LINEAR_ERROR_METHODS = ("hessian",)
RESOLUTION_UNITS = ("R", "fwhm_A", "sigma_A", "fwhm_kms", "sigma_kms")
# No default: all three are defensible and the wrong one is silent. Raising
# unconditionally refuses roughly half of a MUSE band against either shipped
# basis, and clamping to zero fits data with templates coarser than it.
LSF_UNDERSAMPLED = ("raise", "degrade_data", "ignore")

# The backend names are appended rather than listed, so adding a backend is
# one edit to BACKENDS rather than three scattered literals.
TOP_KEYS_BASE = ("schema_version", "backend", "name", "bands_include",
                 "min_valid_bands", "min_snr_broadband", "err_floor",
                 "mu_lensing", "z_ref", "qa_gates")
TOP_KEYS = TOP_KEYS_BASE + BACKENDS
TOP_REQUIRED = ("schema_version", "backend", "name")
# Shared fields that only mean something against a photometry table. Under a
# backend with USES_PHOTOMETRY False they are refused when written and forced
# to null otherwise, so a spectrum-only run's identity does not carry, say,
# min_valid_bands = 5. mu_lensing is NOT here: it divides a spectrum's flux
# exactly as it divides a band's.
PHOTOMETRY_ONLY_TOP = ("bands_include", "min_valid_bands",
                       "min_snr_broadband", "err_floor", "qa_gates")

EAZY_KEYS = ("engine", "mode", "z_min", "z_max", "z_step", "z_step_type",
             "z_fixed", "templates", "template_pattern", "tef", "tef_file",
             "tef_scale", "tef_lnp", "prior", "prior_file", "prior_filter",
             "fitter", "n_proc", "extra_params", "save_zcoeffs")

ZRED_KEYS = ("prior", "mean", "sigma", "bounds")
SP_PRIOR_TYPES = ("clipped_normal", "uniform")
PROSPECTOR_KEYS = ("stellar_library", "fit_redshift", "zred", "sfh",
                   "nebular", "gas_logu_free", "gas_logu_prior",
                   "gas_logu_init", "agn", "dust_emission",
                   "free_norm_instruments", "free_norm_prior",
                   "exact_filters", "sampler", "dynesty", "emcee", "seed",
                   "n_agebins", "tie_tage_to_tuniv", "mass_range",
                   "mass_init", "logzsol_prior", "logzsol_prior_type",
                   "logzsol_init", "dust2_prior", "dust2_prior_type",
                   "dust2_init", "tau_range", "tau_init",
                   "tage_range", "tage_init", "tage_tuniv_init",
                   "spectrum")
SPECTRUM_KEYS = ("file", "polyorder", "smooth_sigma_prior",
                 "smooth_sigma_init", "smooth_sigma_fixed",
                 "jitter_prior", "jitter_init", "outlier_prior",
                 "outlier_nsigma", "err_floor", "mask_windows",
                 "eline_sigma_kms")

LINEAR_KEYS = ("templates", "template_pattern", "template_wave_range",
               "template_dv_kms", "normalize_range",
               "z_min", "z_max", "z_step",
               "sigma_min", "sigma_max", "sigma_step",
               "poly_order", "poly_domain", "poly_wave_frame",
               "n_poly_iter", "clip_sigma", "error_method",
               "transmission", "gas", "lsf", "template_resolution",
               "scan", "spectrum")
# Required rather than defaulted, every one of them, because a wrong value is
# SILENT: the fit runs and returns a number that is merely wrong.
#   templates            the basis is the dominant systematic (as for eazy)
#   template_wave_range  arms the coverage assertion; a default disarms it
#   z_min / z_max        no campaign default exists, unlike the eazy block
#   poly_wave_frame      air vs vacuum moves the amplitudes in the 5th decimal
#   spectrum             a linear fit with no spectrum is nothing
LINEAR_REQUIRED = ("templates", "template_wave_range", "z_min", "z_max",
                   "poly_wave_frame", "spectrum")
# The linear block's own spectrum: no sampler, so none of the prospector
# block's prior/jitter/outlier fields apply. mask_windows carries the same
# meaning it does there -- analysis choices, where the file mask records data
# quality -- and is where the masked measurement window goes, the one that
# makes the in-window template a prediction rather than a fit.
LINEAR_SPECTRUM_KEYS = ("file", "err_floor", "mask_windows")
# T(lambda) in Cheb(l) * T(l) * sum_k a_k B_k(l) -- any fixed multiplicative
# correction the model needs and the templates do not carry: telluric
# transmission, aperture loss, instrument response. Null is a flat 1.
# wave_frame is required for the same reason poly_wave_frame is: a telluric
# curve prepared against air data and read as vacuum is wrong by 2.5 A at
# 9155 A, and nothing about the file says which it is.
LINEAR_TRANSMISSION_KEYS = ("file", "wave_frame")
# Emission lines as extra NNLS columns. `lines` has no default for the reason
# `templates` has none: a short list is SILENT -- an unlisted strong line is
# fit by nothing, contributes a large residual, and is then removed by the
# sigma clip, all without a word.
# One shape for both resolution curves: exactly one of a constant or a file,
# the unit it is written in, and -- for a file -- the frame it was prepared
# in, for the reason poly_wave_frame is required.
LINEAR_RESOLUTION_KEYS = ("constant", "file", "unit", "wave_frame")
LINEAR_LSF_KEYS = LINEAR_RESOLUTION_KEYS + ("on_undersampled",)
# The two-stage blind scan. Present means run one; null means fit the
# configured grid whole. z_step_coarse null is derived from the basis rather
# than fixed, because gas columns move the narrowest feature in it by an
# order of magnitude.
LINEAR_SCAN_KEYS = ("z_step_coarse", "n_poly_iter_coarse", "sigma_coarse",
                    "window_steps", "minima_dv_kms")
LINEAR_GAS_KEYS = ("lines", "sigma_kms", "ratio_locked")
LINEAR_GAS_GROUP_KEYS = ("name", "lines", "ratios")
# Ratios fixed by atomic physics -- each pair shares an upper level, so the
# ratio is set by transition probabilities alone and is independent of
# temperature, density, abundance and dust. The pair enters as ONE column and
# NNLS cannot buy chi-square with an unphysical member. Balmer lines are
# deliberately absent: the decrement is a dust measurement, not a constant.
#
# The source is recorded PER GROUP, not once for the table: these four ratios
# do not come from one place, and a single label would assert a provenance
# that is false for half of them. It lives beside the table rather than inside
# a group because ratio_locked resolves into run_id.
RATIO_SOURCES = {
    "[OIII]": "Storey & Zeippen (2000)",
    "[NII]": "Storey & Zeippen (2000)",
    "[OI]": "Osterbrock & Ferland (2006)",
    "[SIII]": "Osterbrock & Ferland (2006)",
}
LINEAR_GAS_RATIO_LOCKED = (
    {"name": "[OIII]", "lines": ("[OIII]4959", "[OIII]5007"),
     "ratios": (1.0, 2.98)},
    {"name": "[NII]", "lines": ("[NII]6548", "[NII]6584"),
     "ratios": (1.0, 2.94)},
    {"name": "[OI]", "lines": ("[OI]6300", "[OI]6364"),
     "ratios": (3.0, 1.0)},
    {"name": "[SIII]", "lines": ("[SIII]9069", "[SIII]9531"),
     "ratios": (1.0, 2.47)},
)

BASE_DEFAULTS = {
    "bands_include": None,
    "min_valid_bands": 5,
    "min_snr_broadband": 2.0,
    "err_floor": 0.05,
    "mu_lensing": 1.0,
    "z_ref": None,
    "qa_gates": None,
}

EAZY_DEFAULTS = {
    "engine": "quick",
    "mode": "combo",
    "z_min": 0.05, "z_max": 0.16, "z_step": 0.001, "z_step_type": "linear",
    "z_fixed": None,
    "templates": "", "template_pattern": "*_spec.dat",
    "tef": True, "tef_file": None, "tef_scale": 1.0, "tef_lnp": True,
    "prior": False, "prior_file": None, "prior_filter": None,
    "fitter": "nnls",
    "n_proc": 4,
    "extra_params": {},
    "save_zcoeffs": False,
}

PROSPECTOR_DEFAULTS = {
    "fit_redshift": True,
    "zred": None,
    "sfh": "continuity",
    "nebular": False,
    "gas_logu_free": None, "gas_logu_prior": None, "gas_logu_init": None,
    "agn": False,
    "dust_emission": False,
    "free_norm_instruments": [],
    "free_norm_prior": [0.1, 10.0],
    "exact_filters": True,
    "sampler": "dynesty",
    "dynesty": {}, "emcee": None,
    "seed": None,
    "n_agebins": None,
    "tie_tage_to_tuniv": None,
    "mass_range": [1.0e10, 1.0e13], "mass_init": 3.0e11,
    "logzsol_prior": [0.0, 0.3, -1.0, 0.5], "logzsol_prior_type":
        "clipped_normal", "logzsol_init": 0.0,
    "dust2_prior": [0.15, 0.2, 0.0, 1.0], "dust2_prior_type":
        "clipped_normal", "dust2_init": 0.15,
    "tau_range": None, "tau_init": None,
    "tage_range": None, "tage_init": None, "tage_tuniv_init": None,
    "spectrum": None,
}
LINEAR_DEFAULTS = {
    "template_pattern": "*.dat",
    # ln-lambda sampling of the template grid; must oversample the narrowest
    # broadening kernel the sigma grid can reach.
    "template_dv_kms": 15.0,
    # null normalizes over the whole template_wave_range. The choice cancels
    # out of chi2 and the model exactly -- nnls_fit renormalizes internally
    # and divides back out -- so it sets only whether the amplitudes read as
    # light fractions over the band you care about. Prefer an INTERIOR band:
    # the default reaches the outermost grid pixels, where each template was
    # interpolated from the last file rows available, so the light fractions
    # pick up a little edge behavior. The reference insets by 150 A at the
    # blue end and 50 A at the red.
    "normalize_range": None,
    "z_step": 4.0e-4,
    "sigma_min": 50.0, "sigma_max": 400.0, "sigma_step": 50.0,
    "poly_order": 8,
    # null spans the fitted spectrum's own extent.
    "poly_domain": None,
    "n_poly_iter": 4,
    "clip_sigma": 4.0,
    "error_method": "hessian",
    "transmission": None,
    "gas": None,
    "scan": None,
    "lsf": None,
    "template_resolution": None,
}
LINEAR_RESOLUTION_DEFAULTS = {"constant": None, "file": None,
                              "wave_frame": None}
LINEAR_SCAN_DEFAULTS = {"z_step_coarse": None, "n_poly_iter_coarse": 1,
                        "sigma_coarse": None, "window_steps": 10,
                        "minima_dv_kms": 1000.0}
LINEAR_SPECTRUM_DEFAULTS = {"err_floor": 0.0, "mask_windows": None}
# ratio_locked resolves to the concrete groups rather than staying null, so
# the identity records which ratios were locked instead of deferring to
# whatever the package shipped that day.
LINEAR_GAS_DEFAULTS = {"sigma_kms": 100.0, "ratio_locked": None}

ZRED_DEFAULTS = {"prior": "uniform", "mean": None, "sigma": None,
                 "bounds": [0.0, 1.0]}
SPECTRUM_DEFAULTS = {
    "polyorder": 12,
    "smooth_sigma_prior": [10.0, 400.0], "smooth_sigma_init": 100.0,
    "smooth_sigma_fixed": None,
    "jitter_prior": None, "jitter_init": None,
    "outlier_prior": None, "outlier_nsigma": None,
    "err_floor": 0.0,
    "mask_windows": None,
    "eline_sigma_kms": None,
}

CONTINUITY_ONLY = ("n_agebins",)
PARAMETRIC_ONLY = ("tau_range", "tau_init", "tage_range", "tage_init",
                   "tage_tuniv_init", "tie_tage_to_tuniv")
PARAMETRIC_DEFAULTS = {
    "tau_range": [0.1, 30.0], "tau_init": 1.0,
    "tage_range": [0.5, 13.5], "tage_init": 10.0, "tage_tuniv_init": 0.9,
    "tie_tage_to_tuniv": True,
}
GAS_LOGU_DEFAULTS = {"gas_logu_free": False,
                     "gas_logu_prior": [-4.0, -1.0], "gas_logu_init": -2.0}

SEED_BITS = 31


# ------------------------------------
# Helpers
# ------------------------------------

def _materialize(label: str, raw: dict, defaults: dict) -> dict:
    out = dict(defaults)
    out.update(raw)
    return out


def _block_at(projected: dict, path: tuple[str, ...]) -> dict | None:
    """The nested block at `path`, or None when it is absent or null."""
    node: object = projected
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node if isinstance(node, dict) else None


def _require_null(label: str, cfg: dict, fields: tuple[str, ...],
                  because: str) -> None:
    filled = [f for f in fields if cfg.get(f) not in (None, {}, [])]
    if filled:
        raise ValueError(f"{label}: {filled} must be null when {because}")


def _require_number(label: str, value: object, *, positive: bool = False,
                    nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label}: expected a number, got {value!r}")
    if positive and value <= 0:
        raise ValueError(f"{label}: must be positive, got {value}")
    if nonnegative and value < 0:
        raise ValueError(f"{label}: must be non-negative, got {value}")
    return float(value)


def _require_interval(label: str, value: object, *,
                      positive: bool = False) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{label}: expected [lo, hi], got {value!r}")
    lo = _require_number(f"{label}[0]", value[0], positive=positive)
    hi = _require_number(f"{label}[1]", value[1], positive=positive)
    if not lo < hi:
        raise ValueError(f"{label}: lo {lo} must be below hi {hi}")
    return lo, hi


# ------------------------------------
# Backend blocks
# ------------------------------------

def _parse_eazy(label: str, raw: dict) -> dict:
    require_keys(label, raw, allowed=EAZY_KEYS, required=())
    cfg = _materialize(label, raw, EAZY_DEFAULTS)
    require_enum(f"{label}.engine", cfg["engine"], allowed=EAZY_ENGINES)
    require_enum(f"{label}.mode", cfg["mode"], allowed=EAZY_MODES)
    require_enum(f"{label}.z_step_type", cfg["z_step_type"],
                 allowed=Z_STEP_TYPES)
    require_enum(f"{label}.fitter", cfg["fitter"], allowed=EAZY_FITTERS)
    # The template set is the dominant systematic in a fit, so every
    # config must name its own basis.
    if not str(cfg["templates"]).strip():
        raise ValueError(f"{label}.templates is required: name the template "
                         f"directory (or an eazy .param file) explicitly")

    z_min = _require_number(f"{label}.z_min", cfg["z_min"], nonnegative=True)
    z_max = _require_number(f"{label}.z_max", cfg["z_max"], positive=True)
    _require_number(f"{label}.z_step", cfg["z_step"], positive=True)
    if z_min >= z_max:
        raise ValueError(f"{label}: z_min {z_min} >= z_max {z_max}")
    if cfg["z_fixed"] is not None:
        z_fixed = _require_number(f"{label}.z_fixed", cfg["z_fixed"])
        if not z_min < z_fixed < z_max:
            raise ValueError(f"{label}: z_fixed {z_fixed} outside the grid "
                             f"({z_min}, {z_max})")

    if not cfg["tef"]:
        _require_null(label, cfg, ("tef_file",), "tef is off")
        if "tef_scale" in raw or "tef_lnp" in raw:
            raise ValueError(f"{label}: tef_scale/tef_lnp are meaningless "
                             f"with tef off")
        cfg["tef_scale"] = None
        cfg["tef_lnp"] = None
    if not cfg["prior"]:
        _require_null(label, cfg, ("prior_file", "prior_filter"),
                      "prior is off")
    elif cfg["prior_file"] is None or cfg["prior_filter"] is None:
        raise ValueError(f"{label}: prior requires prior_file and "
                         f"prior_filter")

    if not isinstance(cfg["extra_params"], dict):
        raise ValueError(f"{label}.extra_params must be an object")

    if cfg["engine"] == "quick":
        remedy = "engine 'eazy-py' is the remedy"
        if cfg["prior"]:
            raise ValueError(f"{label}: the quick engine has no priors; "
                             f"{remedy}")
        if cfg["fitter"] != "nnls":
            raise ValueError(f"{label}: the quick engine is NNLS-only; "
                             f"{remedy}")
        if cfg["extra_params"]:
            raise ValueError(f"{label}: the quick engine takes no "
                             f"extra_params; {remedy}")
    return cfg


def _parse_zred(label: str, raw: dict) -> dict:
    require_keys(label, raw, allowed=ZRED_KEYS, required=())
    cfg = _materialize(label, raw, ZRED_DEFAULTS)
    require_enum(f"{label}.prior", cfg["prior"], allowed=ZRED_PRIORS)
    bounds = cfg["bounds"]
    if (not isinstance(bounds, list) or len(bounds) != 2
            or not bounds[0] < bounds[1]):
        raise ValueError(f"{label}.bounds must be [lo, hi] with lo < hi")
    if cfg["prior"] == "uniform":
        _require_null(label, cfg, ("mean", "sigma"),
                      "the zred prior is uniform")
    else:
        _require_number(f"{label}.sigma", cfg["sigma"], positive=True)
        if cfg["mean"] is not None:
            _require_number(f"{label}.mean", cfg["mean"])
    return cfg


def _parse_spectrum(label: str, raw: dict) -> dict:
    require_keys(label, raw, allowed=SPECTRUM_KEYS, required=("file",))
    cfg = _materialize(label, raw, SPECTRUM_DEFAULTS)
    if not isinstance(cfg["file"], str) or not cfg["file"].strip():
        raise ValueError(f"{label}.file must be a non-empty path")

    order = cfg["polyorder"]
    if isinstance(order, bool) or not isinstance(order, int) or order < 0:
        raise ValueError(f"{label}.polyorder must be an integer >= 0, "
                         f"got {order!r}")

    if cfg["smooth_sigma_fixed"] is not None:
        if "smooth_sigma_prior" in raw or "smooth_sigma_init" in raw:
            raise ValueError(f"{label}: smooth_sigma_prior/smooth_sigma_init "
                             f"are meaningless with smooth_sigma_fixed")
        _require_number(f"{label}.smooth_sigma_fixed",
                        cfg["smooth_sigma_fixed"], positive=True)
        cfg["smooth_sigma_prior"] = None
        cfg["smooth_sigma_init"] = None
    else:
        lo, hi = _require_interval(f"{label}.smooth_sigma_prior",
                                   cfg["smooth_sigma_prior"], positive=True)
        init = _require_number(f"{label}.smooth_sigma_init",
                               cfg["smooth_sigma_init"], positive=True)
        if not lo <= init <= hi:
            raise ValueError(f"{label}.smooth_sigma_init {init} outside the "
                             f"prior [{lo}, {hi}]")

    if cfg["jitter_prior"] is None:
        _require_null(label, cfg, ("jitter_init",), "jitter_prior is null")
    else:
        lo, hi = _require_interval(f"{label}.jitter_prior",
                                   cfg["jitter_prior"], positive=True)
        if cfg["jitter_init"] is None:
            cfg["jitter_init"] = 1.0 if lo <= 1.0 <= hi else 0.5 * (lo + hi)
        init = _require_number(f"{label}.jitter_init", cfg["jitter_init"],
                               positive=True)
        if not lo <= init <= hi:
            raise ValueError(f"{label}.jitter_init {init} outside the "
                             f"prior [{lo}, {hi}]")

    if cfg["outlier_prior"] is None:
        _require_null(label, cfg, ("outlier_nsigma",),
                      "outlier_prior is null")
    else:
        lo, hi = _require_interval(f"{label}.outlier_prior",
                                   cfg["outlier_prior"])
        if not (0.0 <= lo < hi <= 1.0):
            raise ValueError(f"{label}.outlier_prior must lie inside [0, 1], "
                             f"got [{lo}, {hi}]")
        if cfg["outlier_nsigma"] is None:
            cfg["outlier_nsigma"] = 50.0
        nsigma = _require_number(f"{label}.outlier_nsigma",
                                 cfg["outlier_nsigma"], positive=True)
        if nsigma <= 1:
            raise ValueError(f"{label}.outlier_nsigma must exceed 1, "
                             f"got {nsigma}")

    _require_number(f"{label}.err_floor", cfg["err_floor"], nonnegative=True)

    if cfg["eline_sigma_kms"] is not None:
        _require_number(f"{label}.eline_sigma_kms", cfg["eline_sigma_kms"],
                        positive=True)

    if cfg["mask_windows"] is not None:
        windows = cfg["mask_windows"]
        if not isinstance(windows, list) or not windows:
            raise ValueError(f"{label}.mask_windows must be null or a "
                             f"non-empty list of [lo, hi]")
        for i, window in enumerate(windows):
            _require_interval(f"{label}.mask_windows[{i}]", window)
    return cfg


def _parse_linear_spectrum(label: str, raw: dict) -> dict:
    require_keys(label, raw, allowed=LINEAR_SPECTRUM_KEYS, required=("file",))
    cfg = _materialize(label, raw, LINEAR_SPECTRUM_DEFAULTS)
    if not isinstance(cfg["file"], str) or not cfg["file"].strip():
        raise ValueError(f"{label}.file must be a non-empty path")
    _require_number(f"{label}.err_floor", cfg["err_floor"], nonnegative=True)
    if cfg["mask_windows"] is not None:
        windows = cfg["mask_windows"]
        if not isinstance(windows, list) or not windows:
            raise ValueError(f"{label}.mask_windows must be null or a "
                             f"non-empty list of [lo, hi]")
        for i, window in enumerate(windows):
            _require_interval(f"{label}.mask_windows[{i}]", window)
    return cfg


def _parse_linear_transmission(label: str, raw: dict) -> dict:
    require_keys(label, raw, allowed=LINEAR_TRANSMISSION_KEYS,
                 required=LINEAR_TRANSMISSION_KEYS)
    cfg = _materialize(label, raw, {})
    if not isinstance(cfg["file"], str) or not cfg["file"].strip():
        raise ValueError(f"{label}.file must be a non-empty path")
    require_enum(f"{label}.wave_frame", cfg["wave_frame"],
                 allowed=POLY_WAVE_FRAMES)
    return cfg


def _parse_linear_scan(label: str, raw: dict) -> dict:
    require_keys(label, raw, allowed=LINEAR_SCAN_KEYS, required=())
    cfg = _materialize(label, raw, LINEAR_SCAN_DEFAULTS)
    for key in ("z_step_coarse", "sigma_coarse"):
        if cfg[key] is not None:
            _require_number(f"{label}.{key}", cfg[key], positive=True)
    for key in ("n_poly_iter_coarse", "window_steps"):
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{label}.{key} must be an integer >= 1, "
                             f"got {value!r}")
    _require_number(f"{label}.minima_dv_kms", cfg["minima_dv_kms"],
                    positive=True)
    return cfg


def _parse_resolution(label: str, raw: dict, *, is_lsf: bool) -> dict:
    allowed = LINEAR_LSF_KEYS if is_lsf else LINEAR_RESOLUTION_KEYS
    required = ("unit", "on_undersampled") if is_lsf else ("unit",)
    require_keys(label, raw, allowed=allowed, required=required)
    cfg = _materialize(label, raw, LINEAR_RESOLUTION_DEFAULTS)
    require_enum(f"{label}.unit", cfg["unit"], allowed=RESOLUTION_UNITS)
    if is_lsf:
        require_enum(f"{label}.on_undersampled", cfg["on_undersampled"],
                     allowed=LSF_UNDERSAMPLED)

    if (cfg["constant"] is None) == (cfg["file"] is None):
        raise ValueError(f"{label} takes exactly one of constant or file")
    if cfg["constant"] is not None:
        _require_number(f"{label}.constant", cfg["constant"], positive=True)
        if cfg["wave_frame"] is not None:
            raise ValueError(f"{label}.wave_frame means nothing without a "
                             f"file")
    else:
        if not isinstance(cfg["file"], str) or not cfg["file"].strip():
            raise ValueError(f"{label}.file must be a non-empty path")
        if cfg["wave_frame"] is None:
            raise ValueError(f"{label}.wave_frame is required with a file: "
                             f"nothing about the file says which frame it "
                             f"was prepared in")
        require_enum(f"{label}.wave_frame", cfg["wave_frame"],
                     allowed=POLY_WAVE_FRAMES)
    return cfg


def _parse_linear_gas(label: str, raw: dict) -> dict:
    require_keys(label, raw, allowed=LINEAR_GAS_KEYS, required=("lines",))
    cfg = _materialize(label, raw, LINEAR_GAS_DEFAULTS)
    if not isinstance(cfg["lines"], str) or not cfg["lines"].strip():
        raise ValueError(f"{label}.lines must be a packaged list name or a "
                         f"path")
    _require_number(f"{label}.sigma_kms", cfg["sigma_kms"], positive=True)

    if cfg["ratio_locked"] is None:
        cfg["ratio_locked"] = [
            {"name": group["name"], "lines": list(group["lines"]),
             "ratios": list(group["ratios"])}
            for group in LINEAR_GAS_RATIO_LOCKED]
    if not isinstance(cfg["ratio_locked"], list):
        raise ValueError(f"{label}.ratio_locked must be null or a list of "
                         f"groups")
    for i, group in enumerate(cfg["ratio_locked"]):
        where = f"{label}.ratio_locked[{i}]"
        if not isinstance(group, dict):
            raise ValueError(f"{where} must be an object")
        require_keys(where, group, allowed=LINEAR_GAS_GROUP_KEYS,
                     required=LINEAR_GAS_GROUP_KEYS)
        if not isinstance(group["name"], str) or not group["name"].strip():
            raise ValueError(f"{where}.name must be a non-empty string")
        members, ratios = group["lines"], group["ratios"]
        if not isinstance(members, list) or len(members) < 2:
            raise ValueError(f"{where}.lines must list at least two lines")
        if not isinstance(ratios, list) or len(ratios) != len(members):
            raise ValueError(f"{where}.ratios must give one value per line")
        for j, ratio in enumerate(ratios):
            _require_number(f"{where}.ratios[{j}]", ratio, positive=True)
    return cfg


def _parse_linear(label: str, raw: dict) -> dict:
    require_keys(label, raw, allowed=LINEAR_KEYS, required=LINEAR_REQUIRED)
    cfg = _materialize(label, raw, LINEAR_DEFAULTS)
    if not str(cfg["templates"]).strip():
        raise ValueError(f"{label}.templates is required: name the template "
                         f"directory explicitly")
    if not str(cfg["template_pattern"]).strip():
        raise ValueError(f"{label}.template_pattern must be a non-empty glob")

    lo, hi = _require_interval(f"{label}.template_wave_range",
                               cfg["template_wave_range"], positive=True)
    _require_number(f"{label}.template_dv_kms", cfg["template_dv_kms"],
                    positive=True)
    if cfg["normalize_range"] is not None:
        n_lo, n_hi = _require_interval(f"{label}.normalize_range",
                                       cfg["normalize_range"], positive=True)
        # TemplateBasis would raise on this too, but only once a fit is
        # running and a basis has been loaded off disk.
        if not (lo <= n_lo and n_hi <= hi):
            raise ValueError(f"{label}.normalize_range [{n_lo}, {n_hi}] must "
                             f"lie inside template_wave_range [{lo}, {hi}]")

    z_min = _require_number(f"{label}.z_min", cfg["z_min"], nonnegative=True)
    z_max = _require_number(f"{label}.z_max", cfg["z_max"], positive=True)
    _require_number(f"{label}.z_step", cfg["z_step"], positive=True)
    if z_min >= z_max:
        raise ValueError(f"{label}: z_min {z_min} >= z_max {z_max}")

    s_min = _require_number(f"{label}.sigma_min", cfg["sigma_min"],
                            positive=True)
    s_max = _require_number(f"{label}.sigma_max", cfg["sigma_max"],
                            positive=True)
    _require_number(f"{label}.sigma_step", cfg["sigma_step"], positive=True)
    if s_min >= s_max:
        raise ValueError(f"{label}: sigma_min {s_min} >= sigma_max {s_max}")

    order = cfg["poly_order"]
    if isinstance(order, bool) or not isinstance(order, int) or order < 0:
        raise ValueError(f"{label}.poly_order must be an integer >= 0, "
                         f"got {order!r}")
    if cfg["poly_domain"] is not None:
        _require_interval(f"{label}.poly_domain", cfg["poly_domain"],
                          positive=True)
    require_enum(f"{label}.poly_wave_frame", cfg["poly_wave_frame"],
                 allowed=POLY_WAVE_FRAMES)

    n_iter = cfg["n_poly_iter"]
    if isinstance(n_iter, bool) or not isinstance(n_iter, int) or n_iter < 1:
        raise ValueError(f"{label}.n_poly_iter must be an integer >= 1, "
                         f"got {n_iter!r}")
    if cfg["clip_sigma"] is not None:
        _require_number(f"{label}.clip_sigma", cfg["clip_sigma"],
                        positive=True)
    require_enum(f"{label}.error_method", cfg["error_method"],
                 allowed=LINEAR_ERROR_METHODS)

    if cfg["transmission"] is not None:
        if not isinstance(cfg["transmission"], dict):
            raise ValueError(f"{label}.transmission must be null or an "
                             f"object")
        cfg["transmission"] = _parse_linear_transmission(
            f"{label}.transmission", cfg["transmission"])

    for key, is_lsf in (("lsf", True), ("template_resolution", False)):
        if cfg[key] is None:
            continue
        if not isinstance(cfg[key], dict):
            raise ValueError(f"{label}.{key} must be null or an object")
        cfg[key] = _parse_resolution(f"{label}.{key}", cfg[key],
                                     is_lsf=is_lsf)
    # The kernel is a difference and half of it is a property of the basis;
    # a wrong or missing library resolution is silent.
    if cfg["lsf"] is not None and cfg["template_resolution"] is None:
        raise ValueError(f"{label}.template_resolution is required whenever "
                         f"lsf is set")

    if cfg["scan"] is not None:
        if not isinstance(cfg["scan"], dict):
            raise ValueError(f"{label}.scan must be null or an object")
        cfg["scan"] = _parse_linear_scan(f"{label}.scan", cfg["scan"])

    if cfg["gas"] is not None:
        if not isinstance(cfg["gas"], dict):
            raise ValueError(f"{label}.gas must be null or an object")
        cfg["gas"] = _parse_linear_gas(f"{label}.gas", cfg["gas"])

    if not isinstance(cfg["spectrum"], dict):
        raise ValueError(f"{label}.spectrum must be an object")
    cfg["spectrum"] = _parse_linear_spectrum(f"{label}.spectrum",
                                             cfg["spectrum"])
    return cfg


def _parse_prospector(label: str, raw: dict) -> dict:
    require_keys(label, raw, allowed=PROSPECTOR_KEYS,
                 required=("stellar_library",))
    cfg = _materialize(label, raw, PROSPECTOR_DEFAULTS)
    require_enum(f"{label}.stellar_library", cfg["stellar_library"],
                 allowed=STELLAR_LIBRARIES)
    require_enum(f"{label}.sfh", cfg["sfh"], allowed=SFH_FAMILIES)
    require_enum(f"{label}.sampler", cfg["sampler"], allowed=SAMPLERS)

    if cfg["fit_redshift"]:
        if cfg["zred"] is None:
            raise ValueError(f"{label}: fit_redshift requires a zred block")
        cfg["zred"] = _parse_zred(f"{label}.zred", cfg["zred"])
    else:
        _require_null(label, cfg, ("zred",), "fit_redshift is off")

    if cfg["sfh"] == "continuity":
        _require_null(label, cfg, PARAMETRIC_ONLY,
                      "the SFH family is continuity")
        cfg["n_agebins"] = (7 if cfg["n_agebins"] is None
                            else int(_require_number(
                                f"{label}.n_agebins", cfg["n_agebins"],
                                positive=True)))
    else:
        _require_null(label, cfg, CONTINUITY_ONLY,
                      "the SFH family is parametric")
        for key, default in PARAMETRIC_DEFAULTS.items():
            if cfg[key] is None:
                cfg[key] = default

    if cfg["nebular"]:
        for key, default in GAS_LOGU_DEFAULTS.items():
            if cfg[key] is None:
                cfg[key] = default
    else:
        _require_null(label, cfg,
                      ("gas_logu_free", "gas_logu_prior", "gas_logu_init"),
                      "nebular is off")

    if cfg["sampler"] == "dynesty":
        _require_null(label, cfg, ("emcee",), "the sampler is dynesty")
        cfg["dynesty"] = cfg["dynesty"] or {}
        if not isinstance(cfg["dynesty"], dict):
            raise ValueError(f"{label}.dynesty must be an object")
    else:
        _require_null(label, cfg, ("dynesty",), "the sampler is emcee")
        cfg["emcee"] = cfg["emcee"] or {}
        if not isinstance(cfg["emcee"], dict):
            raise ValueError(f"{label}.emcee must be an object")

    for name in ("dust2", "logzsol"):
        ptype = require_enum(f"{label}.{name}_prior_type",
                             cfg[f"{name}_prior_type"],
                             allowed=SP_PRIOR_TYPES)
        spec = cfg[f"{name}_prior"]
        expected = 2 if ptype == "uniform" else 4
        if not (isinstance(spec, list) and len(spec) == expected):
            raise ValueError(
                f"{label}.{name}_prior: a {ptype} prior takes "
                f"{expected} values "
                f"({'[lo, hi]' if expected == 2 else '[mean, sigma, lo, hi]'}), "
                f"got {spec!r}")

    if not isinstance(cfg["free_norm_instruments"], list):
        raise ValueError(f"{label}.free_norm_instruments must be a list")
    if cfg["seed"] is not None:
        cfg["seed"] = int(_require_number(f"{label}.seed", cfg["seed"],
                                          nonnegative=True))
    if cfg["spectrum"] is not None:
        if not isinstance(cfg["spectrum"], dict):
            raise ValueError(f"{label}.spectrum must be null or an object")
        cfg["spectrum"] = _parse_spectrum(f"{label}.spectrum",
                                          cfg["spectrum"])
        if (cfg["spectrum"]["eline_sigma_kms"] is not None
                and not cfg["nebular"]):
            raise ValueError(f"{label}.spectrum.eline_sigma_kms is "
                             f"meaningless with nebular off")
    return cfg


# ------------------------------------
# Loading
# ------------------------------------

def parse_fit_config(raw: dict, *, label: str = "fit config") -> dict:
    """Validate a fit-config dict; returns it with defaults materialized."""
    require_keys(label, raw, allowed=TOP_KEYS_BASE + BACKENDS,
                 required=TOP_REQUIRED)
    if raw["schema_version"] not in SCHEMA_VERSIONS:
        raise ValueError(f"{label}: unrecognized schema_version "
                         f"{raw['schema_version']!r}")
    backend = require_enum(f"{label}.backend", raw["backend"],
                           allowed=BACKENDS)
    strangers = [name for name in BACKENDS
                 if name != backend and name in raw]
    if strangers:
        blocks = ", ".join(repr(name) for name in strangers)
        raise ValueError(f"{label}: a {blocks} block is illegal under "
                         f"backend {backend!r}")
    if not isinstance(raw["name"], str) or not raw["name"]:
        raise ValueError(f"{label}.name must be a non-empty string")

    cfg = _materialize(label, {k: v for k, v in raw.items()
                               if k not in BACKENDS}, BASE_DEFAULTS)
    if backend not in USES_PHOTOMETRY:
        raise ValueError(f"{label}: backend {backend!r} is listed in BACKENDS "
                         f"but does not declare USES_PHOTOMETRY")
    if not USES_PHOTOMETRY[backend]:
        written = [k for k in PHOTOMETRY_ONLY_TOP if k in raw]
        if written:
            raise ValueError(f"{label}: {written} describe a photometry "
                             f"table, which backend {backend!r} never reads")
        # Null rather than their photometry defaults, so they drop out of
        # canonical JSON and leave the run identity alone.
        for key in PHOTOMETRY_ONLY_TOP:
            cfg[key] = None
    _require_number(f"{label}.mu_lensing", cfg["mu_lensing"], positive=True)
    if cfg["min_snr_broadband"] is not None:
        _require_number(f"{label}.min_snr_broadband",
                        cfg["min_snr_broadband"], nonnegative=True)
    if cfg["z_ref"] is not None:
        _require_number(f"{label}.z_ref", cfg["z_ref"])
    if cfg["bands_include"] is not None and (
            not isinstance(cfg["bands_include"], list)
            or not cfg["bands_include"]):
        raise ValueError(f"{label}.bands_include must be null or a non-empty "
                         f"list")
    if cfg["qa_gates"] is not None:
        try:
            validate_qa_gates(cfg["qa_gates"])
        except ValueError as err:
            raise ValueError(f"{label}.{err}") from None

    block_label = f"{label}.{backend}"
    if backend == "eazy":
        cfg["eazy"] = _parse_eazy(block_label, raw.get("eazy", {}))
    elif backend == "prospector":
        if "prospector" not in raw:
            raise ValueError(f"{label}: backend 'prospector' requires a "
                             f"prospector block (stellar_library is "
                             f"required)")
        cfg["prospector"] = _parse_prospector(block_label, raw["prospector"])
    elif backend == "linear":
        if "linear" not in raw:
            raise ValueError(f"{label}: backend 'linear' requires a linear "
                             f"block ({list(LINEAR_REQUIRED)} are required)")
        cfg["linear"] = _parse_linear(block_label, raw["linear"])
    else:
        raise ValueError(f"{label}: backend {backend!r} is listed in BACKENDS "
                         f"but has no parser")
    return cfg


# Path-valued fields that resolve against the config file's own directory,
# keyed like EXECUTION_ONLY by the path from the config root, so configs can
# carry no machine-specific paths.
RELATIVE_TO_CONFIG = {
    ("prospector", "spectrum"): ("file",),
    ("linear", "spectrum"): ("file",),
    ("linear", "transmission"): ("file",),
    ("linear", "lsf"): ("file",),
}


def load_fit_config(path: str | Path) -> dict:
    """Load and validate a fit-config JSON file.

    A relative path in a RELATIVE_TO_CONFIG field resolves against the
    config file's own directory, so configs carry no machine-specific
    paths.
    """
    path = Path(path)
    cfg = parse_fit_config(json.loads(path.read_text(encoding="utf-8")),
                           label=str(path))
    for block_path, keys in RELATIVE_TO_CONFIG.items():
        block = _block_at(cfg, block_path)
        if block is None:
            continue
        for key in keys:
            value = block.get(key)
            if isinstance(value, str) and not Path(value).is_absolute():
                block[key] = str((path.parent / value).resolve())
    return cfg


# ------------------------------------
# Resolution
# ------------------------------------

def resolve_config(
        cfg: dict,
        *,
        target_z_ref: float,
        reference_redshift: float | None = None,
        seed_source: Callable[[], int] | None = None,
    ) -> dict:
    """Materialize the roster-derived fields; returns the resolved config.

    Parameters
    ----------
    cfg : dict
        Output of parse_fit_config.
    target_z_ref : float
        The roster target's resolved z_ref.
    reference_redshift : float or None
        The roster's sample reference redshift, which centers a null
        normal-prior mean. [default: None]
    seed_source : callable or None
        Zero-argument seed factory (tests inject one); None draws from
        SystemRandom. [default: None]

    Returns
    -------
    resolved : dict
        The config with every roster-derived field materialized.
    """
    resolved = json.loads(json.dumps(cfg))
    if resolved["z_ref"] is None:
        resolved["z_ref"] = target_z_ref
    elif resolved["z_ref"] != target_z_ref:
        raise ValueError(f"z_ref {resolved['z_ref']} disagrees with the "
                         f"roster target's {target_z_ref}; change the roster "
                         f"or leave z_ref null")

    prospector = resolved.get("prospector")
    if prospector is not None:
        zred = prospector.get("zred")
        if zred is not None and zred["prior"] == "normal" \
                and zred["mean"] is None:
            if reference_redshift is None:
                raise ValueError(
                    "a null normal zred mean is centered on the roster's "
                    "reference_redshift, which this roster does not declare; "
                    "set the mean explicitly")
            zred["mean"] = reference_redshift
        if prospector["seed"] is None:
            draw = seed_source or (lambda:
                                   random.SystemRandom().getrandbits(SEED_BITS))
            prospector["seed"] = int(draw())
    return resolved


# ------------------------------------
# Hash projection
# ------------------------------------

# Execution-only fields, keyed by the path from the config root to the block
# that carries them; () is the root. A field belongs here when it locates an
# input rather than describing one, and the input's CONTENT is hashed into
# `digests` instead -- identical inputs under any path then share one identity,
# and moving the repo does not fork every run_id.
#
# A table rather than a branch per backend, because a branch cannot notice the
# backend it fails to mention: a third backend's paths would silently enter the
# identity hash. test_every_backend_declares_its_execution_only_fields pins the
# entry, so adding a name to BACKENDS forces an entry here.
EXECUTION_ONLY = {
    (): ("name",),
    # templates/template_pattern/tef_file locate the curves; content_digests
    # hashes their contents, keyed by basename.
    ("eazy",): ("n_proc", "templates", "template_pattern", "tef_file"),
    # spectrum.file locates the spectrum; jobs hashes its content.
    ("prospector", "spectrum"): ("file",),
    # templates/template_pattern select the basis; its content is hashed.
    ("linear",): ("templates", "template_pattern"),
    ("linear", "spectrum"): ("file",),
    ("linear", "transmission"): ("file",),
    # gas.lines is a packaged name or a path; its content is hashed.
    ("linear", "gas"): ("lines",),
    ("linear", "lsf"): ("file",),
    ("linear", "template_resolution"): ("file",),
}


def hash_projection(resolved: dict, *, digests: dict | None = None) -> dict:
    """The resolved config reduced to its scientific identity.

    Strips the EXECUTION_ONLY fields and merges content digests (template,
    TEF, and spectrum file hashes) under 'digests'. provenance.run_id
    consumes the result; the projection is frozen by the golden run_id
    test.
    """
    projected = json.loads(json.dumps(resolved))
    for path, keys in EXECUTION_ONLY.items():
        block = _block_at(projected, path)
        if block is None:
            continue
        for key in keys:
            block.pop(key, None)
    if digests:
        projected["digests"] = digests
    return projected
