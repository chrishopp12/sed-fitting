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
    centers a normal zred prior on the cluster redshift when mean is
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
BACKENDS = ("eazy", "prospector")
Z_STEP_TYPES = ("linear", "log")
EAZY_ENGINES = ("quick", "eazy-py")
EAZY_MODES = ("combo", "single")
EAZY_FITTERS = ("nnls", "bvls", "lstsq")
SFH_FAMILIES = ("delayed-tau", "tau", "continuity")
ZRED_PRIORS = ("uniform", "normal")
SAMPLERS = ("dynesty", "emcee")
STELLAR_LIBRARIES = ("miles", "c3k")

TOP_KEYS = ("schema_version", "backend", "name", "bands_include",
            "min_valid_bands", "min_snr_broadband", "err_floor",
            "mu_lensing", "z_ref", "qa_gates", "eazy", "prospector")
TOP_REQUIRED = ("schema_version", "backend", "name")

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
                   "tage_range", "tage_init", "tage_tuniv_init")

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
}
ZRED_DEFAULTS = {"prior": "uniform", "mean": None, "sigma": None,
                 "bounds": [0.0, 1.0]}

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
    return cfg


# ------------------------------------
# Loading
# ------------------------------------

def parse_fit_config(raw: dict, *, label: str = "fit config") -> dict:
    """Validate a fit-config dict; returns it with defaults materialized."""
    require_keys(label, raw, allowed=TOP_KEYS, required=TOP_REQUIRED)
    if raw["schema_version"] not in SCHEMA_VERSIONS:
        raise ValueError(f"{label}: unrecognized schema_version "
                         f"{raw['schema_version']!r}")
    backend = require_enum(f"{label}.backend", raw["backend"],
                           allowed=BACKENDS)
    other = "prospector" if backend == "eazy" else "eazy"
    if other in raw:
        raise ValueError(f"{label}: a {other!r} block is illegal under "
                         f"backend {backend!r}")
    if not isinstance(raw["name"], str) or not raw["name"]:
        raise ValueError(f"{label}.name must be a non-empty string")

    cfg = _materialize(label, {k: v for k, v in raw.items()
                               if k not in BACKENDS}, BASE_DEFAULTS)
    _require_number(f"{label}.mu_lensing", cfg["mu_lensing"], positive=True)
    _require_number(f"{label}.min_snr_broadband", cfg["min_snr_broadband"],
                    nonnegative=True)
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
    else:
        if "prospector" not in raw:
            raise ValueError(f"{label}: backend 'prospector' requires a "
                             f"prospector block (stellar_library is "
                             f"required)")
        cfg["prospector"] = _parse_prospector(block_label, raw["prospector"])
    return cfg


def load_fit_config(path: str | Path) -> dict:
    """Load and validate a fit-config JSON file."""
    path = Path(path)
    return parse_fit_config(json.loads(path.read_text(encoding="utf-8")), label=str(path))


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

EXECUTION_ONLY_TOP = ("name",)
# templates/template_pattern/tef_file locate the curves; content_digests
# hashes their contents, keyed by basename, so identical curves under any
# path share one identity.
EXECUTION_ONLY_EAZY = ("n_proc", "templates", "template_pattern", "tef_file")


def hash_projection(resolved: dict, *, digests: dict | None = None) -> dict:
    """The resolved config reduced to its scientific identity.

    Strips execution-only fields and merges content digests (template
    and TEF file hashes) under 'digests'. provenance.run_id consumes the
    result; the projection is frozen by the golden run_id test.
    """
    projected = json.loads(json.dumps(resolved))
    for key in EXECUTION_ONLY_TOP:
        projected.pop(key, None)
    if "eazy" in projected:
        for key in EXECUTION_ONLY_EAZY:
            projected["eazy"].pop(key, None)
    if digests:
        projected["digests"] = digests
    return projected
