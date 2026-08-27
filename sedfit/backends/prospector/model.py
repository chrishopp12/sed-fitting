"""
model.py

Prospector Model and SPS Construction
---------------------------------------------------------

Assembles a SpecModel from the resolved config's prospector block, for
either SFH family: parametric (delayed-tau or tau; CSPSpecBasis) with
age fit directly or tied to t_universe(zred), or continuity
(non-parametric step SFH; FastStepBasis) with age bins tied to zred.
The two samplers differ only in the parametric mass/tau
parameterization: dynesty fits mass/tau with LogUniform priors; emcee
fits logmass/logtau with uniform priors (mass/tau derived via
depends_on) so walkers move in a uniform space.

Requirements:
  - numpy, astropy, prospect (lazy), fsps (via build_sps)

Notes:
  - prospect propagates depends_on in dict order; agebins (from zred)
    must precede mass (from agebins) or mass uses stale bins. The
    continuity builder reorders explicitly.
  - The stellar library is a compile-time FSPS choice invisible in
    Python source; assert_stellar_library checks the config's declared
    library against the live SPS build at fit start.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from astropy.cosmology import WMAP9
from prospect.models import SpecModel
from prospect.models.sedmodel import PolySpecModel

if TYPE_CHECKING:
    from prospect.sources import CSPSpecBasis, FastStepBasis

    from sedfit.core.registry import Registry

# ------------------------------------
# Constants
# ------------------------------------

ZRED_INIT_DISP = 0.01
# Config vocabulary -> the live FSPS library stamp (sps.ssp.libraries[1]).
LIBRARY_STAMPS = {"miles": "miles", "c3k": "c3k_a"}


# ------------------------------------
# emcee reparameterization transforms
# ------------------------------------

def logmass_to_mass(logmass: float | np.ndarray = 0,
                    **extras) -> float | np.ndarray:
    """Stellar mass from its log10, for the emcee mass parameterization."""
    return 10 ** logmass


def logtau_to_tau(logtau: float | np.ndarray = 0,
                  **extras) -> float | np.ndarray:
    """SFH tau from its log10, for the emcee tau parameterization."""
    return 10 ** logtau


# ------------------------------------
# Shared model-building helpers
# ------------------------------------

def _set_zred(model_params, cfg: dict) -> None:
    from prospect.models import priors
    from prospect.models.priors import ClippedNormal

    p = cfg["prospector"]
    if not p["fit_redshift"]:
        model_params["zred"]["isfree"] = False
        model_params["zred"]["init"] = cfg["z_ref"]
        return
    zred = p["zred"]
    lo, hi = zred["bounds"]
    model_params["zred"]["isfree"] = True
    model_params["zred"]["init"] = cfg["z_ref"]
    if zred["prior"] == "normal":
        model_params["zred"]["prior"] = ClippedNormal(
            mean=zred["mean"], sigma=zred["sigma"], mini=lo, maxi=hi)
    else:
        model_params["zred"]["prior"] = priors.TopHat(mini=lo, maxi=hi)
    # prospect's default initial_disp (0.1) is absolute and far too wide
    # for a low-z ball.
    model_params["zred"]["init_disp"] = ZRED_INIT_DISP
    model_params["zred"]["disp_floor"] = ZRED_INIT_DISP


def _set_nebular(model_params, p: dict) -> None:
    from prospect.models import priors, transforms
    from prospect.models.templates import TemplateLibrary

    model_params.update(TemplateLibrary["nebular"])
    model_params["gas_logz"]["isfree"] = False
    model_params["gas_logz"]["init"] = -1.0
    model_params["gas_logz"]["depends_on"] = transforms.stellar_logzsol
    if p["gas_logu_free"]:
        lo, hi = p["gas_logu_prior"]
        model_params["gas_logu"]["isfree"] = True
        model_params["gas_logu"]["init"] = p["gas_logu_init"]
        model_params["gas_logu"]["prior"] = priors.TopHat(mini=lo, maxi=hi)
    else:
        model_params["gas_logu"]["isfree"] = False
        model_params["gas_logu"]["init"] = -3.0
        model_params["gas_logu"]["prior"] = priors.TopHat(mini=-3.5,
                                                          maxi=-2.0)


def _set_dust_emission(model_params) -> None:
    from prospect.models import priors
    from prospect.models.templates import TemplateLibrary

    model_params.update(TemplateLibrary["dust_emission"])
    model_params["duste_qpah"]["isfree"] = True
    model_params["duste_qpah"]["init"] = 2.0
    model_params["duste_qpah"]["prior"] = priors.TopHat(mini=0.0, maxi=7.0)


def _set_agn(model_params) -> None:
    from prospect.models import priors
    from prospect.models.templates import TemplateLibrary

    model_params.update(TemplateLibrary["agn"])
    model_params["fagn"]["isfree"] = True
    model_params["fagn"]["init"] = 0.05
    model_params["fagn"]["prior"] = priors.LogUniform(mini=1e-4, maxi=3.0)
    model_params["agn_tau"]["isfree"] = True
    model_params["agn_tau"]["init"] = 20.0
    model_params["agn_tau"]["prior"] = priors.LogUniform(mini=5.0, maxi=150.0)
    model_params["add_agn_dust"]["init"] = True


def _set_spectrum(model_params, spec_cfg: dict) -> None:
    """Model parameters for a joint photometry+spectrum fit.

    Smoothing is always declared explicitly: prospect's smoothspec
    silently applies sigma_smooth = 100 km/s when the parameter is
    absent, so absence is never allowed to choose the physics. The
    calibration polynomial order is a fixed parameter PolySpecModel
    reads; jitter scales the spectral uncertainty; the outlier mixture
    de-weights channels the model cannot reach.
    """
    from prospect.models import priors

    model_params["smoothtype"] = {"N": 1, "isfree": False, "init": "vel"}
    model_params["fftsmooth"] = {"N": 1, "isfree": False, "init": True}
    if spec_cfg["smooth_sigma_fixed"] is not None:
        model_params["sigma_smooth"] = {
            "N": 1, "isfree": False, "units": "km/s",
            "init": spec_cfg["smooth_sigma_fixed"],
        }
    else:
        lo, hi = spec_cfg["smooth_sigma_prior"]
        model_params["sigma_smooth"] = {
            "N": 1, "isfree": True, "units": "km/s",
            "init": spec_cfg["smooth_sigma_init"],
            "prior": priors.TopHat(mini=lo, maxi=hi),
        }

    if spec_cfg["polyorder"] > 0:
        model_params["polyorder"] = {"N": 1, "isfree": False,
                                     "init": spec_cfg["polyorder"]}

    if spec_cfg["jitter_prior"] is not None:
        lo, hi = spec_cfg["jitter_prior"]
        model_params["spec_jitter"] = {
            "N": 1, "isfree": True, "init": spec_cfg["jitter_init"],
            "prior": priors.TopHat(mini=lo, maxi=hi),
        }

    if spec_cfg["outlier_prior"] is not None:
        lo, hi = spec_cfg["outlier_prior"]
        model_params["f_outlier_spec"] = {
            "N": 1, "isfree": True, "init": 0.5 * (lo + hi),
            "prior": priors.TopHat(mini=lo, maxi=hi),
        }
        model_params["nsigma_outlier_spec"] = {
            "N": 1, "isfree": False, "init": spec_cfg["outlier_nsigma"]}

    if spec_cfg.get("eline_sigma_kms") is not None:
        # FSPS's in-spectrum nebular lines carry the library-resolution
        # convention width, not the observed one; drawing them in
        # prospect at an explicitly configured total width (intrinsic
        # (+) instrumental, measured upstream) puts the line profiles
        # at the data's resolution. Photometry still receives the line
        # fluxes through nebline_photometry, so band predictions are
        # unchanged in content.
        model_params["nebemlineinspec"] = {"N": 1, "isfree": False,
                                           "init": False}
        model_params["eline_sigma"] = {
            "N": 1, "isfree": False, "units": "km/s",
            "init": spec_cfg["eline_sigma_kms"]}


def _set_sp_prior(model_params, name: str, prior_type: str, prior_spec,
                  init) -> None:
    from prospect.models import priors
    from prospect.models.priors import ClippedNormal

    model_params[name]["isfree"] = True
    model_params[name]["init"] = init
    if prior_type == "uniform":
        lo, hi = prior_spec
        model_params[name]["prior"] = priors.TopHat(mini=lo, maxi=hi)
    else:
        mean, sigma, lo, hi = prior_spec
        model_params[name]["prior"] = ClippedNormal(mean=mean, sigma=sigma,
                                                    mini=lo, maxi=hi)


# ------------------------------------
# SFH families
# ------------------------------------

def _build_parametric_params(cfg: dict) -> dict:
    from prospect.models import priors, transforms
    from prospect.models.templates import TemplateLibrary

    p = cfg["prospector"]
    mp = TemplateLibrary["parametric_sfh"]
    if p["sfh"] == "tau":
        mp["sfh"]["init"] = 1       # pure exponential (4 = delayed-tau)
    if p["nebular"]:
        _set_nebular(mp, p)

    _set_zred(mp, cfg)
    mp["mass_units"] = {"init": "mstar", "isfree": False, "N": 1}
    _set_sp_prior(mp, "dust2", p["dust2_prior_type"], p["dust2_prior"],
                  p["dust2_init"])
    _set_sp_prior(mp, "logzsol", p["logzsol_prior_type"],
                  p["logzsol_prior"], p["logzsol_init"])

    if p["dust_emission"]:
        _set_dust_emission(mp)
    if p["agn"]:
        _set_agn(mp)

    if p["tie_tage_to_tuniv"]:
        mp["tage"]["isfree"] = False
        mp["tage"]["init"] = 10.0
        mp["tage"]["depends_on"] = transforms.tage_from_tuniv
        mp["tage_tuniv"] = {
            "N": 1, "isfree": True, "init": p["tage_tuniv_init"],
            "prior": priors.TopHat(mini=0.1, maxi=1.0),
        }
    else:
        lo, hi = p["tage_range"]
        mp["tage"]["isfree"] = True
        mp["tage"]["init"] = p["tage_init"]
        mp["tage"]["prior"] = priors.TopHat(mini=lo, maxi=hi)

    m_lo, m_hi = p["mass_range"]
    t_lo, t_hi = p["tau_range"]
    if p["sampler"] == "emcee":
        mp["mass"]["isfree"] = False
        mp["mass"]["depends_on"] = logmass_to_mass
        mp["logmass"] = {
            "N": 1, "isfree": True, "init": np.log10(p["mass_init"]),
            "prior": priors.TopHat(mini=np.log10(m_lo),
                                   maxi=np.log10(m_hi)),
        }
        mp["tau"]["isfree"] = False
        mp["tau"]["depends_on"] = logtau_to_tau
        mp["logtau"] = {
            "N": 1, "isfree": True, "init": np.log10(p["tau_init"]),
            "prior": priors.TopHat(mini=np.log10(t_lo),
                                   maxi=np.log10(t_hi)),
        }
    else:
        mp["mass"]["isfree"] = True
        mp["mass"]["init"] = p["mass_init"]
        mp["mass"]["prior"] = priors.LogUniform(mini=m_lo, maxi=m_hi)
        mp["tau"]["isfree"] = True
        mp["tau"]["init"] = p["tau_init"]
        mp["tau"]["prior"] = priors.LogUniform(mini=t_lo, maxi=t_hi)
    return mp


def make_agebins(redshift: float, n_agebins: int) -> np.ndarray:
    """Seed age-bin edges (log10 yr) for the continuity SFH.

    A 0-30 Myr bin and a 30-100 Myr bin, then log-spaced bins to
    0.95 * t_universe(redshift), with t_universe from WMAP9; retuned per
    zred at fit time by zred_to_agebins_pbeta -- this is only the
    template seed.

    Parameters
    ----------
    redshift : float
        Redshift setting the age of the universe.
    n_agebins : int
        Number of bins to produce.

    Returns
    -------
    agebins : np.ndarray
        Shape (n_agebins, 2): the lower and upper log10-age edge of
        each bin.
    """
    t_universe = WMAP9.age(redshift).value * 1e9
    a_max = np.log10(t_universe * 0.95)
    edges = np.concatenate(
        [[0.0, 7.4772, 8.0], np.linspace(8.0, a_max, n_agebins - 1)[1:]])
    return np.array([edges[:-1], edges[1:]]).T


def _build_continuity_params(cfg: dict) -> dict:
    from prospect.models import priors, transforms
    from prospect.models.priors import StudentT
    from prospect.models.templates import TemplateLibrary

    p = cfg["prospector"]
    mp = TemplateLibrary["continuity_sfh"]
    n = p["n_agebins"]

    mp["agebins"]["isfree"] = False
    mp["agebins"]["init"] = make_agebins(cfg["z_ref"], n)
    mp["agebins"]["N"] = n
    mp["agebins"]["depends_on"] = transforms.zred_to_agebins_pbeta

    mp["mass"]["N"] = n
    mp["logsfr_ratios"]["N"] = n - 1
    mp["logsfr_ratios"]["init"] = np.zeros(n - 1)
    mp["logsfr_ratios"]["prior"] = StudentT(
        mean=np.zeros(n - 1), scale=np.full(n - 1, 0.3),
        df=np.full(n - 1, 2))

    _set_zred(mp, cfg)

    m_lo, m_hi = p["mass_range"]
    mp["logmass"]["isfree"] = True
    mp["logmass"]["init"] = np.log10(p["mass_init"])
    mp["logmass"]["prior"] = priors.TopHat(mini=np.log10(m_lo),
                                           maxi=np.log10(m_hi))

    _set_sp_prior(mp, "logzsol", p["logzsol_prior_type"],
                  p["logzsol_prior"], p["logzsol_init"])
    _set_sp_prior(mp, "dust2", p["dust2_prior_type"], p["dust2_prior"],
                  p["dust2_init"])

    if p["dust_emission"]:
        _set_dust_emission(mp)
    if p["nebular"]:
        _set_nebular(mp, p)
    if p["agn"]:
        _set_agn(mp)

    # depends_on propagates in dict order; agebins (<- zred) must
    # precede mass (<- agebins).
    order = ["zred", "agebins", "mass"] + [
        k for k in mp if k not in ("zred", "agebins", "mass")]
    return {k: mp[k] for k in order if k in mp}


# ------------------------------------
# Free per-instrument normalization
# ------------------------------------

class _NormMixin:
    """Free multiplicative normalization on selected instruments' model
    photometry (instruments with no SPHEREx anchor float their level).
    Pure pass-through when norm_groups is empty. Mixed into a concrete
    SpecModel subclass; both concretions are module-level so dynesty
    checkpointing can pickle the model.
    """

    norm_groups = None

    def predict(self, theta, obs=None, sps=None, sigma_spec=None, **extras):
        """The parent prediction, with each norm group's bands rescaled
        by its own free normalization parameter."""
        spec, phot, x = super().predict(
            theta, obs=obs, sps=sps, sigma_spec=sigma_spec, **extras)
        groups = self.norm_groups
        if groups and phot is not None:
            phot = np.array(phot, dtype=float, copy=True)
            for pname, mask in groups.items():
                phot[mask] *= float(np.atleast_1d(self.params[pname])[0])
        return spec, phot, x


class _NormSpecModel(_NormMixin, SpecModel):
    """The photometry-only concretion: no spectroscopic calibration."""


class _NormPolySpecModel(_NormMixin, PolySpecModel):
    """The joint-fit concretion: PolySpecModel's maximum-likelihood
    Chebyshev calibration vector absorbs the smooth ratio between the
    observed spectrum and the model, so the spectrum constrains features
    while the photometry anchors the absolute scale."""


def _set_free_norms(model_params, p: dict) -> None:
    from prospect.models import priors

    lo, hi = p["free_norm_prior"]
    for inst in p["free_norm_instruments"]:
        model_params[f"{inst}_norm"] = {
            "N": 1, "isfree": True, "init": 1.0,
            "prior": priors.LogUniform(mini=lo, maxi=hi),
        }


def attach_norm_masks(model, obs: dict, cfg: dict,
                      registry: Registry) -> None:
    """Bind each <INST>_norm parameter to its bands in obs order.

    Call after build_obs and build_model (masks index the final
    obs['filters']; filters are named by band, so registry.instrument_of
    resolves membership). An entry naming an unknown instrument or
    matching no fitted band is a hard error, not a silent no-op.
    """
    instruments = cfg["prospector"]["free_norm_instruments"]
    if not instruments:
        return
    names = [getattr(f, "name", "") for f in obs["filters"]]
    groups = {}
    for inst in instruments:
        if inst not in registry.instruments:
            raise ValueError(f"free_norm_instruments: {inst!r} is not a "
                             f"registry instrument")
        mask = np.array([registry.instrument_of(n) == inst for n in names])
        if not mask.any():
            raise ValueError(f"free_norm_instruments: {inst!r} matches no "
                             f"band in this fit")
        groups[f"{inst}_norm"] = mask
    model.norm_groups = groups


# ------------------------------------
# Public builders
# ------------------------------------

def build_model(cfg: dict) -> SpecModel:
    """Build the Prospector SpecModel for a resolved config.

    A config with a spectrum block gets the PolySpecModel concretion
    (unless polyorder is 0, which turns the calibration vector off) and
    the explicit smoothing/jitter/outlier parameters; a photometry-only
    config gets exactly the model it always did.
    """
    p = cfg["prospector"]
    if p["sfh"] == "continuity":
        model_params = _build_continuity_params(cfg)
    else:
        model_params = _build_parametric_params(cfg)
    _set_free_norms(model_params, p)
    # .get(): staged configs echoed before the spectrum key existed
    # rebuild (for replotting) exactly as photometry-only fits.
    spec_cfg = p.get("spectrum")
    if spec_cfg is None:
        return _NormSpecModel(model_params)
    _set_spectrum(model_params, spec_cfg)
    if spec_cfg["polyorder"] > 0:
        return _NormPolySpecModel(model_params)
    return _NormSpecModel(model_params)


def build_sps(cfg: dict) -> CSPSpecBasis | FastStepBasis:
    """The FSPS SPS object: FastStepBasis (continuity) or CSPSpecBasis."""
    from prospect.sources import CSPSpecBasis, FastStepBasis

    if cfg["prospector"]["sfh"] == "continuity":
        return FastStepBasis(zcontinuous=1)
    return CSPSpecBasis(zcontinuous=1)


def spectral_library(sps) -> tuple[str, str]:
    """(isochrones, spectral_library) of the compiled FSPS build."""
    try:
        libs = sps.ssp.libraries
        iso, spec = libs[0], libs[1]
        return (iso.decode() if isinstance(iso, bytes) else str(iso),
                spec.decode() if isinstance(spec, bytes) else str(spec))
    except Exception:
        return "unknown", "unknown"


def assert_stellar_library(sps, required: str) -> None:
    """Hard error when the live FSPS build is not the declared library.

    The env choice alone must not silently select the physics: a config
    declaring miles run in the c3k env (or vice versa) dies here.
    """
    if required not in LIBRARY_STAMPS:
        raise ValueError(
            f"unknown stellar_library {required!r}; known libraries are "
            f"{sorted(LIBRARY_STAMPS)}")
    expected = LIBRARY_STAMPS[required]
    _, live = spectral_library(sps)
    if live != expected:
        raise RuntimeError(
            f"stellar_library mismatch: the config declares {required!r} "
            f"(FSPS stamp {expected!r}) but the live build reports "
            f"{live!r}; run in the matching sedfit env")
