"""
results.py

Result Loading and Chain Normalization
---------------------------------------------------------

A unified FitResult for dynesty and emcee outputs: all sampler-specific
chain wrangling (burn-in removal, flattening, weight handling, MAP
extraction) happens here. The model is rebuilt from the resolved config
when one is supplied (the run directory's config.json); otherwise the
result carries whatever model the HDF5 file stored.

Requirements:
  - numpy, prospect
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from prospect.io import read_results as reader
from prospect.plotting.corner import quantile

# Parameters displayed in log scale per sampler (dynesty fits mass/tau
# linearly; emcee already fits their logs).
LOGIFY = {"dynesty": {"mass", "tau"}, "emcee": set()}


@dataclass
class FitResult:
    """Normalized output from a Prospector fit.

    Attributes
    ----------
    sampler : str
        ``"dynesty"`` or ``"emcee"``.
    chain : ndarray, shape (n_samples, n_dim)
        Posterior samples, always 2D.  For emcee this is the flattened,
        post-burn-in chain; for dynesty it is the raw chain.
    weights : ndarray, shape (n_samples,)
        Sample weights.  Dynesty provides importance weights; for emcee
        all weights are uniform (1/N).
    lnp : ndarray, shape (n_samples,)
        Log-probability for each sample.
    theta_labels : list of str
        Free-parameter names, in the order they appear in ``chain``.
    map_theta : ndarray
        Maximum a-posteriori parameter vector.
    med_theta : ndarray
        Median parameter vector (weighted for dynesty, unweighted for
        emcee post-burn-in).
    obs : dict
        The observation dictionary stored in the HDF5 file.
    model : SpecModel
        The Prospector model (rebuilt if not stored in the file).
    raw_out : dict
        The full output dict from ``reader.results_from`` (for any
        fields not captured above, e.g. ``bestfit``).
    raw_chain_3d : ndarray or None
        The original (nwalkers, niter, ndim) chain for emcee, needed
        for per-walker trace plots.  ``None`` for dynesty.
    raw_lnp_3d : ndarray or None
        The original (nwalkers, niter) log-probability for emcee.
    burn_cut : int
        Number of iterations discarded as burn-in (0 for dynesty).
    logify : set of str
        Parameter names that should be log-transformed for plots.
    """
    sampler: str
    chain: np.ndarray
    weights: np.ndarray
    lnp: np.ndarray
    theta_labels: list[str]
    map_theta: np.ndarray
    med_theta: np.ndarray
    obs: dict
    model: object
    raw_out: dict
    raw_chain_3d: np.ndarray | None = None
    raw_lnp_3d: np.ndarray | None = None
    burn_cut: int = 0
    logify: set[str] = field(default_factory=set)

    @property
    def ndim(self) -> int:
        return self.chain.shape[1]

    @property
    def n_samples(self) -> int:
        return self.chain.shape[0]


# ------------------------------------
# Sampler detection
# ------------------------------------

def detect_sampler(out: dict) -> str:
    """Determine whether an HDF5 result came from dynesty or emcee.

    Dynesty produces a 2D chain (n_samples, n_dim); emcee produces a
    3D chain (n_walkers, n_iter, n_dim).
    """
    if out["chain"].ndim == 3:
        return "emcee"
    return "dynesty"


# ------------------------------------
# Dynesty helpers
# ------------------------------------

def _map_theta_dynesty(out: dict) -> np.ndarray:
    """MAP parameters from dynesty: stored in bestfit dict."""
    return out["bestfit"]["parameter"]


def _weighted_median_dynesty(out: dict) -> np.ndarray:
    """Component-wise weighted median of the dynesty chain."""
    pcts = quantile(out["chain"].T, q=[0.5], weights=out["weights"])
    return np.array([p[0] for p in pcts])


# ------------------------------------
# emcee helpers
# ------------------------------------

def _flatten_emcee_chain(out: dict, burn_frac: float = 0.25
                         ) -> tuple[np.ndarray, np.ndarray, int]:
    """Flatten emcee's 3D chain after discarding burn-in.

    Parameters
    ----------
    out : dict
        Must contain ``chain`` (nwalkers, niter, ndim) and
        ``lnprobability`` (nwalkers, niter).
    burn_frac : float
        Fraction of iterations to discard.

    Returns
    -------
    flat_chain : ndarray, shape (n_samples, ndim)
    flat_lnp : ndarray, shape (n_samples,)
    burn_cut : int
    """
    chain = out["chain"]
    lnp = out["lnprobability"]
    nwalkers, niter, ndim = chain.shape
    burn_cut = int(niter * burn_frac)

    flat_chain = chain[:, burn_cut:, :].reshape(-1, ndim)
    flat_lnp = lnp[:, burn_cut:].reshape(-1)
    return flat_chain, flat_lnp, burn_cut


def _map_theta_emcee(out: dict) -> np.ndarray:
    """MAP parameters from emcee: best across all walkers and iterations."""
    lnp = out["lnprobability"]
    chain = out["chain"]
    best_w, best_i = np.unravel_index(np.argmax(lnp), lnp.shape)
    return chain[best_w, best_i, :]


# ------------------------------------
# Main entry point
# ------------------------------------

def load_results(
        h5_path: str | Path,
        *,
        burn_frac: float = 0.25,
        sampler: str | None = None,
        cfg: dict | None = None,
    ) -> FitResult:
    """Load an HDF5 result file and return a unified ``FitResult``.

    Parameters
    ----------
    h5_path : str or Path
        Path to the Prospector output ``.h5`` file.
    burn_frac : float
        Fraction of emcee iterations to discard as burn-in.
        Ignored for dynesty. [default: 0.25]
    sampler : str or None
        Force sampler type.  If ``None``, auto-detected from chain
        dimensionality. [default: None]
    cfg : dict or None
        Resolved fit config (the run directory's config.json); when
        given, the model is rebuilt from it. None keeps whatever model
        the file stored. [default: None]

    Returns
    -------
    result : FitResult
        Chain, weights, MAP and median parameters, obs and model.
    """
    out, out_obs, model = reader.results_from(h5_path, dangerous=False)

    if sampler is None:
        sampler = detect_sampler(out)

    if model is None and cfg is not None:
        from sedfit.backends.prospector.model import build_model

        model = build_model(cfg)

    if sampler == "emcee":
        return _build_emcee_result(out, out_obs, model, burn_frac)
    return _build_dynesty_result(out, out_obs, model)


def _build_dynesty_result(out: dict, out_obs: dict,
                          model: object) -> FitResult:
    """Construct a FitResult from dynesty output."""
    chain = out["chain"]
    weights = out["weights"]
    lnp = out["lnprobability"]

    print(f"chain : {chain.shape[0]} samples, {chain.shape[1]} params")
    print(f"labels: {out['theta_labels']}")

    map_theta = _map_theta_dynesty(out)
    med_theta = _weighted_median_dynesty(out)

    return FitResult(
        sampler="dynesty",
        chain=chain,
        weights=weights,
        lnp=lnp,
        theta_labels=list(out["theta_labels"]),
        map_theta=map_theta,
        med_theta=med_theta,
        obs=out_obs,
        model=model,
        raw_out=out,
        logify=set(LOGIFY["dynesty"]),
    )


def _build_emcee_result(out: dict, out_obs: dict, model: object,
                        burn_frac: float) -> FitResult:
    """Construct a FitResult from emcee output."""
    chain_3d = out["chain"]
    lnp_3d = out["lnprobability"]
    nwalkers, niter, ndim = chain_3d.shape

    print(f"chain : {nwalkers} walkers x {niter} iterations x {ndim} params")
    print(f"labels: {out['theta_labels']}")

    flat_chain, flat_lnp, burn_cut = _flatten_emcee_chain(out, burn_frac)
    print(f"burn-in cut: {burn_cut}/{niter} iterations ({burn_frac:.0%})")
    print(f"post-burn-in samples: {flat_chain.shape[0]}")

    # Uniform weights for MCMC samples
    n_samples = flat_chain.shape[0]
    weights = np.ones(n_samples) / n_samples

    map_theta = _map_theta_emcee(out)
    med_theta = np.median(flat_chain, axis=0)

    return FitResult(
        sampler="emcee",
        chain=flat_chain,
        weights=weights,
        lnp=flat_lnp,
        theta_labels=list(out["theta_labels"]),
        map_theta=map_theta,
        med_theta=med_theta,
        obs=out_obs,
        model=model,
        raw_out=out,
        raw_chain_3d=chain_3d,
        raw_lnp_3d=lnp_3d,
        burn_cut=burn_cut,
        logify=set(LOGIFY["emcee"]),
    )


def print_parameter_summary(result: FitResult) -> None:
    """Print MAP and median parameter values.

    A convenience for interactive inspection of a loaded FitResult; the
    package's own output products do not go through it.
    """
    print("\nMAP parameters:")
    for name, val in zip(result.theta_labels, result.map_theta):
        print(f"  {name}: {val:.4f}")

    print("Median parameters:")
    for name, val in zip(result.theta_labels, result.med_theta):
        print(f"  {name}: {val:.4f}")

    if result.sampler == "emcee":
        print("\nPosterior summary (median +/- 68% CI):")
        for i, name in enumerate(result.theta_labels):
            p16, p50, p84 = np.percentile(result.chain[:, i], [16, 50, 84])
            print(f"  {name}: {p50:.4f}  (+{p84 - p50:.4f} / -{p50 - p16:.4f})")
