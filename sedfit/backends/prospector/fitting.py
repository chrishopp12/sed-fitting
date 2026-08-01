"""
fitting.py

Sampler Execution and HDF5 Output
---------------------------------------------------------

Runs the Prospector fit with the resolved config's sampler and seed and
writes the HDF5 result. dynesty runs through DynamicNestedSampler
directly -- seedable (rstate) and checkpointable (crash-safe .save file
with interrupt salvage); emcee runs through prospect's fit_model with
the global numpy RNG seeded. Either way the concrete seed comes from
the resolved config, so a run is reproducible from its echo.

Requirements:
  - numpy, prospect, dynesty (lazy)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from sedfit.backends.prospector.model import spectral_library

if TYPE_CHECKING:
    from prospect.models import SpecModel
    from prospect.sources import CSPSpecBasis, FastStepBasis

# rwalk sampling inside multi-ellipsoid bounds keeps the acceptance rate
# usable on the >10-D correlated posteriors these models produce.
DYNESTY_DEFAULTS = {
    "nested_sample": "rwalk",
    "nested_bound": "multi",
    "nested_walks": 25,
    "nested_nlive_init": 400,
    "nested_nlive_batch": 200,
    "nested_dlogz_init": 0.05,
}
EMCEE_DEFAULTS = {
    "nwalkers": 120,
    "niter": 800,
    "nburn": [200],
}


def run_sampler(
        cfg: dict,
        obs: dict,
        model: SpecModel,
        sps: CSPSpecBasis | FastStepBasis,
        *,
        checkpoint: str | None = None,
    ) -> dict:
    """Run the configured sampler; returns prospect's output structure.

    Parameters
    ----------
    cfg : dict
        Resolved fit config; prospector.seed is concrete by resolution.
    obs : dict
        The prospect-ready observation dictionary.
    model : SpecModel
        The model whose free parameters are sampled.
    sps : CSPSpecBasis or FastStepBasis
        The FSPS source basis the model predicts through.
    checkpoint : str or None
        dynesty only: path for the .save checkpoint file. [default: None]

    Returns
    -------
    output : dict
        prospect's output structure, keyed "sampling".

    Notes
    -----
    The seed is set on the global numpy RNG because emcee reaches its
    randomness through prospect's module globals; dynesty instead takes
    an explicit rstate built from the same seed.
    """
    p = cfg["prospector"]
    seed = int(p["seed"])
    np.random.seed(seed & 0xFFFFFFFF)

    if p["sampler"] == "emcee":
        from prospect.fitting import fit_model

        kwargs = {**EMCEE_DEFAULTS, **p["emcee"]}
        print(f"emcee fit (seed {seed}): {kwargs['nwalkers']} walkers x "
              f"{kwargs['niter']} iterations")
        return fit_model(obs, model, sps, emcee=True, dynesty=False,
                         progress=True, **kwargs)

    import dynesty
    from prospect.fitting.fitting import lnprobfn, wrap_lnp

    kwargs = {**DYNESTY_DEFAULTS, **p["dynesty"]}
    run_kwargs = dict(
        nlive_init=kwargs.get("nested_nlive_init", 400),
        nlive_batch=kwargs.get("nested_nlive_batch", 200),
        dlogz_init=kwargs.get("nested_dlogz_init", 0.05),
    )
    if "nested_target_n_effective" in kwargs:
        run_kwargs["n_effective"] = kwargs["nested_target_n_effective"]

    lnp = wrap_lnp(lnprobfn, obs, model, sps, noise=(None, None),
                   nested=True)
    dsampler = dynesty.DynamicNestedSampler(
        lnp, model.prior_transform, model.ndim,
        bound=kwargs.get("nested_bound", "multi"),
        sample=kwargs.get("nested_sample", "rwalk"),
        walks=kwargs.get("nested_walks", 25),
        rstate=np.random.default_rng(seed))
    print(f"dynesty fit (seed {seed})"
          + (f"; checkpoint -> {checkpoint}" if checkpoint else ""))
    try:
        dsampler.run_nested(checkpoint_file=checkpoint,
                            checkpoint_every=300 if checkpoint else None,
                            print_progress=True, **run_kwargs)
    except KeyboardInterrupt:
        print("\ninterrupted -- keeping the accumulated samples")
    return {"sampling": [dsampler.results, None]}


def save_results(out_file: str | Path, cfg: dict, model: SpecModel,
                 obs: dict, output: dict,
                 sps: CSPSpecBasis | FastStepBasis) -> None:
    """Write the HDF5 result with the library and seed stamped.

    Parameters
    ----------
    out_file : str or Path
        Destination .h5 path.
    cfg : dict
        Resolved fit config; supplies the seed stamp.
    model : SpecModel
        The sampled model.
    obs : dict
        The observation dictionary the fit consumed.
    output : dict
        run_sampler's return; its "sampling" entry is written.
    sps : CSPSpecBasis or FastStepBasis
        The source basis; supplies the isochrone and library stamps.
    """
    from prospect.io import write_results as writer

    isochrones, library = spectral_library(sps)
    run_params = {"isochrones": isochrones, "spectral_library": library,
                  "seed": int(cfg["prospector"]["seed"])}
    writer.write_hdf5(str(out_file), run_params, model, obs,
                      output["sampling"][0], sps=sps)
    print(f"results -> {out_file}")
