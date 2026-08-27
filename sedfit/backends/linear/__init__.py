"""Linear spectral fitting: NNLS template amplitudes, no sampler, no priors."""
from sedfit.backends.linear.basis import TemplateBasis
from sedfit.backends.linear.fitting import (
    LinearFit,
    Minimum,
    SpectrumScan,
    coarse_step,
    fit_at,
    fit_spectrum,
    rank_minima,
    scan_spectrum,
)
from sedfit.backends.linear.runner import (
    FLAM_UNIT,
    MANIFEST_NAME,
    build_basis,
    build_gas,
    build_lsf,
    content_digests,
    plan,
    read_transmission,
    resolve_templates,
    run,
    to_flam,
)

__all__ = ["TemplateBasis", "LinearFit", "Minimum", "SpectrumScan",
           "coarse_step", "fit_at", "fit_spectrum", "rank_minima", "scan_spectrum",
           "FLAM_UNIT", "MANIFEST_NAME", "build_basis", "build_gas", "build_lsf", "content_digests",
           "plan", "read_transmission", "resolve_templates", "run", "to_flam"]
