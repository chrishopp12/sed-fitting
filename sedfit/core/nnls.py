"""
nnls.py

Non-Negative Template Amplitudes
---------------------------------------------------------

One weighted NNLS solve against a design matrix, with eazy's internal
per-template renormalization applied for conditioning and divided back out.

Shared by every backend that fits a non-negative combination of templates. The
solve is indifferent to how a design row was built — a filter integral or a
resampled spectral pixel — so it lives here rather than under either backend.

Requirements:
    numpy, scipy

Notes:
    Rationale in DESIGN.md.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import nnls


def nnls_fit(design: np.ndarray, data: np.ndarray, var: np.ndarray,
             ok: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """One NNLS solve, mirroring eazy's template_lsq (fitter="nnls").

    Applies eazy's internal per-template renormalization
    (RENORM_TEMPLATES=y) for conditioning and divides it back out, so the
    returned coefficients are raw template amplitudes.

    Parameters
    ----------
    design : np.ndarray
        (NTEMP, N) template rows on the data's own sampling.
    data : np.ndarray
        (N,) observed values.
    var : np.ndarray
        (N,) variances.
    ok : np.ndarray
        (N,) boolean mask of the elements to fit.

    Returns
    -------
    chi2 : float
        Sum over the fitted elements.
    coeffs : np.ndarray
        (NTEMP,) raw template amplitudes, renormalization removed.
    model : np.ndarray
        (N,) model over all elements, fitted or not.
    """
    rms = np.sqrt(var)
    anorm = np.linalg.norm((design / rms)[:, ok], axis=1)
    ok_temp = anorm > 0
    anorm[~ok_temp] = 1.0
    normed = design / anorm[:, None]
    coeffs = np.zeros(design.shape[0])
    if ok_temp.any():
        weighted = (normed / rms).T[ok, :]
        solution, _ = nnls(weighted[:, ok_temp], (data / rms)[ok])
        coeffs[ok_temp] = solution
    model = coeffs @ normed
    chi2 = float((((data - model) ** 2 / var)[ok]).sum())
    return chi2, coeffs / anorm, model
