"""
ew.py

Equivalent-Width Measurement
---------------------------------------------------------

Rest- and observed-frame EW on explicit (wave, flux) arrays, using
two-sided continuum fitting with sigma-clipping and dynamic
line-boundary detection. Convention: EW > 0 for absorption (flux
deficit), EW < 0 for emission.

Requirements:
  - numpy
"""

from __future__ import annotations

import numpy as np

_trapz = np.trapezoid

# Mapping from friendly name -> key in the spec_data dictionary
SPEC_KEYS = {
    "map":         "map_spec_ujy",
    "medparam":    "medparam_spec_ujy",
    "post_median": "post_med_spec_ujy",
    "post_p16":    "post_p16_spec_ujy",
    "post_p84":    "post_p84_spec_ujy",
}


# -- Continuum fitting ------------------------------------------------

def _sigclip_polyfit(x, y, deg=1, sigma=2.0, maxiter=10):
    """Polynomial fit with iterative sigma-clipping to reject outliers.

    Parameters
    ----------
    x, y : array-like
        Data to fit.
    deg : int
        Polynomial degree (0 = constant, 1 = linear, ...).
    sigma : float
        Clipping threshold in standard deviations.
    maxiter : int
        Maximum number of clipping iterations.

    Returns
    -------
    poly : np.poly1d
        Best-fit polynomial.
    """
    mask = np.ones(len(x), dtype=bool)
    coeffs = np.polyfit(x, y, deg=deg)
    for _ in range(maxiter):
        if mask.sum() < deg + 2:
            break
        resid = y - np.polyval(coeffs, x)
        std = np.std(resid[mask])
        if std == 0:
            break
        new_mask = np.abs(resid) < sigma * std
        if np.array_equal(new_mask, mask):
            break
        mask = new_mask
        coeffs = np.polyfit(x[mask], y[mask], deg=deg)
    return np.poly1d(coeffs)


def _fit_two_sided_continuum(wave, flux, lc_obs, ci, co):
    """Fit each continuum sideband independently with sigma-clipping.

    Uses a linear (deg=1) fit per sideband - a robust mean level with slope.
    The slope across the line is captured by the interpolation between
    the two sideband levels.

    Parameters
    ----------
    wave : ndarray
        Observed-frame wavelength array.
    flux : ndarray
        Flux array (same length as *wave*).
    lc_obs : float
        Observed-frame line center.
    ci, co : float
        Inner and outer sideband edges (observed frame), measured from
        *lc_obs*.

    Returns
    -------
    blue_func : np.poly1d or None
    red_func : np.poly1d or None
    ok : bool
        False if either sideband has < 3 pixels.
    """
    blue_mask = (wave >= lc_obs - co) & (wave <= lc_obs - ci)
    red_mask = (wave >= lc_obs + ci) & (wave <= lc_obs + co)

    if blue_mask.sum() < 3 or red_mask.sum() < 3:
        return None, None, False

    blue_func = _sigclip_polyfit(wave[blue_mask], flux[blue_mask], deg=1)
    red_func = _sigclip_polyfit(wave[red_mask], flux[red_mask], deg=1)
    return blue_func, red_func, True


# -- Line-boundary detection ------------------------------------------

def _find_line_bounds(wave, flux, blue_func, red_func, lc_obs,
                      max_search=80.0, smooth_npix=5, n_consec=3):
    """Walk outward from line center to find where the line meets the continuum.

    Uses the blue-side continuum when walking blueward and the red-side
    continuum when walking redward.  The flux is smoothed with a boxcar
    before testing crossings; *n_consec* consecutive pixels below (or
    above for absorption) the continuum are required before declaring
    the boundary.

    Returns
    -------
    lo, hi : float
        Observed-frame wavelength bounds of the line.
    """
    kernel = np.ones(smooth_npix) / smooth_npix
    if len(flux) > smooth_npix:
        flux_sm = np.convolve(flux, kernel, mode="same")
    else:
        flux_sm = flux

    i_cen = np.argmin(np.abs(wave - lc_obs))
    avg_cont = 0.5 * (blue_func(lc_obs) + red_func(lc_obs))
    sign = 1.0 if flux_sm[i_cen] >= avg_cont else -1.0

    # Walk blueward
    res_blue = flux_sm - blue_func(wave)
    lo_idx = i_cen
    consec = 0
    for j in range(i_cen - 1, -1, -1):
        if wave[j] < lc_obs - max_search:
            lo_idx = j
            break
        if sign * res_blue[j] <= 0:
            consec += 1
            if consec >= n_consec:
                lo_idx = j + n_consec - 1
                break
        else:
            consec = 0

    # Walk redward
    res_red = flux_sm - red_func(wave)
    hi_idx = i_cen
    consec = 0
    for j in range(i_cen + 1, len(wave)):
        if wave[j] > lc_obs + max_search:
            hi_idx = j
            break
        if sign * res_red[j] <= 0:
            consec += 1
            if consec >= n_consec:
                hi_idx = j - n_consec + 1
                break
        else:
            consec = 0

    return wave[lo_idx], wave[hi_idx]


# -- Core EW computation ----------------------------------------------

def compute_equivalent_width(wave, flux, line_center_rest, redshift,
                             cont_window=(25.0, 65.0),
                             max_line_search=30.0,
                             fixed_bounds=None):
    """Compute equivalent width with two-sided continuum fitting.

    Each continuum sideband is fit independently with sigma-clipping.
    The continuum under the line is linearly interpolated between the
    blue fit at the blue line edge and the red fit at the red line edge.

    Parameters
    ----------
    wave : ndarray
        Observed-frame wavelength array.
    flux : ndarray
        Flux array (uJy or any consistent unit).
    line_center_rest : float
        Rest-frame line center in A.
    redshift : float
        Source redshift.
    cont_window : (float, float)
        (inner, outer) sideband edges in rest-frame A.
    max_line_search : float
        Maximum rest-frame distance from line center to search for
        the line-continuum crossing.
    fixed_bounds : (float, float) or None
        If provided, use these (line_lo, line_hi) instead of auto-
        detecting.  Useful for ensuring all spectrum variants integrate
        over the same region.

    Returns
    -------
    result : dict
        Keys: ``ew_obs``, ``ew_rest``, ``cont_level``,
        ``line_center_obs``, ``line_lo``, ``line_hi``, ``converged``.
    """
    a = 1.0 + redshift
    lc_obs = line_center_rest * a
    inner, outer = cont_window
    ci, co = inner * a, outer * a

    _nan_result = {
        "ew_obs": np.nan, "ew_rest": np.nan, "cont_level": np.nan,
        "line_center_obs": lc_obs, "line_lo": np.nan, "line_hi": np.nan,
        "converged": False,
    }

    blue_func, red_func, ok = _fit_two_sided_continuum(
        wave, flux, lc_obs, ci, co)
    if not ok:
        return _nan_result

    if fixed_bounds is not None:
        line_lo, line_hi = fixed_bounds
    else:
        search_hw = max_line_search * a
        search_mask = (wave >= lc_obs - search_hw) & (wave <= lc_obs + search_hw)
        if search_mask.sum() < 2:
            return _nan_result
        line_lo, line_hi = _find_line_bounds(
            wave[search_mask], flux[search_mask],
            blue_func, red_func, lc_obs, max_search=search_hw)

    line_mask = (wave >= line_lo) & (wave <= line_hi)
    if line_mask.sum() < 2:
        _nan_result.update(line_lo=line_lo, line_hi=line_hi)
        return _nan_result

    w_line = wave[line_mask]
    f_line = flux[line_mask]

    f_lo = blue_func(line_lo)
    f_hi = red_func(line_hi)
    f_cont = f_lo + (f_hi - f_lo) * (w_line - line_lo) / (line_hi - line_lo)

    integrand = 1.0 - f_line / f_cont
    ew_obs = _trapz(integrand, w_line)

    return {
        "ew_obs":          ew_obs,
        "ew_rest":         ew_obs / a,
        "cont_level":      0.5 * (blue_func(lc_obs) + red_func(lc_obs)),
        "line_center_obs": lc_obs,
        "line_lo":         line_lo,
        "line_hi":         line_hi,
        "converged":       True,
    }


# -- Batch EW computation ---------------------------------------------

def compute_ew_batch(spec_data, lines, redshift,
                     spectrum_types=None, **ew_kwargs):
    """Compute EW for multiple lines and spectrum types.

    Line bounds are determined from the MAP spectrum and shared across
    all spectrum types via ``fixed_bounds``, ensuring consistent
    integration regions.

    Parameters
    ----------
    spec_data : dict
        Spectrum arrays keyed by the values of ``SPEC_KEYS``, plus
        ``wave_obs``, the observed-frame wavelength grid they share.
    lines : list of (str, float)
        List of (label, rest_wavelength) tuples.
    redshift : float
        Source redshift.
    spectrum_types : list of str or None
        Which spectra to measure.  Defaults to
        ``["map", "post_median", "post_p16", "post_p84"]``.
    **ew_kwargs
        Passed to ``compute_equivalent_width`` (e.g. ``cont_window``).

    Returns
    -------
    results : list of dict
        One dict per (line, spectrum_type) combination.
    """
    if spectrum_types is None:
        spectrum_types = ["map", "post_median", "post_p16", "post_p84"]
    wave = spec_data["wave_obs"]
    results = []

    for label, rest_wl in lines:
        # Determine line bounds from the MAP spectrum
        map_ew = compute_equivalent_width(
            wave, spec_data[SPEC_KEYS["map"]], rest_wl, redshift,
            **ew_kwargs)
        bounds = None
        if map_ew["converged"]:
            bounds = (map_ew["line_lo"], map_ew["line_hi"])

        for stype in spectrum_types:
            key = SPEC_KEYS.get(stype)
            if key is None or key not in spec_data:
                continue
            if stype == "map":
                ew = map_ew
            else:
                ew = compute_equivalent_width(
                    wave, spec_data[key], rest_wl, redshift,
                    fixed_bounds=bounds, **ew_kwargs)
            ew["line_label"] = label
            ew["rest_wl"] = rest_wl
            ew["spectrum_type"] = stype
            results.append(ew)

    return results


def print_ew_table(results):
    """Pretty-print the output of ``compute_ew_batch``."""
    hdr = (f"{'Line':<16} {'rest_wl':>8} {'Spectrum':<14} "
           f"{'EW_rest':>12} {'EW_obs':>12} "
           f"{'line_lo':>10} {'line_hi':>10} {'OK':>4}")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        flag = "y" if r["converged"] else "N"
        ew_r = f"{r['ew_rest']:+.3f}" if r["converged"] else "---"
        ew_o = f"{r['ew_obs']:+.3f}" if r["converged"] else "---"
        lo = f"{r['line_lo']:.1f}" if r["converged"] else "---"
        hi = f"{r['line_hi']:.1f}" if r["converged"] else "---"
        print(f"{r['line_label']:<16} {r['rest_wl']:>8.2f} "
              f"{r['spectrum_type']:<14} {ew_r:>12} {ew_o:>12} "
              f"{lo:>10} {hi:>10} {flag:>4}")
