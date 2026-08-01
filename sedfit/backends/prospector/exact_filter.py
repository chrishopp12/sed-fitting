"""
exact_filter.py

Fixed-Grid Filter Projection
---------------------------------------------------------

sedpy's default projection integrates the filter transmission on the
redshifting source (FSPS) wavelength grid; for a narrow band, the count
of source points inside the band jumps by +/-1 as redshift slides the
grid past the fixed filter edges, quantizing the band flux and carving a
spurious comb into chi2(z). obj_counts_lores integrates on the filter's
own fixed grid instead -- same integrand and normalization, so this
changes the jitter, not the flux scale. The two projections agree for
ordinary broadbands; the difference matters for the narrow SPHEREx
channels.

Requirements:
  - sedpy
"""

from __future__ import annotations

from sedpy.observate import Filter


class ExactFilter(Filter):
    """A Filter whose default (hires) projection is redirected to the
    fixed-grid obj_counts_lores."""

    def obj_counts_hires(self, sourcewave, sourceflux, **extras):
        """Project on the filter's fixed grid, not the source grid."""
        return self.obj_counts_lores(sourcewave, sourceflux)


def make_exact(filters: list[Filter]) -> None:
    """Upgrade sedpy Filter objects in place to fixed-grid projection."""
    for f in filters:
        if not isinstance(f, ExactFilter):
            f.__class__ = ExactFilter
