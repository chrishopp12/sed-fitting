"""
build.py

SED Assembly
---------------------------------------------------------

One pipeline from roster x recipe to the SED table: select source rows,
ingest SPHEREx, resolve the color reference (raw SPHEREx, one anchor,
or a two-anchor log-linear tilt), scale stitch-role instrument groups
onto the reference, concatenate float-role entries unchanged, and write
the table with its provenance sidecar.

Data products:
  <target.dir>/Photometry/<prefix>_sed_<recipe>.csv
  <target.dir>/Photometry/<prefix>_sed_<recipe>.provenance.json

Requirements:
  - numpy, pandas, astropy (via core modules)

Notes:
  - Uniform error convention: any flux transformation (achromatic scale
    or tilt) multiplies flux, error, and scatter by the same factor.
  - The two-anchor tilt: with anchor pivots lam_b < lam_r and correction
    factors c = 1/scale, t(lam) = c_b * (c_r/c_b)**(ln(lam/lam_b) /
    ln(lam_r/lam_b)), extrapolated with the same power law.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import sedfit
from sedfit.core.dered import deredden
from sedfit.core.provenance import canonical_json, git_state, sha256_bytes, sha256_file
from sedfit.core.recipe import Recipe
from sedfit.core.registry import Registry
from sedfit.core.roster import Roster, Target
from sedfit.core.sources import (
    check_position,
    provider_token,
    read_source_table,
    select_rows,
)
from sedfit.core.spherex import default_split_um, ingest
from sedfit.core.stitch import measure_scale
from sedfit.core.table import COLUMNS, write_sed_table

# ------------------------------------
# Constants
# ------------------------------------

# Provider tokens whose flags column carries the QA-token grammar. Every
# other provider's flags are archive-native and stay in the source table.
MEASURED_PROVIDER_TOKENS = ("aperture", "sersic")

# Subdirectory of a target's dir holding its source tables and the built
# SED table. sedfit.jobs resolves the same path for the fit verb.
SED_TABLE_SUBDIR = "Photometry"


# ------------------------------------
# Data model
# ------------------------------------

@dataclass(frozen=True)
class BuildResult:
    """An assembled SED table, ready to write.

    Attributes
    ----------
    frame : pd.DataFrame
        The table, in the canonical column order.
    sidecar : dict
        Provenance record written beside it.
    out_path : Path
        Where write_build will put the table; not yet written.
    """

    frame: pd.DataFrame
    sidecar: dict
    out_path: Path


# ------------------------------------
# Applicability
# ------------------------------------

def effective_bands(recipe: Recipe, target: Target) -> dict[str, tuple[str, ...]]:
    """Per recipe-source entry, the bands it consumes on this target.

    An entry without an explicit band list consumes every band its
    roster source declares. Hard errors: entry source missing on the
    target; explicit bands outside the source's declaration; the same
    band consumed by two entries.
    """
    label = f"{target.name}/{recipe.name}"
    resolved: dict[str, tuple[str, ...]] = {}
    seen: dict[str, str] = {}
    for entry in recipe.sources:
        if entry.source not in target.sources:
            raise ValueError(f"{label}: recipe source {entry.source!r} is not "
                             f"declared by the target")
        declared = target.sources[entry.source].bands
        bands = entry.bands if entry.bands is not None else declared
        extra = [b for b in bands if b not in declared]
        if extra:
            raise ValueError(f"{label}: entry {entry.source!r} requests bands "
                             f"outside its source declaration: {extra}")
        for band in bands:
            if band in seen:
                raise ValueError(f"{label}: band {band!r} consumed by both "
                                 f"{seen[band]!r} and {entry.source!r}")
            seen[band] = entry.source
        resolved[entry.source] = tuple(bands)
    return resolved


def check_applicability(recipe: Recipe, target: Target) -> None:
    """Everything checkable without touching pixel data; all hard errors."""
    effective_bands(recipe, target)
    label = f"{target.name}/{recipe.name}"
    if target.spherex is not None and not recipe.spherex_declared:
        raise ValueError(f"{label}: target declares a SPHEREx table but the "
                         f"recipe has no spherex block (declare one, or "
                         f"exclude deliberately with \"spherex\": null)")
    if target.spherex is None and recipe.spherex is not None:
        raise ValueError(f"{label}: recipe has a spherex block but the "
                         f"target declares no SPHEREx table")


# ------------------------------------
# Assembly steps
# ------------------------------------

def _check_dered_states(selected: dict[str, pd.DataFrame], label: str,
                        *, dereddening: bool) -> None:
    """The extinction cross-check on consumed rows; all hard errors."""
    states: dict[str, set[bool]] = {}
    for name, rows in selected.items():
        if "dered_applied" not in rows.columns:
            continue
        for value in rows["dered_applied"]:
            if pd.isna(value):
                continue
            state = (bool(value) if isinstance(value, (bool, np.bool_))
                     else str(value).strip().lower() == "true")
            states.setdefault(name, set()).add(state)
    if not states:
        return
    if dereddening:
        dered_sources = sorted(n for n, s in states.items() if True in s)
        if dered_sources:
            raise ValueError(f"{label}: dereddening build consumes already-"
                             f"dereddened rows from {dered_sources}")
    seen = set().union(*states.values())
    if len(seen) > 1:
        raise ValueError(f"{label}: consumed rows mix dereddened and "
                         f"as-measured states across {sorted(states)}; no "
                         f"common extinction scale")


def _qa_flags_of(rows: pd.DataFrame, provider: str) -> np.ndarray:
    """Per-row qa_flags carry: measured-provider rows only."""
    values = np.full(len(rows), np.nan, dtype=object)
    if provider_token(provider) not in MEASURED_PROVIDER_TOKENS \
            or "flags" not in rows.columns:
        return values
    for i, value in enumerate(rows["flags"]):
        if isinstance(value, str) and value:
            values[i] = value
    return values


def _tilt_factors(lam_AA: np.ndarray, lam_b: float, lam_r: float,
                  c_b: float, c_r: float) -> np.ndarray:
    frac = np.log(lam_AA / lam_b) / np.log(lam_r / lam_b)
    return c_b * (c_r / c_b) ** frac


def _apply_factor(spectrum: pd.DataFrame, factor: np.ndarray | float) -> None:
    for column in ("flux_uJy", "flux_err_uJy", "scatter_uJy"):
        spectrum[column] = spectrum[column].to_numpy(dtype=float) * factor


def _resolve_reference(
        recipe: Recipe,
        target: Target,
        spectrum: pd.DataFrame | None,
        selected: dict[str, pd.DataFrame],
        registry: Registry,
    ) -> dict:
    """Measure anchors on the raw spectrum and transform it in place."""
    label = f"{target.name}/{recipe.name}"
    anchors = [e for e in recipe.sources if e.role == "anchor"]
    if recipe.reference != "anchors":
        return {"kind": recipe.reference}
    if spectrum is None:
        raise ValueError(f"{label}: reference 'anchors' needs a SPHEREx "
                         f"spectrum to re-color")

    measured = []
    for entry in anchors:
        rows = selected[entry.source]
        scale, info = measure_scale(
            list(rows["band"]), rows["flux_uJy"].to_numpy(dtype=float),
            spec_wave_um=spectrum["wave_um"].to_numpy(),
            spec_fnu_uJy=spectrum["flux_uJy"].to_numpy(),
            registry=registry, min_coverage=recipe.min_coverage)
        if scale is None:
            raise ValueError(f"{label}: anchor {entry.source!r} has no band "
                             f"fully covered by the SPHEREx spectrum")
        measured.append({"source": entry.source, "band": info["anchor"],
                         "pivot_AA": registry.pivot_wave_AA(info["anchor"]),
                         "scale": scale, "c": 1.0 / scale,
                         "bands": info["bands"]})

    if len(measured) == 1:
        anchor = measured[0]
        _apply_factor(spectrum, anchor["c"])
        return {"kind": "anchors", "anchors": measured, "tilt": None}

    blue, red = sorted(measured, key=lambda a: a["pivot_AA"])
    if blue["pivot_AA"] == red["pivot_AA"]:
        raise ValueError(f"{label}: the two anchor pivots are equal "
                         f"({blue['band']}, {red['band']}); a tilt needs "
                         f"distinct wavelengths")
    lam_AA = spectrum["wave_um"].to_numpy(dtype=float) * 1e4
    factors = _tilt_factors(lam_AA, blue["pivot_AA"], red["pivot_AA"],
                            blue["c"], red["c"])
    _apply_factor(spectrum, factors)
    return {"kind": "anchors", "anchors": measured,
            "tilt": {"blue": blue["band"], "red": red["band"],
                     "c_blue": blue["c"], "c_red": red["c"],
                     "factor_min": float(factors.min()),
                     "factor_max": float(factors.max())}}


def _scale_stitch_groups(
        recipe: Recipe,
        target: Target,
        spectrum: pd.DataFrame | None,
        selected: dict[str, pd.DataFrame],
        registry: Registry,
    ) -> list[dict]:
    """One achromatic scale per stitch-role instrument group, in place."""
    label = f"{target.name}/{recipe.name}"
    diagnostics = []
    for entry in recipe.sources:
        if entry.role != "stitch":
            continue
        if spectrum is None:
            raise ValueError(f"{label}: stitch role {entry.source!r} needs a "
                             f"SPHEREx spectrum to scale onto")
        rows = selected[entry.source]
        instruments = [registry.instrument_of(b) for b in rows["band"]]
        for instrument in dict.fromkeys(instruments):
            group = np.array([i == instrument for i in instruments])
            scale, info = measure_scale(
                list(rows.loc[group, "band"]),
                rows.loc[group, "flux_uJy"].to_numpy(dtype=float),
                spec_wave_um=spectrum["wave_um"].to_numpy(),
                spec_fnu_uJy=spectrum["flux_uJy"].to_numpy(),
                registry=registry, min_coverage=recipe.min_coverage)
            if scale is None:
                raise ValueError(f"{label}: stitch group {entry.source!r}/"
                                 f"{instrument} has no band covered by the "
                                 f"reference spectrum; declare it float")
            rows.loc[group, "flux_uJy"] *= scale
            rows.loc[group, "flux_err_uJy"] *= scale
            diagnostics.append({"source": entry.source,
                                "instrument": instrument,
                                "anchor": info["anchor"], "scale": scale,
                                "bands": info["bands"]})
    return diagnostics


# ------------------------------------
# Build
# ------------------------------------

def build_target(
        roster: Roster,
        target_name: str,
        recipe_name: str,
        *,
        registry: Registry,
        mw_ebv: float | None = None,
    ) -> BuildResult:
    """Assemble one target under one recipe.

    When ``mw_ebv`` is given, every source and the SPHEREx spectrum are
    dereddened in their native frame before the reference/stitch step;
    the output path gains a ``_dered`` suffix.

    Returns
    -------
    result : BuildResult
        The SED table, its provenance sidecar, and the fixed output path
        (nothing is written; see write_build).
    """
    target = roster.targets[target_name]
    recipe = roster.recipes[recipe_name]
    check_applicability(recipe, target)
    bands_by_source = effective_bands(recipe, target)

    selected: dict[str, pd.DataFrame] = {}
    checked_paths: set[Path] = set()
    for entry in recipe.sources:
        source = target.sources[entry.source]
        frame = read_source_table(source.path)
        if source.path not in checked_paths:
            check_position(frame, target.position.ra_deg,
                           target.position.dec_deg,
                           label=f"{target.name}/{entry.source}")
            checked_paths.add(source.path)
        selected[entry.source] = select_rows(
            frame, list(bands_by_source[entry.source]),
            provider=source.provider,
            label=f"{target.name}/{entry.source}")
    _check_dered_states(selected, f"{target.name}/{recipe.name}",
                        dereddening=mw_ebv is not None)

    spectrum = None
    spherex_meta: dict | None = None
    if recipe.spherex is not None and target.spherex is not None:
        split_um = recipe.spherex.binning.split_um
        if split_um is None:
            split_um = default_split_um(target.z_ref)
        spectrum, counts = ingest(
            target.spherex.table, split_um=split_um,
            fit_ql_max=recipe.spherex.cuts.fit_ql_max,
            flags_reject=recipe.spherex.cuts.flags_reject,
            drop_local_bkg=recipe.spherex.cuts.drop_local_bkg,
            blue_merge_dlam_um=recipe.spherex.binning.blue_merge_dlam_um,
            red_bin_resolution=recipe.spherex.binning.red_bin_resolution)
        spherex_meta = {
            "table": str(target.spherex.table),
            "sha256_16": sha256_file(target.spherex.table)[:16],
            "model": target.spherex.model,
            "provenance": target.spherex.provenance,
            "cuts": {"fit_ql_max": recipe.spherex.cuts.fit_ql_max,
                     "flags_reject": recipe.spherex.cuts.flags_reject,
                     "drop_local_bkg": recipe.spherex.cuts.drop_local_bkg},
            "binning": {"split_um": split_um,
                        "blue_merge_dlam_um":
                            recipe.spherex.binning.blue_merge_dlam_um,
                        "red_bin_resolution":
                            recipe.spherex.binning.red_bin_resolution},
            "counts": counts,
        }

    dered_meta = None
    if mw_ebv is not None:
        dered_meta = deredden(selected, spectrum, mw_ebv, registry)

    reference = _resolve_reference(recipe, target, spectrum, selected, registry)
    stitch_diags = _scale_stitch_groups(recipe, target, spectrum, selected,
                                        registry)

    broadband_parts = []
    for entry in recipe.sources:
        rows = selected[entry.source]
        part = pd.DataFrame({
            "band": rows["band"].to_numpy(),
            "flux_uJy": rows["flux_uJy"].to_numpy(dtype=float),
            "flux_err_uJy": rows["flux_err_uJy"].to_numpy(dtype=float),
            "scatter_uJy": np.nan, "wave_um": np.nan,
            "bandwidth_um": np.nan, "n_exp": np.nan,
            "qa_flags": _qa_flags_of(
                rows, target.sources[entry.source].provider),
        })
        broadband_parts.append(part)
    parts = broadband_parts + ([spectrum] if spectrum is not None else [])
    frame = pd.concat(parts, ignore_index=True).reindex(
        columns=list(COLUMNS))

    suffix = "_dered" if mw_ebv is not None else ""
    out_path = (target.dir / SED_TABLE_SUBDIR
                / f"{target.prefix}_sed_{recipe.name}{suffix}.csv")
    sidecar = {
        "builder": "sedfit.core.build",
        "package_version": sedfit.__version__,
        "git": git_state(Path(sedfit.__file__).resolve().parents[1]),
        "generated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "roster": str(roster.path),
        "data_root": str(roster.data_root),
        "target": target.name,
        "position": {"ra_deg": target.position.ra_deg,
                     "dec_deg": target.position.dec_deg,
                     "authority": target.position.authority},
        "z_ref": target.z_ref,
        "z_ref_kind": target.z_ref_kind,
        "recipe": recipe.name,
        "recipe_sha256_16": sha256_bytes(
            canonical_json(recipe).encode("ascii"))[:16],
        "registry": {"path": str(registry.path),
                     "bandpass_sha256_16": registry.bandpass_hash()[:16],
                     "bands": {b: h[:16] for b, h in
                               registry.per_band_hashes().items()
                               if b in set(frame["band"])}},
        "sources": [
            {"source": entry.source,
             "role": entry.role,
             "path": str(target.sources[entry.source].path),
             "sha256_16": sha256_file(target.sources[entry.source].path)[:16],
             "kind": target.sources[entry.source].kind,
             "provider": target.sources[entry.source].provider,
             "bands": list(bands_by_source[entry.source]),
             "source_strings": sorted(set(
                 selected[entry.source]["source"].astype(str)))}
            for entry in recipe.sources
        ],
        "spherex": spherex_meta,
        "reference": reference,
        "stitch": stitch_diags,
        "dered": dered_meta,
        "min_coverage": recipe.min_coverage,
        "n_rows": int(len(frame)),
        "n_broadband": int(sum(len(p) for p in broadband_parts)),
    }
    return BuildResult(frame=frame, sidecar=sidecar, out_path=out_path)


def write_build(result: BuildResult, registry: Registry) -> Path:
    """Write the SED table and its sidecar; returns the table path."""
    write_sed_table(result.frame, result.out_path, registry)
    sidecar_path = result.out_path.parent / (
        result.out_path.stem + ".provenance.json")
    sidecar_path.write_text(json.dumps(result.sidecar, indent=2,
                                       sort_keys=True) + "\n", encoding="utf-8")
    return result.out_path
