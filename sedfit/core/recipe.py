"""
recipe.py

Assembly Recipes
---------------------------------------------------------

Recipe validation: one frame model. `reference` chooses what defines the
color truth -- the raw SPHEREx spectrum ("spherex"), one or two anchor
instruments ("anchors"), or nothing ("none") -- and per-source roles
place everything else on it. Role legality: anchor only under
"anchors" (which requires one or two); stitch under "spherex" and
"anchors"; float everywhere.

Requirements:
  - (stdlib only)

Notes:
  - The spherex block is tri-state: absent (the build hard-errors when
    the target declares a SPHEREx table), null (deliberate exclusion),
    or a block with cuts and binning. Defaults materialize the
    production flag filter (every bit except SOURCE).
"""

from __future__ import annotations

from dataclasses import dataclass

from sedfit.core.spherex import DEFAULT_FLAGS_REJECT
from sedfit.core.validate import require_enum, require_keys

# ------------------------------------
# Constants
# ------------------------------------

REFERENCES = ("spherex", "anchors", "none")
ROLES = ("anchor", "stitch", "float")
DEFAULT_MIN_COVERAGE = 0.98

RECIPE_KEYS = ("description", "reference", "sources", "spherex",
               "min_coverage")
RECIPE_REQUIRED = ("reference", "sources")
SOURCE_ENTRY_KEYS = ("source", "role", "bands")
SOURCE_ENTRY_REQUIRED = ("source", "role")
SPHEREX_KEYS = ("cuts", "binning")
CUTS_KEYS = ("fit_ql_max", "flags_reject", "drop_local_bkg")
BINNING_KEYS = ("split_um", "blue_merge_dlam_um", "red_bin_resolution")


# ------------------------------------
# Data model
# ------------------------------------

@dataclass(frozen=True)
class RecipeSource:
    """One source entry: roster source name, role, optional band subset."""
    source: str
    role: str
    bands: tuple[str, ...] | None = None


@dataclass(frozen=True)
class SpherexCuts:
    """Per-visit quality cuts applied before binning."""

    fit_ql_max: float | None = None
    flags_reject: int = DEFAULT_FLAGS_REJECT
    drop_local_bkg: bool = False


@dataclass(frozen=True)
class SpherexBinning:
    """Observed-frame binning: a split wavelength in micron, the blue
    side's merge width in micron, and the red side's resolution."""

    split_um: float | None = None
    blue_merge_dlam_um: float = 0.001
    red_bin_resolution: float = 1.0


@dataclass(frozen=True)
class SpherexBlock:
    """A recipe's SPHEREx handling: which visits survive, and how they
    bin into channels."""

    cuts: SpherexCuts
    binning: SpherexBinning


@dataclass(frozen=True)
class Recipe:
    """One validated recipe.

    spherex_declared distinguishes an explicit "spherex": null
    (declared, block None: deliberate exclusion) from an absent key
    (undeclared: the build hard-errors when the target has a SPHEREx
    table).
    """
    name: str
    reference: str
    sources: tuple[RecipeSource, ...]
    spherex: SpherexBlock | None
    spherex_declared: bool
    min_coverage: float
    description: str | None = None


# ------------------------------------
# Parsing
# ------------------------------------

def _parse_spherex_block(label: str, raw: dict) -> SpherexBlock:
    require_keys(label, raw, allowed=SPHEREX_KEYS, required=())
    cuts_raw = raw.get("cuts", {})
    require_keys(f"{label}.cuts", cuts_raw, allowed=CUTS_KEYS, required=())
    fit_ql_max = cuts_raw.get("fit_ql_max")
    if fit_ql_max is not None and not isinstance(fit_ql_max, (int, float)):
        raise ValueError(f"{label}.cuts.fit_ql_max must be a number or null")
    flags_reject = cuts_raw.get("flags_reject", DEFAULT_FLAGS_REJECT)
    if not isinstance(flags_reject, int) or flags_reject < 0:
        raise ValueError(f"{label}.cuts.flags_reject must be a non-negative "
                         f"integer bitmask")
    drop_local_bkg = cuts_raw.get("drop_local_bkg", False)
    if not isinstance(drop_local_bkg, bool):
        raise ValueError(f"{label}.cuts.drop_local_bkg must be a boolean")

    binning_raw = raw.get("binning", {})
    require_keys(f"{label}.binning", binning_raw, allowed=BINNING_KEYS,
                 required=())
    split_um = binning_raw.get("split_um")
    if split_um is not None and not (isinstance(split_um, (int, float))
                                     and split_um > 0):
        raise ValueError(f"{label}.binning.split_um must be null or a "
                         f"positive observed-frame wavelength [micron]")
    blue = binning_raw.get("blue_merge_dlam_um", 0.001)
    red = binning_raw.get("red_bin_resolution", 1.0)
    for key, value in (("blue_merge_dlam_um", blue),
                       ("red_bin_resolution", red)):
        if not (isinstance(value, (int, float)) and value > 0):
            raise ValueError(f"{label}.binning.{key} must be positive")

    return SpherexBlock(
        cuts=SpherexCuts(fit_ql_max=fit_ql_max, flags_reject=flags_reject,
                         drop_local_bkg=drop_local_bkg),
        binning=SpherexBinning(split_um=split_um, blue_merge_dlam_um=blue,
                               red_bin_resolution=red))


def parse_recipe(name: str, raw: dict) -> Recipe:
    """Validate one recipe dict; every violation is a hard error."""
    label = f"recipe[{name}]"
    require_keys(label, raw, allowed=RECIPE_KEYS, required=RECIPE_REQUIRED)
    reference = require_enum(f"{label}.reference", raw["reference"],
                             allowed=REFERENCES)

    entries_raw = raw["sources"]
    if not isinstance(entries_raw, list):
        raise ValueError(f"{label}.sources must be a list")
    if not entries_raw and reference != "spherex":
        raise ValueError(f"{label}: an empty sources list is only legal "
                         f"under reference 'spherex'")
    entries = []
    seen_names = set()
    explicit_bands: list[str] = []
    for i, entry_raw in enumerate(entries_raw):
        entry_label = f"{label}.sources[{i}]"
        require_keys(entry_label, entry_raw, allowed=SOURCE_ENTRY_KEYS,
                     required=SOURCE_ENTRY_REQUIRED)
        role = require_enum(f"{entry_label}.role", entry_raw["role"],
                            allowed=ROLES)
        source = entry_raw["source"]
        if not isinstance(source, str) or not source:
            raise ValueError(f"{entry_label}.source must be a non-empty string")
        if source in seen_names:
            raise ValueError(f"{label}: source {source!r} appears more than "
                             f"once")
        seen_names.add(source)
        bands = entry_raw.get("bands")
        if bands is not None:
            if (not isinstance(bands, list) or not bands
                    or not all(isinstance(b, str) for b in bands)):
                raise ValueError(f"{entry_label}.bands must be a non-empty "
                                 f"list of band names")
            explicit_bands.extend(bands)
            bands = tuple(bands)
        entries.append(RecipeSource(source=source, role=role, bands=bands))

    duplicates = sorted({b for b in explicit_bands
                         if explicit_bands.count(b) > 1})
    if duplicates:
        raise ValueError(f"{label}: bands declared by more than one source "
                         f"entry: {duplicates}")

    n_anchor = sum(1 for e in entries if e.role == "anchor")
    n_stitch = sum(1 for e in entries if e.role == "stitch")
    if reference == "anchors":
        if not 1 <= n_anchor <= 2:
            raise ValueError(f"{label}: reference 'anchors' requires one or "
                             f"two anchor entries, got {n_anchor}")
    else:
        if n_anchor:
            raise ValueError(f"{label}: anchor roles are only legal under "
                             f"reference 'anchors'")
    if reference == "none" and n_stitch:
        raise ValueError(f"{label}: stitch roles are illegal under reference "
                         f"'none' (declare them float)")

    if "spherex" in raw:
        spherex_declared = True
        spherex = (None if raw["spherex"] is None
                   else _parse_spherex_block(f"{label}.spherex", raw["spherex"]))
    else:
        spherex_declared = False
        spherex = None

    min_coverage = raw.get("min_coverage", DEFAULT_MIN_COVERAGE)
    if not (isinstance(min_coverage, (int, float))
            and 0.0 < min_coverage <= 1.0):
        raise ValueError(f"{label}.min_coverage must be in (0, 1]")

    description = raw.get("description")
    if description is not None and not isinstance(description, str):
        raise ValueError(f"{label}.description must be a string")

    return Recipe(name=name, reference=reference, sources=tuple(entries),
                  spherex=spherex, spherex_declared=spherex_declared,
                  min_coverage=float(min_coverage), description=description)
