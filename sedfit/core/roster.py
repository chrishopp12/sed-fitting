"""
roster.py

Target Roster
---------------------------------------------------------

Roster loading, resolution, and validation. One roster per campaign,
next to the data it describes: data_root is relative to the roster
file's directory, target dirs to data_root, source paths and SPHEREx
tables to the target dir. The target position is the single coordinate
store, and redshift has one home per meaning: reference-kind targets
derive z_ref from the sample's reference_redshift; spec and phot
targets carry their own.

Requirements:
  - via core.sources: numpy, pandas, astropy

Notes:
  - Loading reads every declared source file and requires the declared
    columns to be present and each declared band to select exactly one
    row under its provider prefix.
  - 'cluster', 'cluster_redshift' and z_ref_kind 'cluster' are accepted
    as deprecated spellings of 'sample', 'reference_redshift' and
    'reference'.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from sedfit.core.recipe import Recipe, parse_recipe
from sedfit.core.sources import match_counts, read_source_table, source_prefix
from sedfit.core.validate import apply_aliases, require_enum, require_keys

if TYPE_CHECKING:
    from sedfit.core.registry import Registry

# ------------------------------------
# Constants
# ------------------------------------

SCHEMA_VERSIONS = (2,)
Z_REF_KINDS = ("reference", "spec", "phot")
Z_REF_KIND_ALIASES = {"cluster": "reference"}
SPHEREX_MODELS = ("psf", "sersic")
POSITION_FRAMES = ("icrs",)

DEFAULT_MANIFEST_RELPATH = Path("sed_fitting/runs.jsonl")

TOP_ALIASES = {"cluster": "sample", "cluster_redshift": "reference_redshift"}
TOP_KEYS = ("schema_version", "description", "sample", "data_root",
            "reference_redshift", "manifest_path", "targets", "recipes")
TOP_REQUIRED = ("schema_version", "sample", "data_root", "targets",
                "recipes")
POSITION_KEYS = ("ra_deg", "dec_deg", "frame", "authority")
TARGET_KEYS = ("position", "z_ref", "z_ref_kind", "dir", "prefix",
               "aperture_arcsec", "spherex", "sources")
TARGET_REQUIRED = ("position", "z_ref_kind", "dir", "prefix", "sources")
TARGET_SPHEREX_KEYS = ("table", "model", "provenance")
SOURCE_KEYS = ("path", "bands", "kind", "provider")
SOURCE_KINDS = ("measured", "catalog")


# ------------------------------------
# Data model
# ------------------------------------

@dataclass(frozen=True)
class Position:
    """A target's sky position and the source that established it."""

    ra_deg: float
    dec_deg: float
    frame: str
    authority: str


@dataclass(frozen=True)
class TargetSource:
    """One declared photometry file: a resolved path, its bands, and
    the provider whose prefix its rows must carry."""

    path: Path
    bands: tuple[str, ...]
    kind: str
    provider: str


@dataclass(frozen=True)
class TargetSpherex:
    """A target's SPHEREx visit table, resolved, with its source model."""

    table: Path
    model: str
    provenance: str


@dataclass(frozen=True)
class Target:
    """One target, fully resolved.

    ``dir`` and every source path are absolute; ``z_ref`` is derived
    from the roster's reference_redshift when z_ref_kind is
    'reference', and carried by the target otherwise. ``prefix`` is the
    filename stem its products are written under.
    """

    name: str
    position: Position
    z_ref_kind: str
    z_ref: float
    dir: Path
    prefix: str
    sources: dict[str, TargetSource]
    spherex: TargetSpherex | None = None
    aperture_arcsec: float | None = None


@dataclass(frozen=True)
class Roster:
    """One campaign's targets, recipes, and the tree they live in.

    Attributes
    ----------
    path : Path
        The roster file itself, resolved.
    sample : str
        The campaign's name for this set of targets.
    data_root : Path
        Absolute root the target directories hang from.
    manifest_path : Path
        The central run manifest, absolute.
    reference_redshift : float or None
        Redshift adopted by targets of z_ref_kind 'reference'; None when
        no target uses that kind.
    targets, recipes : dict
        Name -> Target and name -> Recipe.
    """

    path: Path
    sample: str
    data_root: Path
    manifest_path: Path
    targets: dict[str, Target]
    recipes: dict[str, Recipe]
    reference_redshift: float | None = None
    description: str | None = None


# ------------------------------------
# Parsing
# ------------------------------------

def _parse_position(label: str, raw: dict) -> Position:
    require_keys(label, raw, allowed=POSITION_KEYS, required=POSITION_KEYS)
    require_enum(f"{label}.frame", raw["frame"], allowed=POSITION_FRAMES)
    for key in ("ra_deg", "dec_deg"):
        if not isinstance(raw[key], (int, float)):
            raise ValueError(f"{label}.{key} must be a number")
    if not isinstance(raw["authority"], str) or not raw["authority"]:
        raise ValueError(f"{label}.authority must be a non-empty string")
    return Position(ra_deg=float(raw["ra_deg"]), dec_deg=float(raw["dec_deg"]),
                    frame=raw["frame"], authority=raw["authority"])


def _parse_target(
        name: str,
        raw: dict,
        *,
        data_root: Path,
        reference_redshift: float | None,
        registry: Registry,
    ) -> Target:
    label = f"target[{name}]"
    require_keys(label, raw, allowed=TARGET_KEYS, required=TARGET_REQUIRED)
    position = _parse_position(f"{label}.position", raw["position"])

    kind = require_enum(
        f"{label}.z_ref_kind",
        Z_REF_KIND_ALIASES.get(raw["z_ref_kind"], raw["z_ref_kind"]),
        allowed=Z_REF_KINDS)
    if kind == "reference":
        if "z_ref" in raw:
            raise ValueError(f"{label}: z_ref_kind 'reference' derives z_ref "
                             f"from the roster's reference_redshift; remove "
                             f"the explicit z_ref")
        if reference_redshift is None:
            raise ValueError(f"{label}: z_ref_kind 'reference' requires the "
                             f"roster to declare reference_redshift")
        z_ref = reference_redshift
    else:
        if "z_ref" not in raw or not isinstance(raw["z_ref"], (int, float)):
            raise ValueError(f"{label}: z_ref_kind {kind!r} requires a "
                             f"numeric z_ref")
        z_ref = float(raw["z_ref"])

    for key in ("dir", "prefix"):
        if not isinstance(raw[key], str) or not raw[key]:
            raise ValueError(f"{label}.{key} must be a non-empty string")
    target_dir = (data_root / raw["dir"]).resolve()
    if not target_dir.is_dir():
        raise ValueError(f"{label}: dir does not exist: {target_dir}")

    aperture = raw.get("aperture_arcsec")
    if aperture is not None and not isinstance(aperture, (int, float)):
        raise ValueError(f"{label}.aperture_arcsec must be a number")

    spherex = None
    if "spherex" in raw and raw["spherex"] is not None:
        sx_label = f"{label}.spherex"
        require_keys(sx_label, raw["spherex"], allowed=TARGET_SPHEREX_KEYS,
                     required=TARGET_SPHEREX_KEYS)
        require_enum(f"{sx_label}.model", raw["spherex"]["model"],
                     allowed=SPHEREX_MODELS)
        table = (target_dir / raw["spherex"]["table"]).resolve()
        if not table.is_file():
            raise ValueError(f"{sx_label}: table does not exist: {table}")
        spherex = TargetSpherex(table=table, model=raw["spherex"]["model"],
                                provenance=str(raw["spherex"]["provenance"]))

    sources_raw = raw["sources"]
    if not isinstance(sources_raw, dict) or not sources_raw:
        raise ValueError(f"{label}.sources must be a non-empty object")
    sources = {}
    for source_name, source_raw in sources_raw.items():
        src_label = f"{label}.sources[{source_name}]"
        require_keys(src_label, source_raw, allowed=SOURCE_KEYS,
                     required=SOURCE_KEYS)
        require_enum(f"{src_label}.kind", source_raw["kind"],
                     allowed=SOURCE_KINDS)
        source_prefix(source_raw["provider"])
        bands = source_raw["bands"]
        if (not isinstance(bands, list) or not bands
                or not all(isinstance(b, str) for b in bands)):
            raise ValueError(f"{src_label}.bands must be a non-empty list")
        unknown = [b for b in bands if b not in registry]
        if unknown:
            raise ValueError(f"{src_label}: bands not in the registry: "
                             f"{unknown}")
        path = (target_dir / source_raw["path"]).resolve()
        if not path.is_file():
            raise ValueError(f"{src_label}: path does not exist: {path}")
        sources[source_name] = TargetSource(
            path=path, bands=tuple(bands), kind=source_raw["kind"],
            provider=source_raw["provider"])

    return Target(name=name, position=position, z_ref_kind=kind, z_ref=z_ref,
                  dir=target_dir, prefix=raw["prefix"], sources=sources,
                  spherex=spherex, aperture_arcsec=aperture)


def _check_source_files(target: Target) -> None:
    """Every declared band selects exactly one row in its source file."""
    for source_name, source in target.sources.items():
        label = f"target[{target.name}].sources[{source_name}]"
        frame = read_source_table(source.path)
        counts = match_counts(frame, source.bands, provider=source.provider)
        missing = sorted(b for b, n in counts.items() if n == 0)
        ambiguous = sorted(b for b, n in counts.items() if n > 1)
        if missing:
            raise ValueError(f"{label}: declared bands matching no row "
                             f"under the provider prefix: {missing}")
        if ambiguous:
            raise ValueError(f"{label}: declared bands matching more than "
                             f"one row under the provider prefix: {ambiguous}")


def _check_recipes(recipes: dict[str, Recipe],
                   targets: dict[str, Target]) -> None:
    """Recipe source names resolve on at least one target, with legal bands."""
    for recipe in recipes.values():
        for entry in recipe.sources:
            holders = [t for t in targets.values()
                       if entry.source in t.sources]
            if not holders:
                raise ValueError(f"recipe[{recipe.name}]: source "
                                 f"{entry.source!r} exists on no target")
            if entry.bands is not None:
                for target in holders:
                    declared = set(target.sources[entry.source].bands)
                    extra = [b for b in entry.bands if b not in declared]
                    if extra:
                        raise ValueError(
                            f"recipe[{recipe.name}]: bands {extra} are not "
                            f"declared by source {entry.source!r} on target "
                            f"{target.name!r}")


# ------------------------------------
# Loading
# ------------------------------------

def load_roster(path: str | Path, registry: Registry) -> Roster:
    """Load, resolve, and validate a roster; every violation is a hard error.

    Parameters
    ----------
    path : str or Path
        Roster JSON path; data_root resolves relative to its directory.
    registry : Registry
        Band-identity authority for declared bands.

    Returns
    -------
    roster : Roster
        Fully resolved (absolute paths, derived z_ref) and validated.
    """
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"roster not found: {path}")
    raw = apply_aliases("roster", json.loads(path.read_text(encoding="utf-8")),
                        TOP_ALIASES)
    require_keys("roster", raw, allowed=TOP_KEYS, required=TOP_REQUIRED)
    if raw["schema_version"] not in SCHEMA_VERSIONS:
        raise ValueError(f"roster: unrecognized schema_version "
                         f"{raw['schema_version']!r} (known: {SCHEMA_VERSIONS})")

    reference_redshift = raw.get("reference_redshift")
    if reference_redshift is not None:
        if not isinstance(reference_redshift, (int, float)):
            raise ValueError("roster: reference_redshift must be a number")
        reference_redshift = float(reference_redshift)

    data_root = (path.parent / raw["data_root"]).resolve()
    if not data_root.is_dir():
        raise ValueError(f"roster: data_root does not exist: {data_root}")
    manifest_relpath = Path(raw.get("manifest_path")
                            or DEFAULT_MANIFEST_RELPATH)
    if manifest_relpath.is_absolute():
        raise ValueError(f"roster: manifest_path must be relative to "
                         f"data_root, got {manifest_relpath}")

    if not isinstance(raw["targets"], dict) or not raw["targets"]:
        raise ValueError("roster: targets must be a non-empty object")
    targets = {
        name: _parse_target(name, target_raw, data_root=data_root,
                            reference_redshift=reference_redshift,
                            registry=registry)
        for name, target_raw in raw["targets"].items()
    }
    for target in targets.values():
        _check_source_files(target)

    if not isinstance(raw["recipes"], dict):
        raise ValueError("roster: recipes must be an object")
    recipes = {name: parse_recipe(name, recipe_raw)
               for name, recipe_raw in raw["recipes"].items()}
    _check_recipes(recipes, targets)

    return Roster(path=path, sample=raw["sample"], data_root=data_root,
                  manifest_path=data_root / manifest_relpath,
                  reference_redshift=reference_redshift,
                  targets=targets, recipes=recipes,
                  description=raw.get("description"))
