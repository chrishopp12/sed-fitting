"""
generate.py

Roster Generation
---------------------------------------------------------

Builds a roster from two inputs. A sample catalog holds one row per
target and carries only facts about a galaxy: identity, position,
redshift, and where its directory is. A campaign config holds the policy
choices shared by every target: which sources to expect, which bands,
which recipes, the sample's reference redshift. The catalog is the file
a campaign operator edits, so adding a galaxy leaves what is fit, and
how, unchanged.

Generation reads the photometry to decide what to declare: a source
whose file is absent is omitted, and a declared band that no row
supplies under the provider's prefix is dropped from that source. A
generated roster is therefore true of the tree it was generated
against, and load_roster's checks detect a tree that has since changed.

Data products:
  <out>          the roster JSON
  <out>.report.csv   one row per (target, source) with what was kept

Requirements:
  - pandas (via core.sources)

Notes:
  - Path templates take {name} and {label}; {prefix} is accepted as a
    synonym for {label}. A source may declare a list of templates and
    the first that SUPPLIES a declared band wins.
  - The SPHEREx table is matched by PATTERN (globs allowed): an
    extraction's filename carries a hash over its own configuration, so
    the name differs per galaxy. A pattern matching several tables is an
    error rather than a pick.
  - z_ref must be blank for z_ref_kind 'reference': the roster derives it
    from reference_redshift and rejects a target that restates it, so the
    catalog is not allowed to disagree about where a redshift comes from.
"""

from __future__ import annotations

import collections
import csv
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

from sedfit.core.roster import Z_REF_KIND_ALIASES, Z_REF_KINDS
from sedfit.core.sources import match_counts, read_source_table, source_prefix
from sedfit.core.validate import (apply_aliases, describe_columns,
                                  require_enum, require_keys)

if TYPE_CHECKING:
    from sedfit.core.registry import Registry

# ------------------------------------
# Constants
# ------------------------------------

CAMPAIGN_SCHEMA_VERSIONS = (1,)
CAMPAIGN_ALIASES = {"cluster": "sample",
                    "cluster_redshift": "reference_redshift"}
CAMPAIGN_KEYS = ("schema_version", "description", "sample", "data_root",
                 "reference_redshift", "manifest_path", "position_frame",
                 "position_authority", "sources", "spherex", "recipes")
CAMPAIGN_REQUIRED = ("schema_version", "sample", "data_root",
                     "position_authority", "sources", "recipes")
CAMPAIGN_SOURCE_KEYS = ("path", "bands", "kind", "provider")
CAMPAIGN_SPHEREX_KEYS = ("table", "model", "provenance")

CATALOG_NAME_ALIASES = ("name", "target")
# Position column spellings, in preference order.
CATALOG_POSITION_COLUMNS = (("ra_deg", "dec_deg"), ("ra", "dec"))
# Every column this reader consumes. Anything else in the file belongs
# to another tool and is reported, not refused (core.validate).
CATALOG_COLUMNS = ("name", "target", "ra_deg", "dec_deg", "ra", "dec",
                   "z_ref_kind", "z_ref", "dir", "label")
REPORT_COLUMNS = ("target", "source", "status", "detail", "bands")

STATUS_KEPT = "kept"
STATUS_DROPPED = "dropped"
STATUS_PARTIAL = "partial"


# ------------------------------------
# Inputs
# ------------------------------------

def sanitize_label(name: str) -> str:
    """Reduce a target name to a filesystem-safe output stem.

    Non-word characters collapse to underscores. The stem is the same
    one the photometry products are written under, so one identifier
    names a target's files everywhere.

    Parameters
    ----------
    name : str
        The target name from the sample catalog.

    Returns
    -------
    stem : str
        The output stem, or 'target' if nothing survives.
    """
    label = re.sub(r"[^\w+-]+", "_", name.strip()).strip("_")
    return label or "target"


def read_sample_catalog(path: str | Path) -> tuple[list[dict], str]:
    """One row per target, in file order.

    Parameters
    ----------
    path : str or Path
        Sample catalog CSV.

    Returns
    -------
    targets : list of dict
        name, ra_deg, dec_deg, z_ref_kind, z_ref, dir, label, prefix.
        ``prefix`` is set from the same label, so one identifier names a
        target's files everywhere.
    column_report : str
        One line naming the columns used and ignored, for the caller to
        display.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"sample catalog not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)

    name_col = next((c for c in CATALOG_NAME_ALIASES if c in fields), None)
    if name_col is None:
        raise ValueError(f"{path}: no {' or '.join(CATALOG_NAME_ALIASES)} "
                         f"column (found: {fields})")
    position = next((pair for pair in CATALOG_POSITION_COLUMNS
                     if pair[0] in fields and pair[1] in fields), None)
    if position is None:
        raise ValueError(f"{path}: no ra_deg/dec_deg column pair "
                         f"(found: {fields})")
    if "z_ref_kind" not in fields:
        raise ValueError(f"{path}: missing required column 'z_ref_kind' "
                         f"(found: {fields})")
    column_report = describe_columns(path, len(rows), fields, CATALOG_COLUMNS)

    targets, seen = [], set()
    for number, row in enumerate(rows, start=2):
        name = (row.get(name_col) or "").strip()
        if not name:
            continue
        if name in seen:
            raise ValueError(f"{path} line {number}: duplicate target {name!r}")
        seen.add(name)

        label = f"{path} line {number} ({name})"
        raw_kind = (row.get("z_ref_kind") or "").strip()
        kind = require_enum(f"{label}.z_ref_kind",
                            Z_REF_KIND_ALIASES.get(raw_kind, raw_kind),
                            allowed=Z_REF_KINDS)
        raw_z = (row.get("z_ref") or "").strip()
        if kind == "reference":
            if raw_z:
                raise ValueError(f"{label}: z_ref_kind 'reference' derives "
                                 f"z_ref from the campaign's "
                                 f"reference_redshift; leave z_ref blank")
            z_ref = None
        else:
            if not raw_z:
                raise ValueError(f"{label}: z_ref_kind {kind!r} requires a "
                                 f"numeric z_ref")
            z_ref = float(raw_z)

        directory = (row.get("dir") or "").strip() or name
        # The label is the product filename stem and the roster prefix, so
        # one identifier names a target's files everywhere. Default it from
        # the name: dir is optional and may point anywhere.
        stem = (row.get("label") or "").strip() or sanitize_label(name)
        try:
            ra_deg = float(row[position[0]])
            dec_deg = float(row[position[1]])
        except ValueError as err:
            raise ValueError(f"{label}: unreadable position "
                             f"({position[0]}, {position[1]}): {err}") from None
        targets.append({
            "name": name,
            "ra_deg": ra_deg,
            "dec_deg": dec_deg,
            "z_ref_kind": kind, "z_ref": z_ref,
            "dir": directory, "label": stem, "prefix": stem,
        })
    if not targets:
        raise ValueError(f"{path}: no target rows")
    return targets, column_report


def load_campaign(path: str | Path) -> dict:
    """Load and validate a campaign config; every violation is an error.

    Parameters
    ----------
    path : str or Path
        Campaign JSON path.

    Returns
    -------
    campaign : dict
        The validated config, with deprecated key spellings renamed.
    """
    path = Path(path)
    if not Path(path).is_file():
        raise FileNotFoundError(f"campaign config not found: {path}")
    raw = apply_aliases("campaign",
                        json.loads(path.read_text(encoding="utf-8")),
                        CAMPAIGN_ALIASES)
    require_keys("campaign", raw, allowed=CAMPAIGN_KEYS,
                 required=CAMPAIGN_REQUIRED)
    if raw["schema_version"] not in CAMPAIGN_SCHEMA_VERSIONS:
        raise ValueError(f"campaign: unrecognized schema_version "
                         f"{raw['schema_version']!r} "
                         f"(known: {CAMPAIGN_SCHEMA_VERSIONS})")
    if (raw.get("reference_redshift") is not None
            and not isinstance(raw["reference_redshift"], (int, float))):
        raise ValueError("campaign: reference_redshift must be a number")
    if not isinstance(raw["sources"], dict) or not raw["sources"]:
        raise ValueError("campaign: sources must be a non-empty object")

    for name, source in raw["sources"].items():
        label = f"campaign.sources[{name}]"
        require_keys(label, source, allowed=CAMPAIGN_SOURCE_KEYS,
                     required=CAMPAIGN_SOURCE_KEYS)
        source_prefix(source["provider"])
        paths = source["path"]
        paths = [paths] if isinstance(paths, str) else paths
        if (not isinstance(paths, list) or not paths
                or not all(isinstance(p, str) and p for p in paths)):
            raise ValueError(f"{label}.path must be a template or a "
                             f"non-empty list of templates")
        if (not isinstance(source["bands"], list) or not source["bands"]
                or not all(isinstance(b, str) for b in source["bands"])):
            raise ValueError(f"{label}.bands must be a non-empty list")
    if raw.get("spherex") is not None:
        require_keys("campaign.spherex", raw["spherex"],
                     allowed=CAMPAIGN_SPHEREX_KEYS,
                     required=CAMPAIGN_SPHEREX_KEYS)
        tables = raw["spherex"]["table"]
        tables = [tables] if isinstance(tables, str) else tables
        if (not isinstance(tables, list) or not tables
                or not all(isinstance(t, str) and t for t in tables)):
            raise ValueError("campaign.spherex.table must be a pattern or a "
                             "non-empty list of patterns")
    return raw


# ------------------------------------
# Generation
# ------------------------------------

def _format(template: str, target: dict) -> str:
    try:
        return template.format(name=target["name"], label=target["label"],
                               prefix=target["prefix"])
    except KeyError as err:
        raise ValueError(f"path template {template!r} uses unknown field "
                         f"{err}; known: name, label, prefix") from err


def _resolve_source(target_dir: Path, target: dict, name: str, spec: dict,
                    registry: Registry) -> tuple[dict | None, dict]:
    """One source for one target: what survives, and why."""
    # A source may declare several filename templates, tried in order:
    # a tree assembled over time carries variants, and losing a whole
    # source to a naming difference is worse than naming the variants.
    row = {"target": target["name"], "source": name}
    unknown = [b for b in spec["bands"] if b not in registry]
    if unknown:
        raise ValueError(f"campaign.sources[{name}]: bands not in the "
                         f"registry: {unknown}")

    templates = spec["path"]
    if isinstance(templates, str):
        templates = [templates]
    candidates = [_format(t, target) for t in templates]

    # A variant wins by SUPPLYING a declared band, not by existing. A
    # tree that carries both a combined catalog and a per-provider file
    # has several candidates present at once, and stopping at the first
    # that exists drops the source whenever that one happens to be the
    # file this provider's bands are not in.
    tried = []
    for relative in candidates:
        path = target_dir / relative
        if not path.is_file():
            tried.append(f"{relative}: absent")
            continue
        try:
            frame = read_source_table(path)
            counts = match_counts(frame, spec["bands"],
                                  provider=spec["provider"])
        except Exception as err:
            tried.append(f"{relative}: {type(err).__name__}")
            continue
        ambiguous = sorted(b for b, n in counts.items() if n > 1)
        if ambiguous:
            raise ValueError(f"{path}: bands matching more than one row "
                             f"under the provider prefix: {ambiguous}")
        kept = [b for b in spec["bands"] if counts.get(b, 0) == 1]
        if not kept:
            tried.append(f"{relative}: no declared band")
            continue

        missing = [b for b in spec["bands"] if b not in kept]
        entry = {"path": relative, "bands": kept, "kind": spec["kind"],
                 "provider": spec["provider"]}
        return entry, {**row,
                       "status": STATUS_PARTIAL if missing else STATUS_KEPT,
                       "detail": f"absent: {missing}" if missing else "",
                       "bands": " ".join(kept)}

    return None, {**row, "status": STATUS_DROPPED,
                  "detail": "; ".join(tried), "bands": ""}


def _resolve_spherex(target_dir: Path, target: dict,
                     spec: dict) -> tuple[dict | None, dict]:
    """The target's SPHEREx table, by pattern rather than exact name.

    An extraction's filename carries a hash over its own configuration --
    the Sersic parameters among them -- so the tag differs per galaxy and
    no single name can be written down in advance. Patterns may glob, and
    are tried in order.

    A pattern matching SEVERAL tables is a hard error, not a pick: a
    galaxy with more than one extraction on disk is exactly the case
    where choosing silently would bury which one a flux came from.
    """
    row = {"target": target["name"], "source": "spherex", "bands": ""}
    patterns = spec["table"]
    if isinstance(patterns, str):
        patterns = [patterns]

    tried = []
    for pattern in patterns:
        relative = _format(pattern, target)
        matches = sorted(p for p in target_dir.glob(relative) if p.is_file())
        if len(matches) > 1:
            names = [str(p.relative_to(target_dir)) for p in matches]
            raise ValueError(
                f"target[{target['name']}].spherex: pattern {relative!r} "
                f"matches {len(matches)} tables {names}; name one "
                f"extraction rather than leaving the choice implicit")
        if not matches:
            tried.append(f"{relative}: no match")
            continue
        return ({"table": str(matches[0].relative_to(target_dir)),
                 "model": spec["model"],
                 "provenance": spec["provenance"]},
                {**row, "status": STATUS_KEPT, "detail": ""})
    return None, {**row, "status": STATUS_DROPPED, "detail": "; ".join(tried)}


def generate_roster(targets: list[dict], campaign: dict, *,
                    registry: Registry,
                    campaign_dir: str | Path) -> tuple[dict, list[dict]]:
    """Build a roster declaring only what the tree actually supplies.

    Parameters
    ----------
    targets : list of dict
        From read_sample_catalog.
    campaign : dict
        From load_campaign.
    registry : Registry
        Band-identity authority; a campaign band it does not know is an
        error, since no target could ever satisfy it.
    campaign_dir : str or Path
        Directory the campaign's data_root resolves against.

    Returns
    -------
    roster : dict
        Ready to write; every declaration is true of the tree.
    report : list of dict
        One row per (target, source) considered.
    """
    data_root = (Path(campaign_dir) / campaign["data_root"]).resolve()
    if not data_root.is_dir():
        raise ValueError(f"campaign: data_root does not exist: {data_root}")

    frame = campaign.get("position_frame") or "icrs"
    authority = campaign["position_authority"]
    spherex_spec = campaign.get("spherex")

    roster_targets: dict[str, dict] = {}
    report: list[dict] = []
    for target in targets:
        target_dir = (data_root / target["dir"]).resolve()
        if not target_dir.is_dir():
            report.append({"target": target["name"], "source": "-",
                           "status": STATUS_DROPPED,
                           "detail": f"no directory at {target['dir']}",
                           "bands": ""})
            continue

        sources: dict[str, dict] = {}
        for name, spec in campaign["sources"].items():
            entry, row = _resolve_source(target_dir, target, name, spec,
                                         registry)
            report.append(row)
            if entry is not None:
                sources[name] = entry
        if not sources:
            report.append({"target": target["name"], "source": "-",
                           "status": STATUS_DROPPED,
                           "detail": "no source survived", "bands": ""})
            continue

        entry = {
            "position": {"ra_deg": target["ra_deg"],
                         "dec_deg": target["dec_deg"],
                         "frame": frame, "authority": authority},
            "z_ref_kind": target["z_ref_kind"],
            "dir": target["dir"],
            "prefix": target["prefix"],
            "sources": sources,
        }
        if target["z_ref"] is not None:
            entry["z_ref"] = target["z_ref"]
        if spherex_spec is not None:
            found, row = _resolve_spherex(target_dir, target, spherex_spec)
            report.append(row)
            if found is not None:
                entry["spherex"] = found
        roster_targets[target["name"]] = entry

    if not roster_targets:
        # Naming the dominant reason matters more here than anywhere else:
        # the report is discarded with the exception, and the usual cause
        # is a data_root resolving somewhere real but wrong, which looks
        # identical to a tree with no photometry in it.
        reasons = collections.Counter(
            row["detail"].split(":")[0].strip() or row["status"]
            for row in report if row["status"] == STATUS_DROPPED)
        top = ", ".join(f"{n}x {reason}" for reason, n in reasons.most_common(3))
        raise ValueError(f"no target survived generation against data_root "
                         f"{data_root}: {top}")

    roster = {"schema_version": 2,
              "description": campaign.get("description"),
              "sample": campaign["sample"],
              "data_root": campaign["data_root"],
              "targets": roster_targets,
              "recipes": campaign["recipes"]}
    if campaign.get("reference_redshift") is not None:
        roster["reference_redshift"] = campaign["reference_redshift"]
    if campaign.get("manifest_path"):
        roster["manifest_path"] = campaign["manifest_path"]
    return roster, report


def write_roster(roster: dict, report: list[dict],
                 path: str | Path) -> tuple[Path, Path]:
    """Write the roster and its generation report."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(roster, indent=1, sort_keys=True) + "\n",
                    encoding="utf-8")

    report_path = path.with_suffix(path.suffix + ".report.csv")
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REPORT_COLUMNS))
        writer.writeheader()
        writer.writerows(report)
    return path, report_path
