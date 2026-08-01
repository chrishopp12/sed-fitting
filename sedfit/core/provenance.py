"""
provenance.py

Run Identity and Provenance Primitives
---------------------------------------------------------

Canonical JSON, content hashing, git state, and run_id construction.
The canonical form is version-stable: changing it changes every run
identity.

Requirements:
  - numpy

Notes:
  - Canonical form: sorted keys, compact separators, ASCII, dataclasses
    materialized with their defaults, None-valued dict keys dropped
    (null == absent), tuples become lists, numpy scalars become Python
    scalars. NaN/inf, None inside a list, non-string dict keys, and any
    other type are hard errors.
  - run_id: first RUN_ID_HEX_LEN hex digits of sha256 over the canonical
    JSON of {"config": config, "phot": phot_sha256,
    "bandpass": bandpass_sha256}.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

# ------------------------------------
# Constants
# ------------------------------------

RUN_ID_HEX_LEN = 12


# ------------------------------------
# Canonical JSON
# ------------------------------------

def _normalize(obj: object) -> object:
    """Reduce obj to plain JSON types under the frozen canonical rules."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _normalize(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            if not isinstance(key, str):
                raise TypeError(f"non-string dict key has no canonical form: {key!r}")
            if value is None:
                continue
            out[key] = _normalize(value)
        return out
    if isinstance(obj, (list, tuple)):
        items = []
        for value in obj:
            if value is None:
                raise ValueError("None inside a list has no canonical form")
            items.append(_normalize(value))
        return items
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (np.bool_, np.integer, np.floating)):
        return _normalize(obj.item())
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        if not np.isfinite(obj):
            raise ValueError(f"non-finite float has no canonical form: {obj!r}")
        return obj
    if isinstance(obj, str):
        return obj
    raise TypeError(f"type with no canonical form: {type(obj).__name__}")


def canonical_json(obj: object) -> str:
    """Serialize obj as its canonical JSON string.

    Parameters
    ----------
    obj : object
        JSON-like structure (dicts, lists/tuples, scalars, dataclasses,
        numpy scalars).

    Returns
    -------
    text : str
        The canonical serialization.
    """
    return json.dumps(_normalize(obj), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


# ------------------------------------
# Hashing
# ------------------------------------

def sha256_bytes(data: bytes) -> str:
    """Full sha256 hex digest of a byte string."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Full sha256 hex digest of a file's bytes."""
    return sha256_bytes(Path(path).read_bytes())


def run_id(config: object, phot_sha256: str, bandpass_sha256: str) -> str:
    """Run identity from the three scientific inputs.

    Parameters
    ----------
    config : object
        Hash-projected resolved fit config (canonical-JSON-able).
    phot_sha256 : str
        sha256 hex of the photometry file bytes.
    bandpass_sha256 : str
        sha256 hex over the resolved registry and bandpass arrays.

    Returns
    -------
    run_id : str
        First RUN_ID_HEX_LEN hex digits of the identity hash.
    """
    payload = {"config": config, "phot": phot_sha256, "bandpass": bandpass_sha256}
    return sha256_bytes(canonical_json(payload).encode("ascii"))[:RUN_ID_HEX_LEN]


# ------------------------------------
# Git state
# ------------------------------------

def git_state(repo_dir: str | Path) -> dict:
    """Short HEAD revision and dirty flag for a repository directory.

    Returns
    -------
    state : dict
        {"rev": str | None, "dirty": bool | None}; dirty is True when
        `git status --porcelain` reports anything, untracked files
        included. Nones when repo_dir is not inside a git repository or
        git is unavailable.
    """
    try:
        rev = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(repo_dir), "status", "--porcelain"],
            capture_output=True, text=True, check=True,
        ).stdout
        return {"rev": rev, "dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"rev": None, "dirty": None}
