"""
runs.py

Run Directories and the Central Manifest
---------------------------------------------------------

Run-directory lifecycle and the append-only central manifest. A run
directory is staged under a temporary name with its full skeleton
(resolved config echo, byte-copied photometry, build sidecar, an
initial manifest stamp) and atomically renamed before the fit runs, so
a crashed fit leaves a self-describing directory. An existing directory
with the same run_id is the directory -- reused or replaced, never
siblinged -- and replacing it when the machinery stamps differ from the
live ones requires force.

Data products:
  <backend_dir>/<run_id[-label]>/
      config.json  phot.csv  phot.provenance.json  manifest.json  plots/
  <manifest_path>          one JSON row appended per finalized run

Requirements:
  - (stdlib only)
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from sedfit.core.provenance import sha256_bytes

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

# ------------------------------------
# Constants
# ------------------------------------

MACHINERY_KEYS = ("package_version", "git_rev", "git_dirty",
                  "fsps_libraries", "versions")
_LABEL_OK = re.compile(r"[^A-Za-z0-9._-]+")


# ------------------------------------
# Helpers
# ------------------------------------

def sanitize_label(label: str) -> str:
    """Reduce a label to filesystem-safe characters."""
    cleaned = _LABEL_OK.sub("-", label).strip("-")
    if not cleaned:
        raise ValueError(f"label {label!r} has no filesystem-safe characters")
    return cleaned


def now_iso() -> str:
    """The current local time as a seconds-resolution ISO 8601 string."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def locate_run(backend_dir: Path, run_id: str) -> Path | None:
    """The existing directory for a run_id, if any (never more than one)."""
    matches = sorted(Path(backend_dir).glob(f"{run_id}*"))
    matches = [m for m in matches if m.is_dir()]
    if len(matches) > 1:
        raise ValueError(f"{backend_dir}: multiple directories for run_id "
                         f"{run_id}: {[m.name for m in matches]}")
    return matches[0] if matches else None


def _machinery_of(run_dir: Path) -> dict | None:
    manifest = run_dir / "manifest.json"
    if not manifest.exists():
        return None
    row = json.loads(manifest.read_text(encoding="utf-8"))
    return {key: row.get(key) for key in MACHINERY_KEYS}


# ------------------------------------
# Staging
# ------------------------------------

def stage_run(
        backend_dir: str | Path,
        run_id: str,
        *,
        config: dict,
        phot_bytes: bytes,
        phot_sha256: str,
        sidecar: dict,
        machinery: dict,
        label: str | None = None,
        force: bool = False,
    ) -> Path:
    """Create the run directory with its skeleton; returns the final path.

    Parameters
    ----------
    config : dict
        The fully resolved config (echoed verbatim as config.json).
    phot_bytes, phot_sha256 : bytes, str
        The photometry file bytes (read exactly once by the caller) and
        their sha256; the written copy is re-hashed as a self-check.
    sidecar : dict
        The build provenance sidecar to copy alongside the photometry.
    machinery : dict
        Live machinery stamps (MACHINERY_KEYS subset); compared against
        an existing run's stamps for the overwrite guard.
    force : bool
        Allow replacing an existing run whose machinery stamps differ.
        [default: False]
    """
    backend_dir = Path(backend_dir)
    backend_dir.mkdir(parents=True, exist_ok=True)

    existing = locate_run(backend_dir, run_id)
    if existing is not None:
        stored = _machinery_of(existing) or {}
        live = {key: machinery.get(key) for key in MACHINERY_KEYS}
        stored = {key: stored.get(key) for key in MACHINERY_KEYS}
        if stored != live and not force:
            raise ValueError(
                f"{existing.name}: existing run was produced by different "
                f"machinery (stored {stored}, live {live}); pass force to "
                f"replace it")
        shutil.rmtree(existing)

    name = run_id if label is None else f"{run_id}-{sanitize_label(label)}"
    final_dir = backend_dir / name
    staging = backend_dir / f".staging-{run_id}-{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()

    (staging / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (staging / "phot.csv").write_bytes(phot_bytes)
    written = sha256_bytes((staging / "phot.csv").read_bytes())
    if written != phot_sha256:
        shutil.rmtree(staging)
        raise ValueError(f"phot.csv hash mismatch after write: {written} != "
                         f"{phot_sha256}")
    (staging / "phot.provenance.json").write_text(
        json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (staging / "plots").mkdir()
    stamp = {"run_id": run_id, "status": "staged", "written": now_iso(),
             "phot_sha256_16": phot_sha256[:16], **machinery}
    (staging / "manifest.json").write_text(
        json.dumps(stamp, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    os.replace(staging, final_dir)
    return final_dir


# ------------------------------------
# Finalizing and the central manifest
# ------------------------------------

@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    """Hold an exclusive lock on the first byte of path for the block.

    The lock is taken on its own file descriptor, so the caller is free
    to open the same file separately for appending. Windows uses
    msvcrt.locking, every other platform fcntl.flock.
    """
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        if sys.platform == "win32":
            msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
        else:
            fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            if sys.platform == "win32":
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def append_row(manifest_path: str | Path, row: dict) -> None:
    """Append one complete JSON line under an exclusive lock."""
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, sort_keys=True) + "\n"
    with _exclusive_lock(manifest_path):
        fd = os.open(manifest_path,
                     os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, line.encode("utf-8"))
        finally:
            os.close(fd)


def finalize_run(run_dir: str | Path, manifest_path: str | Path,
                 row: dict) -> None:
    """Write the run's manifest.json, then append the central row."""
    run_dir = Path(run_dir)
    (run_dir / "manifest.json").write_text(
        json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    append_row(manifest_path, row)


def read_manifest(manifest_path: str | Path) -> tuple[list[dict], list[str]]:
    """All manifest rows plus problems (a torn final line is reported)."""
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        return [], []
    rows, problems = [], []
    lines = manifest_path.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                problems.append(f"torn final line ({len(line)} chars) ignored")
            else:
                problems.append(f"unparseable line {i + 1}")
    return rows, problems
