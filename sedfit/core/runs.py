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
      config.json  manifest.json  plots/
      phot.csv  phot.provenance.json          (a photometry backend)
      spectrum.csv  spectrum.provenance.json  (whenever a spectrum is fit)
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
import time
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
# The manifest lock's sentinel byte, past any real manifest, and how long
# to wait for it. Windows locks a byte range; POSIX ignores the offset.
LOCK_OFFSET = 1 << 40
LOCK_TIMEOUT_S = 120.0
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
        machinery: dict,
        phot_bytes: bytes | None = None,
        phot_sha256: str | None = None,
        sidecar: dict | None = None,
        label: str | None = None,
        force: bool = False,
        spectrum_bytes: bytes | None = None,
        spectrum_sha256: str | None = None,
        spectrum_sidecar: dict | None = None,
    ) -> Path:
    """Create the run directory with its skeleton; returns the final path.

    Parameters
    ----------
    config : dict
        The fully resolved config (echoed verbatim as config.json).
    phot_bytes, phot_sha256, sidecar : optional
        The photometry file bytes (read exactly once by the caller),
        their sha256, and the build provenance sidecar; the written copy
        is re-hashed as a self-check. None for a backend that fits no
        photometry table, which then writes no phot.csv. [default: None]
    machinery : dict
        Live machinery stamps (MACHINERY_KEYS subset); compared against
        an existing run's stamps for the overwrite guard.
    force : bool
        Allow replacing an existing run whose machinery stamps differ.
        [default: False]
    spectrum_bytes, spectrum_sha256, spectrum_sidecar : optional
        A joint fit's spectrum file bytes, their sha256 (self-checked
        like the photometry), and its provenance sidecar; staged as
        spectrum.csv and spectrum.provenance.json. [default: None]
    """
    if phot_bytes is None and spectrum_bytes is None:
        raise ValueError("stage_run needs photometry, a spectrum, or both; a "
                         "run directory describing neither is not a record "
                         "of anything")
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
    if phot_bytes is not None:
        (staging / "phot.csv").write_bytes(phot_bytes)
        written = sha256_bytes((staging / "phot.csv").read_bytes())
        if written != phot_sha256:
            shutil.rmtree(staging)
            raise ValueError(f"phot.csv hash mismatch after write: {written} "
                             f"!= {phot_sha256}")
        (staging / "phot.provenance.json").write_text(
            json.dumps(sidecar or {}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
    if spectrum_bytes is not None:
        (staging / "spectrum.csv").write_bytes(spectrum_bytes)
        written = sha256_bytes((staging / "spectrum.csv").read_bytes())
        if written != spectrum_sha256:
            shutil.rmtree(staging)
            raise ValueError(f"spectrum.csv hash mismatch after write: "
                             f"{written} != {spectrum_sha256}")
        (staging / "spectrum.provenance.json").write_text(
            json.dumps(spectrum_sidecar or {}, indent=2, sort_keys=True)
            + "\n", encoding="utf-8")
    (staging / "plots").mkdir()
    stamp = {"run_id": run_id, "status": "staged", "written": now_iso(),
             "phot_sha256_16": phot_sha256[:16] if phot_sha256 else None,
             **machinery}
    (staging / "manifest.json").write_text(
        json.dumps(stamp, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    os.replace(staging, final_dir)
    return final_dir


# ------------------------------------
# Finalizing and the central manifest
# ------------------------------------

@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    """Hold an exclusive lock on path for the duration of the block.

    The lock is taken on its own file descriptor, so the caller is free
    to open the same file separately for appending.

    Windows byte-range locks are mandatory and are enforced against
    every handle, including other handles in the same process, so the
    locked byte sits at LOCK_OFFSET -- far past any manifest -- and the
    append never collides with it. POSIX uses whole-file flock, where
    the offset is irrelevant.
    """
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        if sys.platform == "win32":
            _lock_windows(fd)
        else:
            fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            if sys.platform == "win32":
                os.lseek(fd, LOCK_OFFSET, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _lock_windows(fd: int) -> None:
    """Block on the sentinel byte until the lock is ours.

    msvcrt.locking gives up after about ten seconds, which a busy batch
    can exceed, so the wait is retried up to LOCK_TIMEOUT_S.
    """
    deadline = time.monotonic() + LOCK_TIMEOUT_S
    while True:
        os.lseek(fd, LOCK_OFFSET, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
            return
        except OSError:
            if time.monotonic() >= deadline:
                raise


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
