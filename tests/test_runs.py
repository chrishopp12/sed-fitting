from __future__ import annotations

import json
import multiprocessing
from pathlib import Path

import pytest

from sedfit.core.provenance import sha256_bytes
from sedfit.core.runs import (
    append_row,
    finalize_run,
    locate_run,
    read_manifest,
    sanitize_label,
    stage_run,
)

MACHINERY = {"package_version": "0.1.0", "git_rev": "abc1234",
             "git_dirty": False, "fsps_libraries": None,
             "versions": {"numpy": "2.0.2"}}
PHOT = b"band,flux_uJy\nCFHT_u,10.0\n"


def _stage(tmp_path: Path, run_id: str = "a1b2c3d4e5f6", *,
           label: str | None = "test", machinery: dict = MACHINERY,
           force: bool = False) -> Path:
    return stage_run(tmp_path / "backend", run_id,
                     config={"name": "x"}, phot_bytes=PHOT,
                     phot_sha256=sha256_bytes(PHOT),
                     sidecar={"recipe": "r"}, machinery=machinery,
                     label=label, force=force)


def test_stage_skeleton(tmp_path) -> None:
    run_dir = _stage(tmp_path)
    assert run_dir.name == "a1b2c3d4e5f6-test"
    assert (run_dir / "config.json").exists()
    assert (run_dir / "phot.csv").read_bytes() == PHOT
    assert (run_dir / "phot.provenance.json").exists()
    assert (run_dir / "plots").is_dir()
    stamp = json.loads((run_dir / "manifest.json").read_text())
    assert stamp["status"] == "staged"
    assert stamp["git_rev"] == "abc1234"
    assert not list((tmp_path / "backend").glob(".staging*"))


def test_phot_hash_self_check(tmp_path) -> None:
    with pytest.raises(ValueError, match="hash mismatch"):
        stage_run(tmp_path / "backend", "ffffffffffff",
                  config={}, phot_bytes=PHOT, phot_sha256="0" * 64,
                  sidecar={}, machinery=MACHINERY)


def test_never_siblinged(tmp_path) -> None:
    first = _stage(tmp_path, label="alpha")
    assert locate_run(tmp_path / "backend", "a1b2c3d4e5f6") == first
    second = _stage(tmp_path, label="beta")
    dirs = [d for d in (tmp_path / "backend").iterdir() if d.is_dir()]
    assert dirs == [second]
    assert second.name == "a1b2c3d4e5f6-beta"


def test_overwrite_guard(tmp_path) -> None:
    _stage(tmp_path)
    drifted = dict(MACHINERY, git_rev="def5678")
    with pytest.raises(ValueError, match="different machinery"):
        _stage(tmp_path, machinery=drifted)
    replaced = _stage(tmp_path, machinery=drifted, force=True)
    stamp = json.loads((replaced / "manifest.json").read_text())
    assert stamp["git_rev"] == "def5678"


def test_finalize_and_read(tmp_path) -> None:
    run_dir = _stage(tmp_path)
    manifest = tmp_path / "runs.jsonl"
    row = {"run_id": "a1b2c3d4e5f6", "status": "ok", **MACHINERY}
    finalize_run(run_dir, manifest, row)
    assert json.loads((run_dir / "manifest.json").read_text())["status"] == "ok"
    rows, problems = read_manifest(manifest)
    assert rows == [row] and problems == []


def test_torn_final_line(tmp_path) -> None:
    manifest = tmp_path / "runs.jsonl"
    append_row(manifest, {"run_id": "one"})
    with open(manifest, "a") as fh:
        fh.write('{"run_id": "tw')
    rows, problems = read_manifest(manifest)
    assert len(rows) == 1
    assert problems and "torn" in problems[0]


def _append_many(args) -> None:
    path, worker = args
    for i in range(25):
        append_row(path, {"worker": worker, "i": i})


def test_concurrent_appends(tmp_path) -> None:
    manifest = tmp_path / "runs.jsonl"
    with multiprocessing.get_context("spawn").Pool(4) as pool:
        pool.map(_append_many, [(str(manifest), w) for w in range(4)])
    rows, problems = read_manifest(manifest)
    assert len(rows) == 100 and problems == []


def test_sanitize_label() -> None:
    assert sanitize_label("BCG tilt/c3k") == "BCG-tilt-c3k"
    with pytest.raises(ValueError):
        sanitize_label("///")
