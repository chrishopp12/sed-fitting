from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from sedfit.core.provenance import (
    RUN_ID_HEX_LEN,
    canonical_json,
    git_state,
    run_id,
    sha256_bytes,
    sha256_file,
)


@dataclass
class _Cfg:
    name: str = "x"
    z_max: float = 0.16
    seed: int | None = None
    bands: tuple = ("a", "b")


FIXTURE = {
    "b": 1,
    "a": {"y": None, "x": np.float64(0.5), "n": np.int32(3), "flag": np.bool_(True)},
    "cfg": _Cfg(),
    "list": [1, 2.5, "s", True],
}

GOLDEN_CANONICAL = ('{"a":{"flag":true,"n":3,"x":0.5},"b":1,'
                    '"cfg":{"bands":["a","b"],"name":"x","z_max":0.16},'
                    '"list":[1,2.5,"s",true]}')
GOLDEN_RUN_ID = "4d0a93344beb"


def test_canonical_golden() -> None:
    assert canonical_json(FIXTURE) == GOLDEN_CANONICAL


def test_run_id_golden() -> None:
    rid = run_id(FIXTURE, "p" * 64, "q" * 64)
    assert rid == GOLDEN_RUN_ID
    assert len(rid) == RUN_ID_HEX_LEN


def test_run_id_sensitivity() -> None:
    base = run_id(FIXTURE, "p" * 64, "q" * 64)
    assert run_id(FIXTURE, "r" * 64, "q" * 64) != base
    assert run_id(FIXTURE, "p" * 64, "r" * 64) != base
    assert run_id({**FIXTURE, "b": 2}, "p" * 64, "q" * 64) != base


def test_null_equals_absent() -> None:
    assert canonical_json({"a": 1, "b": None}) == canonical_json({"a": 1})


def test_rejections() -> None:
    with pytest.raises(ValueError):
        canonical_json({"x": float("nan")})
    with pytest.raises(ValueError):
        canonical_json({"x": float("inf")})
    with pytest.raises(ValueError):
        canonical_json({"x": [None]})
    with pytest.raises(TypeError):
        canonical_json({1: "x"})
    with pytest.raises(TypeError):
        canonical_json({"x": {2, 3}})
    with pytest.raises(TypeError):
        canonical_json({"x": np.arange(3)})
    with pytest.raises(TypeError):
        canonical_json({"x": Path("/")})


def test_sha256_helpers(tmp_path: Path) -> None:
    payload = tmp_path / "x.bin"
    payload.write_bytes(b"abc")
    assert sha256_file(payload) == sha256_bytes(b"abc")
    assert len(sha256_bytes(b"")) == 64


def test_git_state() -> None:
    repo = Path(__file__).resolve().parent.parent
    state = git_state(repo)
    assert state["rev"] is None or len(state["rev"]) >= 7
    assert git_state(Path("/")) == {"rev": None, "dirty": None}
