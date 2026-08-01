"""
registry.py

Band Registry
---------------------------------------------------------

Single authority for band identity: instrument membership and bandpass
curves. Every band carries exactly one of a sedpy filter name or a
packaged curve file. SPHEREx channels are not registry entries (their
geometry is per-object, carried by the SED table); they resolve to the
SPHEREx instrument by prefix, and their bandpasses come from
core.synth, not from here.

Requirements:
  - numpy, sedpy

Notes:
  - Effective wavelengths are computed (pivot wavelength), never stored.
  - All loader violations are hard errors: unknown keys, unknown
    instruments, sedpy-XOR-curve violations, missing curve files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sedfit.core.provenance import canonical_json, sha256_bytes, sha256_file
from sedfit.core.synth import pivot_wave_AA as _pivot_from_arrays

# ------------------------------------
# Constants
# ------------------------------------

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_REGISTRY = DATA_DIR / "registry.json"

SCHEMA_VERSIONS = (1,)
PIVOT_BOUNDS_AA = (500.0, 1.0e6)

TOP_KEYS = ("schema_version", "instruments", "spherex", "bands")
SPHEREX_KEYS = ("prefix", "instrument")
BAND_KEYS = ("instrument", "sedpy", "curve")


# ------------------------------------
# Helpers
# ------------------------------------

def _require_exact_keys(label: str, mapping: dict, allowed: tuple[str, ...],
                        required: tuple[str, ...]) -> None:
    unknown = sorted(set(mapping) - set(allowed))
    if unknown:
        raise ValueError(f"{label}: unknown keys {unknown}")
    missing = sorted(set(required) - set(mapping))
    if missing:
        raise ValueError(f"{label}: missing keys {missing}")


@dataclass(frozen=True)
class Band:
    """One registry band: instrument plus exactly one bandpass source."""
    instrument: str
    sedpy: str | None = None
    curve: str | None = None


# ------------------------------------
# Registry
# ------------------------------------

class Registry:
    """Band-identity authority loaded from a registry JSON file.

    Parameters
    ----------
    path : str or Path
        Registry JSON; curve paths resolve relative to its directory.
        [default: the packaged sedfit/data/registry.json]
    """

    def __init__(self, path: str | Path = DEFAULT_REGISTRY):
        self.path = Path(path)
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        _require_exact_keys("registry", raw, TOP_KEYS, TOP_KEYS)

        if raw["schema_version"] not in SCHEMA_VERSIONS:
            raise ValueError(f"registry: unrecognized schema_version "
                             f"{raw['schema_version']!r} (known: {SCHEMA_VERSIONS})")

        instruments = raw["instruments"]
        if (not isinstance(instruments, list) or not instruments
                or len(set(instruments)) != len(instruments)
                or not all(isinstance(i, str) for i in instruments)):
            raise ValueError("registry: instruments must be a non-empty list "
                             "of unique strings")
        self.instruments: tuple[str, ...] = tuple(instruments)

        spherex = raw["spherex"]
        _require_exact_keys("registry.spherex", spherex, SPHEREX_KEYS, SPHEREX_KEYS)
        if spherex["instrument"] not in self.instruments:
            raise ValueError(f"registry.spherex: instrument "
                             f"{spherex['instrument']!r} not in instruments")
        if not isinstance(spherex["prefix"], str) or not spherex["prefix"]:
            raise ValueError("registry.spherex: prefix must be a non-empty string")
        self.spherex_prefix: str = spherex["prefix"]
        self.spherex_instrument: str = spherex["instrument"]

        self._bands: dict[str, Band] = {}
        for name, entry in raw["bands"].items():
            label = f"registry.bands[{name}]"
            if name.startswith(self.spherex_prefix):
                raise ValueError(f"{label}: name collides with the SPHEREx "
                                 f"channel prefix {self.spherex_prefix!r}")
            _require_exact_keys(label, entry, BAND_KEYS, ("instrument",))
            if entry["instrument"] not in self.instruments:
                raise ValueError(f"{label}: unknown instrument "
                                 f"{entry['instrument']!r}")
            has_sedpy = "sedpy" in entry
            has_curve = "curve" in entry
            if has_sedpy == has_curve:
                raise ValueError(f"{label}: exactly one of 'sedpy' or 'curve' "
                                 f"is required")
            if has_curve and not (self.path.parent / entry["curve"]).exists():
                raise ValueError(f"{label}: curve file not found: "
                                 f"{self.path.parent / entry['curve']}")
            self._bands[name] = Band(instrument=entry["instrument"],
                                     sedpy=entry.get("sedpy"),
                                     curve=entry.get("curve"))

    # ------------------------------------
    # Membership and identity
    # ------------------------------------

    @property
    def bands(self) -> tuple[str, ...]:
        """Every registry band name, in file order."""
        return tuple(self._bands)

    def __contains__(self, band: str) -> bool:
        return band in self._bands

    def is_spherex(self, band: str) -> bool:
        """True when the name carries the registry's SPHEREx prefix."""
        return band.startswith(self.spherex_prefix)

    def instrument_of(self, band: str) -> str:
        """Instrument of a registry band or SPHEREx channel; KeyError otherwise."""
        if band in self._bands:
            return self._bands[band].instrument
        if self.is_spherex(band):
            return self.spherex_instrument
        raise KeyError(f"unknown band: {band!r} (not in the registry and not "
                       f"a {self.spherex_prefix}* channel)")

    def require_known(self, bands: list[str] | tuple[str, ...]) -> None:
        """Hard error listing every band that the registry cannot identify."""
        unknown = [b for b in bands
                   if b not in self._bands and not self.is_spherex(b)]
        if unknown:
            raise ValueError(f"unknown bands: {unknown}")

    # ------------------------------------
    # Bandpasses
    # ------------------------------------

    def get_bandpass(self, band: str) -> tuple[np.ndarray, np.ndarray]:
        """Bandpass arrays for a registry band.

        Parameters
        ----------
        band : str
            Registry band name. SPHEREx channels are rejected: their
            tophats are built per object by core.synth.

        Returns
        -------
        wave_AA, throughput : np.ndarray
            Wavelength grid [Angstrom], ascending, and dimensionless
            throughput.
        """
        if band not in self._bands:
            raise KeyError(f"unknown band: {band!r}"
                           + (" (SPHEREx channels resolve via core.synth)"
                              if self.is_spherex(band) else ""))
        entry = self._bands[band]
        if entry.sedpy is not None:
            from sedpy.observate import Filter
            filt = Filter(entry.sedpy)
            wave = np.asarray(filt.wavelength, dtype=float)
            thru = np.asarray(filt.transmission, dtype=float)
        else:
            data = np.loadtxt(self.path.parent / entry.curve, dtype=float)
            wave, thru = data[:, 0], data[:, 1]

        if wave.ndim != 1 or wave.size < 2 or wave.shape != thru.shape:
            raise ValueError(f"{band}: malformed bandpass arrays")
        if not (np.isfinite(wave).all() and np.isfinite(thru).all()):
            raise ValueError(f"{band}: non-finite bandpass values")
        order = np.argsort(wave)
        wave, thru = wave[order], thru[order]
        if not (thru > 0).any():
            raise ValueError(f"{band}: bandpass has no positive throughput")
        return wave, thru

    def pivot_wave_AA(self, band: str) -> float:
        """Pivot wavelength [Angstrom], computed from the bandpass."""
        wave, thru = self.get_bandpass(band)
        pivot = _pivot_from_arrays(wave, thru)
        lo, hi = PIVOT_BOUNDS_AA
        if not (lo < pivot < hi):
            raise ValueError(f"{band}: pivot wavelength {pivot:.1f} A outside "
                             f"sanity bounds ({lo}, {hi})")
        return pivot

    # ------------------------------------
    # Identity hashes
    # ------------------------------------

    def per_band_hashes(self) -> dict[str, str]:
        """sha256 per band over its resolved bandpass arrays."""
        hashes = {}
        for band in self._bands:
            wave, thru = self.get_bandpass(band)
            payload = (np.ascontiguousarray(wave, dtype=np.float64).tobytes()
                       + b"|"
                       + np.ascontiguousarray(thru, dtype=np.float64).tobytes())
            hashes[band] = sha256_bytes(payload)
        return hashes

    def bandpass_hash(self) -> str:
        """One digest over the registry file content and every band's arrays."""
        payload = {"registry": sha256_file(self.path),
                   "bands": self.per_band_hashes()}
        return sha256_bytes(canonical_json(payload).encode("ascii"))


def load_registry(path: str | Path | None = None) -> Registry:
    """Load the packaged registry, or one at an explicit path."""
    return Registry(DEFAULT_REGISTRY if path is None else path)
