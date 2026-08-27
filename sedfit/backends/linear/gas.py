"""
gas.py

Analytic Emission-Line Columns
---------------------------------------------------------

Nebular lines as extra NNLS columns, built as Gaussians at observed
wavelength rather than resampled from template files. Each column carries
unit integrated flux, so its amplitude reads as a line flux.

Data products:
    sedfit/data/lines/<set>.txt
        two whitespace-separated columns, component name and rest VACUUM
        wavelength [A]; blank lines and '#' comments ignored

Requirements:
    numpy

Notes:
    Rationale in DESIGN.md section 17.1.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from sedfit.backends.linear.basis import C_KMS
from sedfit.core.fitconfig import LINEAR_GAS_RATIO_LOCKED

PACKAGED_LINE_DIR = Path(__file__).resolve().parents[2] / "data" / "lines"
LINE_SUFFIX = ".txt"

# Intrinsic nebular dispersion. A symmetric kernel does not move a line
# centroid, so the redshift is insensitive to this at first order; it sets
# the amplitude, which is why it is recorded.
GAS_SIGMA_KMS_DEFAULT = 100.0

# Gaussian half-width beyond which a line contributes nothing measurable.
_HALF_WIDTHS = 6.0
_SQRT_2PI = np.sqrt(2.0 * np.pi)


def packaged_line_lists() -> list[str]:
    """Names of the line lists shipped with the package, sorted."""
    if not PACKAGED_LINE_DIR.is_dir():
        return []
    return sorted(p.stem for p in PACKAGED_LINE_DIR.glob(f"*{LINE_SUFFIX}"))


def resolve_line_list(lines: str | Path) -> Path:
    """The file a config's `gas.lines` field names.

    Parameters
    ----------
    lines : str or Path
        A filesystem path, or the name of a packaged list.

    Returns
    -------
    path : Path
        The resolved file.
    """
    spec = Path(lines).expanduser()
    if spec.is_file():
        return spec
    packaged = PACKAGED_LINE_DIR / f"{lines}{LINE_SUFFIX}"
    if packaged.is_file():
        return packaged
    raise ValueError(f"gas.lines={str(lines)!r} is not a file or a packaged "
                     f"list; packaged lists are {packaged_line_lists()}")


def read_line_list(path: str | Path) -> dict[str, float]:
    """Component names and their rest vacuum wavelengths, in file order."""
    path = Path(path)
    lines: dict[str, float] = {}
    for number, text in enumerate(path.read_text(encoding="utf-8").splitlines(),
                                  start=1):
        text = text.split("#", 1)[0].strip()
        if not text:
            continue
        fields = text.split()
        if len(fields) != 2:
            raise ValueError(f"{path}:{number}: expected 'name wavelength', "
                             f"got {text!r}")
        name, wave = fields[0], float(fields[1])
        if name in lines:
            raise ValueError(f"{path}:{number}: {name!r} is listed twice")
        if not np.isfinite(wave) or wave <= 0:
            raise ValueError(f"{path}:{number}: {name!r} has wavelength {wave}")
        lines[name] = wave
    if not lines:
        raise ValueError(f"{path}: no lines")
    return lines


class GasBasis:
    """Emission-line columns for the NNLS design matrix.

    One column per ratio-locked group and one per remaining free line, each
    normalized to unit integrated flux so its amplitude is a line flux in the
    fit's flux unit times Angstrom.

    Parameters
    ----------
    lines : dict[str, float]
        Component names to rest vacuum wavelengths [A].
    sigma_kms : float
        Intrinsic nebular dispersion. [default: 100.0]
    ratio_locked : sequence of dict or None
        Groups entering as one column, each with 'name', 'lines' and
        'ratios'. None takes RATIO_LOCKED_DEFAULT; an empty sequence locks
        nothing. [default: None]
    """

    def __init__(self, lines: dict[str, float], *,
                 sigma_kms: float = GAS_SIGMA_KMS_DEFAULT,
                 ratio_locked=None) -> None:
        if sigma_kms <= 0:
            raise ValueError(f"gas sigma_kms must be positive, got {sigma_kms}")
        self.lines = dict(lines)
        self.sigma_kms = float(sigma_kms)
        groups = (LINEAR_GAS_RATIO_LOCKED if ratio_locked is None
                  else tuple(ratio_locked))

        self.names: list[str] = []
        self.components: list[list[tuple[float, float]]] = []
        locked: dict[str, str] = {}
        for group in groups:
            name, members = group["name"], list(group["lines"])
            present = [m for m in members if m in self.lines]
            # A list that simply has no [OIII] is not an error; one carrying
            # half a locked pair is, since the present member would be fit
            # free and the lock would silently not apply.
            if not present:
                continue
            if len(present) < len(members):
                raise ValueError(
                    f"gas group {name!r} names {sorted(set(members) - set(present))}, "
                    f"which are not in the line list")
            ratios = np.asarray(group["ratios"], float)
            if ratios.size != len(members):
                raise ValueError(f"gas group {name!r}: {len(members)} lines "
                                 f"against {ratios.size} ratios")
            if not np.all(np.isfinite(ratios)) or not np.all(ratios > 0):
                raise ValueError(f"gas group {name!r}: every ratio must be "
                                 f"finite and positive")
            for member in members:
                if member in locked:
                    raise ValueError(f"{member!r} is in both {locked[member]!r} "
                                     f"and {name!r}")
                locked[member] = name
            if name in self.names:
                raise ValueError(f"gas group {name!r} is declared twice")
            self.names.append(name)
            self.components.append(
                [(self.lines[m], float(r / ratios.sum()))
                 for m, r in zip(members, ratios)])

        for name, rest in self.lines.items():
            if name in locked:
                continue
            if name in self.names:
                raise ValueError(f"{name!r} is both a line and a group name")
            self.names.append(name)
            self.components.append([(rest, 1.0)])

    @property
    def n_columns(self) -> int:
        return len(self.names)

    def design(self, wave_vac_obs: np.ndarray, redshift: float, *,
               lsf_sigma_kms: np.ndarray | None = None) -> np.ndarray:
        """(n_columns, npix) line rows at the observed vacuum wavelengths.

        Parameters
        ----------
        wave_vac_obs : np.ndarray
            Observed vacuum wavelengths [A], increasing.
        redshift : float
            The redshift the lines sit at.
        lsf_sigma_kms : np.ndarray or None
            Instrument line-spread width over `wave_vac_obs`, added in
            quadrature to the nebular dispersion. [default: None]

        Returns
        -------
        rows : np.ndarray
            One row per column, zero where a line falls outside the band.
        """
        wave = np.asarray(wave_vac_obs, float)
        rows = np.zeros((self.n_columns, wave.size))
        for row, component in zip(rows, self.components):
            for rest, weight in component:
                centre = rest * (1.0 + redshift)
                sigma_kms = self.sigma_kms
                if lsf_sigma_kms is not None:
                    sigma_kms = float(np.hypot(
                        sigma_kms, np.interp(centre, wave, lsf_sigma_kms)))
                sigma_A = centre * sigma_kms / C_KMS
                lo = int(np.searchsorted(wave, centre - _HALF_WIDTHS * sigma_A))
                hi = int(np.searchsorted(wave, centre + _HALF_WIDTHS * sigma_A))
                if hi <= lo:
                    continue
                offset = (wave[lo:hi] - centre) / sigma_A
                row[lo:hi] += weight * np.exp(-0.5 * offset ** 2) / (
                    _SQRT_2PI * sigma_A)
        return rows
