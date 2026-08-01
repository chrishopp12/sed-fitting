"""
templates.py

Template Set and TEF Resolution
---------------------------------------------------------

Resolves the config's template selection to an eazy .param file and the
template-error curve, and produces the content digests that enter run
identity: editing a template or TEF curve changes the run_id even though
no config text changed.

A config's `templates` field takes either a filesystem path (a directory
of spectra or an eazy .param file) or the bare name of a set packaged
under sedfit/data/templates/, so a config carries no machine-specific
path. Paths are tried first.

Requirements:
  - (stdlib only)
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

from sedfit.core.provenance import sha256_bytes, sha256_file

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
PACKAGED_TEMPLATE_DIR = DATA_DIR / "templates"
DEFAULT_TEF_FILE = DATA_DIR / "TEMPLATE_ERROR.eazy_v1.0"


def packaged_template_sets() -> list[str]:
    """Names of the template sets shipped with the package, sorted."""
    if not PACKAGED_TEMPLATE_DIR.is_dir():
        return []
    return sorted(p.name for p in PACKAGED_TEMPLATE_DIR.iterdir()
                  if p.is_dir())


def resolve_spectra(directory: str | Path, *,
                    pattern: str = "*_spec.dat") -> list[Path]:
    """Sorted spectrum paths in a template directory.

    Falls back to *.dat when the pattern matches nothing, so a generic
    directory of two-column spectra works without configuration. A
    pattern that matches only part of the directory raises a
    UserWarning, since a basis silently missing members is a scientific
    error rather than a preference.
    """
    directory = Path(directory).expanduser()
    spectra = sorted(directory.glob(pattern))
    if not spectra and pattern != "*.dat":
        spectra = sorted(directory.glob("*.dat"))
    if not spectra:
        raise ValueError(f"no template spectra matching {pattern!r} "
                         f"(or *.dat) in {directory}")

    available = sorted(directory.glob("*.dat"))
    if len(spectra) < len(available):
        warnings.warn(
            f"{directory.name}: template_pattern {pattern!r} selects "
            f"{len(spectra)} of {len(available)} spectra; set "
            f'"template_pattern": "*.dat" to fit the whole set',
            stacklevel=2)
    return spectra


def resolve_template_source(templates: str | Path) -> Path:
    """The directory or .param file a config's `templates` field names.

    Parameters
    ----------
    templates : str or Path
        A filesystem path, or the name of a packaged template set.

    Returns
    -------
    source : Path
        The resolved directory or .param file.
    """
    spec = Path(templates).expanduser()
    if spec.exists():
        return spec
    packaged = PACKAGED_TEMPLATE_DIR / str(templates)
    if packaged.is_dir():
        return packaged
    raise ValueError(
        f"templates={str(templates)!r} is not a directory, a .param file, "
        f"or a packaged set; packaged sets are "
        f"{packaged_template_sets()}")


def resolve_templates(eazy_cfg: dict) -> tuple[list[Path], Path | None]:
    """The spectrum list (directory mode) or an existing .param file.

    Returns
    -------
    spectra : list of Path
        Template spectra (empty when a .param file is used directly).
    param_file : Path or None
        An existing .param file, or None in directory mode.
    """
    source = resolve_template_source(eazy_cfg["templates"])
    if source.is_file() and source.suffix == ".param":
        return [], source.resolve()
    if source.is_dir():
        return (resolve_spectra(source, pattern=eazy_cfg["template_pattern"]),
                None)
    raise ValueError(f"templates={eazy_cfg['templates']!r} is neither a "
                     f".param file nor a directory")


def resolve_tef(eazy_cfg: dict) -> Path:
    """The template-error curve for this fit."""
    tef_file = eazy_cfg["tef_file"]
    path = Path(tef_file).expanduser() if tef_file else DEFAULT_TEF_FILE
    if not path.is_file():
        raise ValueError(f"TEF file not found: {path}")
    return path.resolve()


def write_templates_param(spectra: list[Path], out_path: str | Path) -> Path:
    """Write an eazy templates .param file: <n> <absolute path> 1.0 rows."""
    out_path = Path(out_path)
    lines = [f"{i} {path.resolve()} 1.0"
             for i, path in enumerate(spectra, start=1)]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def prepare_templates_param(eazy_cfg: dict, run_dir: str | Path) -> Path:
    """The .param path handed to eazy as TEMPLATES_FILE for this run."""
    spectra, param_file = resolve_templates(eazy_cfg)
    if param_file is not None:
        return param_file
    return write_templates_param(spectra, Path(run_dir) / "templates.param")


def spectra_from_param(param_file: str | Path) -> list[Path]:
    """Spectrum paths listed in an eazy .param file (column 2 per row)."""
    param_file = Path(param_file)
    paths = []
    for line in param_file.read_text(encoding="utf-8").splitlines():
        tokens = line.split()
        if not tokens or tokens[0].startswith("#"):
            continue
        if len(tokens) < 2:
            raise ValueError(f"{param_file}: unreadable templates row {line!r}")
        path = Path(tokens[1])
        if not path.is_absolute():
            path = param_file.parent / path
        paths.append(path)
    return paths


def content_digests(eazy_cfg: dict) -> dict:
    """Template and TEF content hashes for the run-identity projection."""
    spectra, param_file = resolve_templates(eazy_cfg)
    files = spectra if param_file is None else spectra_from_param(param_file)
    digests = {"templates": {p.name: sha256_file(p)[:16] for p in files}}
    if eazy_cfg["tef"]:
        tef = resolve_tef(eazy_cfg)
        digests["tef"] = {tef.name: sha256_file(tef)[:16]}
    return digests


def template_summary(eazy_cfg: dict, template_digests: dict) -> dict:
    """A scannable manifest fingerprint of the resolved template set.

    Derived from the run-identity digests so the manifest and run_id
    agree; set_sha256_16 is one stable hash over every per-file hash.
    """
    per_file = template_digests["templates"]
    set_hash = sha256_bytes(json.dumps(per_file, sort_keys=True).encode())[:16]
    source = str(resolve_template_source(eazy_cfg["templates"]))
    return {"n": len(per_file), "set_sha256_16": set_hash, "source": source}
