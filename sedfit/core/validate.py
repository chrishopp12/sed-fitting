"""
validate.py

Strict-Loader Helpers
---------------------------------------------------------

Shared validation primitives for the JSON loaders. Unknown keys and
out-of-vocabulary values are hard errors everywhere.

CSV readers are the exception: one sample catalog is shared with other
tools, so a column this reader does not consume is reported by
describe_columns rather than refused.

Requirements:
  - (stdlib only)
"""

from __future__ import annotations

from pathlib import Path


def require_keys(
        label: str,
        mapping: dict,
        *,
        allowed: tuple[str, ...],
        required: tuple[str, ...],
    ) -> None:
    """Hard error on unknown or missing keys.

    Parameters
    ----------
    label : str
        Prefix identifying the object in error messages.
    mapping : dict
        The parsed JSON object to check.
    allowed : tuple of str
        Every key the object may carry.
    required : tuple of str
        Keys the object must carry.
    """
    if not isinstance(mapping, dict):
        raise ValueError(f"{label}: expected a JSON object, got "
                         f"{type(mapping).__name__}")
    unknown = sorted(set(mapping) - set(allowed))
    if unknown:
        raise ValueError(f"{label}: unknown keys {unknown} "
                         f"(allowed: {sorted(allowed)})")
    missing = sorted(set(required) - set(mapping))
    if missing:
        raise ValueError(f"{label}: missing required keys {missing}")


def require_enum(label: str, value: str, *,
                 allowed: tuple[str, ...]) -> str:
    """Hard error when value is outside the closed vocabulary.

    Parameters
    ----------
    label : str
        Prefix identifying the field in error messages.
    value : str
        The value to check.
    allowed : tuple of str
        The permitted values.

    Returns
    -------
    value : str
        The value, unchanged.
    """
    if value not in allowed:
        raise ValueError(f"{label}: {value!r} not in {list(allowed)}")
    return value


def apply_aliases(label: str, mapping: dict,
                  aliases: dict[str, str]) -> dict:
    """Rename deprecated keys to their canonical names.

    Parameters
    ----------
    label : str
        Prefix identifying the object in error messages.
    mapping : dict
        The parsed JSON object.
    aliases : dict
        Deprecated key -> canonical key.

    Returns
    -------
    renamed : dict
        A copy carrying canonical names only.

    Raises
    ------
    ValueError
        When both spellings of one key are present.
    """
    renamed = dict(mapping)
    for old, new in aliases.items():
        if old not in renamed:
            continue
        if new in renamed:
            raise ValueError(f"{label}: {old!r} and {new!r} are the same "
                             f"field; declare only {new!r}")
        renamed[new] = renamed.pop(old)
    return renamed


def describe_columns(path: str | Path, n_rows: int, fields: list[str],
                     recognized: tuple[str, ...] | set[str]) -> str:
    """One line naming the columns a CSV reader used and the ones it did not.

    Naming both sets surfaces a typo in an optional column, which would
    otherwise fall through to a default unremarked.

    Parameters
    ----------
    path : str or Path
        The file being described; only its name is shown.
    n_rows : int
        Row count to report.
    fields : list of str
        Column names as they appear in the file.
    recognized : tuple or set of str
        Every column name this reader would consume, whether or not the
        file supplies it.

    Returns
    -------
    summary : str
        A one-line description of the split.
    """
    used = [c for c in fields if c in recognized]
    ignored = [c for c in fields if c not in recognized]
    return (f"{Path(path).name}: {n_rows} rows; "
            f"using {', '.join(used) if used else '(none)'}"
            + (f"; ignoring {', '.join(ignored)}" if ignored else ""))
