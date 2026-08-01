from __future__ import annotations

import ast
from pathlib import Path

import sedfit

FORBIDDEN_ROOTS = {"eazy", "prospect", "fsps"}
LIGHT_SUBPACKAGES = ("core", "analysis")


def test_light_stack_only() -> None:
    package_dir = Path(sedfit.__file__).resolve().parent
    offenders = []
    for sub in LIGHT_SUBPACKAGES:
        for source in sorted((package_dir / sub).rglob("*.py")):
            tree = ast.parse(source.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    modules = [node.module or ""]
                else:
                    continue
                for module in modules:
                    if module.split(".")[0] in FORBIDDEN_ROOTS:
                        offenders.append(
                            f"{source.relative_to(package_dir)}: {module}")
    assert not offenders, f"backend imports in light subpackages: {offenders}"
