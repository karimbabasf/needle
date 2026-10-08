"""Step 3: look up the registry row for a kind and a level."""

import os
import tomllib
from functools import lru_cache
from pathlib import Path

LEVELS = ("easy", "medium", "hard", "extreme")


def registry(path: str | None = None) -> dict:
    """The registry table. Pass a path, or set NEEDLE_REGISTRY, to use your own file."""
    return _load(path or os.environ.get("NEEDLE_REGISTRY") or str(Path(__file__).with_name("registry.toml")))


@lru_cache(maxsize=8)
def _load(path: str) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def level_of(kind: str, score: float, path: str | None = None) -> str:
    """Probe score to level, using the cut-offs in the registry."""
    for name, cutoff in zip(LEVELS, registry(path)[kind]["levels"]):
        if score >= cutoff:
            return name
    return "extreme"


def route(kind: str, level: str, path: str | None = None) -> dict:
    """The row to run, with its name attached."""
    row = dict(registry(path)[kind][level])
    row["name"] = f"{kind}.{level}"
    return row
