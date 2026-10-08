"""The cache. An exact repeat of a prompt reuses the whole result. A near repeat (same question,
small edits) reuses only the level, so the probe is skipped but the answer is still worked out."""

import hashlib
import json
import random
import re
from dataclasses import asdict
from pathlib import Path

from .task import Result, Task

_PRIME = (1 << 61) - 1
_HASHES = 64
_BANDS = 16
_rng = random.Random(1729)
_COEFS = [(_rng.randrange(1, _PRIME), _rng.randrange(_PRIME)) for _ in range(_HASHES)]


def signature(text: str) -> tuple[int, ...]:
    """MinHash of the word triples in the end of the prompt, where the actual question sits."""
    words = re.findall(r"\w+", text[-1000:].lower())
    triples = {" ".join(words[i : i + 3]) for i in range(max(1, len(words) - 2))}
    hashes = [int.from_bytes(hashlib.blake2b(t.encode(), digest_size=8).digest(), "big") for t in triples]
    return tuple(min((a * h + b) % _PRIME for h in hashes) for a, b in _COEFS)


def similarity(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    return sum(x == y for x, y in zip(a, b)) / len(a)


def cacheable(task: Task, ladder: list) -> bool:
    """Only things with a stable name can be recognised again: named samplers and test lists.
    A function (an unnamed sampler, a custom runner, a checker) cannot, so those runs skip the cache."""
    return all(getattr(g, "name", None) for g in ladder) and not callable(task.tests) and task.check is None


class Cache:
    def __init__(self, path: str | None = None, near: float = 0.9):
        self.path = Path(path) if path else None
        self.near = near
        self.results: dict[str, dict] = {}
        self.levels: dict[str, str] = {}
        self.signatures: dict[str, tuple[int, ...]] = {}
        self.buckets: dict[tuple, list[str]] = {}
        if self.path and self.path.exists():
            for line in self.path.read_text().splitlines():
                entry = json.loads(line)
                self._store(entry["key"], entry["scope"], entry["result"], tuple(entry["signature"]),
                            entry.get("level"))

    @staticmethod
    def key(task: Task, kind: str, model: str) -> tuple[str, str]:
        """(exact key, scope). Scope keeps near matches inside the same model and kind."""
        scope = f"{model}\x00{kind}"
        raw = f"{scope}\x00{task.prompt}\x00{task.tests!r}"
        return hashlib.sha256(raw.encode()).hexdigest(), scope

    def get(self, task: Task, kind: str, model: str = "") -> tuple[Result | None, str | None]:
        """(cached result, None) on an exact repeat; (None, level) on a near repeat; else (None, None)."""
        key, scope = self.key(task, kind, model)
        if key in self.results:
            return Result(**{**self.results[key], "cached": True}), None
        sig = signature(task.prompt)
        seen = set()
        for band in range(_BANDS):
            for other in self.buckets.get(self._band(scope, sig, band), []):
                if other not in seen and similarity(sig, self.signatures[other]) >= self.near:
                    return None, self.levels[other]
                seen.add(other)
        return None, None

    def put(self, task: Task, kind: str, model: str, result: Result, level: str | None = None) -> None:
        """Store a result. level: the level a near repeat should start from (the first model's),
        when it differs from the result's own (the last model tried)."""
        key, scope = self.key(task, kind, model)
        sig = signature(task.prompt)
        data = asdict(result)
        self._store(key, scope, data, sig, level)
        if self.path:
            with open(self.path, "a") as f:
                f.write(json.dumps({"key": key, "scope": scope, "result": data, "signature": sig,
                                    "level": level}) + "\n")

    def _store(self, key: str, scope: str, result: dict, sig: tuple[int, ...], level: str | None = None) -> None:
        self.results[key] = result
        self.levels[key] = level or result["level"]
        self.signatures[key] = sig
        for band in range(_BANDS):
            self.buckets.setdefault(self._band(scope, sig, band), []).append(key)

    @staticmethod
    def _band(scope: str, sig: tuple[int, ...], band: int) -> tuple:
        rows = _HASHES // _BANDS
        return (scope, band) + sig[band * rows : (band + 1) * rows]
