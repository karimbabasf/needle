"""What goes in (Task) and what comes out (Result)."""

from dataclasses import dataclass, field
from typing import Callable

# A sampler: give it a prompt and a count, get back that many samples (fewer when it runs dry).
Generate = Callable[[str, int], list[str]]


@dataclass
class Task:
    prompt: str
    # stdin/stdout pairs the program must pass, or your own runner: sample -> facts dict
    # ({"passed": bool, "partial": int, "fingerprint": str, "flaky": bool, "length": int}).
    tests: list[tuple[str, str]] | Callable[[str], dict] | None = None
    # Any checker that accepts or rejects a whole sample: a proof checker, a schema validator.
    check: Callable[[str], bool] | None = None
    # Force a kind (code, checked, answer, open) instead of letting needle classify.
    kind: str | None = None


@dataclass
class Result:
    answer: str | None  # canonical final answer, the program for code, the sample for open text
    sample: str | None  # the full sample the answer came from
    confidence: float  # the checker's probability that the answer is right
    kind: str
    level: str
    samples: int  # samples spent: probe, strategy and fixes, on every model tried
    cost: float  # samples weighted by each sampler's cost (equals samples with one sampler)
    flags: list[str] = field(default_factory=list)  # what still looks off
    fixes: list[str] = field(default_factory=list)  # what the fixer did about it
    cached: bool = False
