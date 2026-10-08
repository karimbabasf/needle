"""Step 2: how hard is this problem for this model? Draw a small probe batch and measure it.

The measure is the one the long-tail result says matters: how often a single sample gets it.
answer: share of the most common answer. code, checked: share that pass. open: word overlap.
"""

from . import router
from .strategies import Run, agreement

# Below this share of samples stating an answer, the prompt is free text. Kept low on purpose:
# long math solutions often run out of room before the final line, and that is not free text.
NO_ANSWER = 0.25


def measure(run: Run, registry: str | None = None) -> str:
    run.draw(router.registry(registry)[run.kind]["probe"])
    if run.kind == "answer" and run.answered() < NO_ANSWER * len(run.samples):
        run.kind = "open"  # the samples state no answer, so read them as free text
        run.refresh()
        run.fixes.append("samples state no final answer: read as open text")
    return router.level_of(run.kind, score(run), registry)


def score(run: Run) -> float:
    n = len(run.samples)
    if n == 0:
        return 0.0
    if run.kind == "answer":
        cands = run.candidates()
        return len(cands[0].members) / n if cands else 0.0
    if run.kind in ("code", "checked"):
        return run.passers() / n
    return sum(agreement(run, i) for i in range(n)) / n
