"""Step 5: make it right. One fixed remedy per flag, and the checker looks again after each round.

no_answer   the samples state no final answer: read them as open text and pick the most typical
flaky       a passing program failed on a rerun: stop trusting its pass
split, doubt, weak_tests, no_pass
            not enough evidence yet: draw more samples (up to the row's fix budget) until the
            flag clears, then let the checker pick
"""

from . import checker, router
from .strategies import Run

_MORE = ("split", "doubt", "weak_tests", "no_pass")


def fix(run: Run, flags: list[str], budget: float = 1.0) -> list[str]:
    """Apply remedies. Returns the flags that still stand."""
    if "no_answer" in flags and run.kind == "answer":
        run.kind = "open"
        run.row = router.route("open", run.level)
        run.refresh()
        run.fixes.append("samples state no final answer: picked the most typical text")
        flags = checker.check(run)
    flags = _drop_flaky(run, flags)
    stop = len(run.samples) + int(run.row.get("fix", 0) * budget)
    drawn = 0
    while any(f in _MORE for f in flags) and len(run.samples) < stop:
        got = run.draw(min(run.row["batch"], stop - len(run.samples)))
        if got == 0:
            break
        drawn += got
        flags = _drop_flaky(run, checker.check(run))
    if drawn:
        run.fixes.append(f"drew {drawn} more samples")
    return flags


def _drop_flaky(run: Run, flags: list[str]) -> list[str]:
    if "flaky" not in flags:
        return flags
    flaky = [f for f in run.facts if f.get("passed") and f.get("flaky")]
    if len(flaky) == run.passers():
        return flags  # nothing steady passes: keep them, keep the flag
    for f in flaky:
        f["passed"] = False
    run.fixes.append(f"stopped trusting {len(flaky)} passes that failed on a rerun")
    return checker.check(run)
