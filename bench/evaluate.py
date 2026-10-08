"""Run needle itself on a replayed problem, and collect training rows for its checker."""

from needle import Replay, Task, checker, complexity, router, solve, strategies
from needle.answers import answer_of
from needle.classify import classify
from needle.strategies import Run


def task_of(problem) -> Task:
    if problem.facts:
        facts = problem.facts
        empty = {"passed": False, "partial": 0, "fingerprint": "", "flaky": False, "length": 0}
        return Task(problem.prompt, tests=lambda sample: dict(facts.get(sample, empty)))
    return Task(problem.prompt)


def served_index(gen: Replay, problem, sample) -> int | None:
    """Original index of a served sample (identity first, then equal text)."""
    for i in gen.drawn:
        if problem.samples[i] is sample:
            return i
    return next((i for i in gen.drawn if problem.samples[i] == sample), None)


def needle_once(problem, seed: int, scale: float) -> dict:
    gen = Replay(problem.samples, seed)
    result = solve(task_of(problem), gen, budget=scale, cache=None)
    i = served_index(gen, problem, result.sample) if result.sample is not None else None
    return {"used": result.samples, "right": bool(i is not None and problem.correct[i]),
            "kind": result.kind, "level": result.level, "flags": result.flags,
            "confidence": result.confidence}


def checker_rows(problem, seed: int, scale: float) -> list[tuple[str, list[float], int]]:
    """(kind, features, leader right) at every round of a run: after the probe, after each batch."""
    gen = Replay(problem.samples, seed)
    task = task_of(problem)
    run = Run(task, classify(task), gen)
    run.level = complexity.measure(run)
    run.row = router.route(run.kind, run.level)
    rows = []

    def record():
        top, cands = checker.leader(run)
        if run.kind in checker.MODEL and top is not None:
            rep = top.members[0]
            if run.kind == "answer":
                rep = next((m for m in top.members if run.facts[m]["strict"]), rep)
            rows.append((run.kind, checker.features(run, cands), int(problem.correct[gen.drawn[rep]])))

    record()
    cap = int(run.row["cap"] * scale) + int(run.row.get("fix", 0) * scale)
    while len(run.samples) < cap:
        if run.draw(min(run.row["batch"], cap - len(run.samples))) == 0:
            break
        record()
    return rows


def answer_arrays(problem) -> tuple[list, list]:
    pairs = [answer_of(s) for s in problem.samples]
    return [a for a, _ in pairs], [s for _, s in pairs]
