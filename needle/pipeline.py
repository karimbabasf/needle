"""The whole idea, in order."""

from . import checker, complexity, fixer, router, strategies
from .cache import Cache, cacheable
from .classify import classify
from .strategies import Run
from .task import Generate, Result, Task

MEMORY = Cache()


def solve(task: Task | str, generate: Generate | list[Generate], budget: float = 1.0,
          cache: Cache | None = MEMORY) -> Result:
    """Answer one prompt.

    generate: one sampler, or a list of samplers from cheapest to strongest. needle starts with
    the cheapest and moves up only when the checker doubts the result. A sampler may carry a
    `cost` per sample (default 1) and a `name`.
    budget: scales every cap in the registry (0.5 spends half, 2 spends double).
    """
    task = Task(task) if isinstance(task, str) else task
    ladder = list(generate) if isinstance(generate, (list, tuple)) else [generate]
    if not ladder:
        raise ValueError("solve needs at least one sampler")
    if task.kind == "code" and not task.tests:
        raise ValueError("a code task needs tests")
    if task.kind == "checked" and not task.check:
        raise ValueError("a checked task needs a check function")
    if not cacheable(task, ladder):
        cache = None
    model = "+".join(getattr(g, "name", None) or "" for g in ladder)

    kind = classify(task)                                         # 1. what kind of problem is this
    hit, level = cache.get(task, kind, model) if cache else (None, None)
    if hit:
        return hit                                                #    seen it before: reuse the answer
    samples, cost, moves, first_level = 0, 0.0, [], None
    for rung, sampler in enumerate(ladder):
        run = Run(task, kind, sampler)
        run.level = (level if rung == 0 else None) or complexity.measure(run)  # 2. how hard is it
        first_level = first_level or run.level
        run.row = router.route(run.kind, run.level)               # 3. its row in the registry
        strategies.spend(run, run.row, budget)                    #    run that strategy
        flags = checker.check(run)                                # 4. does the result look off
        moving_up = "doubt" in flags and rung + 1 < len(ladder)
        if not moving_up:
            flags = fixer.fix(run, flags, budget)                 # 5. make it right
        samples += len(run.samples)
        cost += len(run.samples) * getattr(sampler, "cost", 1.0)
        if not moving_up:
            break
        moves.append(f"{getattr(sampler, 'name', None) or 'model'} looked unsure: moved up a model")
    run.fixes[:0] = moves
    result = checker.result(run, flags, samples, cost)

    if cache:
        cache.put(task, kind, model, result, level=first_level)
    return result
