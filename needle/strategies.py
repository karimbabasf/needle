"""The strategies. Each kind has three small rules: what a sample says (its facts), how samples
group into candidates, and when there is enough evidence to stop.

vote        answers: group equal answers, stop when the leader is very likely to stay the leader
cluster     code: keep programs that pass the tests, group them by behaviour on new inputs
first-pass  checked: stop at the first sample the checker accepts
medoid      open text: the sample that shares the most words with all the others
"""

from dataclasses import dataclass, field
from math import comb, erf, sqrt

from . import answers, sandbox
from .task import Generate, Task


@dataclass
class Candidate:
    key: str
    members: list[int] = field(default_factory=list)
    score: float = 0.0


class Run:
    """Everything known about one prompt so far: the samples, what each one says, and the plan."""

    def __init__(self, task: Task, kind: str, generate: Generate):
        self.task, self.kind, self.generate = task, kind, generate
        self.samples: list[str] = []
        self.facts: list[dict] = []
        self.level, self.row = "", {}
        self.flags: list[str] = []
        self.fixes: list[str] = []
        self.leaked = answers.prompt_answers(task.prompt)
        self.inputs: list[str] | None = None

    def draw(self, n: int) -> int:
        """Ask the sampler for n more samples. Returns how many came back."""
        if n <= 0:
            return 0
        new = list(self.generate(self.task.prompt, n))[:n]
        for sample in new:
            self.samples.append(sample)
            self.facts.append(self.fact(sample))
        return len(new)

    def fact(self, sample: str) -> dict:
        if self.kind == "answer":
            value, strict = answers.answer_of(sample)
            return {"answer": value, "strict": strict, "length": len(sample), "slip": answers.slips(sample) > 0}
        if self.kind == "code":
            return self.test(sample)
        if self.kind == "checked":
            return {"passed": bool(self.task.check(sample)), "length": len(sample)}
        return {"words": frozenset(sample.lower().split()), "length": len(sample)}

    def test(self, sample: str) -> dict:
        if callable(self.task.tests):
            return self.task.tests(sample)
        tests = self.task.tests
        code = sandbox.code_of(sample)
        verdicts = sandbox.test(code, tests) if code else [False] * len(tests)
        fact = {"passed": all(verdicts), "partial": sum(verdicts), "length": len(code),
                "flaky": False, "fingerprint": ""}
        if fact["passed"]:
            fact["flaky"] = not all(all(sandbox.test(code, tests)) for _ in range(2))
            if self.inputs is None:
                self.inputs = sandbox.mutate([stdin for stdin, _ in tests], 8, seed=len(tests))
            fact["fingerprint"] = sandbox.behaviour(code, self.inputs)
        return fact

    def refresh(self) -> None:
        """Recompute every fact, after the kind changed."""
        self.facts = [self.fact(s) for s in self.samples]

    def candidates(self) -> list[Candidate]:
        """Samples grouped into candidate answers, biggest group first, ties to the earliest."""
        groups: dict[str, Candidate] = {}
        # Samples that get their own arithmetic wrong do not vote, unless every sample does.
        clean = self.kind == "answer" and any(f["answer"] is not None and not f["slip"] for f in self.facts)
        for i, f in enumerate(self.facts):
            if self.kind == "answer":
                key = None if clean and f["slip"] else f["answer"]
            elif self.kind == "code":
                key = (f.get("fingerprint") or f"#{i}") if f["passed"] else None
            elif self.kind == "checked":
                key = f"#{i}" if f["passed"] else None
            else:
                key = f"#{i}"
            if key is not None:
                groups.setdefault(key, Candidate(key)).members.append(i)
        return sorted(groups.values(), key=lambda c: (-len(c.members), c.members[0]))

    def done(self) -> bool:
        """The row's stop rule."""
        cands = self.candidates()
        if self.kind == "answer":
            a = len(cands[0].members) if cands else 0
            b = len(cands[1].members) if len(cands) > 1 else 0
            return a > 0 and leader_holds(a, b) >= self.row.get("stop", 0.95)
        if self.kind == "code":
            return bool(cands) and len(cands[0].members) >= self.row.get("enough", 3)
        if self.kind == "checked":
            return bool(cands)
        return False

    def answered(self) -> int:
        return sum(f.get("answer") is not None for f in self.facts)

    def passers(self) -> int:
        return sum(bool(f.get("passed")) for f in self.facts)


def leader_holds(a: int, b: int) -> float:
    """Chance the leading answer is truly more common than the runner-up, given a and b votes
    (Beta posterior with a flat prior; the Adaptive-Consistency stop rule)."""
    n = a + b + 1
    if n > 1000:
        return 0.5 * (1 + erf((a + 0.5 - n / 2) / sqrt(n / 4) / sqrt(2)))
    return sum(comb(n, i) for i in range(a + 1)) / 2**n


def agreement(run: Run, i: int) -> float:
    """Mean word overlap (Jaccard) between sample i and every other sample."""
    words = run.facts[i]["words"]
    others = [f["words"] for j, f in enumerate(run.facts) if j != i]
    if not others:
        return 1.0
    return sum(len(words & w) / max(1, len(words | w)) for w in others) / len(others)


def spend(run: Run, row: dict, budget: float = 1.0) -> None:
    """Run the row: draw batch after batch until its stop rule fires or its cap is spent."""
    cap = int(row["cap"] * budget)
    while len(run.samples) < cap and not run.done():
        if run.draw(min(row["batch"], cap - len(run.samples))) == 0:
            break
