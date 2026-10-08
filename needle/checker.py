"""Step 4: does the result look off?

The strategy has already picked a leader (the most common answer, the biggest group of passing
programs, the first accepted sample, the most typical text). A small trained model (logistic
regression over a dozen facts about how the samples came out) gives the probability that the
leader is right. The row's `normal` says what a healthy result looks like. Anything worse becomes
a flag for the fixer. This is the one learned part of needle: weights in checker.json, rebuilt by
bench/train_checker.py.

On the Large Language Monkeys data, re-ranking candidates with these facts did not beat the plain
vote, so the model judges the leader instead of replacing it.
"""

import json
import math
import re
from pathlib import Path

from . import sandbox
from .strategies import Candidate, Run, agreement
from .task import Result

MODEL = json.loads(Path(__file__).with_name("checker.json").read_text())


def leader(run: Run) -> tuple[Candidate | None, list[Candidate]]:
    cands = run.candidates()
    if run.kind == "open" and cands:
        return max(cands, key=lambda c: (agreement(run, c.members[0]), -c.members[0])), cands
    return (cands[0] if cands else None), cands


def features(run: Run, cands: list[Candidate]) -> list[float]:
    """Facts about how the samples came out, in the order of MODEL[kind]["features"]."""
    sizes = [len(c.members) for c in cands]
    total = sum(sizes)
    shares = [s / total for s in sizes]
    k = len(cands)
    entropy = -sum(p * math.log(p) for p in shares) / math.log(k) if k > 1 else 0.0
    lengths = [run.facts[i]["length"] for c in cands for i in c.members]
    top = cands[0]
    length = sum(run.facts[i]["length"] for i in top.members) / len(top.members) / (sum(lengths) / len(lengths) or 1) - 1
    common = [shares[0], shares[0] - (shares[1] if k > 1 else 0.0), math.log2(sizes[0] + 1) / 10]
    if run.kind == "answer":
        answered = [f for f in run.facts if f["answer"] is not None]
        return common + [
            len(answered) / len(run.samples), entropy, k / total,
            sum(run.facts[i]["strict"] for i in top.members) / len(top.members), length,
            float(top.key in run.leaked), float(bool(re.fullmatch(r"-?\d+(/\d+)?", top.key))),
            sum(f["slip"] for f in answered) / len(answered),
        ]
    flaky = sum(bool(run.facts[i].get("flaky")) for i in top.members) / len(top.members)
    return common + [run.passers() / len(run.samples), k / total, flaky, length]


def confidence(run: Run) -> float:
    """Probability that the leader is right."""
    top, cands = leader(run)
    if top is None:
        return 0.0
    if run.kind == "checked":
        return 1.0
    if run.kind == "open":
        return agreement(run, top.members[0])
    if run.kind not in MODEL:  # no trained model for this kind: fall back to the vote share
        return len(top.members) / sum(len(c.members) for c in cands)
    model = MODEL[run.kind]
    z = model["bias"] + sum(w * x for w, x in zip(model["weights"], features(run, cands)))
    return 1 / (1 + math.exp(-max(-30.0, min(30.0, z))))


def check(run: Run) -> list[str]:
    """Flags for whatever looks off, compared with the row's normal."""
    top, cands = leader(run)
    normal, trust = run.row.get("normal", {}), run.row.get("trust", 0.5)
    flags = []
    if run.kind == "answer":
        if top is None or run.answered() < len(run.samples) / 2:
            return ["no_answer"]
        if len(top.members) / sum(len(c.members) for c in cands) < normal.get("share", 0.0):
            flags.append("split")  # the vote is weaker than this row expects
    elif run.kind == "code":
        if top is None:
            return ["no_pass"]
        if any(run.facts[i].get("flaky") for i in top.members):
            flags.append("flaky")  # a passing program failed when run again
        if len(cands) > normal.get("clusters", 1 << 30) and len(top.members) < run.passers() / 2:
            flags.append("weak_tests")  # passing programs disagree on new inputs
    elif run.kind == "checked":
        return [] if top else ["no_pass"]
    elif top and agreement(run, top.members[0]) < normal.get("agreement", 0.0):
        flags.append("split")  # the texts barely agree
    if confidence(run) < trust:
        flags.append("doubt")  # the checker does not trust the leader
    return flags


def result(run: Run, flags: list[str], samples: int | None = None, cost: float | None = None) -> Result:
    """The final pick: the leader, with the checker's confidence."""
    top, _ = leader(run)
    answer = sample = None
    if top is not None:
        first = top.members[0]
        if run.kind == "answer":
            first = next((i for i in top.members if run.facts[i]["strict"]), first)
        sample = run.samples[first]
        answer = top.key if run.kind == "answer" else sample
    elif run.kind == "code" and run.samples:
        first = max(range(len(run.samples)), key=lambda i: (run.facts[i].get("partial", 0), -i))
        sample = answer = run.samples[first]  # nothing passes: the program that got furthest
    if run.kind == "code" and answer is not None:
        answer = sandbox.code_of(answer)
    samples = len(run.samples) if samples is None else samples
    cost = float(samples) if cost is None else cost
    return Result(answer, sample, round(confidence(run), 4), run.kind, run.level, samples, round(cost, 2),
                  flags, run.fixes)
