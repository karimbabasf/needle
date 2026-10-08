"""Step 1: what kind of problem is this? Kinds are named by how an answer can be checked.

code     you gave tests, so programs can be run
checked  you gave a checker that accepts or rejects a whole sample (a proof checker, a schema)
answer   the prompt wants one short final answer, so samples can vote
open     free text, so the most typical sample wins
"""

import re

from .task import Task

_ANSWER = [
    r"####", r"final answer", r"\\boxed", r"answer is", r"\bhow (many|much|long|far|old)\b",
    r"\bwhat is the (value|sum|product|number|probability|remainder|area|smallest|largest)\b",
    r"\b(compute|calculate|evaluate|simplify|solve|find)\b", r"\$[^$\n]+\$", r"\\frac", r"\d+\s*[-+*/^]\s*\d+",
    r"(?m)^\s*\(?[A-E][).]\s", r"\bwhich of the following\b", r"\btrue or false\b",
]
_OPEN = [
    r"\bwrite (a|an|me|the)\b", r"\bexplain\b", r"\bdescribe\b", r"\bsummari[sz]e\b", r"\bessay\b",
    r"\bstory\b", r"\bpoem\b", r"\bdraft\b", r"\bopinion\b", r"\bwhy\b", r"\bcompare\b", r"\bideas?\b",
    r"\bbrainstorm\b", r"\brewrite\b", r"\btranslate\b",
]


def classify(task: Task) -> str:
    if task.kind:
        return task.kind
    if task.tests:
        return "code"
    if task.check:
        return "checked"
    answer = sum(bool(re.search(p, task.prompt, re.I)) for p in _ANSWER)
    open_ = sum(bool(re.search(p, task.prompt, re.I)) for p in _OPEN)
    return "answer" if answer > open_ else "open"
