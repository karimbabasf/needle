"""Load the replay data: one Problem per question, with every stored sample and its hidden verdict."""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw"
ANSWER_CONFIGS = ["GSM8K_Llama-3-8B-Instruct", "GSM8K_Llama-3-70B-Instruct",
                  "MATH_Llama-3-8B-Instruct", "MATH_Llama-3-70B-Instruct"]
CODE_CONFIGS = ["CodeContests_Llama-3-8B-Instruct", "CodeContests_Llama-3-70B-Instruct"]


@dataclass
class Problem:
    config: str
    id: int
    prompt: str
    samples: list[str]
    correct: list[bool]
    facts: dict = field(default_factory=dict)  # code only: sample -> precomputed test facts

    @property
    def fold(self) -> int:
        """Problems split in two halves by question, the same split for every model."""
        dataset = self.config.split("_")[0]
        return hashlib.sha256(f"{dataset}:{self.id}".encode()).digest()[0] % 2


def load(config: str) -> list[Problem]:
    rows = json.load(open(RAW / f"{config}.json"))
    if not config.startswith("CodeContests"):
        return [Problem(config, r["orig_dset_idx"], r["prompt"], r["samples"], r["is_corrects"]) for r in rows]
    model = config.split("_", 1)[1]
    table = json.load(open(ROOT / "data" / f"code_{model}.json"))
    problems = []
    for r in rows:
        done = table.get(str(r["orig_dset_idx"]))
        if done is None:
            continue
        n = len(done["samples"])
        facts = {}
        for sample, (passed, fingerprint, flaky, correct, length) in zip(r["samples"][:n], done["samples"]):
            facts.setdefault(sample, {"passed": bool(passed), "partial": int(passed) * done["tests"],
                                      "fingerprint": fingerprint, "flaky": bool(flaky), "length": length})
        problems.append(Problem(config, r["orig_dset_idx"], r["prompt"], r["samples"][:n], r["is_corrects"][:n], facts))
    return problems
