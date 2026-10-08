from helpers import cycle

from needle import Task, checker, fixer
from needle.strategies import Run


def run_of(kind, generate, n, row, task=None):
    run = Run(task or Task("q", kind=kind), kind, generate)
    run.row = row
    run.draw(n)
    return run


def test_split_vote_is_flagged():
    run = run_of("answer", cycle("#### 1", "#### 2", "#### 3", "#### 1"), 8, {"normal": {"share": 0.6}, "trust": 0.0})
    assert "split" in checker.check(run)


def test_clear_vote_is_not_flagged():
    run = run_of("answer", cycle("#### 4"), 8, {"normal": {"share": 0.6}, "trust": 0.5})
    assert checker.check(run) == []


def test_flaky_passes_are_dropped_by_the_fixer():
    facts = {"steady": {"passed": True, "partial": 1, "fingerprint": "a", "flaky": False, "length": 6},
             "shaky": {"passed": True, "partial": 1, "fingerprint": "b", "flaky": True, "length": 5}}
    task = Task("q", tests=lambda s: dict(facts[s]))
    run = run_of("code", cycle("shaky", "shaky", "steady"), 3, {"batch": 1, "fix": 0, "trust": 0.0, "normal": {}}, task)
    flags = checker.check(run)
    assert "flaky" in flags
    fixer.fix(run, flags)
    assert checker.leader(run)[0].key == "a" and any("rerun" in f for f in run.fixes)


def test_fixer_draws_more_when_nothing_passes():
    facts = {"bad": {"passed": False, "partial": 0, "fingerprint": "", "flaky": False, "length": 3},
             "good": {"passed": True, "partial": 1, "fingerprint": "g", "flaky": False, "length": 4}}
    task = Task("q", tests=lambda s: dict(facts[s]))
    run = run_of("code", cycle("bad", "bad", "bad", "good"), 2, {"batch": 2, "fix": 4, "trust": 0.0, "normal": {}}, task)
    flags = fixer.fix(run, checker.check(run))
    assert "no_pass" not in flags and checker.leader(run)[0].key == "g"
