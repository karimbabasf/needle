from helpers import cycle, finite

from needle import Task
from needle.strategies import Run, leader_holds, spend


def test_leader_holds_grows_with_the_lead():
    assert leader_holds(4, 0) > 0.95 > leader_holds(3, 1)
    assert abs(leader_holds(5, 5) - 0.5) < 0.2
    assert leader_holds(600, 450) > 0.99  # large counts use the normal approximation


def test_vote_groups_equal_answers_and_breaks_ties_by_first_seen():
    run = Run(Task("q", kind="answer"), "answer", cycle("#### 2", "#### 1", "#### 1.0", "#### 2"))
    run.draw(4)
    cands = run.candidates()
    assert [c.key for c in cands] == ["2", "1"]  # 2 and 1 tie at two votes; 2 came first


def test_samples_with_wrong_arithmetic_do_not_vote():
    run = Run(Task("q", kind="answer"), "answer", cycle("2 * 3 = 7 so #### 7", "2 * 3 = 6 so #### 6"))
    run.draw(4)
    assert [c.key for c in run.candidates()] == ["6"]


def test_spend_stops_when_the_sampler_runs_dry():
    run = Run(Task("q", kind="answer"), "answer", finite("#### 1", "#### 2", "#### 3"))
    run.row = {"batch": 2, "cap": 100, "stop": 0.99}
    spend(run, run.row)
    assert len(run.samples) == 3


def test_spend_stops_once_the_vote_is_settled():
    run = Run(Task("q", kind="answer"), "answer", cycle("#### 5"))
    run.row = {"batch": 1, "cap": 100, "stop": 0.95}
    spend(run, run.row)
    assert len(run.samples) == 4  # four agreeing votes clear 0.95
