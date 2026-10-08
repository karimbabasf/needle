from helpers import cycle, finite

from needle import Cache, Task, solve


def test_answer_task_end_to_end():
    r = solve("How many legs do 3 cats have? Compute it.", cycle("#### 12", "#### 12", "#### 9", "#### 12"), cache=None)
    assert (r.kind, r.answer) == ("answer", "12")
    assert 0 < r.confidence <= 1 and r.samples >= 4


def test_no_answer_anywhere_falls_back_to_open_text():
    r = solve(Task("Compute the vibe.", kind="answer"), cycle("it feels warm", "it feels warm and bright", "cold"), cache=None)
    assert r.kind == "open"
    assert r.answer.startswith("it feels warm")


def test_sampler_that_runs_dry_still_returns():
    r = solve("Find x if x + 1 = 3.", finite("#### 2", "#### 3"), cache=None)
    assert r.samples == 2 and r.answer in ("2", "3")


def test_code_task_picks_a_program_that_passes():
    progs = cycle("print(int(input()) * 2)", "print(int(input()) + 2)", "print(int(input()) * 2)")
    r = solve(Task("Double it.", tests=[("2\n", "4\n"), ("5\n", "10\n")]), progs, cache=None)
    assert r.kind == "code" and r.answer == "print(int(input()) * 2)"
    assert "no_pass" not in r.flags


def test_code_task_where_nothing_passes_returns_the_closest_program():
    progs = cycle("print(4)", "print('nope')")
    r = solve(Task("Double it.", tests=[("2\n", "4\n"), ("5\n", "10\n")]), progs, cache=None)
    assert r.answer == "print(4)"  # passes the first test, gets furthest
    assert "no_pass" in r.flags and r.confidence == 0.0


def test_checked_task_stops_at_the_first_accepted_sample():
    r = solve(Task("prove it", check=lambda s: s == "QED"), cycle("no", "no", "QED"), cache=None)
    assert r.answer == "QED" and r.confidence == 1.0


def test_ladder_moves_up_only_when_doubtful():
    sure = cycle("#### 7")
    unsure = cycle("#### 1", "#### 2", "#### 3", "#### 4")
    strong = cycle("#### 9")
    strong.name, strong.cost = "strong", 10.0
    r = solve("What is 3 + 4? Compute it.", [sure, strong], cache=None)
    assert r.answer == "7" and r.cost == r.samples
    r = solve("What is 3 + 6? Compute it.", [unsure, strong], cache=None)
    assert r.answer == "9" and r.cost > r.samples and "moved up" in r.fixes[0]


def test_budget_scales_spending():
    gen = cycle("#### 1", "#### 2", "#### 3", "#### 1", "#### 2", "#### 3", "#### 4")
    small = solve("Find it. Compute it.", gen, budget=0.25, cache=None)
    large = solve("Find it. Compute it.", gen, budget=4, cache=None)
    assert small.samples < large.samples


def test_exact_repeat_comes_from_the_cache():
    cache = Cache()
    gen = cycle("#### 5")
    first = solve("What is 2 + 3? Compute it.", gen, cache=cache)
    again = solve("What is 2 + 3? Compute it.", gen, cache=cache)
    assert again.cached and again.answer == first.answer
