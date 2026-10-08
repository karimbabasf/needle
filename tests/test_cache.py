from needle import Cache, Result, Task


def result(answer="5", level="easy"):
    return Result(answer, f"#### {answer}", 0.9, "answer", level, 4, 4.0)


def test_exact_hit_returns_the_result():
    cache = Cache()
    cache.put(Task("What is 2 + 3?"), "answer", "m", result())
    hit, level = cache.get(Task("What is 2 + 3?"), "answer", "m")
    assert hit.cached and hit.answer == "5" and level is None


def test_near_hit_returns_only_the_level():
    cache = Cache()
    long = "A farmer has twelve cows and buys seven more cows at the market on a sunny Tuesday morning. "
    cache.put(Task(long + "How many cows does the farmer have now?"), "answer", "m", result("19", "medium"))
    hit, level = cache.get(Task(long + "How many cows does the farmer have now??"), "answer", "m")
    assert hit is None and level == "medium"


def test_different_model_or_kind_never_matches():
    cache = Cache()
    cache.put(Task("What is 2 + 3?"), "answer", "m", result())
    assert cache.get(Task("What is 2 + 3?"), "answer", "other") == (None, None)
    assert cache.get(Task("What is 2 + 3?"), "open", "m") == (None, None)


def test_cache_survives_a_restart(tmp_path):
    path = tmp_path / "cache.jsonl"
    Cache(path).put(Task("What is 2 + 3?"), "answer", "m", result())
    hit, _ = Cache(path).get(Task("What is 2 + 3?"), "answer", "m")
    assert hit.answer == "5"
