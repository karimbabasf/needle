import sys

import pytest

from needle import Replay, openai_compatible, sandbox


def test_run_reports_ok_error_and_timeout():
    assert sandbox.run("print(int(input()) + 1)", "41\n") == ("ok", "42\n")
    assert sandbox.run("raise SystemExit(3)", "")[0] == "error"
    assert sandbox.run("while True: pass", "", timeout=0.5)[0] == "timeout"


@pytest.mark.skipif(sys.platform != "darwin", reason="sandbox-exec is macOS only")
def test_programs_cannot_reach_the_network():
    code = "import socket; socket.create_connection(('example.com', 80), timeout=2); print('out')"
    assert sandbox.run(code, "")[0] == "error"


def test_output_compare_is_forgiving_about_format_only():
    assert sandbox.same_output("YES\n1.0000000001 ", "yes 1")
    assert not sandbox.same_output("1 2", "1 2 3")
    assert not sandbox.same_output("1.1", "1")


def test_mutated_inputs_are_deterministic():
    assert sandbox.mutate(["3\n1 2 3\n"], 4, seed=7) == sandbox.mutate(["3\n1 2 3\n"], 4, seed=7)


def test_code_of_reads_fenced_blocks():
    assert sandbox.code_of("Here:\n```python\nprint(1)\n```\nbye") == "print(1)"
    assert sandbox.code_of("print(2)") == "print(2)"


def test_replay_is_a_seeded_shuffle_that_runs_dry():
    gen = Replay(["a", "b", "c"], seed=3)
    first = gen("p", 2) + gen("p", 5)
    assert sorted(first) == ["a", "b", "c"] and gen("p", 1) == []
    assert Replay(["a", "b", "c"], seed=3)("p", 3) == first
    assert [gen.samples[i] for i in gen.drawn] == first


def test_credentials_travel_as_a_pair(monkeypatch):
    monkeypatch.delenv("NEEDLE_BASE_URL", raising=False)
    monkeypatch.delenv("NEEDLE_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-not-ours")
    with pytest.raises(ValueError):
        openai_compatible("m")
    with pytest.raises(ValueError):
        openai_compatible("m", base_url="https://x.example/v1")
    gen = openai_compatible("m", base_url="https://x.example/v1", api_key="k", cost=3)
    assert gen.name == "m" and gen.cost == 3
