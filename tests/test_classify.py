from needle import Task
from needle.classify import classify


def test_structure_decides_first():
    assert classify(Task("anything", tests=[("1\n", "1\n")])) == "code"
    assert classify(Task("anything", check=lambda s: True)) == "checked"
    assert classify(Task("Write a poem", kind="answer")) == "answer"


def test_prompts_that_want_one_answer():
    assert classify(Task("How many apples are left if I eat 3 of 10?")) == "answer"
    assert classify(Task("Compute $\\frac{3}{4} + 1$.")) == "answer"
    assert classify(Task("Which of the following is prime?\n(A) 4\n(B) 7")) == "answer"


def test_free_text_prompts():
    assert classify(Task("Write a short story about a lighthouse keeper")) == "open"
    assert classify(Task("Explain why the sky looks blue")) == "open"
