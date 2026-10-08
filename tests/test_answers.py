from needle.answers import answer_of, extract, normalize, prompt_answers, slips


def test_extracts_each_answer_format():
    assert extract("so 3 + 4 = 7.\n#### 7")[0] == "7"
    assert extract("Final Answer: The final answer is $\\frac{1}{2}$. I hope it is correct.")[0] == "$\\frac{1}{2}$"
    assert extract("blah. The answer is 12.")[0] == "12"
    assert extract("we get \\boxed{\\frac{3}{4}} done") == ("\\frac{3}{4}", False)
    assert extract("no answer here") == (None, False)


def test_first_answer_line_wins_over_invented_follow_ups():
    assert extract("#### 5\n\nQuestion: next one?\nAnswer: ... #### 9")[0] == "5"


def test_equal_values_share_one_form():
    same = ["$57,500", "57500", "57500.00", "57,500 dollars"]
    assert {normalize(x) for x in same} == {"57500"}
    assert normalize("\\frac{1}{2}") == normalize("0.5") == normalize("1/2") == "1/2"
    assert normalize("18^2") == "324"
    assert normalize("\\dfrac{2^7}{3}") == "128/3"
    assert normalize("48grams") == "48"
    assert normalize("x = 5") == "5"


def test_non_numbers_are_kept_and_cleaned():
    assert normalize("[2, 5)") == "[2,5)"
    assert normalize("\\text{(B)}") == "(B)"


def test_answer_of_reports_missing_answers():
    assert answer_of("I am not sure.") == (None, False)
    assert answer_of("#### 42") == ("42", True)


def test_slips_counts_wrong_arithmetic_only():
    assert slips("15 * 50 * 30 = <<15*50*30=45000>>45000") == 1
    assert slips("20 * 35 * 50 = <<20*35*50=35000>>35000 and 6+1+7 = 14") == 0


def test_prompt_answers_finds_worked_examples():
    prompt = "Q: a?\nA: so #### 15\n\nQ: b?\nA: The final answer is 2.\n\nQ: the real question?\nA:"
    assert prompt_answers(prompt) == frozenset({"15", "2"})
