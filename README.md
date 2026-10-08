# needle

Ask a model the same question many times and one of the answers is usually right. The hard part is knowing which one. needle does that part for you.

Give it a prompt and a model. It works out what kind of problem it is and how hard it is, runs the sampling strategy that fits, checks the result with a small trained model, and fixes what looks off. You never pick a strategy, a sample count or a stopping rule.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="bench/charts/code-dark.png">
  <img alt="CodeContests: needle against public-test filtering and clustering" src="bench/charts/code.png">
</picture>

## How it works

```
prompt
  │  cache      seen this prompt before? reuse the answer
  │  1 classify code (you gave tests), checked (you gave a checker),
  │             answer (it wants one final answer), open (free text)
  │  2 probe    a few samples: how often does one sample agree with the others?
  │             that sets the level: easy, medium, hard or extreme
  │  3 route    look up the registry row for that kind and level:
  │             strategy, batch, budget, stop rule, and what normal looks like
  │    run      vote, test and cluster, first pass, or most typical text
  │  4 check    a small trained model: is the leader right? flag what looks off
  │  5 fix      one remedy per flag: drop flaky passes, draw more samples,
  ▼             or move up to a stronger model
result: answer, confidence, samples spent, flags, fixes
```

Everything except step 4 is fixed rules, so the same samples always give the same result. Step 4 is a logistic regression over a dozen facts about how the samples came out (how big the leading group is, how far ahead it is, how many samples state an answer, how many got their own arithmetic wrong). It runs in microseconds.

## Results

Measured by replaying the public samples behind [Large Language Monkeys](https://arxiv.org/abs/2407.21787), the paper in the CS329A lecture: Llama-3 8B and 70B, 10,000 samples per problem on GSM8K and MATH, and 1,000 per problem on CodeContests (public tests run here, hidden verdicts from the dataset). Every method sees the same sample orders, 20 shuffles per problem. The checker is trained on half the problems and scored on the other half, both ways round, so no number comes from a problem the checker saw.

<!-- results:start -->
**Answers** (no checker exists, so samples vote):

| Problems, model | needle | Adaptive-Consistency, same samples | Majority vote, same samples | Majority vote, 10,000 samples |
|---|---|---|---|---|
| GSM8K, 8B | 87.6% with 12 samples | 87.1% | 85.3% | 88.2% |
| GSM8K, 70B | 96.7% with 6 samples | 96.8% | 96.1% | 96.9% |
| MATH, 8B | 41.3% with 49 samples | 41.6% | 40.8% | 43.7% |
| MATH, 70B | 59.4% with 33 samples | 60.6% | 59.7% | 60.0% |

**Code** (public tests are the checker, hidden tests decide right or wrong):

| Problems, model | needle | Filter + cluster, same samples | Public-test filter, same samples | Right program in the pile, same samples |
|---|---|---|---|---|
| CodeContests, 8B | 12.0% with 786 samples | 11.9% | 11.2% | 16.2% |
| CodeContests, 70B | 24.6% with 664 samples | 23.8% | 23.2% | 27.3% |

**What the flags are worth** (budget 1):

| Problems, model | Answers flagged | Right when not flagged | Right when flagged |
|---|---|---|---|
| GSM8K, 8B | 8.5% | 90.8% | 52.5% |
| GSM8K, 70B | 2.0% | 97.2% | 70.0% |
| MATH, 8B | 65.5% | 73.3% | 24.5% |
| MATH, 70B | 39.8% | 80.4% | 27.6% |
| CodeContests, 8B | 96.5% | 74.1% | 9.8% |
| CodeContests, 70B | 86.5% | 74.4% | 16.9% |

Checker AUROC on problems it never saw: answers 0.88 to 0.90, code 0.66 to 0.73.

**Two models, cheapest first** (cost in 8B samples; one 70B sample costs 70/8):

| Problems | needle, 8B then 70B | Moved up to 70B | 70B alone (Adaptive-Consistency) | 8B alone, best |
|---|---|---|---|---|
| GSM8K | 90.1% at cost 18 | 8.1% | 96.3% at cost 40 | 87.9% |
| MATH | 56.6% at cost 299 | 64.9% | 57.1% at cost 58 | 42.7% |
<!-- results:end -->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="bench/charts/answers-dark.png">
  <img alt="GSM8K and MATH: needle against majority vote, Adaptive-Consistency and early stopping" src="bench/charts/answers.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="bench/charts/ladder-dark.png">
  <img alt="Two models: needle starts with 8B and moves up to 70B only when unsure" src="bench/charts/ladder.png">
</picture>

## Use cases

- **Offline jobs where being right beats being fast.** The lecture's "$1k per problem, could be offline" setup: code with tests, data questions with one answer, proofs with a checker. needle spends where the problem is hard and stops where it is easy.
- **Training data.** Generating synthetic data or RL rollouts at scale: keep the samples that pass, with a confidence on every kept answer and a flag on the doubtful ones.
- **Evals.** The same accuracy as a fixed sample count, with fewer samples.
- **Cheap model first.** Answer with a small model and send only the doubtful prompts to a large one.
- **Agent steps with checkable output.** Generated code with tests, SQL that must run, JSON that must match a schema: pass any `check` function and needle samples until one passes.

## Quick start

Python 3.11 or newer, no dependencies.

```bash
pip install git+https://github.com/karimbabasf/needle
```

```python
from needle import Task, openai_compatible, solve

model = openai_compatible("openai/gpt-oss-120b", base_url="https://your-endpoint/v1", api_key="...")

r = solve("A shop sells 3 pens for $2. What do 27 pens cost?", model)
print(r.answer, r.confidence, r.samples, r.flags)

r = solve(Task("Read two ints, print their sum.", tests=[("2 3\n", "5\n")]), model)   # code
r = solve(Task("Prove it in Lean.", check=lean_accepts), model)                        # any checker
r = solve(prompt, [small_model, large_model])                                          # cheapest first
```

Any function `generate(prompt, n) -> list[str]` works as a model. From the shell: `NEEDLE_BASE_URL=... NEEDLE_API_KEY=... NEEDLE_MODEL=... python -m needle "your prompt"`.

## The registry

Every strategy needle can run is one row in [`needle/registry.toml`](needle/registry.toml), filed by kind and level. Change a row and the router follows it. Point `NEEDLE_REGISTRY` at your own file to swap the whole table.

```toml
[code.medium]
strategy = "cluster"   # pass the tests, then group programs by what they print on new inputs
batch = 32             # samples per round
cap = 256              # most samples to spend
enough = 4             # stop once the biggest group of passing programs has 4 members
fix = 128              # extra samples the fixer may spend on a flagged result
trust = 0.5            # flag any leader the checker rates below this
normal = { clusters = 3 }
```

## Tests

```bash
pip install pytest && pytest
```

## Reproduce the benchmark

About 3.5 GB of data, and roughly two hours for the code track on a laptop.

```bash
pip install -e ".[bench]"
python bench/download.py                  # samples, CodeContests tests, chart font
python bench/codetrack.py Llama-3-8B-Instruct && python bench/codetrack.py Llama-3-70B-Instruct
python bench/train_checker.py             # cross-fitted checker
python bench/run.py --weights bench/results/checker_folds.json
python bench/ladder.py --weights bench/results/checker_folds.json
python bench/plot.py && python bench/report.py
```

## License

[FSL-1.1-ALv2](LICENSE.md): use it, change it, share it, build on it, inside any company. You may not sell needle, or something that does the same job, as a competing product. Each version becomes Apache 2.0 two years after its release.

The replayed samples come from [Monkey Business](https://huggingface.co/datasets/ScalingIntelligence/monkey_business) (MIT), the CodeContests tests from [DeepMind](https://huggingface.co/datasets/deepmind/code_contests).
