"""Write the README result tables from bench/results. Numbers are never typed by hand.

Usage: python bench/report.py   (rewrites the block between the results markers in README.md)
"""

import json
import math
import random
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
README = ROOT.parent / "README.md"
START, END = "<!-- results:start -->", "<!-- results:end -->"
NAMES = {"GSM8K_Llama-3-8B-Instruct": "GSM8K, 8B", "GSM8K_Llama-3-70B-Instruct": "GSM8K, 70B",
         "MATH_Llama-3-8B-Instruct": "MATH, 8B", "MATH_Llama-3-70B-Instruct": "MATH, 70B",
         "CodeContests_Llama-3-8B-Instruct": "CodeContests, 8B", "CodeContests_Llama-3-70B-Instruct": "CodeContests, 70B"}
DEFAULT = "1.0"  # needle at budget 1, the default


def at(curve, x):
    """A curve's value at x, interpolating on log x. Past the end, the last value (that method has
    stopped spending); before the start, None."""
    pts = sorted(curve)
    if not pts or x < pts[0][0]:
        return None
    if x >= pts[-1][0]:
        return pts[-1][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            f = 0.0 if x1 == x0 else (math.log(x) - math.log(x0)) / (math.log(x1) - math.log(x0))
            return y0 + f * (y1 - y0)
    return pts[-1][1]


def baseline_curve(probs, name):
    """(mean samples, mean right) points of a baseline over a set of problems."""
    first = probs[0]["base"][name]
    if name in ("ac", "esc"):
        return [(mean(p["base"][name][c][0] for p in probs), mean(p["base"][name][c][1] for p in probs)) for c in first]
    return [(int(k), mean(p["base"][name][k] for p in probs)) for k in first]


def versus(probs, name):
    """needle at budget 1 against a baseline at the same mean samples: (needle, used, baseline, diff)."""
    used = mean(p["needle"][DEFAULT]["used"] for p in probs)
    right = mean(p["needle"][DEFAULT]["right"] for p in probs)
    other = at(baseline_curve(probs, name), used)
    return right, used, other, (None if other is None else right - other)


def interval(probs, name, reps=2000, seed=0):
    """95% interval of the needle-minus-baseline difference, resampling problems (paired)."""
    rng = random.Random(seed)
    diffs = []
    for _ in range(reps):
        sample = [rng.choice(probs) for _ in probs]
        d = versus(sample, name)[3]
        if d is not None:
            diffs.append(d)
    diffs.sort()
    return diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs)) - 1]


def pct(v, signed=False):
    if v is None:
        return "n/a"
    return f"{100 * v:+.1f}" if signed else f"{100 * v:.1f}%"


def load(config, fold=-1):
    path = RESULTS / f"{config}.json"
    if not path.exists():
        return None
    probs = json.load(open(path))["problems"]
    return [p for p in probs if fold < 0 or p["fold"] == fold]


def comparison_table(configs, name, label, extra_label, extra) -> list[str]:
    rows = [f"| Problems, model | needle | {label}, same samples | Difference, points (95% interval) | {extra_label} |",
            "|---|---|---|---|---|"]
    for config in configs:
        probs = load(config)
        if not probs:
            continue
        right, used, other, diff = versus(probs, name)
        low, high = interval(probs, name)
        rows.append(f"| {NAMES[config]} ({len(probs)}) | {pct(right)} with {used:.0f} samples | {pct(other)} | "
                    f"{pct(diff, True)} ({pct(low, True)} to {pct(high, True)}) | {extra(probs, used)} |")
    return rows


def untouched_table() -> list[str]:
    rows = ["| Problems, model | needle | Best rival, same samples | Difference, points (95% interval) |", "|---|---|---|---|"]
    for config in NAMES:
        probs = load(config, fold=1)
        if not probs:
            continue
        name, label = ("cluster", "filter + cluster") if config.startswith("Code") else ("ac", "Adaptive-Consistency")
        right, used, other, diff = versus(probs, name)
        low, high = interval(probs, name)
        rows.append(f"| {NAMES[config]} ({len(probs)}) | {pct(right)} with {used:.0f} samples | {pct(other)} ({label}) | "
                    f"{pct(diff, True)} ({pct(low, True)} to {pct(high, True)}) |")
    return rows


def flags_table() -> list[str]:
    rows = ["| Problems, model | Answers flagged | Right when not flagged | Right when flagged |", "|---|---|---|---|"]
    for config in NAMES:
        probs = load(config)
        if not probs:
            continue
        runs = [p["needle"][DEFAULT] for p in probs]
        ok = [x for r in runs for x in r["right_unflagged"]]
        bad = [x for r in runs for x in r["right_flagged"]]
        rows.append(f"| {NAMES[config]} | {pct(mean(r['flagged'] for r in runs))} | {pct(mean(ok) if ok else None)} | "
                    f"{pct(mean(bad) if bad else None)} |")
    return rows


def ladder_table() -> list[str]:
    rows = ["| Problems | needle, 8B then 70B | Moved up to 70B | 8B alone, best | 70B alone, cheapest point |",
            "|---|---|---|---|---|"]
    for dataset in ("GSM8K", "MATH"):
        path = RESULTS / f"ladder_{dataset}.json"
        if not path.exists():
            continue
        res = json.load(open(path))["problems"]
        cost, right, moved = (mean(r["needle"]["1"][i] for r in res) for i in range(3))
        large = min((mean(r["ac_large"][k][0] for r in res), mean(r["ac_large"][k][1] for r in res)) for k in res[0]["ac_large"])
        small = max(mean(r["ac_small"][k][1] for r in res) for k in res[0]["ac_small"])
        rows.append(f"| {dataset} ({len(res)}) | {pct(right)} at cost {cost:.0f} | {pct(moved)} | {pct(small)} | "
                    f"{pct(large[1])} at cost {large[0]:.0f} |")
    return rows


def checker_line() -> str:
    path = RESULTS / "checker_report.json"
    if not path.exists():
        return ""
    rep = json.load(open(path))
    parts = []
    for kind, word in (("answer", "answers"), ("code", "code")):
        scores = [v["auroc"] for k, v in rep.items() if k.startswith(kind)]
        if scores:
            parts.append(f"{word} {min(scores):.2f} to {max(scores):.2f}")
    return "Checker AUROC (how well its confidence separates right from wrong) on problems it never saw: " + ", ".join(parts) + "."


def main() -> None:
    majority = lambda probs, used: pct(mean(p["base"]["majority"]["10000"] for p in probs))  # noqa: E731
    pile = lambda probs, used: pct(at(baseline_curve(probs, "coverage"), used))  # noqa: E731
    answers = comparison_table(list(NAMES)[:4], "ac", "Adaptive-Consistency", "Majority vote, 10,000 samples", majority)
    code = comparison_table(list(NAMES)[4:], "cluster", "Filter + cluster", "Right program in the pile, same samples", pile)
    block = ["", "Budget 1 (the default). Problem counts in brackets. Intervals come from resampling problems.", "",
             "**Answers.** No checker exists, so samples vote.", "", *answers, "",
             "**Code.** Public tests are the checker; hidden tests decide right or wrong.", "", *code, "",
             "**The half never used for tuning.** Registry rows and two answer rules were picked by looking at the "
             "other half (fold 0) of GSM8K 8B, MATH 8B and both CodeContests models. This half was not looked at.", "",
             *untouched_table(), "",
             "**What the flags are worth.**", "", *flags_table(), "", checker_line(), "",
             "**Two models, cheapest first.** Cost in 8B samples; one 70B sample costs 70/8.", "", *ladder_table(), ""]
    text = README.read_text()
    if START in text and END in text:
        head, rest = text.split(START, 1)
        _, tail = rest.split(END, 1)
        README.write_text(head + START + "\n".join(block) + END + tail)
    print("\n".join(block))


if __name__ == "__main__":
    main()
