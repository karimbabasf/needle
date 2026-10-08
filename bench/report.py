"""Write the README result tables from bench/results. Numbers are never typed by hand.

Usage: python bench/report.py   (rewrites the block between the results markers in README.md)
"""

import json
import math
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent))
import summary  # noqa: E402

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
README = ROOT.parent / "README.md"
START, END = "<!-- results:start -->", "<!-- results:end -->"
NAMES = {"GSM8K_Llama-3-8B-Instruct": "GSM8K, 8B", "GSM8K_Llama-3-70B-Instruct": "GSM8K, 70B",
         "MATH_Llama-3-8B-Instruct": "MATH, 8B", "MATH_Llama-3-70B-Instruct": "MATH, 70B",
         "CodeContests_Llama-3-8B-Instruct": "CodeContests, 8B", "CodeContests_Llama-3-70B-Instruct": "CodeContests, 70B"}


def at(curve, x):
    """A curve's value at x, interpolating on log x. None outside the curve."""
    pts = sorted(curve)
    if not pts or x < pts[0][0] or x > pts[-1][0]:
        return None
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            if x1 == x0:
                return y1
            f = (math.log(x) - math.log(x0)) / (math.log(x1) - math.log(x0))
            return y0 + f * (y1 - y0)
    return pts[-1][1]


def pct(v):
    return "n/a" if v is None else f"{100 * v:.1f}%"


def default_point(c):
    return next(n for n in c["needle"] if n["scale"] == 1.0)


def answers_table() -> list[str]:
    rows = ["| Problems, model | needle | Adaptive-Consistency, same samples | Majority vote, same samples | Majority vote, 10,000 samples |",
            "|---|---|---|---|---|"]
    for config in list(NAMES)[:4]:
        path = RESULTS / f"{config}.json"
        if not path.exists():
            continue
        c = summary.curves(path)
        n = default_point(c)
        rows.append(f"| {NAMES[config]} | **{pct(n['right'])}** with {n['used']:.0f} samples | "
                    f"{pct(at(c['ac'], n['used']))} | {pct(at(c['majority'], n['used']))} | {pct(c['majority'][-1][1])} |")
    return rows


def flags_table() -> list[str]:
    rows = ["| Problems, model | Answers flagged | Right when not flagged | Right when flagged |", "|---|---|---|---|"]
    for config in NAMES:
        path = RESULTS / f"{config}.json"
        if not path.exists():
            continue
        n = default_point(summary.curves(path))
        rows.append(f"| {NAMES[config]} | {pct(n['flagged'])} | {pct(n['right_unflagged'])} | {pct(n['right_flagged'])} |")
    return rows


def code_table() -> list[str]:
    rows = ["| Problems, model | needle | Filter + cluster, same samples | Public-test filter, same samples | Right program in the pile, same samples |",
            "|---|---|---|---|---|"]
    for config in list(NAMES)[4:]:
        path = RESULTS / f"{config}.json"
        if not path.exists():
            continue
        c = summary.curves(path)
        n = default_point(c)
        rows.append(f"| {NAMES[config]} | **{pct(n['right'])}** with {n['used']:.0f} samples | "
                    f"{pct(at(c['cluster'], n['used']))} | {pct(at(c['filter'], n['used']))} | {pct(at(c['coverage'], n['used']))} |")
    return rows


def ladder_table() -> list[str]:
    rows = ["| Problems | needle, 8B then 70B | Moved up to 70B | 70B alone (Adaptive-Consistency) | 8B alone, best |",
            "|---|---|---|---|---|"]
    for dataset in ("GSM8K", "MATH"):
        path = RESULTS / f"ladder_{dataset}.json"
        if not path.exists():
            continue
        res = json.load(open(path))["problems"]
        n = [mean(r["needle"]["1"][i] for r in res) for i in range(3)]
        large = [(mean(r["ac_large"][k][0] for r in res), mean(r["ac_large"][k][1] for r in res)) for k in res[0]["ac_large"]]
        small = max(mean(r["ac_small"][k][1] for r in res) for k in res[0]["ac_small"])
        cheapest = min(large)
        rows.append(f"| {dataset} | **{pct(n[1])}** at cost {n[0]:.0f} | {pct(n[2])} | "
                    f"{pct(cheapest[1])} at cost {cheapest[0]:.0f} | {pct(small)} |")
    return rows


def checker_line() -> str:
    path = RESULTS / "checker_report.json"
    if not path.exists():
        return ""
    rep = json.load(open(path))
    parts = []
    for kind in ("answer", "code"):
        scores = [v["auroc"] for k, v in rep.items() if k.startswith(kind)]
        if scores:
            parts.append(f"{kind}s {min(scores):.2f} to {max(scores):.2f}")
    return "Checker AUROC on problems it never saw: " + ", ".join(parts) + "."


def main() -> None:
    block = ["", "**Answers** (no checker exists, so samples vote):", "", *answers_table(), "",
             "**Code** (public tests are the checker, hidden tests decide right or wrong):", "", *code_table(), "",
             "**What the flags are worth** (budget 1):", "", *flags_table(), "", checker_line(), "",
             "**Two models, cheapest first** (cost in 8B samples; one 70B sample costs 70/8):", "", *ladder_table(), ""]
    text = README.read_text()
    if START in text and END in text:
        head, rest = text.split(START, 1)
        _, tail = rest.split(END, 1)
        README.write_text(head + START + "\n".join(block) + END + tail)
    print("\n".join(block))


if __name__ == "__main__":
    main()
