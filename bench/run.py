"""Replay every problem under many shuffles: baselines and needle, on identical sample orders.

Usage: python bench/run.py [--configs A B] [--seeds 20] [--workers 8] [--weights bench/results/checker_folds.json]
Writes bench/results/<config>.json. With --weights, a problem is scored by the checker trained on
the other half of the problems, so no number comes from a checker that saw that problem.
"""

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import baselines as B  # noqa: E402
import data  # noqa: E402
import evaluate  # noqa: E402
from needle import checker  # noqa: E402

KS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 10000]
SCALES = [0.125, 0.25, 0.5, 1, 2, 4, 8]
AC_CAPS = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
ESC_CAPS = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
FOLD_WEIGHTS: dict = {}


def one_problem(job):
    problem, seeds, scales = job
    if FOLD_WEIGHTS:
        checker.MODEL = FOLD_WEIGHTS[str(1 - problem.fold)]  # trained on the other half
    n = len(problem.samples)
    ks = [k for k in KS if k <= n]
    out = {"id": problem.id, "fold": problem.fold, "pass1": mean(problem.correct), "base": {}, "needle": {}}
    code = bool(problem.facts)
    if code:
        empty = {"passed": False, "fingerprint": ""}
        facts = [problem.facts.get(s, empty) for s in problem.samples]
        passed = [f["passed"] and not f.get("flaky") for f in facts]
        prints = [f["fingerprint"] for f in facts]
    else:
        answers, strict = evaluate.answer_arrays(problem)
    rows = {"majority": [], "coverage": [], "filter": [], "cluster": [], "ac": {c: [] for c in AC_CAPS},
            "esc": {c: [] for c in ESC_CAPS}}
    for seed in range(seeds):
        perm = B.order(n, seed)
        rows["coverage"].append(B.coverage(problem.correct, perm, ks))
        if code:
            rows["filter"].append(B.code_filter(passed, problem.correct, perm, ks))
            rows["cluster"].append(B.code_cluster(passed, prints, problem.correct, perm, ks))
        else:
            rows["majority"].append(B.majority(answers, strict, problem.correct, perm, ks))
            for cap in AC_CAPS:
                rows["ac"][cap].append(B.adaptive_consistency(answers, strict, problem.correct, perm, cap))
            for cap in ESC_CAPS:
                rows["esc"][cap].append(B.early_stopping(answers, strict, problem.correct, perm, cap))
    for name in ("coverage", "majority", "filter", "cluster"):
        if rows[name]:
            out["base"][name] = {k: mean(float(r[k]) for r in rows[name]) for k in ks}
    if not code:
        for name in ("ac", "esc"):
            out["base"][name] = {c: [mean(u for u, _ in rows[name][c]), mean(float(r) for _, r in rows[name][c])]
                                 for c in rows[name]}
    for scale in scales:
        runs = [evaluate.needle_once(problem, seed, scale) for seed in range(seeds)]
        out["needle"][scale] = {
            "used": mean(r["used"] for r in runs), "right": mean(float(r["right"]) for r in runs),
            "levels": [r["level"] for r in runs], "kinds": sorted({r["kind"] for r in runs}),
            "flagged": mean(float(bool(r["flags"])) for r in runs),
            "right_unflagged": [float(r["right"]) for r in runs if not r["flags"]],
            "right_flagged": [float(r["right"]) for r in runs if r["flags"]],
            "confidence": [[r["confidence"], float(r["right"])] for r in runs],
        }
    return out


def init(weights):
    global FOLD_WEIGHTS
    FOLD_WEIGHTS = weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="*", default=data.ANSWER_CONFIGS + data.CODE_CONFIGS)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--scales", type=float, nargs="*", default=SCALES)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--weights", default="")
    parser.add_argument("--fold", type=int, default=-1, help="only this half of the problems")
    parser.add_argument("--out", default=str(data.ROOT / "results"))
    args = parser.parse_args()
    weights = json.load(open(args.weights)) if args.weights else {}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    for config in args.configs:
        problems = [p for p in data.load(config) if args.fold < 0 or p.fold == args.fold]
        jobs = [(p, args.seeds, args.scales) for p in problems]
        with Pool(args.workers, initializer=init, initargs=(weights,)) as pool:
            results = pool.map(one_problem, jobs, chunksize=1)
        json.dump({"config": config, "seeds": args.seeds, "problems": results},
                  open(Path(args.out) / f"{config}.json", "w"))
        print(f"{config}: {len(results)} problems", flush=True)


if __name__ == "__main__":
    main()
