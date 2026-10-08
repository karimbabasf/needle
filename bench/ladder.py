"""Two models, cheapest first: needle answers with Llama-3-8B and moves up to Llama-3-70B only when
its checker doubts the 8B result. Cost is counted in 8B samples; one 70B sample costs 70/8.

Usage: python bench/ladder.py [--seeds 20] [--workers 8] [--weights bench/results/checker_folds.json]
Writes bench/results/ladder_<dataset>.json.
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
from needle import Replay, Task, checker, solve  # noqa: E402

COST = 70 / 8
SCALES = [0.125, 0.25, 0.5, 1, 2, 4, 8]
CAPS = [8, 16, 32, 64, 128, 256, 512, 1024]
FOLD_WEIGHTS: dict = {}


def one(job):
    small, large, seeds = job
    if FOLD_WEIGHTS:
        checker.MODEL = FOLD_WEIGHTS[str(1 - small.fold)]
    out = {"id": small.id, "fold": small.fold, "needle": {}, "ac_small": {}, "ac_large": {}}
    a1, s1 = evaluate.answer_arrays(small)
    a2, s2 = evaluate.answer_arrays(large)
    for cap in CAPS:
        r1 = [B.adaptive_consistency(a1, s1, small.correct, B.order(len(a1), seed), cap) for seed in range(seeds)]
        r2 = [B.adaptive_consistency(a2, s2, large.correct, B.order(len(a2), seed), cap) for seed in range(seeds)]
        out["ac_small"][cap] = [mean(u for u, _ in r1), mean(float(r) for _, r in r1)]
        out["ac_large"][cap] = [COST * mean(u for u, _ in r2), mean(float(r) for _, r in r2)]
    for scale in SCALES:
        costs, rights, moved = [], [], []
        for seed in range(seeds):
            g1 = Replay(small.samples, seed, "Llama-3-8B", 1.0)
            g2 = Replay(large.samples, seed, "Llama-3-70B", COST)
            result = solve(Task(small.prompt), [g1, g2], budget=scale, cache=None)
            i = evaluate.served_index(g1, small, result.sample)
            right = small.correct[i] if i is not None else False
            if g2.drawn:
                j = evaluate.served_index(g2, large, result.sample)
                right = large.correct[j] if j is not None else False
            costs.append(result.cost)
            rights.append(float(right))
            moved.append(float(bool(g2.drawn)))
        out["needle"][scale] = [mean(costs), mean(rights), mean(moved)]
    return out


def init(weights):
    global FOLD_WEIGHTS
    FOLD_WEIGHTS = weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--weights", default="")
    parser.add_argument("--fold", type=int, default=-1)
    parser.add_argument("--out", default=str(data.ROOT / "results"))
    args = parser.parse_args()
    weights = json.load(open(args.weights)) if args.weights else {}
    for dataset in ("GSM8K", "MATH"):
        small = {p.id: p for p in data.load(f"{dataset}_Llama-3-8B-Instruct")}
        large = {p.id: p for p in data.load(f"{dataset}_Llama-3-70B-Instruct")}
        jobs = [(small[i], large[i], args.seeds) for i in sorted(small) if i in large
                and (args.fold < 0 or small[i].fold == args.fold)]
        with Pool(args.workers, initializer=init, initargs=(weights,)) as pool:
            results = pool.map(one, jobs, chunksize=1)
        json.dump({"dataset": dataset, "cost_ratio": COST, "seeds": args.seeds, "problems": results},
                  open(Path(args.out) / f"ladder_{dataset}.json", "w"))
        print(f"{dataset}: {len(results)} problems", flush=True)
        for key in ("ac_small", "ac_large"):
            print(f"  {key:9s}", "  ".join(f"{mean(r[key][c][0] for r in results):7.1f}:{mean(r[key][c][1] for r in results):.3f}"
                                         for c in results[0][key]))
        for scale in results[0]["needle"]:
            rows = [r["needle"][scale] for r in results]
            print(f"  needle x{scale:<5} cost {mean(x[0] for x in rows):7.1f}  right {mean(x[1] for x in rows):.3f}"
                  f"  moved up {mean(x[2] for x in rows):.2f}")


if __name__ == "__main__":
    main()
