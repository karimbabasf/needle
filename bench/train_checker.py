"""Train the checker: logistic regression that predicts whether the leader is right.

Rows come from replayed runs, recorded after every round. Two models are fit per kind, one on each
half of the problems (bench/run.py scores each half with the model trained on the other), plus one
on everything, which ships as needle/checker.json.

Usage: python bench/train_checker.py [--seeds 6] [--workers 8]
"""

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import data  # noqa: E402
import evaluate  # noqa: E402
from needle import checker  # noqa: E402

SCALES = [0.5, 1, 2, 4]


def rows_for(job):
    problem, seeds = job
    out = []
    for seed in range(seeds):
        for scale in SCALES:
            for kind, feats, label in evaluate.checker_rows(problem, 1000 + seed, scale):
                out.append((problem.fold, kind, feats, label))
    return out


def fit(X: np.ndarray, y: np.ndarray, ridge: float = 1.0, steps: int = 50) -> np.ndarray:
    """Ridge-penalised logistic regression by Newton's method. Last weight is the bias."""
    X = np.hstack([X, np.ones((len(X), 1))])
    w = np.zeros(X.shape[1])
    penalty = ridge * np.diag(np.r_[np.ones(X.shape[1] - 1), 0.0])
    for _ in range(steps):
        p = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))
        grad = X.T @ (p - y) + penalty @ w
        hess = (X * (p * (1 - p))[:, None]).T @ X + penalty
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / max(1, n_pos * n_neg))


def model_of(w: np.ndarray, kind: str) -> dict:
    return {"features": checker.MODEL[kind]["features"], "weights": [round(float(x), 4) for x in w[:-1]],
            "bias": round(float(w[-1]), 4)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=6)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--configs", nargs="*", default=data.ANSWER_CONFIGS + data.CODE_CONFIGS)
    args = parser.parse_args()
    rows = []
    for config in args.configs:
        problems = data.load(config)
        with Pool(args.workers) as pool:
            for chunk in pool.imap_unordered(rows_for, [(p, args.seeds) for p in problems]):
                rows += chunk
        print(f"{config}: {len(rows)} rows so far", flush=True)
    folds, full, report = {"0": {}, "1": {}}, {}, {}
    for kind in sorted({k for _, k, _, _ in rows}):
        X = np.array([f for _, k, f, _ in rows if k == kind])
        y = np.array([l for _, k, _, l in rows if k == kind], dtype=float)
        fold = np.array([fo for fo, k, _, _ in rows if k == kind])
        for train in (0, 1):
            w = fit(X[fold == train], y[fold == train])
            folds[str(train)][kind] = model_of(w, kind)
            test = fold != train
            z = np.hstack([X[test], np.ones((test.sum(), 1))]) @ w
            report[f"{kind} trained on half {train}"] = {"rows": int(test.sum()), "auroc": round(auroc(z, y[test]), 4),
                                                        "base rate": round(float(y[test].mean()), 4)}
        full[kind] = model_of(fit(X, y), kind)
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    shipped = Path(__file__).parents[1] / "needle" / "checker.json"
    # Kinds not trained this time keep their previous models.
    for path, new in ((out / "checker_folds.json", folds), (out / "checker_report.json", report), (shipped, full)):
        old = json.load(open(path)) if path.exists() else {}
        if path.name == "checker_folds.json":
            merged = {f: {**old.get(f, {}), **new[f]} for f in ("0", "1")}
        else:
            merged = {**old, **new}
        json.dump(merged, open(path, "w"), indent=1 if path.parent == out else 2)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
