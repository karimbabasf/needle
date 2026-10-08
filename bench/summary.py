"""Turn per-problem results into curves: (mean samples per problem, accuracy) for every method."""

import json
import sys
from pathlib import Path
from statistics import mean


def curves(path: str | Path, fold: int = -1) -> dict:
    res = json.load(open(path))
    probs = [p for p in res["problems"] if fold < 0 or p["fold"] == fold]
    out = {"config": res["config"], "problems": len(probs)}
    first = probs[0]
    for name, table in first["base"].items():
        if name in ("ac", "esc"):
            out[name] = [(mean(p["base"][name][c][0] for p in probs), mean(p["base"][name][c][1] for p in probs))
                         for c in table]
        else:
            out[name] = [(int(k), mean(p["base"][name][k] for p in probs)) for k in table]
    out["needle"] = []
    for scale in first["needle"]:
        rows = [p["needle"][scale] for p in probs]
        unflagged = [x for r in rows for x in r["right_unflagged"]]
        flagged = [x for r in rows for x in r["right_flagged"]]
        out["needle"].append({
            "scale": float(scale), "used": mean(r["used"] for r in rows), "right": mean(r["right"] for r in rows),
            "flagged": mean(r["flagged"] for r in rows),
            "right_unflagged": mean(unflagged) if unflagged else None,
            "right_flagged": mean(flagged) if flagged else None,
        })
    return out


def show(path, fold=-1) -> None:
    c = curves(path, fold)
    print(f"== {c['config']} ({c['problems']} problems)")
    for name in ("coverage", "majority", "filter", "cluster", "ac", "esc"):
        if name in c:
            print(f"  {name:9s}", "  ".join(f"{x:>7.1f}:{y:.3f}" for x, y in c[name]))
    for n in c["needle"]:
        print(f"  needle x{n['scale']:<5} used {n['used']:7.1f}  right {n['right']:.3f}  flagged {n['flagged']:.2f}"
              f"  right|ok {n['right_unflagged'] if n['right_unflagged'] is None else round(n['right_unflagged'], 3)}"
              f"  right|flag {n['right_flagged'] if n['right_flagged'] is None else round(n['right_flagged'], 3)}")


if __name__ == "__main__":
    fold = int(sys.argv[2]) if len(sys.argv) > 2 else -1
    for path in sorted(Path(sys.argv[1]).glob("*.json")) if Path(sys.argv[1]).is_dir() else [sys.argv[1]]:
        show(path, fold)
