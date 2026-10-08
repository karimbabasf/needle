"""Run every CodeContests sample against its public tests once, so replays never execute code again.

For each problem and each of the first N samples it records: passes public tests, a behaviour
fingerprint on mutated inputs (passers only), whether a rerun disagreed (flaky), code length, and
the hidden verdict from the dataset. Usage: python bench/codetrack.py <model> [N]
"""

import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from needle import sandbox  # noqa: E402

RAW = Path(__file__).parent / "data" / "raw"
OUT = Path(__file__).parent / "data"
TIMEOUT = 2.0
RERUNS = 2
MUTATIONS = 8


def one(job):
    sample, tests, extra, correct = job
    code = sandbox.code_of(sample)
    if not code:
        return [0, "", 0, int(correct), 0]
    passed = all(sandbox.test(code, tests, TIMEOUT))
    fingerprint, flaky = "", 0
    if passed:
        reruns = [all(sandbox.test(code, tests, TIMEOUT)) for _ in range(RERUNS)]
        flaky = int(not all(reruns))
        fingerprint = sandbox.behaviour(code, extra, TIMEOUT / 2)
    return [int(passed), fingerprint, flaky, int(correct), len(code)]


def main():
    model = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    problems = json.load(open(RAW / f"CodeContests_{model}.json"))
    table = pq.read_table(next(RAW.glob("test-*.parquet")))
    out_path = OUT / f"code_{model}.json"
    done = json.load(open(out_path)) if out_path.exists() else {}
    with Pool(8) as pool:
        for p in problems:
            key = str(p["orig_dset_idx"])
            if key in done:
                continue
            row = table.slice(p["orig_dset_idx"], 1).to_pylist()[0]
            tests = list(zip(row["public_tests"]["input"], row["public_tests"]["output"]))
            extra = sandbox.mutate(row["public_tests"]["input"], MUTATIONS, seed=p["orig_dset_idx"])
            jobs = [(s, tests, extra, c) for s, c in zip(p["samples"][:n], p["is_corrects"][:n])]
            start = time.time()
            rows = pool.map(one, jobs, chunksize=8)
            done[key] = {"name": row["name"], "tests": len(tests), "samples": rows}
            json.dump(done, open(out_path, "w"))
            passers = sum(r[0] for r in rows)
            print(f"{model} {len(done)}/{len(problems)} {row['name'][:30]}: {passers} pass, "
                  f"{sum(r[3] for r in rows)} correct, {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
