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


def repair(model: str) -> None:
    """Re-test, on an idle machine, the verdicts a busy one can corrupt: programs the hidden tests
    accept that failed here (usually a timeout), and passes that failed a rerun (flaky)."""
    problems = {str(p["orig_dset_idx"]): p for p in json.load(open(RAW / f"CodeContests_{model}.json"))}
    table = pq.read_table(next(RAW.glob("test-*.parquet")))
    out_path = OUT / f"code_{model}.json"
    done = json.load(open(out_path))
    fixed = 0
    for key, entry in done.items():
        suspects = [i for i, r in enumerate(entry["samples"]) if (r[3] and not r[0]) or (r[0] and r[2])]
        if not suspects:
            continue
        row = table.slice(int(key), 1).to_pylist()[0]
        tests = list(zip(row["public_tests"]["input"], row["public_tests"]["output"]))
        extra = sandbox.mutate(row["public_tests"]["input"], MUTATIONS, seed=int(key))
        for i in suspects:
            code = sandbox.code_of(problems[key]["samples"][i])
            runs = [all(sandbox.test(code, tests, TIMEOUT)) for _ in range(3)]
            passed, flaky = sum(runs) >= 2, int(len(set(runs)) > 1)
            old = entry["samples"][i]
            new = [int(passed), sandbox.behaviour(code, extra, TIMEOUT / 2) if passed else "", flaky if passed else 0,
                   old[3], old[4]]
            if new[:3] != old[:3]:
                fixed += 1
                print(f"{entry['name']} sample {i}: {old[:3]} -> {new[:3]}", flush=True)
            entry["samples"][i] = new
    json.dump(done, open(out_path, "w"))
    print(f"{model}: {fixed} verdicts changed", flush=True)


def main():
    if sys.argv[1] == "--repair":
        return repair(sys.argv[2])
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
