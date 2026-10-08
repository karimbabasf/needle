"""Download the benchmark data: the Large Language Monkeys samples and the CodeContests tests.

Usage: python bench/download.py [out_dir]
"""

import json
import sys
import urllib.request
from pathlib import Path

MONKEYS = "https://huggingface.co/datasets/ScalingIntelligence/monkey_business/resolve/main/"
CONFIGS = [
    "GSM8K_Llama-3-8B-Instruct",
    "GSM8K_Llama-3-70B-Instruct",
    "MATH_Llama-3-8B-Instruct",
    "MATH_Llama-3-70B-Instruct",
    "CodeContests_Llama-3-8B-Instruct",
    "CodeContests_Llama-3-70B-Instruct",
]
CODE_CONTESTS_TREE = "https://huggingface.co/api/datasets/deepmind/code_contests/tree/main/data"
CODE_CONTESTS_FILE = "https://huggingface.co/datasets/deepmind/code_contests/resolve/main/"
FONTS = "https://raw.githubusercontent.com/vercel/geist-font/v1.7.2/packages/next/dist/fonts/geist-sans/"


def fetch(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"have {dest.name}", flush=True)
        return
    part = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as response, open(part, "wb") as out:
        total = int(response.headers.get("Content-Length", 0))
        done = 0
        while chunk := response.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if done % (100 << 20) < (1 << 20):
                print(f"{dest.name}: {done >> 20} / {total >> 20} MB", flush=True)
    part.rename(dest)
    print(f"got {dest.name}", flush=True)


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "bench/data/raw")
    out.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(CODE_CONTESTS_TREE) as response:
        files = [f["path"] for f in json.load(response) if f["path"].startswith("data/test-")]
    for path in files:
        fetch(CODE_CONTESTS_FILE + path, out / Path(path).name)
    for name in CONFIGS:
        fetch(f"{MONKEYS}{name}.json", out / f"{name}.json")
    fonts = out.parent / "fonts"  # Geist, for the charts
    fonts.mkdir(exist_ok=True)
    for weight in ("Regular", "Medium", "SemiBold"):
        fetch(f"{FONTS}Geist-{weight}.ttf", fonts / f"Geist-{weight}.ttf")


if __name__ == "__main__":
    main()
