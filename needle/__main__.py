"""python -m needle "your prompt"   (needs NEEDLE_BASE_URL, NEEDLE_API_KEY and NEEDLE_MODEL)"""

import json
import os
import sys
from dataclasses import asdict

from . import openai_compatible, solve


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    model = os.environ.get("NEEDLE_MODEL")
    if not model:
        sys.exit("set NEEDLE_MODEL (and NEEDLE_BASE_URL, NEEDLE_API_KEY)")
    result = solve(" ".join(sys.argv[1:]), openai_compatible(model))
    print(json.dumps({k: v for k, v in asdict(result).items() if k != "sample"}, indent=2))


if __name__ == "__main__":
    main()
