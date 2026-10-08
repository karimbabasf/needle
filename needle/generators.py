"""Samplers. Anything with the shape generate(prompt, n) -> list of n strings works; these are two
ready ones: Replay serves stored samples, openai_compatible calls any OpenAI-style endpoint."""

import json
import os
import random
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor


class Replay:
    """Serve stored samples as if a model were writing them: one seeded shuffle, then in order.
    Runs dry when the pool is spent. `drawn` holds the original index of every sample served."""

    def __init__(self, samples: list[str], seed: int = 0, name: str = "replay", cost: float = 1.0):
        self.samples = samples
        self.order = list(range(len(samples)))
        random.Random(seed).shuffle(self.order)
        self.drawn: list[int] = []
        self.name, self.cost = name, cost

    def __call__(self, prompt: str, n: int) -> list[str]:
        start = len(self.drawn)
        picked = self.order[start : start + n]
        self.drawn += picked
        return [self.samples[i] for i in picked]


def openai_compatible(model: str, base_url: str | None = None, api_key: str | None = None,
                      temperature: float = 0.8, max_tokens: int = 1024, workers: int = 8, cost: float = 1.0):
    """A sampler for any OpenAI-compatible chat endpoint. The URL and key travel together: pass both,
    or set NEEDLE_BASE_URL and NEEDLE_API_KEY. A lone key is never paired with someone else's URL."""
    if (base_url is None) != (api_key is None):
        raise ValueError("pass base_url and api_key together")
    if base_url is None:
        base_url, api_key = os.environ.get("NEEDLE_BASE_URL"), os.environ.get("NEEDLE_API_KEY")
        if not (base_url and api_key):
            raise ValueError("set NEEDLE_BASE_URL and NEEDLE_API_KEY together")
    url = base_url.rstrip("/") + "/chat/completions"

    def one(prompt: str) -> str:
        body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                           "temperature": temperature, "max_tokens": max_tokens}).encode()
        request = urllib.request.Request(url, body, {"Authorization": f"Bearer {api_key}",
                                                     "Content-Type": "application/json"})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    return json.load(response)["choices"][0]["message"]["content"] or ""
            except urllib.error.HTTPError as e:
                if e.code != 429 and e.code < 500:
                    raise RuntimeError(f"{url} answered {e.code} {e.reason}") from None  # auth, credit, bad model
                if attempt == 3:
                    raise
            except OSError:  # timeouts and dropped connections
                if attempt == 3:
                    raise
            time.sleep(2**attempt)
        return ""

    def generate(prompt: str, n: int) -> list[str]:
        with ThreadPoolExecutor(min(workers, max(1, n))) as pool:
            return list(pool.map(one, [prompt] * n))

    generate.name, generate.cost = model, cost
    return generate
