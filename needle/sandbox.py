"""Run a Python program on stdin tests, with hard timeouts and a CPU limit.

On macOS, sandbox-exec also blocks the network and every write outside a temp dir. Programs can still
read files, and elsewhere there is no sandbox at all: run untrusted code inside a container."""

import hashlib
import os
import random
import re
import resource
import shutil
import signal
import subprocess
import sys
import tempfile

_PROFILE = """(version 1)
(allow default)
(deny network*)
(deny file-write*)
(allow file-write* (subpath "{tmp}") (literal "/dev/null") (literal "/dev/stdout") (literal "/dev/stderr"))
"""
_SANDBOX = shutil.which("sandbox-exec") if sys.platform == "darwin" else None


def _limits(timeout: float):
    """CPU time everywhere, memory where the OS enforces it (Linux)."""
    def apply():
        cpu = int(timeout) + 1
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
        if sys.platform.startswith("linux"):
            resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
    return apply


def code_of(sample: str) -> str:
    """The program inside a sample: the first fenced block if there is one, else the whole text."""
    m = re.search(r"```[a-zA-Z0-9]*\n(.*?)(?:```|$)", sample, re.S)
    return (m.group(1) if m else sample).strip()


def run(code: str, stdin: str, timeout: float = 2.0) -> tuple[str, str]:
    """Run code with stdin. Returns (status, stdout); status is ok, error or timeout."""
    tmp = tempfile.mkdtemp(prefix="needle-")
    path = os.path.join(tmp, "main.py")
    with open(path, "w") as f:
        f.write(code)
    cmd = [sys.executable, "-I", "-B", path]
    if _SANDBOX:
        cmd = [_SANDBOX, "-p", _PROFILE.format(tmp=os.path.realpath(tmp))] + cmd
    try:
        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            cwd=tmp, start_new_session=True, text=True, preexec_fn=_limits(timeout),
        )
        try:
            out, _ = proc.communicate(stdin, timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.communicate()
            return "timeout", ""
        return ("ok" if proc.returncode == 0 else "error"), out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def same_output(got: str, want: str) -> bool:
    """Token-by-token compare: whitespace ignored, numbers equal within 1e-6, yes/no case ignored."""
    a, b = got.split(), want.split()
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x == y or (x.lower() == y.lower() and x.lower() in ("yes", "no")):
            continue
        try:
            fx, fy = float(x), float(y)
        except ValueError:
            return False
        if abs(fx - fy) > 1e-6 * max(1.0, abs(fy)):
            return False
    return True


def test(code: str, tests: list[tuple[str, str]], timeout: float = 2.0) -> list[bool]:
    """One verdict per (stdin, expected stdout) test. Stops at the first failure to save time."""
    verdicts = []
    for stdin, want in tests:
        status, out = run(code, stdin, timeout)
        verdicts.append(status == "ok" and same_output(out, want))
        if not verdicts[-1]:
            verdicts += [False] * (len(tests) - len(verdicts))
            break
    return verdicts


def behaviour(code: str, inputs: list[str], timeout: float = 2.0) -> str:
    """A short fingerprint of what the program prints on each input. Equal fingerprints, same behaviour."""
    parts = []
    for stdin in inputs:
        status, out = run(code, stdin, timeout)
        parts.append(status if status != "ok" else " ".join(out.split()))
    return hashlib.sha256("\x00".join(parts).encode()).hexdigest()[:16]


def mutate(inputs: list[str], n: int = 8, seed: int = 0) -> list[str]:
    """New inputs shaped like the given ones: same layout, nearby numbers, shuffled letters.
    Used only to tell programs apart, so an input that breaks the format is still useful."""
    rng = random.Random(seed)
    made = []
    for i in range(n):
        base = inputs[i % len(inputs)]
        lines = base.split("\n")
        out = []
        for li, line in enumerate(lines):
            if li == 0:
                out.append(line)
                continue
            tokens = line.split(" ")
            for ti, tok in enumerate(tokens):
                if re.fullmatch(r"-?\d+", tok):
                    v = int(tok)
                    tokens[ti] = str(max(0 if v >= 0 else v - 3, v + rng.choice([-2, -1, 1, 2, 3])))
                elif re.fullmatch(r"[a-zA-Z]+", tok) and len(tok) > 1:
                    chars = list(tok)
                    rng.shuffle(chars)
                    tokens[ti] = "".join(chars)
            out.append(" ".join(tokens))
        made.append("\n".join(out))
    return made
