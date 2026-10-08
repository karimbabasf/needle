"""The strategies needle is measured against, run on the same shuffles as needle's replays.

answer: majority vote at fixed k, Adaptive-Consistency (Aggarwal et al. 2023), Early-Stopping
Self-Consistency (Li et al. 2024), coverage (an oracle: right if any sample so far is right).
code: public-test filter, filter + behaviour clustering (AlphaCode), coverage. (One random sample is
coverage at k = 1.)
"""

import random

from needle.strategies import leader_holds


def order(n: int, seed: int) -> list[int]:
    """The exact shuffle Replay(samples, seed) serves."""
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    return idx


class Votes:
    """Running vote over a stream of samples, leader ties going to the answer seen first."""

    def __init__(self, answers, strict):
        self.answers, self.strict = answers, strict
        self.count, self.first, self.rep = {}, {}, {}

    def add(self, i: int) -> None:
        a = self.answers[i]
        if a is None:
            return
        if a not in self.count:
            self.count[a], self.first[a], self.rep[a] = 0, len(self.first), i
        if self.strict[i] and not self.strict[self.rep[a]]:
            self.rep[a] = i
        self.count[a] += 1

    def top2(self) -> tuple[int, int]:
        counts = sorted(self.count.values(), reverse=True) + [0, 0]
        return counts[0], counts[1]

    def leader(self):
        if not self.count:
            return None
        return min(self.count, key=lambda a: (-self.count[a], self.first[a]))


def majority(answers, strict, correct, perm, ks):
    """Right or wrong at each k in ks."""
    votes, out, ks = Votes(answers, strict), {}, sorted(ks)
    for pos, i in enumerate(perm[: ks[-1]], 1):
        votes.add(i)
        if pos in ks:
            lead = votes.leader()
            out[pos] = bool(lead is not None and correct[votes.rep[lead]])
    return out


def coverage(correct, perm, ks):
    first = next((pos for pos, i in enumerate(perm, 1) if correct[i]), None)
    return {k: first is not None and first <= k for k in ks}


def adaptive_consistency(answers, strict, correct, perm, cap, stop=0.95):
    """(samples used, right) for one problem: stop once the leader is likely to stay the leader."""
    votes = Votes(answers, strict)
    used = 0
    for i in perm[:cap]:
        votes.add(i)
        used += 1
        a, b = votes.top2()
        if a > 0 and leader_holds(a, b) >= stop:
            break
    lead = votes.leader()
    return used, bool(lead is not None and correct[votes.rep[lead]])


def early_stopping(answers, strict, correct, perm, cap, window=8):
    """(samples used, right): draw windows, stop when one window is unanimous."""
    votes = Votes(answers, strict)
    used = 0
    while used < cap:
        chunk = perm[used : min(cap, used + window)]
        for i in chunk:
            votes.add(i)
        used += len(chunk)
        seen = {answers[i] for i in chunk}
        if len(chunk) == window and len(seen) == 1 and None not in seen:
            break
    lead = votes.leader()
    return used, bool(lead is not None and correct[votes.rep[lead]])


def code_filter(passed, correct, perm, ks):
    """Expected right rate of a random program that passes the public tests (0 when none pass)."""
    out, ok, hits = {}, 0, 0
    ks = sorted(ks)
    for pos, i in enumerate(perm[: ks[-1]], 1):
        if passed[i]:
            ok += 1
            hits += correct[i]
        if pos in ks:
            out[pos] = hits / ok if ok else 0.0
    return out


def code_cluster(passed, fingerprint, correct, perm, ks):
    """Right or wrong when picking the biggest group of passing programs that behave the same."""
    out, size, first, rep = {}, {}, {}, {}
    ks = sorted(ks)
    for pos, i in enumerate(perm[: ks[-1]], 1):
        if passed[i]:
            key = fingerprint[i] or f"#{i}"
            if key not in size:
                size[key], first[key], rep[key] = 0, pos, i
            size[key] += 1
        if pos in ks:
            if size:
                best = min(size, key=lambda k: (-size[k], first[k]))
                out[pos] = bool(correct[rep[best]])
            else:
                out[pos] = False
    return out
