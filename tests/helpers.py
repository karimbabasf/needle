import itertools


def cycle(*samples):
    """A sampler that repeats the given samples in order, forever."""
    it = itertools.cycle(samples)

    def generate(prompt, n):
        return [next(it) for _ in range(n)]

    return generate


def finite(*samples):
    """A sampler that runs dry after the given samples."""
    pool = list(samples)

    def generate(prompt, n):
        out = pool[:n]
        del pool[:n]
        return out

    return generate
