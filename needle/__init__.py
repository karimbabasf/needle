"""needle: route every prompt to the sampling strategy that fits it, check the result, fix what looks off."""

from .cache import Cache
from .generators import Replay, openai_compatible
from .pipeline import solve
from .task import Result, Task

__all__ = ["solve", "Task", "Result", "Cache", "Replay", "openai_compatible"]
__version__ = "0.1.0"
