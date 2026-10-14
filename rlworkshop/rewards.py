"""Framework-independent accounting of reward components."""

from collections import defaultdict
from typing import Dict, Optional


class RewardTracker:
    """Accumulate named reward components over one episode.

    ``add`` returns the contribution so existing ``reward += ...`` code works.
    Define all desired component names up front when possible. Unused components
    then appear with zero values in episode summaries, avoiding conditional
    averages over only episodes in which a reward occurred.

    Reset at the beginning of each episode, NOT before taking its final snapshot.
    """

    def __init__(self, names=()):
        self._names = tuple(dict.fromkeys(names))
        self.reset()

    def reset(self) -> None:
        self.components = defaultdict(float, {key: 0.0 for key in self._names})
        self.counts = defaultdict(int, {key: 0 for key in self._names})

    def add(self, name: str, value: float) -> float:
        if not isinstance(name, str) or not name:
            raise ValueError("name must be a non-empty string")
        value = float(value)
        self.components[name] += value
        self.counts[name] += 1
        return value

    @property
    def total(self) -> float:
        return float(sum(self.components.values()))

    def snapshot(self) -> Dict[str, dict]:
        """Return a detached episode summary, safe to place in Gymnasium info."""
        keys = tuple(self.components.keys())
        return {
            "reward_components": {name: float(self.components[name]) for name in keys},
            "reward_counts": {name: int(self.counts[name]) for name in keys},
        }

    def add_to_info(self, info: Optional[dict] = None) -> dict:
        """Attach a snapshot to ``info`` (call on termination or truncation)."""
        info = {} if info is None else dict(info)
        info.update(self.snapshot())
        return info
