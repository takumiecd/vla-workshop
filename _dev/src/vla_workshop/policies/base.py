"""The small interface shared by environment checks and learned policies."""

from typing import Any, Protocol

import numpy as np


class Policy(Protocol):
    metadata: dict[str, Any]

    def reset(self, *, seed: int) -> None:
        """Seed sampling and clear episode-local state, including queued actions."""
        ...

    def select_action(self, obs: dict[str, np.ndarray], instruction: str) -> np.ndarray:
        """Return one unnormalized 7D OSC_POSE action, before environment clipping."""
        ...
