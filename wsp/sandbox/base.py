"""
What a cell of agent code returns, and the errors the sandbox can raise.
"""

from dataclasses import dataclass

import numpy as np

Weights = dict[str, np.ndarray]


@dataclass(frozen=True)
class Execution:
    """
    The result of running one cell of code.

    `changed` maps each weight that changed to how many values changed.
    """

    output: str
    changed: dict[str, int]


class SandboxError(Exception):
    """
    The sandbox stopped working. Restarting it is the way to recover.
    """


class SandboxTimeout(SandboxError):
    """
    A cell ran too long and could not be interrupted.
    """


class SandboxCrashed(SandboxError):
    """
    The sandbox process died, or sent something that was not a valid reply.
    """
