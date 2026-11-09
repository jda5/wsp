"""
Everything one run needs, in one place. Tools receive the session.
"""

from dataclasses import dataclass
from typing import Any

from wsp.experiments import Experiment, Split
from wsp.log import RunLog
from wsp.sandbox import Sandbox
from wsp.workspace import Commit, Weights, Workspace


@dataclass
class Session:
    experiment: Experiment
    sandbox: Sandbox
    workspace: Workspace
    log: RunLog

    def current_weights(self) -> Weights:
        """
        The agent's weights as they are now, validated but not committed.
        """
        return self.workspace.validate(self.sandbox.get_weights())

    def commit(self, message: str) -> Commit:
        commit = self.workspace.commit(self.sandbox.get_weights(), message)
        self.log.event(
            "commit", number=commit.number, message=message, path=commit.path
        )
        return commit

    def score(self, split: Split, weights: Weights | None = None) -> dict[str, Any]:
        """
        Score ``weights`` on ``split``. Uses the last commit by default.
        """
        self.experiment.set_weights(
            weights if weights is not None else self.workspace.head
        )
        return self.experiment.score(split)
