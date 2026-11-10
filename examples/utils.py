import argparse
from typing import Literal

from wsp.agents import BaseAgent, ClaudeAgent, HumanAgent

EFFORTS = ["low", "medium", "high", "xhigh", "max"]


def agent_arguments(
    parser: argparse.ArgumentParser, model: str = "claude-opus-5-5"
) -> None:
    """
    The command-line options every example shares: who runs the experiment,
    and, for Claude, which model at what effort.
    """
    parser.add_argument("--agent", choices=["claude", "human"], default="claude")
    parser.add_argument("--model", default=model)
    parser.add_argument("--effort", choices=EFFORTS, default="high")


def get_agent(agent_type: Literal["human", "claude"], kwargs: dict) -> BaseAgent:
    match agent_type:
        case "human":
            agent = HumanAgent()
        case "claude":
            agent = ClaudeAgent(**kwargs)
        case _:
            raise ValueError(f"Unknown agent: {agent_type} with kwargs: {kwargs}")

    return agent
