from typing import Literal

from wsp.agents import BaseAgent, ClaudeAgent, HumanAgent


def get_agent(agent_type: Literal["human", "claude"], kwargs: dict) -> BaseAgent:
    match agent_type:
        case "human":
            agent = HumanAgent()
        case "claude":
            agent = ClaudeAgent(**kwargs)
        case _:
            raise ValueError(f"Unknown agent: {agent_type} with kwargs: {kwargs}")

    return agent
