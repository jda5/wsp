import anyio
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    query,
)

PROMPT = """
In the folder /Users/jacob/github.com/jda5/wsp/evals/mnist there are 5 files:

- model.py
- dense_layer_weights.json
- dense_layer_biases.json
- output_layer_weights.json
- output_layer_biases.json

Your task is to update the four JSON files containing the weights and biases of the neural network layers
described in the model.py file.

Specifically, you should update the values in the JSON files so that when the neural network defined in model.py is used, 
it produces the highly accurate outputs for the given inputs.

Do NOT modify the model.py file - though you may read it to understand the model structure.

Do NOT train the model yourself; only update the weights and biases in the JSON files, based on your understanding of the model structure
and intuition about how the weights and biases should be adjusted.
""".strip()


async def main():
    options = ClaudeAgentOptions(
        cwd="/Users/jacob/github.com/jda5/wsp/evals/mnist",  # folder the agent works in
        allowed_tools=["Read", "Edit", "Write"],  # file tools only, no Bash
        permission_mode="acceptEdits",  # apply edits without prompting
        model="claude-fable-5-1",
        system_prompt=None,
        thinking={"type": "adaptive"},
        effort="high",
    )

    async for message in query(prompt=PROMPT, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    print("Text block:")
                    print(block.text)

                if isinstance(block, ThinkingBlock):
                    print("Thinking block:")
                    print(block.signature)

                if isinstance(block, ToolUseBlock):
                    print("Tool use block:")
                    print(block.name)

        # print(message)


anyio.run(main)
