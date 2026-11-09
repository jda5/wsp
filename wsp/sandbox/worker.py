"""
The Docker sandbox's worker. It runs inside the container, keeps the agent's
namespace alive, and answers one request at a time:

    run           {code}         -> {output, changed}
    set_weights   {file}         load W from a file in /inbox
    get_weights   {}             save W to /workspace/.weights.npz
    put           {name, file}   load a variable from a file in /inbox

Agent code is ordinary Python. The container provides the limits: no
network, no secrets, and nothing installed but numpy.
"""

import os
from pathlib import Path
from typing import IO

from . import wire
from .namespace import Namespace

# Longer cell output is cut before it is sent; the harness cuts it further.
MAX_OUTPUT_CHARS = 1_000_000


def main() -> None:
    requests, replies = _protocol_streams()
    namespace = Namespace()

    while True:
        try:
            message = wire.receive(requests)
        except KeyboardInterrupt:
            continue  # an interrupt that arrived between cells

        if message is None:
            return  # the harness has gone

        wire.send(replies, _handle(namespace, message))


def _handle(namespace: Namespace, message: dict) -> dict:
    reply = {"id": message.get("id")}

    try:
        reply["result"] = _dispatch(namespace, message)
        reply["ok"] = True
    except (Exception, KeyboardInterrupt) as error:
        reply["error"] = f"{type(error).__name__}: {error}"
        reply["ok"] = False

    return reply


def _dispatch(namespace: Namespace, message: dict) -> object:
    match message["command"]:
        case "run":
            execution = namespace.run(message["code"])
            return {"output": execution.output[:MAX_OUTPUT_CHARS], "changed": execution.changed}

        case "set_weights":
            namespace.set_weights(wire.load_tree(_inbox(message["file"])))
            return None

        case "get_weights":
            if not isinstance(namespace.weights, dict):
                raise TypeError(f"W must be a dict, not {type(namespace.weights).__name__}")
            _save_weights(namespace.weights)
            return None

        case "put":
            tree = wire.load_tree(_inbox(message["file"]))
            namespace.put(message["name"], tree["value"])
            return None

    raise ValueError(f"Unknown command: {message['command']}")


def _inbox(file: str) -> Path:
    return Path(wire.INBOX) / Path(file).name


def _save_weights(weights: dict) -> None:
    """
    Write to a temporary file, then rename it into place. Renaming replaces
    whatever was at the destination, even a symlink, rather than following it.
    """
    destination = Path(wire.WORKSPACE) / wire.WEIGHTS_FILE
    temporary = destination.with_suffix(".tmp.npz")

    wire.save_tree(temporary, weights)
    os.replace(temporary, destination)


def _protocol_streams() -> tuple[IO[str], IO[str]]:
    """
    Keep private copies of stdin and stdout for the protocol, then point the
    originals elsewhere. Agent code that reads stdin or writes straight to
    file descriptor 1 can then never mix with the messages.
    """
    requests = os.fdopen(os.dup(0), "r", encoding="utf-8")
    replies = os.fdopen(os.dup(1), "w", encoding="utf-8")

    os.dup2(os.open(os.devnull, os.O_RDONLY), 0)
    os.dup2(2, 1)

    return requests, replies


if __name__ == "__main__":
    main()
