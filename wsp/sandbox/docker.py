"""
The Docker sandbox: full Python for the agent, in a container that has

    no network        --network none
    no secrets        no environment variables passed in; only two folders mounted
    no frameworks     the image holds Python and numpy, nothing else (docker/Dockerfile)

The worker inside (worker.py) keeps the agent's namespace alive. This class
is the host side: it starts the container, sends requests, and moves data
through two folders in the run:

    <run>/sandbox/inbox      -> /inbox (read-only)   weights and data sent in
    <run>/sandbox/workspace  -> /workspace           the agent's files; weights sent out

The held-out data, the evaluate tool and the logs never enter the container.
"""

import hashlib
import io
import json
import os
import queue
import re
import stat
import subprocess
import threading
from pathlib import Path

import numpy as np

from . import wire
from .base import Execution, SandboxCrashed, SandboxTimeout, Weights

REPOSITORY = Path(__file__).resolve().parents[2]
DOCKERFILE = REPOSITORY / "docker" / "Dockerfile"
IMAGE_NAME = "wsp-sandbox"

# Weights larger than this are refused when read back from the container.
MAX_WEIGHTS_BYTES = 2**30

# A sentinel the reader thread sends when the worker's output ends.
_ENDED = object()


class DockerSandbox:
    def __init__(
        self,
        weights: Weights,
        folder: str | Path,
        name: str = "wsp-sandbox",
        timeout: float = 600.0,
        grace: float = 30.0,
        memory: str = "4g",
        cpus: float = 2,
    ):
        """
        Args:
            weights: The starting weights.
            folder: Where the inbox and workspace folders go, inside the run.
            name: The container's name, as `docker ps` shows it. A restart
                adds a number, since the old container may still be removing.
            timeout: Seconds a cell may run before it is interrupted (Ctrl-C).
                Deliberately generous: a guard against hangs, not a budget.
            grace: Seconds an interrupted cell gets to stop before the
                container is killed and restarted.
            memory, cpus: Resource limits for the container.
        """
        self.folder = Path(folder).resolve()
        self.inbox = self.folder / "inbox"
        self.workspace = self.folder / "workspace"
        self.inbox.mkdir(parents=True, exist_ok=True)
        self.workspace.mkdir(parents=True, exist_ok=True)

        self.name = name
        self.timeout = timeout
        self.grace = grace
        self.limits = ["--memory", memory, "--cpus", str(cpus)]
        self.image = ensure_image()

        self._starts = 0
        self._files_written = 0
        self._puts: dict[str, str] = {}  # name -> inbox file, replayed after a restart

        self._start()
        self.set_weights(weights)

    # -------------------------------------------------------------------------------- used by the agent

    def run(self, code: str) -> Execution:
        result, interrupted = self._request("run", code=code)

        if not isinstance(result, dict):
            raise SandboxCrashed("The sandbox sent an unexpected reply.")

        output = result["output"]

        if interrupted:
            output += (
                f"\n\n[The cell ran for more than {self.timeout / 60:.0f} minutes and was "
                "interrupted. Your variables are kept.]"
            )

        return Execution(output=output, changed=result["changed"])

    # -------------------------------------------------------------------------------- used by the harness

    def get_weights(self) -> Weights:
        self._request("get_weights")
        return self._read_weights()

    def set_weights(self, weights: Weights) -> None:
        file = self._write_inbox("weights", weights)
        self._request("set_weights", file=file)

    def put(self, name: str, value: object) -> None:
        """
        Place a variable in the namespace. Values are dicts of arrays, arrays
        or numbers. They are replayed if the container is restarted.
        """
        file = self._write_inbox(name, {"value": value})
        self._puts[name] = file
        self._request("put", name=name, file=file)

    def restart(self, weights: Weights) -> None:
        """
        Start a fresh container with ``weights`` and every variable placed
        with `put`. Variables the agent made are lost.
        """
        self.close()
        self._start()
        self.set_weights(weights)

        for name, file in self._puts.items():
            self._request("put", name=name, file=file)

    def close(self) -> None:
        if self._process.poll() is None:
            # check=False: the container may already have stopped on its own.
            subprocess.run(
                ["docker", "kill", self._container], capture_output=True, check=False
            )
        self._process.wait()
        self._stderr.close()

    # -------------------------------------------------------------------------------- the container

    def _start(self) -> None:
        self._starts += 1
        self._container = (
            self.name if self._starts == 1 else f"{self.name}-{self._starts}"
        )

        command = [
            "docker", "run", "--rm", "--interactive", "--init",
            "--name", self._container,
            "--network", "none",
            "--read-only", "--tmpfs", "/tmp:size=512m",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--pids-limit", "256", *self.limits,
            "--user", f"{os.getuid()}:{os.getgid()}",
            "--volume", f"{self.inbox}:{wire.INBOX}:ro",
            "--volume", f"{self.workspace}:{wire.WORKSPACE}",
            self.image,
        ]  # fmt: skip

        # No `env=`: docker passes nothing from the host environment to the
        # container unless asked with --env, which this never does.
        self._stderr = (self.folder / "stderr.log").open("a")
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr,
            text=True,
            encoding="utf-8",
        )

        self._replies: queue.Queue = queue.Queue()
        threading.Thread(target=self._read_replies, daemon=True).start()
        self._next_id = 0

    def _read_replies(self) -> None:
        """
        Runs in a thread: pass each line the worker writes to the queue.
        """
        assert self._process.stdout is not None

        for line in self._process.stdout:
            self._replies.put(line)
        self._replies.put(_ENDED)

    def _request(self, command: str, **arguments) -> tuple[object, bool]:
        """
        Send one request and wait for its reply. Returns the result, and
        whether the request had to be interrupted.
        """
        self._next_id += 1
        message = {"id": self._next_id, "command": command, **arguments}

        assert self._process.stdin is not None

        try:
            wire.send(self._process.stdin, message)
        except (BrokenPipeError, OSError) as error:
            raise SandboxCrashed("The sandbox has stopped.") from error

        interrupted = False
        reply = self._wait(self.timeout)

        if reply is None:
            interrupted = True
            subprocess.run(
                ["docker", "kill", "--signal", "SIGINT", self._container],
                capture_output=True,
                check=False,  # if it fails, the wait below times out and says so
            )
            reply = self._wait(self.grace)

        if reply is None:
            raise SandboxTimeout(
                f"The cell ran for more than {self.timeout / 60:.0f} minutes and did not "
                "stop when interrupted."
            )

        if reply.get("id") != self._next_id:
            raise SandboxCrashed("The sandbox sent an unexpected reply.")

        if not reply.get("ok"):
            raise ValueError(reply.get("error", "The sandbox reported an error."))

        return reply.get("result"), interrupted

    def _wait(self, seconds: float) -> dict | None:
        """
        The next reply, or None if none arrives in time.
        """
        try:
            line = self._replies.get(timeout=seconds)
        except queue.Empty:
            return None

        if line is _ENDED:
            raise SandboxCrashed(
                f"The sandbox stopped (see {self.folder / 'stderr.log'})."
            )

        try:
            reply = json.loads(line)
        except ValueError as error:
            raise SandboxCrashed(
                "The sandbox sent something that is not a reply."
            ) from error

        if not isinstance(reply, dict):
            raise SandboxCrashed("The sandbox sent something that is not a reply.")
        return reply

    # -------------------------------------------------------------------------------- moving data

    def _write_inbox(self, name: str, tree: dict) -> str:
        """
        Save data for the worker. The container sees the inbox read-only, so
        agent code cannot change what the harness writes there.
        """
        self._files_written += 1
        safe_name = re.sub(r"[^A-Za-z0-9_]", "_", name)
        file = f"{self._files_written:04d}-{safe_name}.npz"

        wire.save_tree(self.inbox / file, tree)
        return file

    def _read_weights(self) -> Weights:
        """
        Read the weights the worker saved. Agent code can write to the
        workspace, so refuse anything that is not a plain file: a symlink
        planted there could otherwise point the harness at a file of yours.
        """
        path = self.workspace / wire.WEIGHTS_FILE

        try:
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        except OSError as error:
            raise ValueError(
                f"Could not read the weights from the sandbox: {error}"
            ) from error

        with os.fdopen(descriptor, "rb") as file:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_WEIGHTS_BYTES:
                raise ValueError(
                    "The weights file in the sandbox is not a regular file."
                )
            data = file.read()

        return wire.load_tree(io.BytesIO(data))


# -------------------------------------------------------------------------------- the image


def ensure_image() -> str:
    """
    The sandbox image, built if needed. It is tagged with a hash of the code
    and numpy version it contains, so a change to the worker rebuilds it.
    """
    tag = f"{IMAGE_NAME}:{_image_hash()}"

    # check=False: a non-zero exit just means the image has not been built yet.
    exists = subprocess.run(
        ["docker", "image", "inspect", tag], capture_output=True, check=False
    )
    if exists.returncode == 0:
        return tag

    print(f"Building the sandbox image {tag} (first run only)...")
    subprocess.run(
        [
            "docker", "build",
            "--file", str(DOCKERFILE),
            "--build-arg", f"NUMPY_VERSION={np.__version__}",
            "--tag", tag,
            str(REPOSITORY),
        ],
        check=True,
    )  # fmt: skip
    return tag


def _image_hash() -> str:
    digest = hashlib.sha256(np.__version__.encode())

    for path in [DOCKERFILE, *sorted((REPOSITORY / "wsp").rglob("*.py"))]:
        digest.update(str(path.relative_to(REPOSITORY)).encode())
        digest.update(path.read_bytes())

    return digest.hexdigest()[:12]
