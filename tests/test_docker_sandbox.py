"""
The Docker sandbox: full Python works, and the hard cheats do not.

These need Docker running, and are skipped otherwise. The first run builds
the sandbox image, which takes a minute.

    uv run pytest tests/test_docker_sandbox.py
"""

import shutil
import subprocess
import uuid
from pathlib import Path

import numpy as np
import pytest

from wsp.sandbox import DockerSandbox, SandboxCrashed, SandboxTimeout


def _docker_running() -> bool:
    if shutil.which("docker") is None:
        return False
    return subprocess.run(["docker", "info"], capture_output=True).returncode == 0


pytestmark = pytest.mark.skipif(not _docker_running(), reason="Docker is not running")

WEIGHTS = {
    "dense.kernel": np.arange(12, dtype=np.float32).reshape(4, 3),
    "dense.bias": np.zeros(3, dtype=np.float32),
}


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    # A secret in the harness's environment that must never reach the container.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-leak")

    box = DockerSandbox(
        WEIGHTS, tmp_path, name=f"wsp-test-{uuid.uuid4().hex[:8]}", timeout=3, grace=3
    )
    yield box
    box.close()


def output(sandbox: DockerSandbox, code: str) -> str:
    return sandbox.run(code).output.strip()


# -------------------------------------------------------------------------------- it works


def test_weights_round_trip_exactly(sandbox):
    weights = sandbox.get_weights()

    assert weights.keys() == WEIGHTS.keys()
    for name, array in WEIGHTS.items():
        assert weights[name].dtype == array.dtype
        np.testing.assert_array_equal(weights[name], array)


def test_edits_are_counted_and_read_back(sandbox):
    execution = sandbox.run("W['dense.bias'][:] = 1.5")

    assert execution.changed == {"dense.bias": 3}
    np.testing.assert_array_equal(sandbox.get_weights()["dense.bias"], [1.5, 1.5, 1.5])


def test_full_python_is_available(sandbox):
    code = """
import json, math

def double(x):
    return 2 * x

total = 0
for i in range(4):
    total += double(i)

json.dumps([math.floor(v) for v in (total, 1.5)])
"""
    assert output(sandbox, code) == "'[12, 1]'"


def test_variables_persist_between_cells(sandbox):
    sandbox.run("x = np.ones(3)")
    assert output(sandbox, "x.sum()") == "3.0"


def test_put_places_nested_data(sandbox):
    sandbox.put(
        "forward_result",
        {"labels": np.array([1, 2]), "activations": {"dense": np.ones((2, 3))}},
    )
    assert output(sandbox, "forward_result['activations']['dense'].shape") == "(2, 3)"


def test_replacing_w_with_a_non_dict_is_reported(sandbox):
    sandbox.run("W = 5")
    with pytest.raises(ValueError, match="W must be a dict"):
        sandbox.get_weights()


# -------------------------------------------------------------------------------- the hard cheats are blocked


def test_no_network(sandbox):
    code = """
import socket
try:
    socket.create_connection(("1.1.1.1", 80), timeout=3)
    print("connected")
except OSError as error:
    print("blocked")
"""
    assert output(sandbox, code) == "blocked"


@pytest.mark.parametrize(
    "module", ["tensorflow", "keras", "torch", "jax", "sklearn", "scipy"]
)
def test_no_machine_learning_libraries(sandbox, module):
    assert "ModuleNotFoundError" in output(sandbox, f"import {module}")


def test_no_secrets_from_the_host_environment(sandbox):
    assert output(sandbox, "import os; os.environ.get('ANTHROPIC_API_KEY')") == ""
    assert "must-not-leak" not in output(sandbox, "import os; dict(os.environ)")


def test_host_files_are_not_visible(sandbox):
    repository = Path(__file__).resolve().parents[1]

    for path in [Path.home(), repository / ".env", Path.home() / ".keras"]:
        assert output(sandbox, f"import os; os.path.exists({str(path)!r})") == "False"


def test_inbox_is_read_only(sandbox):
    assert "Read-only file system" in output(sandbox, "open('/inbox/planted.npz', 'w')")


# -------------------------------------------------------------------------------- recovering from trouble


def test_a_long_cell_is_interrupted_and_keeps_variables(sandbox):
    sandbox.run("x = 41")

    interrupted = output(sandbox, "while True:\n    pass")
    assert "KeyboardInterrupt" in interrupted
    assert "Your variables are kept" in interrupted

    assert output(sandbox, "x + 1") == "42"


def test_a_cell_that_ignores_interrupts_times_out_then_restarts(sandbox):
    code = "import signal\nsignal.signal(signal.SIGINT, signal.SIG_IGN)\nwhile True:\n    pass"

    with pytest.raises(SandboxTimeout):
        sandbox.run(code)

    sandbox.restart(WEIGHTS)
    assert output(sandbox, "1 + 1") == "2"


def test_writing_to_file_descriptor_1_does_not_reach_the_harness(sandbox):
    code = "import os\nos.write(1, b'{\"id\": 999, \"ok\": true}\\n')\nprint('done')"
    assert output(sandbox, code) == "done"


def test_corrupting_the_protocol_crashes_the_sandbox_not_the_harness(sandbox):
    code = """
import os
for fd in range(3, 32):
    try:
        os.write(fd, b"junk\\n")
    except OSError:
        pass
"""
    with pytest.raises(SandboxCrashed):
        sandbox.run(code)

    sandbox.restart(WEIGHTS)
    assert output(sandbox, "1 + 1") == "2"


def test_a_planted_symlink_is_replaced_not_followed(sandbox):
    sandbox.run("import os\nos.symlink('/etc/hostname', '/workspace/.weights.npz')")
    np.testing.assert_array_equal(
        sandbox.get_weights()["dense.kernel"], WEIGHTS["dense.kernel"]
    )


def test_the_harness_refuses_to_read_a_symlink(sandbox, tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("private")

    path = sandbox.workspace / ".weights.npz"
    path.unlink(missing_ok=True)
    path.symlink_to(secret)

    with pytest.raises(ValueError, match="Could not read"):
        sandbox._read_weights()


# -------------------------------------------------------------------------------- the guide


def test_every_example_in_the_guide_runs(sandbox):
    from wsp.sandbox.guide import examples, weights_guide

    for line in examples(weights_guide(WEIGHTS, max_output_chars=25_000, memory_gb=4)):
        result = output(sandbox, line)
        assert "Traceback" not in result, f"{line!r} failed:\n{result}"
