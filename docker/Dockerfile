# The Docker sandbox: Python and numpy, and nothing else.
#
# No machine learning libraries are installed, so they cannot be imported.
# The harness builds this image itself (wsp/sandbox/docker.py), tagged with a
# hash of the code it contains, and starts it with no network.

FROM python:3.14-slim

# The same numpy as the harness, so weights round-trip exactly.
ARG NUMPY_VERSION
RUN pip install --no-cache-dir "numpy==${NUMPY_VERSION}"

COPY wsp /harness/wsp

ENV PYTHONPATH=/harness \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace
ENTRYPOINT ["python", "-m", "wsp.sandbox.worker"]
