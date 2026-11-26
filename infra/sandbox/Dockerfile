# Isolated Python for LLM-written analysis code. Build: make sandbox
# Run flags (set by DockerSandbox): no network, read-only root, no capabilities, non-root user,
# memory/CPU/pid limits; code and data arrive on stdin, results leave on stdout.
FROM python:3.12-slim

RUN pip install --no-cache-dir "pandas==3.1.*" "numpy==2.5.*" "scipy==1.*" \
    && useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin sandbox

COPY src/analytics_platform/tools/sandbox/runner.py /opt/sandbox/runner.py

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HOME=/tmp
USER 10001:10001
WORKDIR /tmp
ENTRYPOINT ["python", "-I", "/opt/sandbox/runner.py"]
