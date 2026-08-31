# Sidecar image for the tunix rollout decision service.
# Build from the repo root:
#   docker build -f integration/tunix/Dockerfile -t py-rl-scheduler:latest .
#
# Deliberately skips `pip install .` (which drags in ray[serve] etc. for the
# other integration shells): the decision service only needs the scheduler
# core, run straight from the source tree.
FROM python:3.12-slim

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pyyaml \
    aiohttp \
    prometheus-client \
    setproctitle

# src layout: package under src/, integration/ shells at the repo root.
ENV PYTHONPATH=/app:/app/src
EXPOSE 8100

ENTRYPOINT ["python", "-m", "integration.tunix"]
CMD ["--host", "0.0.0.0", "--port", "8100", "--config", "integration/tunix/examples/scheduler.yaml"]
