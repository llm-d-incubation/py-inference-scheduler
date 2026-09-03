# Pinned Ray/verl image for the SWE RL pipeline ("pets -> cattle": replacement
# pods must come up fully provisioned; see docs/swe_bench_guide.md open items).
#
# Base: the exact digest the cluster was validated against (verlai/verl
# vllm011 as of 2026-08; `latest` is a moving tag - never use it here).
FROM docker.io/verlai/verl@sha256:3ce56ff018516b28ab9c4f4fc09d3aa67589074495ace75e2674b720aa4d0e5d

# verl source at the commit the whole pipeline is validated against
# (hook compat check, SWEAgentLoop integration check, 20-step training run).
# Layout matches the historical manual installs so tooling (preflight commit
# check) keeps working: /tmp/verl/verl is the repo root, installed editable.
RUN mkdir -p /tmp/verl && cd /tmp/verl && \
    git init verl && cd verl && \
    git remote add origin https://github.com/volcengine/verl.git && \
    git fetch --depth 1 origin 334d9f8b03816382cbd72e898bc8ae04efca6fbe && \
    git checkout --quiet FETCH_HEAD && \
    pip install --no-cache-dir -e . --no-deps && \
    pip install --no-cache-dir kubernetes
