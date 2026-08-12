# vllm-router Integration with py-inference-scheduler

## Compatibility Notice

**This integration is designed for [our vllm-router fork](https://github.com/aniketmohanty82/router/tree/external-policy)**
(branch `external-policy`, based on [vllm-router v0.1.15](https://github.com/vllm-project/router/releases/tag/v0.1.15)),
validated against vLLM 0.23.0 engines. Stock vllm-router has no external-policy hook — the fork is required.

## Architecture

vllm-router is the vLLM ecosystem's Rust router. Our fork adds an external-policy hook: at launch,
`--external-policy-factory` imports a factory and calls it once to obtain a selection callable; on every
request the router calls `select(workers, request_text, headers)` on a Rust thread and routes to the
returned worker index. `None` or any exception falls back to a built-in policy
(`--external-fallback-policy`, default `round_robin`) — the scheduler can never fail a request. Engine
metrics are scraped off the request path by the shared `MetricsPoller`; router-side inflight counts arrive
with each call in the worker dicts. The router keeps full ownership of serving (streaming, retries,
circuit breakers, worker management); we only decide which worker serves each request.

Key components:
- [adapter.py](./adapter.py): bridges the callable contract to `Scheduler` — lazy worker registry,
  metrics polling, index mapping.
- [factory.py](./factory.py): the target of `--external-policy-factory`; builds the `Scheduler` and
  returns `adapter.select`.
- [`__main__.py`](./__main__.py): the `python -m integration.vllm_router` launcher.

---

## Prerequisites (Step 1)

Build and install the fork into the same environment as this repo (the Rust extension needs a Rust
toolchain, and builds against the Python it is installed with):

```bash
git clone -b external-policy https://github.com/aniketmohanty82/router.git
cd router
pip install .
```

> [!NOTE]
> The build compiles the Rust extension silently for several minutes — this is normal.

## Integration Configuration (Step 2)

The routing policy reuses slime's [`examples/scheduler.yaml`](../slime/examples/scheduler.yaml) (the scorers
are engine-agnostic). Edit that file directly to customize, or pass
`--external-scheduler-config /path/to/your.yaml`. See the
[Scheduler Customization Guide](../../docs/scheduler_customization.md).

## Running the Router (Step 3)

Clone this repo and install the scheduler (editable, without its heavyweight optional deps) alongside
the fork from Step 1:

```bash
git clone https://github.com/llm-d-incubation/py-inference-scheduler.git
cd py-inference-scheduler
pip install -e . --no-deps
pip install aiohttp prometheus-client pyyaml setproctitle
```

**Start the router** — run from the repo root. `--external-scheduler-config` is ours; every other flag is
forwarded to vllm-router unchanged (worker discovery, PD flags, timeouts — see `vllm-router --help`):

```bash
python -m integration.vllm_router \
  --external-scheduler-config integration/slime/examples/scheduler.yaml \
  --worker-urls http://worker1:8000 http://worker2:8000 \
  --host 0.0.0.0 --port 30000
```

Workers may also register at runtime through the router's `POST /workers` API; the adapter tracks whatever
worker set the router offers per request. The `/metrics` poll interval is set by
`--external-metrics-interval-ms` (default 100ms).

What the router expects of each worker: it health-gates on `GET /health` **before serving** (it blocks up
to 600s waiting for all `--worker-urls` to answer), scrapes Prometheus text from `GET /metrics` — the
scorers read `vllm:num_requests_waiting`, `vllm:num_requests_running`, and `vllm:kv_cache_usage_perc`
(missing gauges are treated as zero) — and proxies the inference endpoints (`/v1/completions`,
`/v1/chat/completions`, `/generate`, ...). Real vLLM servers provide all of this out of the box.
Note the router renames its process to `vllm::router` (setproctitle), so match that name with
`pkill`/`pgrep`, not `python`.

Set `RLS_DECISION_LOG=1` in the environment to log each request's per-endpoint stats at decision
time. The line is formatted and written inside the scheduling call itself, so it adds routing
latency on every request while enabled — leave it off for benchmarks.

## Verifying Results (Step 4)

The router prints to **stdout** — the terminal where you started it in Step 3. On startup, watch for the
scheduler config loading, the fork assigning registered workers to us (`Assigning policy external`), and
the adapter picking workers up: `Seeded worker <url>` for the startup `--worker-urls` set,
`Tracking worker <url>` for workers registered later via `POST /workers`.

Routing decisions are visible with `RLS_DECISION_LOG=1` (each request's per-endpoint stats at decision
time) or `--log-level debug` (per-scorer raw scores and the selected endpoint for every request).
