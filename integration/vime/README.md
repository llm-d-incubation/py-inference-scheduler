# vime Integration with py-inference-scheduler

## Compatibility Notice

**This integration is designed for [vime v0.3.0](https://github.com/vllm-project/vime/tree/v0.3.0)**, validated
against the official image `inferactinc/public:vime-latest` (vLLM 0.23.0).

## Architecture

vime manages its own vLLM rollout engines and, by default, launches its own `vllm-router` to load balance
across them. When you set `--vllm-router-ip/--vllm-router-port`, vime skips that router and instead each engine
self-registers with ours (`POST /workers`) and the rollout posts generations to it
(`POST /inference/v1/generate`). On each request the router scrapes the engines' Prometheus `/metrics` and
delegates the routing decision to `py-inference-scheduler`. vime keeps full ownership of the rollout lifecycle;
we only decide which engine serves each request.

Key components:
- [server.py](./server.py): the router — worker registry + the scheduled `/inference/v1/generate` proxy
  (reuses slime's shared router core).
- [`__main__.py`](./__main__.py): the `python -m integration.vime` launcher.
- [datalayer/metrics/vime/](../../src/py_inference_scheduler/datalayer/metrics/vime): per-request vLLM Prometheus `/metrics` scrape.

---

## Prerequisites (Step 1)

This integration follows vime's
[Quick Start](https://github.com/vllm-project/vime/blob/v0.3.0/docs/en/get_started/quick_start.md). For all
non-scheduler steps, follow the guide as directed:

| Task | vime Quick Start |
|---|---|
| Environment / image / install vime | [Basic Environment Setup](https://github.com/vllm-project/vime/blob/v0.3.0/docs/en/get_started/quick_start.md#basic-environment-setup) |
| Download model + dataset | [Model and Dataset Download](https://github.com/vllm-project/vime/blob/v0.3.0/docs/en/get_started/quick_start.md#model-and-dataset-download) |
| Convert HF → Megatron checkpoint | [Model Weight Conversion](https://github.com/vllm-project/vime/blob/v0.3.0/docs/en/get_started/quick_start.md#model-weight-conversion) |

> [!NOTE]
> - The guide's model repo `zai-org/Qwen3-4B` is stale — use `Qwen/Qwen3-4B`.
> - Validated against **vime v0.3.0** (image `inferactinc/public:vime-latest`), **vLLM 0.23.0**, and
>   **vllm-router 0.1.14**. If you hit issues on a newer image, pin to these versions.

## Integration Configuration (Step 2)

The routing policy reuses slime's [`examples/scheduler.yaml`](../slime/examples/scheduler.yaml) (the scorers
are engine-agnostic). Edit that file directly to customize, or pass `--config /path/to/your.yaml`. See the
[Scheduler Customization Guide](../../docs/guides/scheduler_customization.md).

## Running a Training Job (Step 3)

Clone this repo onto the VM (the head node for multi-node) and install the router's dependencies on top of the
vime image:

```bash
git clone https://github.com/llm-d-incubation/py-inference-scheduler.git
cd py-inference-scheduler
pip install fastapi uvicorn aiohttp prometheus-client pyyaml setproctitle
```

**Start the router** — CPU-only, run from the repo root, before the vime job (engines register at boot). It
renames its process to `scheduler` so vime's run scripts' `pkill -9 python` cleanup won't kill it. For multi-node,
run it on **node 0 (the head)**; `--host 0.0.0.0` makes it reachable from worker nodes.

```bash
python -m integration.vime --host 0.0.0.0 --port 8000
```

Then point vime at it — the **only** change to vime's launch is two flags added to `VLLM_ARGS` in the run
script:

### Single node — [Quick Start: Training Script](https://github.com/vllm-project/vime/blob/v0.3.0/docs/en/get_started/quick_start.md#training-script-and-parameter-overview)

Run `bash scripts/run-<model>.sh` as documented, adding the two flags to its `VLLM_ARGS`. The router and
the engines run on the same VM, so the engines reach the router at `127.0.0.1`:

```bash
   --vllm-router-ip   127.0.0.1
   --vllm-router-port 8000
```

### Multi node — [Quick Start: Multi-Node Training](https://github.com/vllm-project/vime/blob/v0.3.0/docs/en/get_started/quick_start.md#multi-node-training-for-large-scale-moe-models)

Follow the Ray cluster and `ray job submit` exactly as documented. Start the router on **node 0** (as above),
and set `--vllm-router-ip` to the **head node's IP** (the `${MASTER_ADDR}` you gave `ray start --head`):

```bash
   --vllm-router-ip   ${MASTER_ADDR}
   --vllm-router-port 8000
```

> [!NOTE]
> vime's example script in the README is single-node, so the script needs a few edits. Set
> `--actor-num-nodes <N>` to however many nodes you have and adjust the Megatron parallelism to span all nodes' GPUs.

## Verifying Results (Step 4)

The router prints its routing decisions to **stdout** — the terminal where you started it in Step 3. Watch for
the engines registering (`POST /workers`) and generations routing through the scheduler
(`POST /inference/v1/generate`).

The scheduler only affects the **rollout** (generation) phase, so vime's per-rollout `perf` line is the most
direct signal:
```
perf 0: {'perf/rollout_time': 79.76, 'perf/tokens_per_gpu_per_sec': 2594.0, 'perf/longest_sample_tokens_per_sec': 102.7, ...}
```
- `perf/rollout_time` — wall-clock for the rollout to complete (**lower is better**); the headline number.
- `perf/tokens_per_gpu_per_sec` — aggregate generation throughput per GPU (**higher is better**).
- `perf/longest_sample_tokens_per_sec` — throughput of the tail (slowest) sample; rises when routing cuts
  contention on the bottleneck (**higher is better**).
