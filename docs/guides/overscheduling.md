# Overscheduling Guide

Overscheduling puts more trajectories on each sampler than its GPU KV can hold, by admitting only turns that fit and reloading evicted KV from a cross-node Mooncake store instead of recomputing it. It lets an agentic RL rollout run on fewer samplers.

## When it helps

- Trajectories alternate model turns with tool calls. While a trajectory waits on a tool, its KV can leave the GPU and come back from the store for its next turn.
- The rollout is limited by KV memory, not compute. If decode is already compute-bound, fewer samplers only make the rollout longer.

## What it improves

Overscheduling is a price/performance tool: the same rollout needs fewer GPUs.

| Metric | Why it improves |
|---|---|
| Samples per second per GPU | The same rollout runs on fewer samplers, so each GPU produces more samples per second and a rollout costs fewer GPU-hours. |
| KV cache utilization | More trajectories share each sampler, and the gate fills its KV budget up to `budget_fraction` without the preemptions that overfilling would cause. |
| GPU utilization | While some trajectories wait on tools, turns from others keep each sampler busy, and loading evicted KV from the store saves recomputing those prompts. |

## How it works

| Part | What it does |
|---|---|
| Prefill-based flow control | `kv_saturation` admits a turn only to a sampler whose KV budget can hold its prompt plus the average of the trajectory's output lengths. Turns that fit nowhere wait in one queue in the verl fleet actor. |
| Affinity routing | `kv_saturation` keeps each trajectory on the sampler that ran its last turn while it fits there. Otherwise the scorers pick a sampler with room, weighing a hash of the trajectory id (`sticky_session`), cached prefixes (`prefix_cache`) and load (the backpressure scorers). |
| KV offload | vLLM's Mooncake store connector saves each turn's prompt KV to host memory. A turn whose KV was evicted from the GPU loads it back instead of recomputing it. |
| Cross-node KV transfer | Mooncake spreads that host memory across every node over RDMA, so any sampler can load KV saved on another node. |

## Requirements

| Component | Requirement |
|---|---|
| verl | 0.9.x, with the py-inference-scheduler hook set up as in the [verl integration guide](../../integration/verl/README.md) |
| vLLM | 0.29 rollout engines. `kv_saturation` reads each engine's KV capacity from vLLM. |
| Mooncake | `mooncake-transfer-engine-cuda13` 0.3.13.post1, which the vLLM 0.29 image ships, and a Mooncake master of the same version |
| Network | RDMA between nodes, for cross-node transfer |

## Setup

### 1. Scheduler profile

Point `ROUTER_CONFIG_PATH` at a profile like this one:

```yaml
profile_handler:
  type: single_profile
profiles:
  overscheduling:
    flow_control:
      type: kv_saturation
      default_osl: 256        # output tokens you expect per turn
      budget_fraction: 0.94   # headroom so engines never need to preempt
    scorers:
      - type: sticky_session
        header_name: x-rls-session-id
        weight: 4.0
      - type: prefix_cache
        weight: 4.0
      - type: least_queue
        weight: 1.0
      - type: waiting_queue
        weight: 1.0
      - type: kv_cache
        weight: 1.0
    picker:
      type: max_score
```

- `budget_fraction` below 1 leaves headroom, so engines never need to preempt.
- The hook sends each trajectory's id in the `x-rls-session-id` header, which `sticky_session` reads.
- The fleet actor reads the profile once, at the first request, so restart the job to change it.
- See the [Scheduler Customization Guide](scheduler_customization.md) for every parameter.

### 2. Mooncake master and client config

Run a Mooncake master with the same version as the client, with at least 2 CPUs.

Write a client config like this one and make it available on every node:

```json
{
  "mode": "embedded",
  "metadata_server": "P2PHANDSHAKE",
  "master_server_address": "<master-host>:50051",
  "protocol": "rdma",
  "device_name": "",
  "global_segment_size": "128gb",
  "local_buffer_size": "4gb"
}
```

- `global_segment_size` is the host memory each GPU process gives the store. Size the total to hold one rollout's KV: trajectories × peak context × KV bytes per token.
- KV bytes per token is 2 × layers × KV heads × head size × bytes per value.
- An empty `device_name` uses every RDMA NIC on the node.

### 3. Connector and environment

Add these overrides to the verl command:

```
+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config.kv_connector=MooncakeStoreConnector
+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config.kv_connector_module_path=py_inference_scheduler.datalayer.connectors.mooncake.store_connector
+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config.kv_role=kv_both
+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config.kv_connector_extra_config.save_decode_cache=false
```

- Keep `kv_connector_module_path`, or vLLM loads its own connector of the same name, which lacks two fixes this setup needs on vLLM 0.29.
- Keep `save_decode_cache=false`, or engines can stall under KV pressure.

Set these on every Ray worker, in the job's runtime env `env_vars` or the pod spec:

| Variable | Value | Why |
|---|---|---|
| `PYTHONHASHSEED` | `0` | Every engine must hash KV blocks the same way to find each other's keys in the store. |
| `MOONCAKE_CONFIG_PATH` | Path to the client config | Where each engine reads its Mooncake settings. |
| `MC_FORCE_HCA` | `1` | Fails when no RDMA device is found, instead of falling back to TCP. |
| `WITH_NVIDIA_PEERMEM` | `0` on hosts without the nvidia-peermem module, such as GKE COS | Registers GPU memory through dmabuf instead. |

## Check that it works

Watch each engine's vLLM metrics during a rollout:

| Metric | What to expect |
|---|---|
| `vllm:num_preemptions_total` | Stays at 0. If it grows, lower `budget_fraction`. |
| `vllm:kv_cache_usage_perc` | Peaks near `budget_fraction`. |
| `vllm:prompt_tokens_by_source_total{source="external_kv_transfer"}` | Grows as turns load KV from the store. |
| `vllm:mooncake_store_operation_failed_keys_total` | Stays at 0. |

If a check fails, the [overscheduling skill](../../.agents/skills/overscheduling/SKILL.md) maps each symptom to its cause and fix.
