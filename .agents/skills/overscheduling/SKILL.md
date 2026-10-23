---
name: overscheduling
description: >-
  Code map, configuration rationale and diagnostic tree for overscheduling with verl: prefill-based
  flow control through the fleet actor, affinity routing, and Mooncake KV offload with cross-node
  transfer. Use when engines preempt or stall, the store serves no KV, KV stays far below its
  budget, turns pile onto one engine, or rollouts run slower than without flow control.
---

# Overscheduling Skill

Overscheduling runs an agentic RL rollout on fewer samplers than its trajectories' combined KV needs. It admits only turns that fit and reloads evicted KV from a Mooncake store instead of recomputing it.

Before debugging, read the [overscheduling guide](../../../docs/guides/overscheduling.md) for the user's setup steps. For problems in the verl hook itself, such as pods, job submission or metrics scraping, use the [verl integration skill](../verl-integration/SKILL.md).

---

## 1. Initial Triage

Ask the user, unless the conversation already answers it:

1. **Samplers**: how many, which model, and what tensor-parallel size?
2. **Nodes**: one node, or several with RDMA between them?
3. **Versions**: which vLLM runs in the engines, and which Mooncake version do the engines and the master run?
4. **Workload**: about how many trajectories run at once, and how long do tool calls take?

---

## 2. Architectural Blueprint

### 2.1 Code map

| Part | What it does | Code |
|---|---|---|
| Flow control | Admits a turn only where its prompt plus the trajectory's last output fits the engine's KV budget | `kv_saturation` in [kv_saturation.py](../../../src/py_inference_scheduler/plugins/flow_control/kv_saturation.py) |
| Admission | Keeps one ledger and one queue for every agent-loop worker | `AdmissionFleet` in [fleet.py](../../../integration/verl/fleet.py), `Admission` in [admission.py](../../../integration/verl/admission.py) |
| Affinity | Keeps a trajectory on the engine that holds its KV | `kv_saturation`'s last-engine rule, [sticky_session.py](../../../src/py_inference_scheduler/plugins/scorers/sticky_session.py), [prefix_plugin.py](../../../src/py_inference_scheduler/plugins/scorers/prefix_plugin.py) |
| Session id | Lets `sticky_session` tell trajectories apart | The hook sends verl's request id as `x-rls-session-id` in [verl_hook.py](../../../integration/verl/verl_hook.py) |
| KV capacity | What `budget_fraction` is a fraction of | vLLM's block count times block size, read in [vllm.py](../../../src/py_inference_scheduler/datalayer/metrics/verl/vllm.py) and kept by [fetch_metrics.py](../../../src/py_inference_scheduler/datalayer/metrics/verl/fetch_metrics.py) |
| KV offload and cross-node transfer | Saves each turn's prompt KV to Mooncake and loads it back for the next turn | `MooncakeStoreConnector` in [store_connector.py](../../../src/py_inference_scheduler/datalayer/connectors/mooncake/store_connector.py) |

### 2.2 One turn with flow control

1. verl asks the hook for an engine (`_acquire_server` in [verl_hook.py](../../../integration/verl/verl_hook.py)).
2. The hook waits until the trajectory's previous `finish` has run (`PendingFinishes`). verl reuses one request id for every turn, so a late finish would otherwise free the new turn's reservation.
3. The `AdmissionFleet` finds the engines that can hold the turn: its prompt tokens plus the trajectory's last output (`default_osl` on turn one), within `budget_fraction` of the engine's capacity minus what admitted turns reserved.
4. If the engine that ran the trajectory's last turn fits, it wins. Otherwise the profile's scorers pick among the engines that fit. The fleet reserves the tokens and returns the engine.
5. If no engine fits, the turn waits in the fleet's queue. Each finish and each metrics poll retries the queue, longest prompt first.
6. When generation ends, the hook sends `finish` with the turn's output length. That frees the reservation and becomes the trajectory's last output for its next turn.

The fleet polls engine metrics every `RLS_METRICS_INTERVAL_MS` (default 100). It loads the profile once, at the first request, because a reload would replace the plugin and drop its reservations.

---

## 3. Configuration and Why Each Setting Matters

| Setting | Required value | Why | Symptom when wrong |
|---|---|---|---|
| `flow_control` in the profile | Exactly one `kv_saturation` | The fleet admits through a single flow-control plugin | The fleet raises "verl admission needs exactly one flow_control plugin." |
| `budget_fraction` | Below 1, start at 0.94 | The gate counts tokens while vLLM allocates whole blocks, and a burst of first turns can land before engine metrics catch up | Preemptions |
| `default_osl` | Near the real output tokens per turn | First turns reserve their prompt plus `default_osl`, later turns use the last real output | KV peaks far below the budget when too high, first-turn preemptions when too low |
| `sticky_session` `header_name` | `x-rls-session-id` | `sticky_session` hashes each trajectory to an engine, which spreads trajectories out | Trajectories whose prompts share a prefix all land on one engine |
| Engines | vLLM 0.29 | `kv_saturation` reads each engine's KV capacity from vLLM's cache config | "No engine reports its KV capacity, which kv_saturation needs." |
| `kv_connector_module_path` | `py_inference_scheduler.datalayer.connectors.mooncake.store_connector` | Without it vLLM loads its own connector of the same name. That one names its lookup socket after the host, so engines on one host collide, and it leaves KV blocks pinned after store jobs queued on a step with no forward pass. | Engines on one host collide at startup, or an engine's KV fills up with nothing running |
| `save_decode_cache` | `false` | On vLLM 0.29, saving decode KV pins a request's whole block table, so under KV pressure an engine can loop between preempting and re-saving | An engine sits at full KV with nothing running while its saves repeat |
| `PYTHONHASHSEED` | `0` on every worker | Every engine must hash KV blocks the same way to find each other's keys | The store serves no KV |
| Mooncake master | Same version as the client, at least 2 CPUs | Engines look keys up in the master on every scheduler step | Failed store operations, or slower scheduler steps |
| `MC_FORCE_HCA` | `1` | Fails loudly when no RDMA device is found | Silent fallback to TCP |
| `WITH_NVIDIA_PEERMEM` | `0` on hosts without the nvidia-peermem module | Registers GPU memory through dmabuf instead | GPU memory registration fails |

---

## 4. Diagnostic Flow

### 4.1 No turn gets an engine at job start
*   **Symptom**: the rollout hangs before the first turn, or fails with "No engine reports its KV capacity, which kv_saturation needs."
*   **Cause**: the engines report no KV capacity, so no turn fits. SGLang engines never report it. vLLM reports it once its cache is sized and the fleet has polled it.
*   **Fix**: use vLLM engines. If they are vLLM, check that the fleet's poller reaches them (the verl integration skill, metrics section).

### 4.2 Preemptions grow
*   **Symptom**: `vllm:num_preemptions_total` rises.
*   **Cause**: real KV use overshoots the gate's token count, from block rounding, a burst of first turns before metrics catch up, a `default_osl` below the real first-turn output, or the connector holding blocks while it saves.
*   **Fix**: lower `budget_fraction` a little at a time, for example from 0.94 to 0.9. Set `default_osl` near the real output length. Confirm `save_decode_cache=false`.

### 4.3 An engine stalls
*   **Symptom**: one engine sits at full KV with nothing running and requests waiting, while its store saves repeat.
*   **Cause**: `save_decode_cache=true` on vLLM 0.29, or vLLM's own connector instead of ours.
*   **Fix**: set `save_decode_cache=false` and keep `kv_connector_module_path`.

### 4.4 The store serves no KV
*   **Symptom**: `vllm:prompt_tokens_by_source_total{source="external_kv_transfer"}` stays flat, or `vllm:mooncake_store_operation_failed_keys_total` grows.
*   **Check, in order**: `kv_connector_module_path` points at our connector, `MOONCAKE_CONFIG_PATH` exists on every node, `PYTHONHASHSEED=0` is set on every worker, the master is reachable and runs the client's version, and RDMA devices are found (`MC_FORCE_HCA=1` makes a missing device fail loudly).

### 4.5 KV stays far below the budget
*   **Symptom**: `vllm:kv_cache_usage_perc` peaks far below `budget_fraction`.
*   **Cause**: the samplers hold more KV than the workload needs, which is the case overscheduling is for, or a `default_osl` far above real output lengths makes first turns over-reserve.
*   **Fix**: run fewer samplers, or lower `default_osl` toward the real output length.

### 4.6 Turns pile onto one or two engines
*   **Check**: count decisions per engine. Each decision logs `Selected endpoint [ScoredEndpoint(endpoint=Endpoint(name='<engine>'`, so count those lines by engine name.
*   **Cause**: the hook sends no `x-rls-session-id` header, or `sticky_session` reads a different header name. When prompts share a prefix, `prefix_cache` then pulls every trajectory to one engine, and a KV budget that never fills never pushes them elsewhere.
*   **Fix**: make sure the hook sends the header and that `header_name` matches it.

### 4.7 Rollouts slower than without flow control
*   **Check first**: the per-engine spread (4.6), and that both runs use the same scorers, so the comparison changes only the `flow_control` block.
*   **Check the fleet's node**: it should run on a node that runs agent-loop workers. Its log lines read `(AdmissionFleet pid=..., ip=...)`, and a missing `ip=` means the driver's node.
*   **Known limit**: every admit and finish crosses the one `AdmissionFleet` actor. With many trajectories, short turns and almost no tool time, that actor can become the bottleneck. Rollouts with real tool calls stay far below that rate.

### 4.8 Fallback to verl's balancer
*   **Symptom**: "py-inference-scheduler returned no endpoints, falling back to verl global LB."
*   **Fix**: see step 4.3 of the [verl integration skill](../verl-integration/SKILL.md), which covers `ROUTER_CONFIG_PATH`.
