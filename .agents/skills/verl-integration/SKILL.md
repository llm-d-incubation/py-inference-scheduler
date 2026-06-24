---
name: verl-integration
description: >-
  Self-healing guide to troubleshoot and fix issues encountered while integrating 
  py-inference-scheduler with the verl framework (v0.7.1) for GRPO training. 
  Use when pods fail, jobs hang, or metrics/routing are broken.
---

# verl Scheduler Integration Skill (Self-Healing Guide)

This skill provides the precise architectural context and diagnostic flows required to debug and self-heal the integration between `py-inference-scheduler` and `verl`. 

Before debugging, always read the [verl Integration README](../../../integration/verl/README.md) to understand the user's setup.

---

## 1. Initial Triage: Establish the Setup

Before executing any diagnostics, you **MUST** ask the user to clarify their setup if it is not already explicitly clear from the conversation history. Ask the user:
1.  **Orchestrator**: Are they running on **Kubernetes (K8s)** or **VMs (Non-K8s)**?
2.  **Backend**: Are they using **vLLM** or **SGLang**?

*This immediately prunes the diagnostic tree (e.g., ignoring K8s ConfigMap errors if they are on VMs).*

---

## 2. Architectural Blueprint (Code Boundaries)

To debug effectively without making assumptions, you must understand how the integration is wired.

### 2.1 Control Flow (Scheduling & Routing)
1.  **Entry Point**: verl loads `PyInferenceAgentLoopManager` ([verl_hook.py](../../../integration/verl/verl_hook.py)), which overrides the worker actor class with `PyInferenceAgentLoopWorker`.
2.  **Manager**: The worker spawns `InferenceSchedulerServerManager` which owns the `Scheduler` (engine) and the `InflightStore` (local queue tracker).
3.  **Scheduling Loop**: When verl requests a generation:
    *   `_acquire_server()` is called. It acquires a lock to prevent concurrent scheduling during batching.
    *   It triggers `fetch_worker_metrics()` for all endpoints.
    *   It queries the `Scheduler` to select the best worker.
    *   It increments the `InflightStore` for the selected worker (to account for lag in Prometheus metrics).
    *   It returns the selected Ray worker handle to verl to execute the generation.
4.  **Release**: Once generation completes, `_release_server()` decrements the `InflightStore`.

### 2.2 Data Flow (Metrics Collection)
*   **Step 1 (Exposure)**: vLLM/SGLang engines expose Prometheus metrics locally on the worker node (e.g., `http://localhost:8000/metrics`).
*   **Step 2 (Monkey Patch)**: `VllmEnginePatch` ([backends/verl/vllm.py](../../../backends/verl/vllm.py)) / `SglangEnginePatch` ([backends/verl/sglang.py](../../../backends/verl/sglang.py)) injects `get_routing_stats()` into the worker server class.
*   **Step 3 (Local Scrape)**: The worker actor performs a local HTTP GET request to its own `/metrics` endpoint and parses it via regex ([datalayer/metrics/verl/vllm.py](../../../datalayer/metrics/verl/vllm.py) / [sglang.py](../../../datalayer/metrics/verl/sglang.py)) to return a clean dictionary (waiting, running, KV cache).
*   **Step 4 (RPC Collection)**: The manager on the head node collects these via Ray RPC (`actor.get_routing_stats.remote()`) inside [fetch_metrics.py](../../../datalayer/metrics/verl/fetch_metrics.py) and merges them with the `InflightStore`.

---

## 3. Pre-Flight Checklist & API Stability

Before deep-diving into logs, verify the environment meets these requirements:

1.  **verl Version**: Recommended to be exactly `verl==0.7.1`.
    *   **Why this strictness?** The integration relies on subclassing and monkey-patching verl's *experimental* agent loop API (`verl.experimental.agent_loop`), which is highly unstable and underwent a major architectural rewrite between `v0.7.0` and `v0.7.1`.
    *   **Key API Dependencies & Historical Context:**
        *   **Worker Spawning:** `PyInferenceAgentLoopManager` ([verl_hook.py](../../../integration/verl/verl_hook.py)) overrides `self.agent_loop_workers_class` in `AgentLoopManager`. If verl changes how it spawns workers or renames this property, the hook will fail to load.
        *   **Manager Injection:** `PyInferenceAgentLoopWorker` injects `InferenceSchedulerServerManager` into `self.server_manager`. This relies on the parent `AgentLoopWorker` class using this exact property name to route requests.
        *   **The 0.7.0 vs 0.7.1 Architectural Shift:**
            *   *In v0.7.0 and earlier:* `AsyncLLMServerManager` managed its own local load balancing. Its `__init__` took `server_handles: list[ray.actor.ActorHandle]` (raw handles without IDs) and used a local method `_choose_server(request_id) -> ray.actor.ActorHandle` to route requests. There was no release/cleanup mechanism.
            *   *In v0.7.1:* verl introduced a global `GlobalRequestLoadBalancer` actor. `AsyncLLMServerManager.__init__` was rewritten to accept `servers: list[tuple[str, ray.actor.ActorHandle]]` (adding string IDs to handles) and a `load_balancer_handle`. Crucially, it introduced the `async def _acquire_server(request_id)` and `def _release_server(server_id)` hooks.
            *   *Our Override:* We rely on the `v0.7.1` structure to intercept `_acquire_server` and `_release_server` to route requests through our native `Scheduler` and track inflight requests.
    *   **How to adapt to other verl versions (Porting Guide):** If you must use a different verl version, you **must** inspect verl's source code (specifically `verl/experimental/agent_loop/agent_loop.py`) to verify:
        1. Whether it uses the `_choose_server` (<=0.7.0) or `_acquire_server` (>=0.7.1) paradigm.
        2. Whether `servers` is passed as a list of raw handles or `(id, handle)` tuples.
        You must then adapt the overrides in [verl_hook.py](../../../integration/verl/verl_hook.py) to match those exact signatures.
2.  **Supported Images**: `verlai/verl:vllm011.latest` or `verlai/verl:sgl059.latest`.
3.  **Shared Metrics Directory**: 
    - **K8s**: An `emptyDir` volume must be mounted at `/tmp/metrics` on **both** head and worker pods.
    - **VM**: `/tmp/metrics` must exist and be writable by Ray on all nodes.
4.  **ConfigMap (K8s Only)**: The `scheduler-config` ConfigMap must be applied *before* the Ray cluster is deployed.

---

## 4. Self-Healing Diagnostic Flow

Follow this progressive diagnostic tree to isolate and fix the exact failure point:

### Step 4.1: Are Pods failing to start? (K8s Phase)
*   **Symptom**: Pods stuck in `PENDING`, `ContainerCreating`, or `ImagePullBackOff`.
*   **Diagnostic**: Run `kubectl describe pod` and look for `CreateContainerConfigError`.
*   **Root Cause**: The `scheduler-config` ConfigMap was not applied before deploying the cluster.
*   **Fix**: Apply the ConfigMap (`kubectl apply -f configs/scheduler_config.yaml`) and recreate the Ray service.

### Step 4.2: Did the Job fail instantly or hang on initialization? (Submission Phase)
*   **Symptom**: Ray job fails immediately after submission, or hangs indefinitely before the first training step.
*   **Diagnostic 1 (Check Mandatory Flags)**: Ensure the submission command contains:
    *   `+actor_rollout_ref.rollout.agent.agent_loop_manager_class=integration.verl.verl_hook.PyInferenceAgentLoopManager`
    *   `actor_rollout_ref.rollout.disable_log_stats=False`
    *   **For SGLang**: `actor_rollout_ref.rollout.prometheus.enable=True`
*   **Diagnostic 2 (Verify Imports)**: If it fails with `ImportError`, verify the worker images have `verl` and the backend engines installed.
*   **Diagnostic 3 (Resource Mismatch)**: If it hangs, verify that the training script resource allocations (GPUs, nodes, TP size) exactly match the Ray cluster's physical resources.

### Step 4.3: Is the Scheduler routing, but using fallback? (Configuration Phase)
*   **Symptom**: Logs show "py-inference-scheduler returned no endpoints, falling back to verl global LB."
*   **Diagnostic**: Check the `ROUTER_CONFIG_PATH` in `runtime-env.yaml`.
    *   If **K8s**: Must be `/etc/scheduler/scheduler.yaml` (absolute path to the mount).
    *   If **VM/Default**: Must be `./integration/verl/examples/scheduler.yaml` (relative path).
*   **Root Cause**: The scheduler engine failed to initialize with the correct configuration, leading to an empty routing decision.

### Step 4.4: Are metrics missing or stuck at 0? (Scraping Phase)
*   **Symptom**: `routing_stats` in logs show 0 waiting/running requests, or KV cache usage is always 0.
*   **Diagnostic 1 (Check Env Vars)**: Verify `PROMETHEUS_MULTIPROC_DIR: "/tmp/metrics"` is set in `runtime-env.yaml`. If missing, Prometheus cannot aggregate metrics across multiproc workers.
*   **Diagnostic 2 (Test Local Scrape)**: Exec into a worker pod and run `curl http://localhost:{port}/metrics` (find port in worker logs).
    *   *If connection refused*: The engine's Prometheus server is not running (verify SGLang prometheus flag is enabled).
    *   *If 200 OK but metrics missing in scheduler*: The monkey patch did not apply. Verify `verl_hook.py` is being loaded and `apply()` is called.
*   **Diagnostic 3 (Regex Mismatch)**: If the page loads with metrics but the scheduler logs show 0, the engine version might have changed the metric names. Check [datalayer/metrics/verl/vllm.py](../../../datalayer/metrics/verl/vllm.py) or [sglang.py](../../../datalayer/metrics/verl/sglang.py) and compare the regexes against the raw `curl` output.