---
name: verl-integration
description: >-
  Self-healing guide to troubleshoot and fix issues encountered while integrating 
  py-inference-scheduler with the veRL framework (v0.7.1) for GRPO training. 
  Use when pods fail, jobs hang, or metrics/routing are broken.
---

# veRL Scheduler Integration Skill (Self-Healing Guide)

This skill provides the precise architectural context and diagnostic flows required to debug and self-heal the integration between `py-inference-scheduler` and `veRL`. 

Before debugging, always read the [veRL Integration README](../../../integration/verl/README.md) to understand the user's setup.

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
1.  **Entry Point**: veRL loads `PyInferenceAgentLoopManager` ([verl_hook.py](../../../integration/verl/verl_hook.py)), which overrides the worker actor class with `PyInferenceAgentLoopWorker`.
2.  **Manager**: The worker spawns `InferenceSchedulerServerManager` which owns the `Scheduler` (engine) and the `InflightStore` (local queue tracker).
3.  **Scheduling Loop**: When veRL requests a generation:
    *   `_acquire_server()` is called. It acquires a lock to prevent concurrent scheduling during batching.
    *   It triggers `fetch_worker_metrics()` for all endpoints.
    *   It queries the `Scheduler` to select the best worker.
    *   It increments the `InflightStore` for the selected worker (to account for lag in Prometheus metrics).
    *   It returns the selected Ray worker handle to veRL to execute the generation.
4.  **Release**: Once generation completes, `_release_server()` decrements the `InflightStore`.

### 2.2 Data Flow (Metrics Collection)
*   **Step 1 (Exposure)**: vLLM/SGLang engines expose Prometheus metrics locally on the worker node (e.g., `http://localhost:8000/metrics`).
*   **Step 2 (Monkey Patch)**: `VllmEnginePatch` ([backends/verl/vllm.py](../../../backends/verl/vllm.py)) / `SglangEnginePatch` ([backends/verl/sglang.py](../../../backends/verl/sglang.py)) injects `get_routing_stats()` into the worker server class.
*   **Step 3 (Local Scrape)**: The worker actor performs a local HTTP GET request to its own `/metrics` endpoint and parses it via regex ([datalayer/metrics/verl/vllm.py](../../../datalayer/metrics/verl/vllm.py) / [sglang.py](../../../datalayer/metrics/verl/sglang.py)) to return a clean dictionary (waiting, running, KV cache).
*   **Step 4 (RPC Collection)**: The manager on the head node collects these via Ray RPC (`actor.get_routing_stats.remote()`) inside [fetch_metrics.py](../../../datalayer/metrics/verl/fetch_metrics.py) and merges them with the `InflightStore`.

---

## 3. Pre-Flight Checklist

Before deep-diving into logs, verify the environment meets these hard requirements:
1.  **veRL Version**: Must be exactly `verl==0.7.1`.
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