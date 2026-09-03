# verl Integration with py-inference-scheduler

## Compatibility Notice

**This integration supports two verl layouts, auto-detected at import time:**

- **verl v0.7.1** ("legacy"): `AsyncLLMServerManager` in `verl.experimental.agent_loop.agent_loop`, server list passed to workers.
- **verl v0.9.x** ("modern"): `LLMServerClient` in `verl.workers.rollout.llm_server` with a `GlobalRequestLoadBalancer` actor owning the server registry; the hook bootstraps its endpoint set by draining the balancer once at first use.

Both expose the same entrypoint flag (`agent_loop_manager_class=integration.verl.verl_hook.PyInferenceAgentLoopManager`). The modern path is validated GPU-free by [hook_compat_check.py](./hook_compat_check.py) (fake server actors + a real load balancer; verifies bootstrap, prefix-sticky routing, and inflight accounting). It uses internal API signatures and is **not backwards compatible** with verl versions earlier than v0.7.1; intermediate releases (v0.8.x) are untested.

## Architecture

`verl` manages its own set of Ray Actors for rollouts. By default, it uses a simple LRU cache or least-requests routing. This integration overrides `verl`'s `AgentLoopManager` to delegate routing decisions to the `py-inference-scheduler` engine.

Key components:
- [verl_hook.py](./verl_hook.py): Contains `InferenceSchedulerServerManager` and `PyInferenceAgentLoopManager` which are injected into the `verl` training loop.
- `InflightStore`: Tracks active requests per worker in real-time to augment slow Prometheus metrics.
- `backends/verl/`: Contains monkey-patches for `vllm` and `sglang` to enable metrics extraction and correct environment propagation.
- `datalayer/metrics/verl/`: Contains backend-specific logic (HTTP scraping) to fetch and parse metrics from the workers.

---

## Prerequisites & Cluster Requirements (Step 1)

Before integrating the scheduler, you must set up a Ray cluster and ensure specific infrastructure conditions are met. If your cluster already satisfies the prerequisites and works with verl, skip to [Integration Configuration](#integration-configuration-step-2).

### Environment & Images
*   **Ray Cluster**: You must have a running Ray cluster. For setup instructions, refer to:
    *   [verl Installation Guide](https://verl.readthedocs.io/en/latest/start/install.html)
    *   [verl Multinode VM Setup](https://verl.readthedocs.io/en/latest/start/multinode.html)
    *   [KubeRay verl Post-Training Guide (K8s)](https://docs.ray.io/en/latest/cluster/kubernetes/examples/verl-post-training.html)
*   **Supported Images**: We strongly recommend using the official pre-built verl images, as they come pre-configured with all necessary dependencies:
    *   **vLLM Backend**: `verlai/verl:vllm011.latest`
    *   **SGLang Backend**: `verlai/verl:sgl059.latest`
> [!NOTE]
> If you must use a custom image, you must ensure that the vLLM or SGLang versions match those in the official images, and that **`verl==0.7.1`** is installed in the environment.

> [!IMPORTANT]
> **You must configure the following resources when creating your Ray cluster:**
> *   **Shared Metrics Directory**: The integration requires a shared, writable directory for Prometheus multiproc metrics.
>     *   **Kubernetes (K8s)**: You must define a shared volume (e.g., an `emptyDir` named `metrics-dir`) and mount it at `/tmp/metrics` on **both** the head and all worker pods.
>     *   **Non-Kubernetes (VMs)**: A directory (defaulting to `/tmp/metrics`) must exist and be writable by the Ray process on **every** node in the cluster.
> *   **Scheduler Config Visibility (K8s Only)**: If running on K8s, a ConfigMap named `scheduler-config` (containing your `scheduler.yaml`) must be mounted to `/etc/scheduler` on all pods.
>     *   *Reference*: See [verl-inference-scheduler.yaml](./examples/verl-inference-scheduler.yaml#L55-L56) to see how these mounts are configured.
>     *   The ConfigMap must be applied **before** deploying the cluster. Failing to do so will cause the pods to get stuck in a `CreateContainerConfigError` state. Refer to [Custom Configuration on Kubernetes (K8s)](#custom-configuration-on-kubernetes-k8s) for instructions on how to apply it.

### Dataset Preprocessing
*   **Your own Training Script**: Ensure your own training dataset is preprocessed and stored in a location accessible by all Ray worker nodes (e.g., via a PVC, GCS bucket, or shared NFS).
*   **Walkthrough Example (Optional)**: If you don't have a dataset and would like to follow along this integration with an example, you can use verl's preprocessed GSM8K/MATH datasets. Refer to the [verl Quickstart Data Preparation](https://verl.readthedocs.io/en/latest/start/quickstart.html) for instructions.
> [!WARNING]
> **Data Path Alignment for walkthrough script**: Our example training scripts contain hardcoded paths (e.g., `/home/ray/data/...`). You must ensure your preprocessed data is mounted/copied to this exact path on the workers, or edit the path variables at the top of the scripts before running them.

---

## Integration Configuration (Step 2)

The scheduler configuration (`scheduler.yaml`) must be made visible to the Ray job. Choose one of the following paths depending on your customization needs:

### Default Configuration
By default, the integration uses the `scheduler.yaml` bundled with the repository, located at `integration/verl/examples/scheduler.yaml` relative to the repository root. This is resolved automatically by the default `runtime_env` configuration.

You can use our default [runtime-env.yaml](./examples/runtime-env.yaml) file:
```yaml
working_dir: "https://github.com/llm-d-incubation/py-inference-scheduler/archive/refs/heads/main.zip"
env_vars:
  PYTHONPATH: "."
  PROMETHEUS_MULTIPROC_DIR: "/tmp/metrics"
  ROUTER_CONFIG_PATH: "./integration/verl/examples/scheduler.yaml" # Relative to CWD (repo root)
```
Ray unpacks the zip, sets the CWD to the repo root, and resolves the relative path to the default config.

### Custom Configuration on Kubernetes (K8s)
If you want to customize the scheduler settings on K8s:
*   Modify the [scheduler_config.yaml](../../configs/scheduler_config.yaml) ConfigMap manifest locally.
*   Apply the updated ConfigMap to your cluster:
    ```bash
    kubectl apply -f configs/scheduler_config.yaml
    ```
*   **Update [runtime-env.yaml](./examples/runtime-env.yaml)**: You **must** change `ROUTER_CONFIG_PATH` to the absolute mount path:
    ```yaml
    env_vars:
      ROUTER_CONFIG_PATH: "/etc/scheduler/scheduler.yaml" # Absolute path to K8s mount, or wherever your scheduler config resides
    ```

### Custom Configuration on VMs (Non-K8s)
If you want to customize the scheduler settings in a VM-based cluster:
*   **Clone the repository** locally:
    ```bash
    git clone https://github.com/llm-d-incubation/py-inference-scheduler.git
    cd py-inference-scheduler
    ```
*   **Modify the configuration** locally at `integration/verl/examples/scheduler.yaml`.
*   **Update [runtime-env.yaml](./examples/runtime-env.yaml)** to use your local directory as the `working_dir` (using `.`):
    ```yaml
    working_dir: "." # Packages the local repo root when submitted from there
    pip:
      - "verl==0.7.1"
    env_vars:
      PYTHONPATH: "."
      PROMETHEUS_MULTIPROC_DIR: "/tmp/metrics"
      ROUTER_CONFIG_PATH: "./integration/verl/examples/scheduler.yaml" # Relative to CWD (repo root)
    ```
*   **Submit the job from the repository root** (see Section 3). Running from the root with `working_dir: "."` ensures Ray packages the entire repository (including the `scheduling` source code and your modified config), preventing import errors on the workers.

> [!TIP]
> **Designing Custom Profiles**: To learn more about the available scorers, filters, pickers, and flow-control plugins you can use to customize your scheduling policies, refer to the comprehensive [Scheduler Customization Guide](../../docs/scheduler_customization.md).

---

## Running a Training Job (Step 3)

You submit training jobs to the Ray cluster using `ray job submit`.

### Connecting to the Cluster
Ensure your Ray cluster is running, then establish connection:
*   **Kubernetes (K8s)**: Once your pods are in the `Running` state, open a local tunnel to the Ray Head Node dashboard:
    ```bash
    kubectl port-forward svc/<ray-head-svc-name> 8265:8265 -n <namespace> &
    export RAY_ADDRESS="http://127.0.0.1:8265"
    ```
*   **Non-Kubernetes (VMs)**: Ensure you have network access to the Ray Head node, and export its address:
    ```bash
    export RAY_ADDRESS="http://<head-node-ip>:8265"
    ```

### Submission Commands
Ensure you are in the directory containing your configured [runtime-env.yaml](./examples/runtime-env.yaml) (or the repository root if using `working_dir: "."`).

*   **Your own Training Script**:
    Submit your job by pointing to your custom Python script and passing the required integration flags:
    ```bash
    ray job submit \
        --address http://localhost:8265 \
        --runtime-env integration/verl/examples/runtime-env.yaml \
        -- python3 path/to/your/custom_training_script.py \
            --your-training-flags... \
            actor_rollout_ref.rollout.disable_log_stats=False \
            +actor_rollout_ref.rollout.agent.agent_loop_manager_class=integration.verl.verl_hook.PyInferenceAgentLoopManager  # This override registers our custom scheduler hook with verl
    ```
> [!IMPORTANT]
> **Required Flags for your own scripts**:
> If you are writing your own training script, you **must** pass the following flags to `python3 -m verl.trainer.main_ppo` for the integration to function:
> *   `+actor_rollout_ref.rollout.agent.agent_loop_manager_class=integration.verl.verl_hook.PyInferenceAgentLoopManager`
>     *   *Purpose*: Registers our custom scheduler hook with verl.
> *   `actor_rollout_ref.rollout.disable_log_stats=False`
>     *   *Purpose*: Enables logging of statistics necessary for metrics parsing.
> *   `actor_rollout_ref.rollout.prometheus.enable=True` (**SGLang Only**)
>     *   *Purpose*: Required for SGLang to enable the Prometheus metrics endpoint. Without this, the scheduler cannot retrieve backend stats.

*   **Walkthrough Example (Optional)**:
    Submit the job by pointing to one of our pre-configured shell scripts (which already include the required flags):
    ```bash
    ray job submit \
        --address http://localhost:8265 \
        --runtime-env integration/verl/examples/runtime-env.yaml \
        -- bash integration/verl/examples/run_qwen2_5-32b_math.sh
    ```
    *(Replace `run_qwen2_5-32b_math.sh` with `run_qwen2_5-32b_math_sglang.sh` if using the SGLang backend).*
> [!WARNING]
> **Resource Alignment**: If you're following walkthrough example, ensure the resource allocation flags in your training script (e.g., `trainer.n_gpus_per_node`, `trainer.nnodes`, and `tensor_model_parallel_size`) exactly match your Ray cluster's physical resources. Mismatches will cause the job to hang indefinitely in Ray or fail with OOMs.

---

## 4. Verifying Results (Step 4)

Once the job is submitted, you can monitor its progress:

*   Open the Ray Dashboard (typically at `http://localhost:8265`).
*   Check the **Jobs** tab and view the logs for your submission.
*   If the scheduler is active and making decisions, you will see logs indicating config reloads (if modified) and routing decisions in the worker/driver logs.

Viewing the logs on the actual ray submit job where you've ran the training job is the best place for logs. This can be found either in the *Overview* or *Jobs* tab of the Ray Dashboard (localhost:8265). verl gives us output by step. Don't be concerned if you see `ppo` tags on the labels for the logs—veRL uses the same testing infrastructure for its GRPO and PPO runs. Our script is a GRPO trainer. Enabling the scheduler does not change verl's standard telemetry behavior - if you have your `WANDB_API_KEY` set, metrics will still populate in Weights & Biases as normal.

Logs look as follows (from verl-vllm using gsm8k):

```bash
(TaskRunner pid=15930, ip=10.4.1.33) step:1
 - global_seqlen/min:255486
 - global_seqlen/max:304956
 - global_seqlen/minmax_diff:49470
 - global_seqlen/balanced_min:274237
 - global_seqlen/balanced_max:274268
 - global_seqlen/mean:274260.8125
 - actor/entropy:0.36423397064208984
 - perf/mfu/actor_infer:0
 - actor/pg_loss:0.05410893690350349
 - actor/kl_loss:0.0013340971445359173
 - actor/pg_clipfrac:0.00044460127605816524
 - actor/ppo_kl:2.8409333591383756e-05
 - actor/pg_clipfrac_lower:0.0
 - actor/kl_coef:0.0010000000000000005
 - actor/grad_norm:0.167236328125
 - perf/mfu/actor:0.3707644223640674
 - perf/max_memory_allocated_gb:86.77569150924683
 - perf/max_memory_reserved_gb:94.625
 - perf/cpu_memory_used_gb:144.76549291610718
 - actor/lr:1e-06
 - training/global_step:1
 - training/epoch:0
 - critic/score/mean:0.5760498046875
 - critic/score/max:1.0
 - critic/score/min:0.0
 - critic/rewards/mean:0.5760498046875
 - critic/rewards/max:1.0
 - critic/rewards/min:0.0
 - critic/advantages/mean:-0.0660763531923294
 - critic/advantages/max:2.4748666286468506
 - critic/advantages/min:-2.4748666286468506
 - critic/returns/mean:-0.0660763531923294
 - critic/returns/max:2.4748666286468506
 - critic/returns/min:-2.4748666286468506
 - response_length/mean:424.5494384765625
 - response_length/max:1024.0
 - response_length/min:26.0
 - response_length/clip_ratio:0.07373046875
 - response_length_non_aborted/mean:424.5494384765625
 - response_length_non_aborted/max:1024.0
 - response_length_non_aborted/min:26.0
 - response_length_non_aborted/clip_ratio:0.07373046875
 - response/aborted_ratio:0.0
 - prompt_length/mean:111.1162109375
 - prompt_length/max:994.0
 - prompt_length/min:42.0
 - prompt_length/clip_ratio:0.0
 - num_turns/min:2
 - num_turns/max:2
 - num_turns/mean:2.0
 - timing_s/start_profile:0.00027797603979706764
 - timing_s/agent_loop/num_preempted/min:-1
 - timing_s/agent_loop/num_preempted/max:-1
 - timing_s/agent_loop/num_preempted/mean:-1.0
 - timing_s/agent_loop/generate_sequences/min:1.597659296123311
 - timing_s/agent_loop/generate_sequences/max:20.618110499111935
 - timing_s/agent_loop/generate_sequences/mean:8.733964191658202
 - timing_s/agent_loop/tool_calls/min:0.0
 - timing_s/agent_loop/tool_calls/max:0.0
 - timing_s/agent_loop/tool_calls/mean:0.0
 - timing_s/agent_loop/slowest/generate_sequences:20.618110499111935
 - timing_s/agent_loop/slowest/tool_calls:0.0
 - timing_s/agent_loop/slowest/prompt_length:54
 - timing_s/agent_loop/slowest/response_length:1024
 - timing_s/agent_loop/slowest/num_preempted:-1
 - timing_s/gen:24.533394969068468
 - timing_s/reward:3.863382153213024e-05
 - timing_s/old_log_prob:13.635552056133747
 - timing_s/ref:12.547897297888994
 - timing_s/adv:0.17872913694009185
 - timing_s/update_actor:35.83383133285679
 - timing_s/update_weights:5.458046867046505
 - timing_s/step:93.52434759191237
 - timing_s/stop_profile:5.5649783462285995e-05
 - timing_per_token_ms/gen:0.0070540646604233944
 - timing_per_token_ms/adv:4.0729738080082955e-05
 - timing_per_token_ms/ref:0.0028594809953684584
 - timing_per_token_ms/update_actor:0.008166002418969533
 - perf/total_num_tokens:4388173
 - perf/time_per_step:93.52434759191237
 - perf/throughput:2932.5070910595373
```

Key metrics to look out for are `perf/throughput` for sampling throughput and `timing_s/agent_loop/slowest/generate_sequences` for your tail latency. Additionally, if you enable preemptions, `timing_s/agent_loop/slowest/num_preempted` can be useful too.