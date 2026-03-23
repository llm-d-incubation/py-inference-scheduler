# verl Integration with py-inference-scheduler

## Architecture

`verl` manages its own set of Ray Actors for rollouts. By default, it uses a simple LRU cache or least-requests routing. This integration overrides `verl`'s `AgentLoopManager` to delegate routing decisions to the `py-inference-scheduler` engine.

Key components:
- `verl_hook.py`: Contains `InferenceSchedulerServerManager` and `PyInferenceAgentLoopManager` which are injected into the `verl` training loop.
- `InflightStore`: Tracks active requests per worker in real-time to augment slow Prometheus metrics.
- Prometheus Metric polling: The manager polls worker metrics at 50ms to maintain a view of cluster load.

## Setup Guide

If you have a `verl` instance and want to enable scheduling:

### 1. Requirements

Ensure you have both `verl` and `py-inference-scheduler` repositories available.

### 2. Configure Docker Registry

The `RayCluster` example configuration (`examples/verl-inference-scheduler.yaml`) points to a placeholder image. You must build your own Docker image and host it in your own container registry to ensure reproducibility.

1. **Build and push your image**: Build the `verl-ray` image using the provided Dockerfiles in the repository and push it to your own container registry (e.g., Docker Hub, AWS ECR, or your own GCP Artifact Registry).
1. **Update the YAML configuration**: Open `examples/verl-inference-scheduler.yaml` and replace all occurrences of `us-central1-docker.pkg.dev/gke-shared-ai-dev/verl-grpo/verl-ray:latest` with your newly pushed image URL.
1. **Configure Image Pull Secrets (Optional)**: If your registry is private, you need to [create a Kubernetes docker-registry secret](https://kubernetes.io/docs/tasks/configure-pod-container/pull-image-private-registry/#create-a-secret-by-providing-credentials-on-the-command-line) to allow nodes to pull the image. Ensure the `imagePullSecrets` configuration in the `verl-inference-scheduler.yaml` file points to the name of your new secret instead of the default `artifact-registry-secret`.

Run the following command as an example to create a secret if you are using GCP Artifact Registry:

```bash
kubectl create secret docker-registry artifact-registry-secret \
    --docker-server=us-central1-docker.pkg.dev \
    --docker-username=oauth2accesstoken \
    --docker-password=$(gcloud auth print-access-token) \
    --dry-run=client -o yaml | kubectl apply -f -
```

### 3. Configure the Ray Cluster

Use the provided example `RayCluster` configuration in `examples/` as a starting point. It specifically targets GKE A3 Ultra instances with H200 GPUs.

```bash
kubectl apply -f examples/verl-inference-scheduler.yaml
```

### 4. Establish Head Node Connection

Before you can submit a job, you must open a local tunnel to the Ray Head Node dashboard.

1.  **Establish port forward**:
    ```bash
    kubectl port-forward svc/verl-inference-scheduler-head-svc 8265:8265 -n default &
    ```
1.  **Export the Ray address** so the CLI knows where to target:
    ```bash
    export RAY_ADDRESS="http://127.0.0.1:8265"
    ```
1.  **View the Dashboard**: Visit `http://localhost:8265` in your browser to monitor job progress and cluster health.

### 5. Submission Configuration

To use the scheduler, you need to provide a `runtime-env.yaml` and a `scheduler.yaml` (`scheduler.yaml` and `configs/scheduler-configmap.yaml` are the same file - you modify these with your ideal scorer config) in your `verl` working directory.

1.  **Copy the examples** to your `verl` root:
    ```bash
    cp ../py-inference-scheduler/integration/verl/examples/runtime-env.yaml .
    cp ../py-inference-scheduler/integration/verl/examples/scheduler.yaml .
    ```
1.  **Adjust `py_modules` paths** in `runtime-env.yaml` to point to your local `py-inference-scheduler` installation.

Example `runtime-env.yaml`:
```yaml
env_vars:
  ROUTER_CONFIG_PATH: "./scheduler.yaml"
py_modules:
  - "../py-inference-scheduler/integration"
  - "../py-inference-scheduler/scheduling"
```

### 6. Hook Injection via Hydra Overrides

To activate the scheduler, use the `+actor_rollout_ref.rollout.agent.agent_loop_manager_class` Hydra override. Removing it would run the job with vERL's native scheduler.

```bash
ray job submit \
    --working-dir . \
    --runtime-env runtime-env.yaml \
    -- bash examples/grpo_trainer/run_qwen2-7b_math.sh \
    algorithm.adv_estimator=grpo \
    data.train_files="['code_benchmark/train.parquet']" \
    data.val_files="['code_benchmark/test.parquet']" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=64 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    trainer.logger="['console']" \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.prometheus.enable=True \
    trainer.total_training_steps=10 \
    +actor_rollout_ref.rollout.agent.agent_loop_manager_class=integration.verl.verl_hook.PyInferenceAgentLoopManager
```

- **`agent_loop_manager_class`**: Points to the integration hook.
- **`disable_log_stats=False`**: Required for vLLM to emit local metrics.
- **`prometheus.enable=True`**: Required to expose the `/metrics` endpoint that the scheduler polls.

### 7. View the results

This is a very small training loop for tetsing with 10 steps configured in the `trainer.total_training_steps=10` flag. Viewing the logs on the actual ray submit job where you've ran the training job is the best place for logs in my opinion. This can be found either in the *Overview* or *Jobs* tab of the Ray Dashboard. (localhost:8265). vERL gives us output by step, don't be concerned if you se `ppo` tags on the labels for the logs - vERL uses the same tetsing infrastructure for its GRPO and PPO runs. Our script is a GRPO trainer.

Logs look as follows:

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