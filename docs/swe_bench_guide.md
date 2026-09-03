# Running SWE-bench RL Training with verl and the Scheduler

This guide describes how to run an agentic RL training job on [verl](https://github.com/volcengine/verl) using SWE-bench-style software engineering tasks, with rollout inference routed through `py-inference-scheduler`.


## Architecture

verl's agent loop stack is client/server: agent loops (clients) call `generate(prompt_ids) -> response_ids` against a pool of vLLM/SGLang server actors. Our [existing verl integration](../integration/verl/README.md) already replaces verl's load balancer at exactly this seam — [`InferenceSchedulerServerManager._acquire_server`](../integration/verl/verl_hook.py) routes every `generate` call through the scheduler engine, at the **token level**, with `prompt_ids` available for prefix scoring.

This means the SWE agent loop rides on top of the existing hook unchanged: any agent loop that calls `server_manager.generate(...)` gets scheduler routing for free. Unlike proxy-based setups (e.g. the Alibaba guide's ProxyServer), no HTTP hop or token-capture middleware is needed — verl's native path is already token-in/token-out, which is what RL training requires for correct advantage computation.

```mermaid
graph TD
    Trainer[verl PPO/GRPO Trainer] --> ALM[PyInferenceAgentLoopManager]
    ALM --> ALW[AgentLoopWorkers]
    ALW --> SWE[SWEAgentLoop - one per trajectory]
    SWE -->|"generate(prompt_ids)"| SM[InferenceSchedulerServerManager]
    SM -->|scored routing| S1[vLLM server actor 1]
    SM -->|scored routing| S2[vLLM server actor N]
    SWE <-->|"bash / edit / run tests"| SB[Sandbox pod - per trajectory]
    SB -->|F2P + P2P test result| SWE
    SWE -->|"AgentLoopOutput(reward_score)"| ALM
```

One design note on stickiness: verl's native `LLMServerClient` pins a trajectory to one server for all of its turns. Our hook re-routes **every** `generate` call instead. With the `prefix_cache` scorer weighted appropriately, follow-up turns naturally land on the replica holding the prefix ("soft stickiness"), while the scheduler retains freedom to move work when a replica saturates — this is the behavior we want to measure.

---

## Phase 0 — Cluster foundation

Everything in the [verl integration prerequisites](../integration/verl/README.md#prerequisites--cluster-requirements-step-1) applies (KubeRay cluster, shared `/tmp/metrics` volume, `scheduler-config` ConfigMap). SWE-bench adds:

1. **Agent sandboxes**: install [agent-sandbox on GKE](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/how-install-agent-sandbox#update-existing-gke-cluster). Sandboxes give each trajectory an isolated environment to run untrusted, model-generated code. Findings from bringing this up (July 2026, GKE 1.35):
   - The managed install ships a `secure-sandbox-policy` ValidatingAdmissionPolicy requiring gVisor, `runAsNonRoot`, dropped capabilities, resource limits, and the gVisor nodeSelector + toleration on every Sandbox.
   - **R2E/SWE task images require root** — the uv-managed interpreter lives under `/root` (mode 700) and `/testbed` is root-owned, so under `runAsNonRoot` the agent can neither run tests nor edit code. The policy binding excludes the `agents-system` namespace: run SWE sandboxes there as root while keeping gVisor, no SA token, and dropped caps voluntarily (root-inside-gVisor is the standard posture for these images). Validated template: [swe_sandbox_example.yaml](../configs/swe_sandbox_example.yaml).
   - Measured startup latency (474 MB R2E image): **~43 s cold** (~30 s of it image pull, first pull through the mirror), **~34 s** with AR cache warm but node cold (pull is dominated by AR→node transfer/extract, not the Docker Hub fetch), **~5–10 s** when the image is already on the node. With ~4.6k distinct training images, node-cache hits are rare, so budget ~35–45 s per sandbox: per-trajectory creation is viable, warm pools add little (they can't pre-stage 4.6k images), and co-scheduling the `rollout.n` siblings that share an image onto the same nodes is the real optimization lever.
2. **A CPU node pool for sandboxes**: rollouts need `train_batch_size × rollout.n` concurrent sandboxes at peak. Sandboxes are CPU/memory bound (git, pip, pytest) — keep them off the GPU pool. Enable autoscaling; VerlTool and DeepSWE both report needing 1000+ CPU cores at scale. (The GKE agent-sandbox install creates a gVisor node pool — size it for this concurrency.)
3. **Image mirror**: SWE task images are per-instance and large (1–5 GiB); R2E-Gym-Lite alone is ~3.2k unique images, so bulk-copying into Artifact Registry is impractical. Use an AR **remote repository** — a pull-through cache of Docker Hub — instead: no bulk transfer, images cache on first pull, and GKE pulls at GCP-internal speed afterward.

   ```bash
   gcloud artifacts repositories create swe-mirror \
       --repository-format=docker --location=<REGION> \
       --mode=remote-repository --remote-docker-repo=DOCKER-HUB
   ```

   Point the dataset at the mirror when preprocessing (Phase 2):

   ```bash
   python integration/verl/helpers/prepare_swe_dataset.py --local_save_dir ~/data/swe \
       --registry_rewrite "namanjain12/=<REGION>-docker.pkg.dev/<PROJECT>/swe-mirror/namanjain12/" \
       --registry_rewrite "docker.io/=<REGION>-docker.pkg.dev/<PROJECT>/swe-mirror/"
   ```

   Then pre-warm the cache so rollouts never cold-pull from Docker Hub: [list_swe_images.py](../integration/verl/helpers/list_swe_images.py) emits the unique image refs from the parquet files, and [swe_image_prewarm_job.yaml](../configs/swe_image_prewarm_job.yaml) is a sharded in-cluster Job that pulls them through the mirror (layers stream to `/dev/null`; nothing lands on node disk):

   ```bash
   python integration/verl/helpers/list_swe_images.py \
       ~/data/swe/train.parquet ~/data/swe/test.parquet > /tmp/swe_images.txt
   kubectl create configmap swe-image-list --from-file=images.txt=/tmp/swe_images.txt
   kubectl apply -f configs/swe_image_prewarm_job.yaml
   ```

   Caveats: the *initial* caching pulls still count against Docker Hub anonymous rate limits — if the pre-warm Job logs 429s, attach Docker Hub credentials to the remote repo (Secret Manager upstream credentials). The node/pod service account needs `artifactregistry.reader` on the mirror. And the AR cache is not the node image cache — sandbox pods still pull from AR on first use, so expect warm-up on the first rollout wave per node.
4. **Shared storage** for the parquet datasets and checkpoints, reachable from all Ray workers (GCS FUSE or PVC).

## Phase 1 — Smoke test with the existing example

Before adding SWE variables, verify the cluster + integration with the stock math walkthrough from the [integration README](../integration/verl/README.md#running-a-training-job-step-3). This proves KubeRay, the scheduler hook, metrics scraping, and FSDP all work. Only then move on.

## Phase 2 — Dataset preparation

**Do not train on SWE-bench Verified** — it is the community's eval set. The standard recipe:

- **Train**: [R2E-Gym](https://github.com/R2E-Gym/R2E-Gym) (8.1k procedurally generated instances). Filter out instances from repos that appear in SWE-bench Verified (e.g. sympy) to avoid contamination.
- **Eval**: [SWE-bench Verified](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified) (500 instances) as `val_files`, run at `trainer.test_freq`.

Convert to verl parquet with a preprocessing script (pattern: `examples/data_preprocess/*.py` in verl). Each row needs:

| Column | Content |
|---|---|
| `prompt` | Chat messages: system prompt (agent instructions + tool schema) and user message (issue text) |
| `agent_name` | `"swe_agent"` — selects our custom agent loop per sample (verl falls back to `rollout.agent.default_agent_loop` when absent) |
| `reward_model` | `{"style": "rule", "ground_truth": ""}` (reward comes from the sandbox, not from a comparator) |
| `extra_info` | `instance_id`, `dataset_kind` (`r2e` \| `swebench`), sandbox image ref, `repo`, `base_commit`, and the grading spec: `expected_output_json` for R2E-Gym rows; `fail_to_pass`/`pass_to_pass` **plus `test_patch`** for SWE-bench rows — official grading applies `test_patch` (it contains the fail-to-pass tests) before running the suite. The gold `patch` is deliberately *not* carried: no answers in the dataset. All JSON/text-encoded strings so both splits share one flat schema. |

The `extra_info` fields are delivered to the agent loop's `run(**kwargs)` as dataset fields, which is how the loop knows which sandbox image to claim and which tests to grade with. Note the grading difference: R2E-Gym ships `expected_output_json` (test name → expected status after a correct fix) instead of SWE-bench's F2P/P2P lists, so the reward step branches on `dataset_kind`.

This is implemented in [prepare_swe_dataset.py](../integration/verl/helpers/prepare_swe_dataset.py):

```bash
# full run
python integration/verl/helpers/prepare_swe_dataset.py --local_save_dir ~/data/swe

# smoke test (streams a small slice instead of downloading everything)
python integration/verl/helpers/prepare_swe_dataset.py \
    --local_save_dir /tmp/swe_data --max_train 50 --max_eval 20
```

It applies the contamination filter automatically (drops R2E-Gym rows whose repo appears in the eval set) and supports `--registry_rewrite OLD=NEW` to point image refs at an Artifact Registry mirror once one exists. Caveat: with `--max_eval` the filter only sees the sampled eval repos, so contamination filtering is only complete on a full run.

## Phase 3 — The SWE agent loop

Implemented in [integration/verl/swe/](../integration/verl/swe/). Four modules, with only the loop itself importing verl:

| Module | Role |
|---|---|
| [swe_agent_loop.py](../integration/verl/swe/swe_agent_loop.py) | The verl `AgentLoopBase` implementation; `@register("swe_agent")` |
| [scaffold.py](../integration/verl/swe/scaffold.py) | Pure helpers: bash-block protocol, observation formatting, diff filtering, system prompt |
| [pristine_grader.py](../integration/verl/swe/pristine_grader.py) | Scores an agent patch in a fresh sandbox (anti-reward-hacking) |
| [sandbox.py](../integration/verl/swe/sandbox.py) | Sandbox CR client (create/wait/exec/write_file/delete); shared by loop + grader + calibration |

### Registration

```yaml
# integration/verl/examples/swe_agent_loop.yaml
- name: swe_agent
  _target_: integration.verl.swe.swe_agent_loop.SWEAgentLoop
```

Passed with `+actor_rollout_ref.rollout.agent.agent_loop_config_path=integration/verl/examples/swe_agent_loop.yaml`. Dataset rows select it via `agent_name == "swe_agent"`.

### Rollout flow

`SWEAgentLoop.run(sampling_params, **kwargs)`:

1. **Sandbox** — create an agent-sandbox pod from the instance image in `agents-system` ([swe_sandbox_rbac.yaml](../configs/swe_sandbox_rbac.yaml) grants the Ray SA cross-namespace access).
2. **Prompt** — build the system prompt (task framing + bash-block protocol from `scaffold.py`) and tokenize.
3. **Loop** until submit / max-turns / token-budget:
   - `server_manager.generate(prompt_ids=...)` — the call the scheduler routes (prefix-aware).
   - Decode assistant tokens → parse the last fenced bash block → execute in the sandbox with a per-command timeout (`SWE_CMD_TIMEOUT_S`, default 60s) → middle-truncate output (`SWE_OBS_MAX_CHARS`, default 6k chars) → tokenize as a delta user message via `apply_chat_template(remove_system_prompt=True)` → append with mask 0.
   - If no bash block: return a protocol nudge as the observation (recoverable).
   - If the bash block is `submit`: break and grade.
4. **Pristine grading** — `git diff HEAD` in the rollout sandbox → `filter_patch()` strips test-path chunks → apply the filtered diff in a *fresh* sandbox → run the graded tests → `reward_score` back on `AgentLoopOutput`.
5. **Cleanup** — delete both sandboxes; return the output with `extra_fields: {swe_reason, swe_submitted}`.

### Token discipline

- LLM tokens are never re-tokenized — verl requires the exact token ids that flowed through the model for advantage computation.
- Observations use `apply_chat_template(remove_system_prompt=True)` on the delta message only (tool-agent-loop convention), appended with `response_mask=0` and `response_logprobs=0.0`.
- Response is truncated to `rollout.response_length` (budget ~28k for SWE trajectories); turns that would overflow it terminate the episode.

### Environment variables (tuning knobs without a config change)

| Var | Default | Purpose |
|---|---|---|
| `SWE_SANDBOX_NAMESPACE` | `agents-system` | Namespace for rollout + grading sandboxes |
| `SWE_CMD_TIMEOUT_S` | `60` | Per-command timeout in the rollout sandbox |
| `SWE_OBS_MAX_CHARS` | `6000` | Character cap for observation truncation |

### Prerequisites

- RBAC: `kubectl apply -f configs/swe_sandbox_rbac.yaml` — lets the Ray pods (default/default SA) manage Sandboxes in `agents-system`.
- The `kubernetes` Python package available in the Ray runtime env ([runtime-env-swe.yaml](../integration/verl/examples/runtime-env-swe.yaml) includes it).

### Validation status (2026-07-30): PASSING end-to-end without GPUs

[integration_check.py](../integration/verl/swe/integration_check.py) runs the real loop on the head pod with a *scripted* model (explore → write the gold fix via heredoc → submit) against real sandboxes and the installed verl build:

```bash
# copy the package to the head pod, then:
kubectl exec <head-pod> -- bash -c "cd /tmp/swe_itest && PYTHONPATH=/tmp/swe_itest \
  python3 -m integration.verl.swe.integration_check \
  --parquet /home/ray/data/swe/train.parquet --instance aiohttp-f0d74880deec"
# => reward=1.0 extra={'swe_reason': 'pass', 'swe_submitted': True} ... INTEGRATION CHECK: PASS
```

This validates the verl 0.9.0.dev `AgentLoopBase` contract, sandbox lifecycle via in-cluster RBAC, observation tokenization, and the full submit → diff → pristine-grade → reward-1.0 path. Re-run it whenever verl or the loop changes.

> [!IMPORTANT]
> **R2E images ship with a dirty git state** (the harness's baked-in bug edits to tracked files). The loop commits a baseline (`BASELINE_CMD`) right after sandbox start so `git diff HEAD` is agent-only — without this, the diff includes image-baked changes and fails to apply in the fresh grading sandbox.

## Phase 4 — Reward

Sparse outcome reward, computed inside the sandbox at trajectory end:

- **+1.0** — SWE-bench rows: all `fail_to_pass` tests now pass **and** all `pass_to_pass` tests still pass. R2E rows: the per-test status map from running `/r2e_tests` **exactly equals** `expected_output_json`. Time-capped (~5 min for training, vs 30 min in official eval).
- **0.0** otherwise (including timeout, crash, or no patch).

> [!IMPORTANT]
> **R2E grading is exact status matching, not "all tests pass".** Validated in-sandbox on `aiohttp-f0d74880deec`: the spec expects `test_add_route_with_invalid_re` to remain FAILED *even after a correct fix* — an "all tests green" reward would score gold patches as failures. Pre-fix, exactly one test (the fail-to-pass one) mismatches the spec, so reward is 0 as intended. Mechanics: `cd /testbed && .venv/bin/python -m pytest /r2e_tests --junitxml=/tmp/report.xml` and compare per-test statuses; the graded suite runs in seconds, so the 5-minute cap is generous headroom for slow repos.

**Both grading paths are validated end-to-end with gold patches** (2026-07-28, real sandboxes on the cluster):

- **R2E** (`aiohttp-f0d74880deec`): pre-fix → 62/2 pass/fail, spec mismatch, reward 0. Gold commit applied → 63/1 with the sole failure being the expected-FAILED test → exact spec match, reward 1.
- **SWE-bench** (`astropy__astropy-12907`): `git apply test_patch` → both F2P tests fail pre-fix (reward 0). Gold patch applied → F2P 2/2 and P2P 13/13 pass (reward 1). Run tests via the image's conda env: `/opt/miniconda3/envs/testbed/bin/python -m pytest <test ids>`.

Implementation notes from the validation: write files into sandboxes via exec streams, **not `kubectl cp`** — the sandbox drops `CAP_CHOWN`, so tar exits nonzero even though file content lands. Grade order for SWE-bench rows is `test_patch` → candidate patch → F2P + P2P.

**Decided: grade against a pristine state.** The agent is root in the container where tests live, so a policy can learn to delete failing tests or patch pytest. The agent loop must extract the agent's diff (filtered to non-test paths), then apply and grade it in a **fresh sandbox** the policy never touched (~35–45 s extra per trajectory, fully parallel).

Grading and calibration tooling lives in [integration/verl/swe/](../integration/verl/swe/):

- [grader.py](../integration/verl/swe/grader.py) — pure grading logic (junitxml → status map → `grade_r2e` exact-match / `grade_swebench` F2P+P2P), unit-tested in [tests/test_swe_grader.py](../tests/test_swe_grader.py). Name mapping handles class-based, module-level, parametrized, and underscore-prefixed (`_BaseTest`) tests.
- [sandbox.py](../integration/verl/swe/sandbox.py) — Sandbox CR client (create/wait/exec/write_file/delete) encoding the validated pod shape. Gotchas baked in: file writes go over exec+base64 (`CAP_CHOWN` breaks `kubectl cp`), exec retries transient websocket races, and clients must be **per-thread** — the kubernetes client's GKE auth races when shared across threads.
- [calibrate.py](../integration/verl/swe/calibrate.py) — the **calibration sweep**: runs the pre-fix suite for every instance in a real gVisor sandbox and keeps only instances that are (1) key-exact vs the spec, (2) deterministic across two runs, (3) not already solved pre-fix, (4) within the time cap. Catches runc→gVisor drift and flaky tests before they poison GRPO groups, and emits a keep-list to filter the training parquet with. The first sweep surfaced two systematic R2E spec conventions, now baked into the grader's key normalization: specs mangle `::` inside parametrized values into `.` (their node-id splitting), and **skipped tests are excluded from specs** (platform-conditional tests skip under Linux/gVisor), so the grader drops SKIPPED before comparing:

  ```bash
  uv run --with kubernetes --with pandas --with pyarrow python \
      -m integration.verl.swe.calibrate --parquet <train.parquet> \
      --out calibration.jsonl --concurrency 24
  ```

Return it via `AgentLoopOutput.reward_score` (field verified present in verl). Optional shaping (small bonuses for patch minimality / speed, per the ACK guide) only after the sparse signal is confirmed working. DeepSWE-style *compact filtering* — masking loss on trajectories that hit max-context/timeout instead of scoring them 0 — is a known stabilizer worth adopting if training oscillates.

> [!TIP]
> "All rewards are 0.0" is the most common failure mode and usually means broken sandbox plumbing (image, test command, network policy), not a bad policy. Validate the reward path standalone first: replay a known-good gold patch through the grading step and confirm it returns 1.0 before any RL is run.

## Phase 5 — Training configuration

Ready to run: [run_swe.sh](../integration/verl/examples/run_swe.sh) with [runtime-env-swe.yaml](../integration/verl/examples/runtime-env-swe.yaml):

```bash
ray job submit --address http://localhost:8265 \
    --runtime-env integration/verl/examples/runtime-env-swe.yaml \
    -- bash integration/verl/examples/run_swe.sh
```

A 7B–8B instruct model (Qwen2.5-7B-Instruct in the script) is the proven starting point for SWE RL on a single 8-GPU node. The scheduler-hook override is left commented in the script until the hook is ported to the cluster's verl version (see Version compatibility). Deltas vs the math example that matter:

```bash
swe_data_dir=$HOME/data/swe   # output of prepare_swe_dataset.py (Phase 2)

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$swe_data_dir/train.parquet \
    data.val_files=$swe_data_dir/test.parquet \
    data.return_raw_chat=True \
    data.max_prompt_length=4096 \
    data.max_response_length=28672 \        # trajectories are long; budget 32k total
    data.train_batch_size=64 \              # issues per step; sandbox pool must cover batch × n
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=8 \         # GRPO group size = sandboxes per issue
    actor_rollout_ref.actor.use_kl_loss=False \   # DeepSWE/DAPO: no KL for agentic tasks
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    +actor_rollout_ref.rollout.agent.agent_loop_config_path=integration/verl/examples/swe_agent_loop.yaml \
    actor_rollout_ref.rollout.disable_log_stats=False \
    +actor_rollout_ref.rollout.agent.agent_loop_manager_class=integration.verl.verl_hook.PyInferenceAgentLoopManager \
    ...
```

Long-context notes: enable `use_remove_padding`, sequence parallelism if 32k contexts OOM, and keep `gpu_memory_utilization` moderate (0.4–0.6) since training and rollout share GPUs. Hyperparameter fallback: the [practitioner's guide](https://arxiv.org/abs/2510.01132) recipe.

## Phase 6 — Measuring scheduler impact

The point of this exercise for the repo: quantify prefix-aware routing on a real agentic RL workload. Run A/B at identical config:

- **Baseline**: drop the `agent_loop_manager_class` override → verl's native load balancer (sticky least-loaded).
- **Treatment**: scheduler hook with a profile weighting `prefix_cache` (see [scheduler.yaml](../integration/verl/examples/scheduler.yaml); tune weights per the [customization guide](./scheduler_customization.md)).

Compare, per step (all already emitted — see the [integration README log reference](../integration/verl/README.md#4-verifying-results-step-4)):

- `timing_s/gen` and `perf/throughput` — rollout wall clock / sampling throughput (headline number)
- `timing_s/agent_loop/slowest/generate_sequences` — tail latency, where sampler imbalance shows up
- `num_turns/*` and `timing_s/agent_loop/tool_calls/*` — workload shape sanity check
- vLLM prefix-cache hit rate from the scraped backend metrics — the mechanism behind any win

### First A/B results (2026-08-27)

**Conditions**: 12 GRPO steps per arm, batch 8 × n 8 (64 concurrent trajectories), Qwen2.5-7B, 4 vLLM endpoints (TP2) on one 8-GPU node, calibrated 3,315-instance set, `data.seed=42` (identical instance order), treatment profile = `prefix_cache` only. W&B: `ab7b_base` (`xuufp5zy`) vs `ab7b_sched` (`pqcozz5f`). Step 1 dropped from aggregates (cold caches).

| Metric (steps 2–12 mean) | Baseline | Scheduler | Δ | Consistency |
|---|---|---|---|---|
| Mean generation / trajectory (s) | 21.2 | 19.1 | **−10.0%** | scheduler better **8/11 steps** |
| Slowest-trajectory generation (s) | 29.3 | 24.6 | −16.1% | 5/11 (inconsistent) |
| Throughput (tok/s) | 98.3 | 89.2 | −9.3% | noise (tracks wall clock) |
| Turns / trajectory (sanity) | 47.5 | 46.8 | −1.5% | comparable work ✓ |
| Response length (sanity) | 8,955 | 9,186 | +2.6% | comparable work ✓ |

## Version compatibility

The hook supports **verl v0.7.1 (legacy) and v0.9.x (modern) layouts, auto-detected at import** ([compat notice](../integration/verl/README.md#compatibility-notice)). The modern port was required because 0.9.x moved routing into a `GlobalRequestLoadBalancer` Ray actor (`verl/workers/rollout/llm_server.py`) that owns the server registry; the hook's client bootstraps its endpoint set by draining the balancer once at first use, then routes via the scheduler engine with verl's LB as fallback. Validated GPU-free on the cluster's **0.9.0.dev** build by [hook_compat_check.py](../integration/verl/hook_compat_check.py) — PASS: 3 endpoints bootstrapped, all same-prefix requests prefix-routed to one server, inflight and LB counters clean. The **SWEAgentLoop is likewise verified against 0.9.0.dev** ([integration_check.py](../integration/verl/swe/integration_check.py)) and tolerates older builds where `generate` returned a bare token list. One robustness fix that came out of this: the vLLM/SGLang engine patches now skip on *any* import failure, not just ImportError — CPU-only nodes (the Ray head) raise `AttributeError` from triton during vLLM import.

## Prior art

| Resource | What it gives us |
|---|---|
| [verl Agentic RL docs](https://verl.readthedocs.io/en/latest/start/agentic_rl.html) | Core async rollout + AgentLoop architecture |
| [verl Agent Loop internals](https://verl.readthedocs.io/en/latest/advance/agent_loop.html) | `AgentLoopBase` interface, token-in/token-out contract |
| [Alibaba ACK verl + SWE-bench guide](https://help.aliyun.com/en/ack/training-agentic-reinforcement-learning-on-ack-using-the-verl-framework) | End-to-end k8s reference: sandbox pods, reward design, GRPO params, troubleshooting. Note: they inject an HTTP proxy to capture tokens; we don't need one (see below). |
| [Practitioner's Guide to Multi-turn Agentic RL](https://arxiv.org/abs/2510.01132) | Tuned hyperparameter recipe on verl, incl. SWE-Gym |
| [DeepSWE recipe](https://www.together.ai/blog/deepswe) | Algorithm tricks for convergence (clip-high, no KL, compact filtering) and reward design |
| [SWE-Gym](https://github.com/SWE-Gym/SWE-Gym) / [R2E-Gym](https://github.com/R2E-Gym/R2E-Gym) | Training environments with per-instance Docker images |
