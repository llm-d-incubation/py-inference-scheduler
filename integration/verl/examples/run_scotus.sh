#!/usr/bin/env bash
# run_test.sh  --mode <native|epp|llm-d>  [options]
#
# Usage examples:
#   bash run_test.sh --mode native
#   bash run_test.sh --mode epp
#   bash run_test.sh --mode epp --steps 20 --tp 2 --n 4
#   bash run_test.sh --mode llm-d          # (not yet implemented)
#
# Options:
#   --mode   native | epp | py-sched | llm-d   (required)
#   --steps  total_training_steps          (default: 40)
#   --tp     tensor-parallel size          (default: 1)
#   --n      rollout group size            (default: 8)
#   --task   any folder under workloads/ (gsm8k | hotpotqa | musique | quality |
#            searchr1 | scotus_xl | arxiv | geo3k)   (default: gsm8k)
#   --name   override experiment name      (default: auto-generated)
#   --reqlog enable per-request JSONL log  (default: on for all modes)

set -euo pipefail

# -- defaults -----------------------------------------------------------------
MODE=""
STEPS=40
TP=1
N=8
CUSTOM_NAME=""
REQLOG=""          # empty = auto (on for non-native modes)
TASK="gsm8k"       # name of a folder under workloads/ (each has a task.env)

# -- arg parsing ---------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)   MODE="$2";        shift 2 ;;
    --steps)  STEPS="$2";       shift 2 ;;
    --tp)     TP="$2";          shift 2 ;;
    --n)      N="$2";           shift 2 ;;
    --task)   TASK="$2";        shift 2 ;;   # any folder name under workloads/
    --name)   CUSTOM_NAME="$2"; shift 2 ;;
    --reqlog) REQLOG="$2";      shift 2 ;;   # "on" or "off"
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [[ -z "$MODE" ]]; then
  echo "ERROR: --mode is required  (native | epp | py-sched | llm-d)"
  exit 1
fi

# -- per-mode config -----------------------------------------------------------
EXTRA_HYDRA=""

case "$MODE" in
  native)
    DEFAULT_NAME="qwen3_4b_grpo_baseline_tp${TP}_n${N}_${STEPS}s"
    [[ -z "$REQLOG" ]] && REQLOG="on"
    # Native verl routing (GlobalRequestLoadBalancer), but with a logging client so
    # the run produces the same per-request reqlog as EPP, plus the endpoints YAML
    # for the vLLM /metrics scraper. Routing behaviour is unchanged from stock native.
    EXTRA_HYDRA="
  +actor_rollout_ref.rollout.agent.agent_loop_manager_class=llm_d_rl_verl_integration.native_logging.agent_loop_manager.NativeLoggingAgentLoopManager \
  +actor_rollout_ref.rollout.custom.epp_endpoints_file=/tmp/epp-endpoints.yaml"
    ;;

  epp)
    DEFAULT_NAME="qwen3_4b_grpo_epp_tp${TP}_n${N}_${STEPS}s"
    [[ -z "$REQLOG" ]] && REQLOG="on"
    EXTRA_HYDRA="
  +actor_rollout_ref.rollout.agent.agent_loop_manager_class=llm_d_rl_verl_integration.llmd_epp.agent_loop_manager.LlmdRouterAgentLoopManager \
  +actor_rollout_ref.rollout.custom.epp_config_file=/etc/llmd-configs/epp-config.yaml \
  +actor_rollout_ref.rollout.custom.epp_endpoints_file=/tmp/epp-endpoints.yaml"
    ;;

  py-sched)
    DEFAULT_NAME="qwen3_4b_grpo_pysched_tp${TP}_n${N}_${STEPS}s"
    [[ -z "$REQLOG" ]] && REQLOG="off"   # reqlog is llm-d-rl-specific; not emitted by this client
    # py-inference-scheduler (upstream main) routing via the pyis_port bridge
    # (port of integration/verl/verl_hook.py to verl commit 334d9f8b).
    # Requires on every pod: repo clone at /tmp/pyis, port pkg at /tmp/pyis-port,
    # scheduler config mounted at /etc/scheduler/scheduler.yaml.
    EXTRA_HYDRA="
  +actor_rollout_ref.rollout.agent.agent_loop_manager_class=pyis_port.agent_loop_manager.PyInferenceAgentLoopManager \
  +ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH=/tmp/pyis:/tmp/pyis-port \
  +ray_kwargs.ray_init.runtime_env.env_vars.ROUTER_CONFIG_PATH=/etc/scheduler/scheduler.yaml \
  +ray_kwargs.ray_init.runtime_env.env_vars.PROMETHEUS_MULTIPROC_DIR=/tmp/metrics"
    ;;

  llm-d)
    echo "ERROR: --mode llm-d is not yet implemented"
    exit 1
    ;;

  *)
    echo "ERROR: unknown mode '${MODE}'. Choose: native | epp | py-sched | llm-d"
    exit 1
    ;;
esac

EXPERIMENT_NAME="${CUSTOM_NAME:-$DEFAULT_NAME}"

# -- reqlog override -----------------------------------------------------------
if [[ "$REQLOG" == "on" ]]; then
  EXTRA_HYDRA="
  +ray_kwargs.ray_init.runtime_env.env_vars.VERL_REQLOG_DIR=/tmp/verl/reqlog${EXTRA_HYDRA}"
fi

# -- task config: sourced from the self-contained workload folder --------------
# Each workloads/<name>/task.env sets FSDP_SCRIPT, DEF_MODEL, DEF_PROJECT, DEF_TRAIN,
# DEF_TEST, DEF_MAXP, DEF_MAXR and the TASK_OVERRIDES array (fully - including any
# env-var-driven logic like QUALITY_SHUFFLE or the geo3k image sizing). Adding a
# workload means adding a folder; this driver does not change.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Resolve the workloads dir: explicit WORKLOADS_DIR override, else the repo layout
# (benchmarks/scripts -> ../workloads), else /tmp/workloads (where run_on_head.sh copies
# the selected workload folder alongside run_test.sh on the head pod).
WORKLOADS_DIR="${WORKLOADS_DIR:-}"
if [[ -z "$WORKLOADS_DIR" ]]; then
  if [[ -d "$SCRIPT_DIR/../workloads" ]]; then
    WORKLOADS_DIR="$(cd "$SCRIPT_DIR/../workloads" && pwd)"
  elif [[ -d /tmp/workloads ]]; then
    WORKLOADS_DIR=/tmp/workloads
  fi
fi
TASK_ENV="$WORKLOADS_DIR/$TASK/task.env"
if [[ ! -f "$TASK_ENV" ]]; then
  echo "ERROR: no task.env for --task '$TASK' (looked at: $TASK_ENV)"
  echo "       available workloads: $(ls -1 "$WORKLOADS_DIR" 2>/dev/null | tr '\n' ' ')"
  exit 1
fi
TASK_OVERRIDES=()
# shellcheck disable=SC1090
source "$TASK_ENV"

TRAIN_RESOLVED=${TRAIN_FILE:-$DEF_TRAIN}
TEST_RESOLVED=${TEST_FILE:-$DEF_TEST}

# Optional extra hydra overrides, appended LAST so they win over the per-task defaults
# (e.g. raise ppo/log_prob token budgets for a bigger max_prompt). Space-separated;
# values must not contain spaces. Empty by default.
read -r -a EXTRA_OV <<< "${EXTRA_OVERRIDES:-}"

# -- launch --------------------------------------------------------------------
cd /tmp/verl/verl/examples/grpo_trainer

ROLLOUT_N=$N ROLLOUT_TP=$TP NGPUS_PER_NODE=8 TRAIN_BATCH_SIZE=256 PPO_MINI_BATCH_SIZE=128 \
MODEL_PATH=${MODEL_PATH:-$DEF_MODEL} \
TRAIN_FILE=$TRAIN_RESOLVED \
TEST_FILE=$TEST_RESOLVED \
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-$DEF_MAXP} MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-$DEF_MAXR} \
SAVE_FREQ=-1 PROJECT_NAME=${PROJECT_NAME:-$DEF_PROJECT} \
EXPERIMENT_NAME=$EXPERIMENT_NAME \
bash "$FSDP_SCRIPT" \
  trainer.logger='["console","file","wandb"]' \
  trainer.total_training_steps=$STEPS \
  trainer.default_local_dir=/tmp/checkpoints \
  trainer.rollout_data_dir=/tmp/verl/generations/train \
  +ray_kwargs.ray_init.runtime_env.env_vars.VERL_FILE_LOGGER_ROOT=/tmp/verl/logs \
  actor_rollout_ref.rollout.disable_log_stats=False \
  actor_rollout_ref.rollout.n=$N \
  ${TASK_OVERRIDES[@]+"${TASK_OVERRIDES[@]}"} \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.enable_prompt_tokens_details=true \
  ${EXTRA_OV[@]+"${EXTRA_OV[@]}"} \
  hydra.run.dir=/tmp/hydra-outputs${EXTRA_HYDRA}