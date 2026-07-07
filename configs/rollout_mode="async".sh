rollout_mode="async"
rollout_name="vllm" # sglang or vllm
if [ "$rollout_mode" = "async" ]; then
    export VLLM_USE_V1=1
    return_raw_chat="True"
fi

train_prompt_bsz=0
gen_prompt_bsz=32
n_resp_per_prompt=16
train_prompt_mini_bsz=32
total_rollout_steps=$(((512*400)))
test_freq=10
staleness_threshold=1
trigger_parameter_sync_step=16
partial_rollout=False
NNODES_TRAIN=1
NNODES_ROLLOUT=8
NGPUS_PER_NODE_TRAIN=8
NGPUS_PER_NODE_ROLLOUT=2


python -m verl.experimental.fully_async_policy.fully_async_main \
	train_batch_size=${train_prompt_bsz} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.return_raw_chat=${return_raw_chat} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.actor.strategy=fsdp \
    critic.strategy=fsdp \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.rollout.name=${rollout_name} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    trainer.nnodes="${NNODES_TRAIN}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE_TRAIN}" \
    rollout.nnodes="${NNODES_ROLLOUT}" \
    rollout.n_gpus_per_node="${NGPUS_PER_NODE_ROLLOUT}" \
    rollout.total_rollout_steps="${total_rollout_steps}" \
    rollout.test_freq="${test_freq}" \
    async_training.staleness_threshold="${staleness_threshold}" \
    async_training.trigger_parameter_sync_step="${trigger_parameter_sync_step}" \
    async_training.partial_rollout="${partial_rollout}"

    # Manually Updating the file that contains _target_: verl.workers.config.FSDPCriticConfig


python3 -m verl.trainer.main_ppo_sync \
    algorithm.adv_estimator=grpo \
    data.train_files="[/root/data/gsm8k/train.parquet]" \
    data.val_files="[/root/data/gsm8k/test.parquet]" \
    data.train_batch_size=1024 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-32B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name='verl_grpo_32b_math' \
    trainer.experiment_name='qwen32b_math' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=2 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_training_steps=10 \
    actor_rollout_ref.rollout.disable_log_stats=False 