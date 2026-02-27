#!/usr/bin/env bash
set -xeuo pipefail

# ============================================================================
# GPPO Multihead Ablation Script
# Parallel to old GRPO ablation recipe, but enables action-conditioned advantage:
#   u_{t,a_t} = U(h_{t-1})[a_t]
# ============================================================================

export RAY_memory_monitor_refresh_ms=0

# --- 1. User Configuration ---
LOG_DIR="${LOG_DIR:-./logs}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_NAME="${EXPERIMENT_NAME:-gppo_${TIMESTAMP}}"
LOG_FILE="${LOG_DIR}/train_log_${EXPERIMENT_NAME}.log"
mkdir -p "${LOG_DIR}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}"
TRAIN_DATA="${TRAIN_DATA:-}"
TEST_DATA="${TEST_DATA:-}"

# --- 2. Core Parameters ---
offload="${OFFLOAD:-false}"

project_name="${PROJECT_NAME:-GPPO-Ablation}"
adv_estimator="gppo"
rollout_engine="${ROLLOUT_ENGINE:-vllm}"
rollout_mode="${ROLLOUT_MODE:-sync}"
gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.6}"

test_freq="${TEST_FREQ:-5}"
save_freq="${SAVE_FREQ:-10}"
total_epochs="${TOTAL_EPOCHS:-10}"
total_training_steps="${TOTAL_TRAINING_STEPS:-100}"
val_before_train="${VAL_BEFORE_TRAIN:-false}"

norm_adv_by_std_in_grpo="${NORM_ADV_BY_STD_IN_GRPO:-true}"

use_kl_in_reward="false"
kl_coef="0.0"
use_kl_loss="${USE_KL_LOSS:-true}"
kl_loss_coef="${KL_LOSS_COEF:-0.001}"
kl_loss_type="${KL_LOSS_TYPE:-low_var_kl}"

clip_ratio_low="${CLIP_RATIO_LOW:-0.2}"
clip_ratio_high="${CLIP_RATIO_HIGH:-0.2}"

# For GPPO multihead we use sequence-sum aggregation (not token-mean / seq-mean-token-mean).
loss_agg_mode="${LOSS_AGG_MODE:-seq-mean-token-sum}"

train_batch_size="${TRAIN_BATCH_SIZE:-512}"
ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-128}"
ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE:-8}"
n_resp_per_prompt="${N_RESP_PER_PROMPT:-8}"

max_prompt_length="${MAX_PROMPT_LENGTH:-512}"
max_response_length="${MAX_RESPONSE_LENGTH:-1024}"

CKPTS_DIR="${CKPTS_DIR:-./checkpoints/${EXPERIMENT_NAME}}"
mkdir -p "${CKPTS_DIR}"

temperature="${TEMPERATURE:-1.0}"
top_p="${TOP_P:-1.0}"
top_k="${TOP_K:--1}"
val_top_p="${VAL_TOP_P:-0.95}"

sp_size="${SP_SIZE:-1}"
gen_tp="${GEN_TP:-1}"
use_dynamic_bsz="${USE_DYNAMIC_BSZ:-true}"
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2))
critic_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 4))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 3))

actor_lr="${ACTOR_LR:-1e-6}"
actor_weight_decay="${ACTOR_WEIGHT_DECAY:-0.1}"
actor_entropy_coeff="${ACTOR_ENTROPY_COEFF:-0}"
actor_grad_clip="${ACTOR_GRAD_CLIP:-1.0}"

critic_lr="${CRITIC_LR:-1e-6}"
critic_warmup="${CRITIC_WARMUP:-0}"

lr_warmup_steps_ratio="${LR_WARMUP_STEPS_RATIO:-0.05}"
enable_gradient_checkpointing="${ENABLE_GRADIENT_CHECKPOINTING:-true}"
use_remove_padding="${USE_REMOVE_PADDING:-true}"

truncation="${TRUNCATION:-error}"
filter_overlong_prompts="${FILTER_OVERLONG_PROMPTS:-true}"

val_do_sample="${VAL_DO_SAMPLE:-true}"
val_n="${VAL_N:-1}"

N_GPUS="${N_GPUS:-8}"
N_NODES="${N_NODES:-1}"

# --- 3. Execution ---
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator="${adv_estimator}" \
  algorithm.norm_adv_by_std_in_grpo="${norm_adv_by_std_in_grpo}" \
  algorithm.use_kl_in_reward="${use_kl_in_reward}" \
  algorithm.kl_ctrl.kl_coef="${kl_coef}" \
  data.train_files="['${TRAIN_DATA}']" \
  data.val_files="['${TEST_DATA}']" \
  data.prompt_key=prompt \
  data.truncation="${truncation}" \
  data.filter_overlong_prompts="${filter_overlong_prompts}" \
  data.train_batch_size="${train_batch_size}" \
  data.max_prompt_length="${max_prompt_length}" \
  data.max_response_length="${max_response_length}" \
  actor_rollout_ref.rollout.n="${n_resp_per_prompt}" \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding="${use_remove_padding}" \
  actor_rollout_ref.model.enable_gradient_checkpointing="${enable_gradient_checkpointing}" \
  actor_rollout_ref.actor.use_dynamic_bsz="${use_dynamic_bsz}" \
  actor_rollout_ref.actor.use_kl_loss="${use_kl_loss}" \
  actor_rollout_ref.actor.kl_loss_coef="${kl_loss_coef}" \
  actor_rollout_ref.actor.kl_loss_type="${kl_loss_type}" \
  actor_rollout_ref.actor.clip_ratio_low="${clip_ratio_low}" \
  actor_rollout_ref.actor.clip_ratio_high="${clip_ratio_high}" \
  actor_rollout_ref.actor.loss_agg_mode="${loss_agg_mode}" \
  actor_rollout_ref.actor.predict_advantage_head=true \
  actor_rollout_ref.actor.optim.lr="${actor_lr}" \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio="${lr_warmup_steps_ratio}" \
  actor_rollout_ref.actor.optim.weight_decay="${actor_weight_decay}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${ppo_mini_batch_size}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${ppo_micro_batch_size_per_gpu}" \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${actor_ppo_max_token_len}" \
  actor_rollout_ref.actor.entropy_coeff="${actor_entropy_coeff}" \
  actor_rollout_ref.actor.grad_clip="${actor_grad_clip}" \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size="${sp_size}" \
  actor_rollout_ref.actor.fsdp_config.param_offload="${offload}" \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="${offload}" \
  actor_rollout_ref.rollout.name="${rollout_engine}" \
  actor_rollout_ref.rollout.mode="${rollout_mode}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${gpu_memory_utilization}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${gen_tp}" \
  actor_rollout_ref.rollout.temperature="${temperature}" \
  actor_rollout_ref.rollout.top_p="${top_p}" \
  actor_rollout_ref.rollout.top_k="${top_k}" \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz="${use_dynamic_bsz}" \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${infer_ppo_max_token_len}" \
  actor_rollout_ref.rollout.val_kwargs.temperature="${temperature}" \
  actor_rollout_ref.rollout.val_kwargs.top_p="${val_top_p}" \
  actor_rollout_ref.rollout.val_kwargs.top_k="${top_k}" \
  actor_rollout_ref.rollout.val_kwargs.do_sample="${val_do_sample}" \
  actor_rollout_ref.rollout.val_kwargs.n="${val_n}" \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz="${use_dynamic_bsz}" \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${infer_ppo_max_token_len}" \
  actor_rollout_ref.ref.fsdp_config.param_offload="${offload}" \
  actor_rollout_ref.ref.ulysses_sequence_parallel_size="${sp_size}" \
  critic.model.path="${MODEL_PATH}" \
  critic.model.use_remove_padding="${use_remove_padding}" \
  critic.model.enable_gradient_checkpointing="${enable_gradient_checkpointing}" \
  critic.optim.lr="${critic_lr}" \
  critic.ppo_max_token_len_per_gpu="${critic_ppo_max_token_len}" \
  critic.ulysses_sequence_parallel_size="${sp_size}" \
  critic.model.fsdp_config.param_offload="${offload}" \
  critic.model.fsdp_config.optimizer_offload="${offload}" \
  trainer.critic_warmup="${critic_warmup}" \
  trainer.logger='["console","tensorboard"]' \
  trainer.project_name="${project_name}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.n_gpus_per_node="${N_GPUS}" \
  trainer.nnodes="${N_NODES}" \
  trainer.val_before_train="${val_before_train}" \
  trainer.test_freq="${test_freq}" \
  trainer.save_freq="${save_freq}" \
  trainer.total_epochs="${total_epochs}" \
  trainer.total_training_steps="${total_training_steps}" \
  trainer.default_local_dir="${CKPTS_DIR}" \
  trainer.resume_mode=auto \
  "$@" 2>&1 | tee "${LOG_FILE}"
