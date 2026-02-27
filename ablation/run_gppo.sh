#!/bin/bash
set -x

# ============================================================================
# GPPO Training Script
# - Multiple rollouts per prompt
# - Grouped outcome advantages + action-conditioned advantage head
# - Actor-only training (no critic)
# ============================================================================

# --- 1. USER CONFIGURATION (Data, Model, Logs) ---
export RAY_memory_monitor_refresh_ms=0

# Your Log Directory
LOG_DIR="/mnt/workspace/MLLM/zz/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_NAME="qwen3_8b_gppo_${TIMESTAMP}"
LOG_FILE="${LOG_DIR}/train_log_${EXPERIMENT_NAME}_${TIMESTAMP}.log"
mkdir -p "$LOG_DIR"

# Your Model and Data Paths
MODEL_PATH="/mnt/storage/models/Qwen/Qwen2.5-7B-Instruct"
TRAIN_DATA="/mnt/data/MLLM/zz/verl_train/data/gsm8k/train.parquet"
TEST_DATA="/mnt/data/MLLM/zz/verl_train/data/gsm8k/test.parquet"
CKPTS_DIR="/mnt/storage/MLLM/zz/models/${EXPERIMENT_NAME}"
mkdir -p "$CKPTS_DIR"

# --- 2. CORE PARAMETERS (GPPO Configuration) ---
# PERFORMANCE SETTING: Set to 'true' if you encounter Out Of Memory (OOM) errors.
offload=${OFFLOAD:-false}

project_name='GPPO'
adv_estimator=gppo
rollout_engine=vllm
rollout_mode=async
gpu_memory_utilization=0.6

# Training schedule
test_freq=${TEST_FREQ:-5}
save_freq=${SAVE_FREQ:-10}
total_epochs=${TOTAL_EPOCHS:-10}
total_training_steps=${TOTAL_TRAINING_STEPS:-100}
val_before_train=false

# KL configuration: use as auxiliary loss, not in reward
use_kl_in_reward=false
kl_coef=0.0
use_kl_loss=${USE_KL_LOSS:-true}
kl_loss_coef=${KL_LOSS_COEF:-0.001}
kl_loss_type=low_var_kl

# PPO clipping
clip_ratio_low=0.2
clip_ratio_high=0.2

# GPPO multihead requires sequence-sum style aggregation.
loss_agg_mode=${LOSS_AGG_MODE:-seq-mean-token-sum}

# Batch sizes
train_batch_size=${TRAIN_BATCH_SIZE:-512}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-128}
ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE:-8}
n_resp_per_prompt=${N_RESP_PER_PROMPT:-8}

# Context Lengths
max_prompt_length=${MAX_PROMPT_LENGTH:-512}
max_response_length=${MAX_RESPONSE_LENGTH:-1024}

# Checkpoint directory
CKPTS_DIR="${CKPTS_DIR:-./checkpoints/${EXPERIMENT_NAME}}"
mkdir -p "$CKPTS_DIR"

# Sampling params
temperature=1.0
top_p=1.0
top_k=-1
val_top_p=0.95

# Performance Related Parameters
sp_size=1
gen_tp=1
use_dynamic_bsz=true
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 3))

# Optimization
actor_lr=1e-6
actor_weight_decay=0.1
actor_entropy_coeff=0
actor_grad_clip=1.0
lr_warmup_steps_ratio=0.05
enable_gradient_checkpointing=true
use_remove_padding=true

# Data config
truncation='error'
filter_overlong_prompts=true

# Valid config
val_do_sample=true
val_n=1

# Number of GPUs / nodes
# If N_GPUS is not set, detect from CUDA_VISIBLE_DEVICES first, then nvidia-smi.
if [ -z "${N_GPUS:-}" ]; then
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        if [ "${CUDA_VISIBLE_DEVICES}" = "NoDevFiles" ]; then
            N_GPUS=0
        else
            IFS=',' read -r -a CUDA_DEVICES <<< "${CUDA_VISIBLE_DEVICES}"
            N_GPUS=${#CUDA_DEVICES[@]}
        fi
    elif command -v nvidia-smi >/dev/null 2>&1; then
        N_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')
    else
        N_GPUS=0
    fi
fi
N_NODES=${N_NODES:-1}

if ! [[ "${N_GPUS}" =~ ^[0-9]+$ ]] || [ "${N_GPUS}" -le 0 ]; then
    echo "ERROR: No visible GPUs detected (N_GPUS=${N_GPUS}). Set N_GPUS explicitly or check CUDA_VISIBLE_DEVICES."
    exit 1
fi

# --- 3. EXECUTION ---
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    data.train_files="['$TRAIN_DATA']" \
    data.val_files="['$TEST_DATA']" \
    data.prompt_key=prompt \
    data.truncation=${truncation} \
    data.filter_overlong_prompts=${filter_overlong_prompts} \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=${kl_loss_type} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.predict_advantage_head=true \
    actor_rollout_ref.model.use_remove_padding=${use_remove_padding} \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.name=${rollout_engine} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=${enable_gradient_checkpointing} \
    actor_rollout_ref.actor.optim.lr=${actor_lr} \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=${lr_warmup_steps_ratio} \
    actor_rollout_ref.actor.optim.weight_decay=${actor_weight_decay} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=${actor_entropy_coeff} \
    actor_rollout_ref.actor.grad_clip=${actor_grad_clip} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=${val_do_sample} \
    actor_rollout_ref.rollout.val_kwargs.n=${val_n} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.nnodes=${N_NODES} \
    trainer.val_before_train=${val_before_train} \
    trainer.test_freq=${test_freq} \
    trainer.save_freq=${save_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.total_training_steps=${total_training_steps} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    $@ 2>&1 | tee "$LOG_FILE"
