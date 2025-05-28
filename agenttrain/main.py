from datasets import concatenate_datasets
from trl import GRPOConfig

from tools import crop
from utils import preprocess_dataset, get_model_and_tokenizer
from envs.tool_env import ToolEnv
from trainers.grpo_env_trainer import GRPOEnvTrainer
from agenttrain.prompts.system_prompts import CROP_SYSTEM_PROMPT
"""
Multi-GPU training (single node, 4 training + 4 inference)

CUDA_VISIBLE_DEVICES=0,1,2,3 python verifiers/inference/vllm_serve.py --model 'Qwen/Qwen2.5-7B-Instruct' \
    --tensor_parallel_size 4 --max_model_len 8192 --dtype bfloat16 \
    --gpu_memory_utilization 0.9 --enable_prefix_caching True \
    --host 0.0.0.0 --port 8000

CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --config-file configs/zero3.yaml verifiers/examples/math_train.py
"""

# 1. 先加载并预处理好 10k 样本的 train 数据
dataset = preprocess_dataset("osunlp/UGround-V1-Data", "train", n=10000, 
                             cache_dir = "/mnt/data1/huggingface/datasets/datasets--osunlp--UGround-V1-Data-Box")

# 2. 随机打乱并直接按数目（或比例）分出 eval 集合
split = dataset.shuffle(seed=0) \
               .train_test_split(test_size=0.1, seed=0)

train_dataset = split["train"]    # 剩下的 10,000-60 = 9,940 样本
eval_dataset  = split["test"]     # 抽取到的 60 样本


tool_env = ToolEnv(
    dataset=train_dataset,
    eval_dataset=eval_dataset,
    system_prompt=CROP_SYSTEM_PROMPT,
    few_shot=[],
    tools=[crop],
    max_steps=5
)
print(tool_env.system_prompt)

model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
model, tokenizer = get_model_and_tokenizer(model_name, cache_dir="/mnt/data1/huggingface/models")
run_name = "VG-grpo_" + model_name.split("/")[-1].lower()

training_args=GRPOConfig(
    output_dir=f"outputs/{run_name}",
    run_name=run_name,
    learning_rate=1e-6,
    lr_scheduler_type="constant_with_warmup",
    warmup_steps=10,
    num_train_epochs=1,
    temperature=1.0,
    max_steps=1000,
    bf16=True,
    max_grad_norm=0.1,
    num_iterations=2,
    beta=0.002,
    max_prompt_length=1024,
    max_completion_length=2048,
    per_device_train_batch_size=12,
    per_device_eval_batch_size=12,
    num_generations=6,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    eval_strategy="steps",
    eval_steps=100,
    eval_accumulation_steps=1,
    eval_on_start=False,
    save_strategy="steps",
    save_steps=100,
    save_only_model=True,
    use_vllm=True,
    vllm_server_host="0.0.0.0", # replace with your inference server's host for multi-node setups
    vllm_server_port=8000,
    vllm_gpu_memory_utilization=0.9,
    logging_steps=1,
    log_on_each_node=False,
    log_completions=True,
    report_to="wandb",
    reward_weights=tool_env.get_reward_weights()
)
trainer = GRPOEnvTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=tool_env.get_reward_funcs(),
    env=tool_env,
    args=training_args,
    train_dataset=tool_env.get_dataset(),
    eval_dataset=tool_env.get_eval_dataset()
)
trainer.train() 