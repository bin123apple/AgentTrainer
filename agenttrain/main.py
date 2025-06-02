from datasets import Dataset, load_from_disk
from trl import GRPOConfig

from tools import crop
from utils import preprocess_dataset, get_model_and_tokenizer
from envs.tool_env import ToolEnv
from trainers.grpo_env_trainer import GRPOEnvTrainer
from agenttrain.prompts.system_prompts import CROP_SYSTEM_PROMPT
"""
Multi-GPU training (single node, 4 training + 4 inference)

CUDA_VISIBLE_DEVICES=0,1,2,3 python agenttrain/inference/vllm_serve.py --model 'Qwen/Qwen2.5-7B-Instruct' \
    --tensor_parallel_size 4 --max_model_len 8192 --dtype bfloat16 \
    --gpu_memory_utilization 0.9 --enable_prefix_caching True \
    --host 0.0.0.0 --port 8000

CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --config-file configs/zero3.yaml agenttrain/examples/math_train.py
"""
def load_processed_dataset(data_path: str) -> Dataset:
    """
    加载预处理好的数据集
    
    Args:
        data_path: 预处理数据的路径
        
    Returns:
        Dataset: 加载的数据集
    """
    print(f"从 {data_path} 加载预处理数据...")
    dataset = load_from_disk(data_path)
    print(f"数据集加载完成，大小: {len(dataset)}")
    return dataset

def main():
    """主函数"""
    
    # 1. 加载预处理数据
    try:
        PROCESSED_DATA_PATH = "/mnt/data1/processed_datasets/uground_processed_10000"
        dataset = load_processed_dataset(PROCESSED_DATA_PATH)
    except Exception as e:
        print(f"加载数据失败: {e}")
        print("请先运行 preprocess_data.py 生成预处理数据")
        return  # 或者 raise e 来停止程序
        
        # 备用方案：如果预处理数据不存在，可以临时使用原始处理方式
        # from your_preprocess_module import preprocess_dataset
        # print("回退到原始预处理方式...")
        # dataset = preprocess_dataset(
        #     "osunlp/UGround-V1-Data", 
        #     "train", 
        #     n=10000,
        #     cache_dir="/mnt/data1/huggingface/datasets/datasets--osunlp--UGround-V1-Data-Box"
        # )
    
    # 2. 随机打乱并按比例分割数据集
    print("2. 分割训练集和验证集...")
    split = dataset.shuffle(seed=0).train_test_split(test_size=0.1, seed=0)
    
    train_dataset = split["train"]    # 90% 用于训练
    # print(f"Fist record in train dataset: {train_dataset[0]}")
    eval_dataset = split["test"]      # 10% 用于评估
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(eval_dataset)}")
    
    # 3. 设置工具环境
    print("3. 初始化工具环境...")
    tool_env = ToolEnv(
        dataset=train_dataset,
        eval_dataset=eval_dataset,
        system_prompt=CROP_SYSTEM_PROMPT,
        few_shot=[],
        tools=[crop],
        max_steps=5
    )
    
    print("System Prompt:")
    print(tool_env.system_prompt)
    
    # 4. 加载模型
    print("4. 加载模型...")
    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    model, tokenizer = get_model_and_tokenizer(
        model_name, 
        cache_dir="/mnt/data1/huggingface/models"
    )
    
    # 5. 设置训练参数
    run_name = "VG-grpo_" + model_name.split("/")[-1].lower()
    
    training_args = GRPOConfig(
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
        vllm_server_host="0.0.0.0",  # 多节点设置时替换为推理服务器的主机
        vllm_server_port=8000,
        vllm_gpu_memory_utilization=0.9,
        logging_steps=1,
        log_on_each_node=False,
        log_completions=True,
        report_to="wandb",
        reward_weights=tool_env.get_reward_weights()
    )
    
    # 6. 初始化训练器
    print("5. 初始化训练器...")
    trainer = GRPOEnvTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=tool_env.get_reward_funcs(),
        env=tool_env,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )
    
    # 7. 开始训练
    print("6. 开始训练...")
    trainer.train()
    
    print("训练完成！")

if __name__ == "__main__":
    main()