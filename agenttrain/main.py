from datasets import Dataset, load_from_disk
from trl import GRPOConfig, ModelConfig, ScriptArguments, TrlParser, get_peft_config
import torch
import io
import re
import ast
from agenttrain.vlm_modules import *
from tools import crop
from pathlib import Path
from utils import preprocess_dataset, get_model_and_tokenizer
from envs.tool_env import ToolEnv
from PIL import Image, ImageDraw
from trainers.grpo_env_trainer import GRPOEnvTrainer
from agenttrain.prompts.system_prompts import CROP_SYSTEM_PROMPT
from transformers import AutoModelForCausalLM, Qwen2_5_VLForConditionalGeneration
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

def get_vlm_module(model_name_or_path):
    if "qwen" in model_name_or_path.lower():
        return Qwen2VLModule
    elif "internvl" in model_name_or_path.lower():
        return InvernVLModule
    else:
        raise ValueError(f"Unsupported model: {model_name_or_path}")

class GRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = False

def main():
    """主函数"""
    
    # 1. 加载预处理数据
    try:
        PROCESSED_DATA_PATH = "/home/uconn/BinLei/processed_datasets/uground_processed_0_10000"
        dataset = load_processed_dataset(PROCESSED_DATA_PATH)
    except Exception as e:
        print(f"加载数据失败: {e}")
        print("请先运行 agenttrain/utils/data_collection_save.py 生成预处理数据")
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
    split = dataset.shuffle(seed=0).train_test_split(test_size=0.01, seed=0)
    
    train_dataset = split["train"]    # 90% 用于训练
    # print(f"Fist record in train dataset: {train_dataset[0]}")
    eval_dataset = split["test"]      # 10% 用于评估
    
    # 随机打乱，取前 50 条（如果不足 50，则取全部）
    debug_root = Path("debug")
    debug_root.mkdir(parents=True, exist_ok=True)
    subset = train_dataset.shuffle(seed=42).select(range(min(50, len(train_dataset))))

    for idx, sample in enumerate(subset):
        folder = debug_root / f"sample_{idx}"
        folder.mkdir(parents=True, exist_ok=True)
        # 1) 保存问题
        q = sample.get("question", "")
        (folder / "question.txt").write_text(q, encoding="utf-8")

        # 2) 加载原始图像
        img = Image.open(io.BytesIO(sample["image"])).convert("RGB")
        draw = ImageDraw.Draw(img)

        # 3) 解析 answer 中的 bbox
        raw = sample.get("answer", "")
        try:
            # 尝试 Python 语法解析
            bbox = tuple(ast.literal_eval(raw))
        except Exception:
            # 回退到正则提取所有数字
            nums = re.findall(r"-?\\d+", str(raw))
            bbox = tuple(map(int, nums))

        # 4) 在图像上画红框
        draw.rectangle(bbox, outline="red", width=3)

        # 5) 保存带框的图像
        img.save(folder / "image.png")

    print(f"✅ 已将 {len(subset)} 个样本保存到 {debug_root}/ 下，每个子文件夹包含 question.txt 和 image.png。")
    
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
    # model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    model_name = "/home/uconn/BinLei/LLaMA-Factory/saves/qwen2_5vl-7b/full/sft"
    # model, tokenizer = get_model_and_tokenizer(
    #     model_name, 
    #     cache_dir="/mnt/data1/huggingface/models"
    # )
    # Load the VLM module
    vlm_module_cls = get_vlm_module(model_name)
    print("using vlm module:", vlm_module_cls.__name__)
    
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
        # max_steps=1000000,
        bf16=True,
        max_grad_norm=0.1,
        num_iterations=2,
        beta=0.002,
        max_prompt_length=1024,
        max_completion_length=8192,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_generations=4,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        eval_strategy="steps",
        eval_steps=10000,
        eval_accumulation_steps=1,
        eval_on_start=False,
        save_strategy="steps",
        save_steps=400,
        save_only_model=True,
        use_vllm=True,
        vllm_server_host="0.0.0.0",  # 多节点设置时替换为推理服务器的主机
        vllm_server_port=8888,
        vllm_gpu_memory_utilization=0.9,
        logging_steps=1,
        log_on_each_node=False,
        log_completions=True,
        report_to="wandb", # wandb/none
        reward_weights=tool_env.get_reward_weights()
    )
    # steps(梯度更新次数) = data_amount(总训练数据量)*num_iterations(相当于每组数据用几次)*num_generations(每个数据生成多少个回答)
    # / (gradient_accumulation_steps(积累几次梯度更新)*per_device_train_batch_size(每个GPU的batch大小)*num_gpus(使用的GPU数量))
    # model_args = ModelConfig(
    #     use_peft = True,
    #     lora_r = 64,
    #     lora_alpha = 128,
    #     lora_dropout = 0.05,
    #     lora_task_type = "CAUSAL_LM",
    # ) # For lora

    # 保存原始方法并创建补丁
    _original_from_pretrained = AutoModelForCausalLM.from_pretrained

    def _vl_compatible_from_pretrained(pretrained_model_name_or_path, *args, **kwargs):
        if isinstance(pretrained_model_name_or_path, str) and ("VL" in pretrained_model_name_or_path):
            return Qwen2_5_VLForConditionalGeneration.from_pretrained(
                pretrained_model_name_or_path,
                trust_remote_code=True,
                **kwargs
            )
        return _original_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

    # 应用补丁
    AutoModelForCausalLM.from_pretrained = _vl_compatible_from_pretrained
    
    # 6. 初始化训练器
    print("5. 初始化训练器...")
    trainer = GRPOEnvTrainer(
        model=model_name,
        reward_funcs=tool_env.get_reward_funcs(),
        reward_weights=tool_env.get_reward_weights(),
        env=tool_env,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        vlm_module=vlm_module_cls(),
        # peft_config=get_peft_config(model_args), # For lora
    )
    
    # 7. 开始训练
    print("6. 开始训练...")
    # trainer.train(resume_from_checkpoint = '/mnt/data1/home/lei00126/AgentTrainer/outputs/VG-grpo_checkpoint-800/checkpoint-2')
    trainer.train()
    
    print("训练完成！")

if __name__ == "__main__":
    main()