#!/usr/bin/env python3
"""
数据预处理独立脚本
运行一次后将处理好的数据保存为 Arrow 格式，后续可直接加载使用
"""

import os
import json
import glob
import random
from tqdm import tqdm
from multiprocessing import Pool
from typing import List, Dict, Callable, Any, Optional
from datasets import Features, Value, Sequence, ClassLabel
from datasets import Dataset, load_dataset, concatenate_datasets

def load_shard(path: str) -> Dataset:
    return Dataset.from_parquet(path, cache_dir='/home/uconn/.cache/huggingface/datasets/datasets--osunlp--UGround-V1-Data-Box')

def parallel_load_dataset(cache_dir: str, num_proc: int = 8) -> Dataset:
    # 1. 递归搜所有 parquet
    pattern = os.path.join(cache_dir, "**", "*.parquet")
    files = glob.glob(pattern, recursive=True)
    if not files:
        raise FileNotFoundError(f"No parquet shards found under {pattern}")

    # 2. 并行加载并显示进度条
    with Pool(processes=num_proc) as pool:
        shards_iter = pool.imap(load_shard, files)
        shards = []
        for shard in tqdm(shards_iter, total=len(files), desc="Loading shards", unit="shard"):
            shards.append(shard)

    # 3. 合并
    return concatenate_datasets(shards)

def preprocess_dataset_uground(dataset: Dataset) -> Dataset:
    """
    优化版本：使用批处理来避免内存问题和提高效率
    """
    
    def process_batch(examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """批处理函数"""
        batch_widths, batch_heights, batch_images = [], [], []
        batch_questions, batch_answers = [], []
        
        for i, conversations_raw in enumerate(examples["conversations"]):
            try:
                # 解析对话
                if isinstance(conversations_raw, str):
                    convs = json.loads(conversations_raw)
                else:
                    convs = conversations_raw
                
                # 提取 QA 对
                for j in range(len(convs) - 1):
                    cur, nxt = convs[j], convs[j + 1]
                    if cur.get("from") == "human" and nxt.get("from") == "gpt":
                        batch_widths.append(int(examples["width"][i]))
                        batch_heights.append(int(examples["height"][i]))
                        batch_images.append(examples["image"][i])
                        batch_questions.append(cur.get("value", "").strip())
                        batch_answers.append(nxt.get("value", "").strip())
            
            except Exception as e:
                print(f"处理批次中第 {i} 个样本时出错: {e}")
                continue
        
        return {
            "width": batch_widths,
            "height": batch_heights,
            "image": batch_images,
            "question": batch_questions,
            "answer": batch_answers,
        }
    
    # 定义输出特征
    new_features = Features({
        "width": Value("int32"),
        "height": Value("int32"),
        "image": Value("large_binary"), 
        "question": Value("string"),
        "answer": Value("string"),
    })
    
    print(f"开始批处理 {len(dataset)} 个样本...")
    
    # 使用 map 函数进行批处理，设置较小的批次大小
    transformed = dataset.map(
        process_batch,
        batched=True,
        batch_size=100,  # 较小的批次大小避免内存问题
        remove_columns=dataset.column_names,  # 移除所有原始列
        features=new_features,
        desc="Processing conversations"  # 显示进度条描述
    )
    
    print(f"处理完成！生成了 {len(transformed)} 个 QA 对")
    
    return transformed

def preprocess_and_save_dataset(
    cache_dir: str,
    output_path: str,
    n: Optional[int] = 10000,
    num_proc: int = 16
):
    """
    预处理数据并保存为 Arrow 格式
    
    Args:
        cache_dir: 原始数据缓存目录
        output_path: 输出文件路径 (建议以 .arrow 结尾)
        n: 采样数量，None 表示全量
        num_proc: 并行进程数
    """
    print("=== 开始数据预处理 ===")
    
    # 1. 加载原始数据
    print("1. 加载原始数据...")
    ds = parallel_load_dataset(cache_dir, num_proc=num_proc)
    print(f"   原始数据集大小: {len(ds)}")
    
    # 2. 采样
    if n is not None:
        total = len(ds)
        sample_size = min(n, total)
        ds = ds.select(range(sample_size))
        print(f"   采样后数据集大小: {len(ds)}")
    
    # 3. 预处理
    print("2. 开始预处理...")
    processed_ds = preprocess_dataset_uground(ds)
    
    # 4. 保存
    print("3. 保存处理后的数据...")
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存为 Arrow 格式
    processed_ds.save_to_disk(output_path)
    print(f"   数据已保存到: {output_path}")
    print(f"   最终数据集大小: {len(processed_ds)}")
    
    return processed_ds

if __name__ == "__main__":
    # 配置参数
    CACHE_DIR = "/home/uconn/.cache/huggingface/datasets/datasets--osunlp--UGround-V1-Data-Box"
    SAMPLE_SIZE = 500
    OUTPUT_PATH = f"/home/uconn/BinLei/processed_datasets/uground_processed_{SAMPLE_SIZE}"
    
    # 执行预处理
    processed_dataset = preprocess_and_save_dataset(
        cache_dir=CACHE_DIR,
        output_path=OUTPUT_PATH,
        n=SAMPLE_SIZE,
        num_proc=16
    )
    
    # 显示一些统计信息
    print("\n=== 处理结果统计 ===")
    print(f"数据集大小: {len(processed_dataset)}")
    print(f"数据集特征: {processed_dataset.features}")
    
    # 显示第一个样本（不显示图像数据）
    if len(processed_dataset) > 0:
        sample = processed_dataset[0]
        print("\n第一个样本:")
        for key, value in sample.items():
            if key == "image":
                print(f"  {key}: <binary data, size: {len(value)} bytes>")
            else:
                print(f"  {key}: {value}")