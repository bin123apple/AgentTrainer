#!/usr/bin/env python3
"""
数据预处理独立脚本
运行一次后将处理好的数据保存为 Arrow 格式，后续可直接加载使用
"""
import io
import re
import os
import ast
import json
import glob
import random
from tqdm import tqdm
from multiprocessing import Pool
from PIL import Image, ImageDraw
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

def denorm_bbox_to_pixel(bbox, w, h):
    x1, y1, x2, y2 = bbox          # 已经是 int
    return (
        round(x1 * w / 999),
        round(y1 * h / 999),
        round(x2 * w / 999),
        round(y2 * h / 999),
    )
    
def parse_answer(ans_str):
    try:                                   # 先试 ast.literal_eval
        res = ast.literal_eval(ans_str)
        if isinstance(res, (list, tuple)):
            return tuple(map(int, res[:4]))    # 只取前 4 个
    except Exception:
        pass
    nums = re.findall(r"-?\\d+", ans_str)   # 退而求其次：正则抽数字
    return tuple(map(int, nums[:4]))

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
                        w = int(examples["width"][i])
                        h = int(examples["height"][i])
                        norm_str = nxt.get("value", "").strip()
                        norm_ans = parse_answer(norm_str)
                        bbox_pixel = denorm_bbox_to_pixel(norm_ans, w, h)
                        batch_answers.append(str(bbox_pixel))
            
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
    n: Optional[int] = 20000, # 到 n 个样本为止，None 表示全量
    m: Optional[int] = 10000,  # 从第 m 个样本开始采样
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
    if n is not None and m is not None:
        k = max(0, n - m)                   # 要保留的样本数
        indices = range(m, m + k)           # 顺序子区间
        # 或 random.sample(range(total), k) # 随机子集
        ds = ds.select(indices)
        print(f"   采样后数据集大小: {len(ds)}")   # == k
    
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
    start_sample = 0
    end_sample = 10000
    OUTPUT_PATH = f"/home/uconn/BinLei/processed_datasets/uground_processed_{start_sample}_{end_sample}"
    
    # 执行预处理
    processed_dataset = preprocess_and_save_dataset(
        cache_dir=CACHE_DIR,
        output_path=OUTPUT_PATH,
        n=end_sample,
        m=start_sample,  # 从头开始采样
        num_proc=16
    )
    
    # 显示一些统计信息
    print("\n=== 处理结果统计 ===")
    print(f"数据集大小: {len(processed_dataset)}")
    print(f"数据集特征: {processed_dataset.features}")
    
    # 显示第一个样本（并把 image 存盘、画框）
    if len(processed_dataset) > 0:
        sample = processed_dataset[0]
        print("\n第一个样本:")

        # 先打印除 image 外的所有字段
        for key, value in sample.items():
            if key != "image":
                print(f"  {key}: {value}")

        # 处理 image
        img_data = sample["image"]  # raw bytes
        img = Image.open(io.BytesIO(img_data)).convert("RGB")

        # 解析 answer bbox（假设是 "(x1, y1, x2, y2)" 形式的字符串或直接 tuple）
        raw_bbox = sample.get("answer")
        if isinstance(raw_bbox, str):
            try:
                bbox = tuple(ast.literal_eval(raw_bbox))
            except Exception:
                # 回退到正则提取数字
                import re
                nums = re.findall(r"-?\d+", raw_bbox)
                bbox = tuple(map(int, nums))
        else:
            bbox = raw_bbox

        # 在图上画红框
        draw = ImageDraw.Draw(img)
        draw.rectangle(bbox, outline="red", width=3)

        # 保存带框的图片
        out_path = "sample_0_with_bbox.png"
        img.save(out_path)
        print(f"  image: saved with bbox → {out_path}")