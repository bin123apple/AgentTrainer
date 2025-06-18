import random
import json
import re
from typing import List, Dict, Callable, Any
import copy

from datasets import Dataset, load_dataset, concatenate_datasets # type: ignore


import json
import random
from datasets import Dataset
from typing import List, Dict, Optional

def preprocess_dataset(
    dataset_name: str,
    split: str = "train",
    n: Optional[int] = 1000,
    cache_dir: Optional[str] = None,
) -> Dataset:
    """
    通用预处理入口。

    Args:
        dataset_name: HF 上的数据集 ID，例如 "osunlp/UGround-V1-Data" 或其他。
        split:       拆分名称，如 "train"、"validation"。
        n:           要采样的样本数；若为 None 或大于总数则取全量。
        cache_dir:   指定 transformers/datasets 缓存目录。
        system_prompt, few_shot, fewshot_prob: 见 format_prompt/format_dataset。
    """
    # 1. 加载数据集
    ds = load_dataset(dataset_name, split=split, cache_dir=cache_dir)

    # 2. 取前 n 条（保持原序）
    if n is not None:
        total = len(ds)
        sample_size = min(n, total)
        ds = ds.select(range(sample_size))

    # 3. 针对 UGround-V1-Data 做特殊预处理
    if dataset_name.lower().startswith("osunlp/uground"):
        return preprocess_dataset_uground(ds)


def preprocess_dataset_uground(dataset: Dataset) -> Dataset:
    """
    将每条记录中的 human→gpt 对话拆分成多个 {'question', 'answer'} 样本。

    Args:
        dataset: 原始 Dataset，包含 `conversations` 字段（JSON 字符串）

    Returns:
        Dataset: 每条记录都只有两个字段：
          - question: str
          - answer:   str
    """
    def extract_qa_pairs(example):
        convs = json.loads(example["conversations"])
        qa_pairs = []
        for i in range(len(convs) - 1):
            cur, nxt = convs[i], convs[i + 1]
            if cur.get("from") == "human" and nxt.get("from") == "gpt":
                qa_pairs.append({
                    "question": cur.get("value", "").strip(),
                    "answer":   nxt.get("value", "").strip(),
                })
        return qa_pairs

    return dataset.flat_map(extract_qa_pairs)

# def preprocess_dataset_uground(
#     dataset: Dataset,
# ) -> Dataset:
#     """
#     对 osunlp/UGround-V1-Data 进行预处理：
#       1. 从 `conversations` 字段提取以 "Description:" 开头的内容作为 question
#       2. 取该对话列表的最后一个 message.value 作为 groundtruth answer
#       3. 调用 format_dataset 生成最终的 prompt/answer 结构

#     Args:
#         dataset:      原始 Dataset，包含 `conversations` 字段（JSON 字符串）
#         system_prompt: 可选的 system 消息
#         few_shot:     可选的 few-shot 示例列表
#         fewshot_prob: 使用 few-shot 的概率

#     Returns:
#         一个新 Dataset，每个例子包含：
#           - "prompt": List[{"role":..., "content":...}]
#           - "answer": str
#     """
#     # 1) 先 map 出 question & answer
#     def extract_qa(example):
#         """
#         从 example["conversations"] 中：
#         - 抽取位于 'Description:' 与下一个 'Answer:' 之间的内容作为 question
#         - 取最后一条消息的 value 作为 answer
#         """
#         conv_list = json.loads(example["conversations"])
#         question = ""
#         # 在所有消息中找带 Description: 的那条
#         for msg in conv_list:
#             val = msg.get("value", "")
#             if "Description:" in val:
#                 start = val.find("Description:") + len("Description:")
#                 # 尝试找 Answer:
#                 end_idx = val.find("Answer:", start)
#                 if end_idx == -1:
#                     # 如果没找到，取到字符串末尾
#                     question = val[start:].strip()
#                 else:
#                     question = val[start:end_idx].strip()
#                 break

#         # groundtruth 仍取最后一条消息
#         answer = conv_list[-1].get("value", "").strip()
#         return {"question": question, "answer": answer}

#     qa_ds = dataset.map(extract_qa, num_proc=10)
    
#     return qa_ds

def parse_crop_bbox_from_text(text: str):
    """
    从形如 <crop>((x1,y1),(x2,y2))</crop> 的文本中提取坐标，
    返回 (x1, y1, x2, y2) 四元组，找不到时返回 None。
    """
    # 匹配 <crop>( (x1,y1) , (x2,y2) )</crop>
    pattern = re.compile(
        r"<crop>\s*\(\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*\)\s*</crop>"
    )
    m = pattern.search(text)
    if not m:
        return None
    x1, y1, x2, y2 = map(int, m.groups())
    return x1, y1, x2, y2

def sanitize_dialogs(dialogs, placeholder="<IMAGE>"):
    """
    Return a deep-copied list of dialogs with all image_url fields replaced by the placeholder.

    Args:
        dialogs (List[List[Dict]]): A list of dialogs, each dialog is a list of message dicts.
        placeholder (str): The string to substitute for each image URL.

    Returns:
        List[List[Dict]]: A new list of dialogs with image_url fields replaced.
    """
    sanitized = copy.deepcopy(dialogs)
    for dialog in sanitized:
        for msg in dialog:
            content = msg.get("content")
            if isinstance(content, list):
                for piece in content:
                    if piece.get("type") == "image_url" and piece.get("image_url") is not None:
                        piece["image_url"] = placeholder
    return sanitized


def format_prompt(prompt: str,
                  system_prompt: str | None = None,
                  few_shot: List[Dict[str, str]] | None = None,
                  fewshot_prob: float = 1.0) -> List[Dict[str, str]]:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": "You are a visual localization assistant."})
    if few_shot and random.random() < fewshot_prob:
        messages.extend(few_shot)
    messages.append({"role": "user", "content": system_prompt + "\n" + prompt})
    return messages

def format_dataset(dataset: Dataset,
                   system_prompt: str | None = None,
                   few_shot: List[Dict[str, str]] | None = None,
                   fewshot_prob: float = 1.0,
                   question_key: str = "question",
                   answer_key: str = "answer",
                   ) -> Dataset:
    return dataset.map(lambda x: {
        "prompt": format_prompt(x[question_key], system_prompt, few_shot, fewshot_prob),
        "answer": x[answer_key]
    }, num_proc=10)