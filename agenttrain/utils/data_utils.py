import random
import json
import re, textwrap
from typing import List, Dict, Callable, Any
import copy
import re, base64, io
from typing import Any, List, Tuple, Optional
from PIL import Image
from datasets import Dataset, load_dataset, concatenate_datasets # type: ignore

import os
import datetime
import json
import random
from datasets import Dataset
from typing import List, Dict, Optional

import copy, re, base64, io, wandb
from typing import Any, List, Tuple, Optional
from PIL import Image

PLACEHOLDER = "<IMAGE>"
B64_PATTERN = re.compile(r"^data:image\/\w+;base64,(.+)", re.I)

# ❶ 安全占位 —— 不改动原 image_url（保留图片）
def sanitize_dialogs(dialogs: List[List[dict]], placeholder: str = PLACEHOLDER):
    safe = copy.deepcopy(dialogs)
    for dialog in safe:
        for msg in dialog:
            content = msg.get("content")
            if isinstance(content, list):
                for piece in content:
                    if piece.get("type") == "image_url":
                        # 单独存 placeholder，不覆盖真 URL
                        piece["image_url"] = placeholder
    return safe

# ❷ 单条 content → 文本 & 图片列表
def _b64_to_pil(data_url: str) -> Optional[Image.Image]:
    m = B64_PATTERN.match(data_url or "")
    if not m:
        return None
    try:
        return Image.open(io.BytesIO(base64.b64decode(m.group(1)))).convert("RGB")
    except Exception:
        return None

def flatten_text_and_images(
    content: Any, placeholder: str = PLACEHOLDER
) -> Tuple[str, Optional[List[Image.Image]]]:
    if isinstance(content, str):
        return content, None
    save_dir = "backup_images"
    os.makedirs(save_dir, exist_ok=True)
    if isinstance(content, list):
        parts = []
        for piece in content:
            if piece.get("type") == "image_url":
                img = _b64_to_pil(piece.get("image_url").get("url", ""))
                filename = f"image_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
                save_path = os.path.join(save_dir, filename)
                img.save(save_path)
                parts.append(f"{placeholder}(saved to {save_path})")
            else:
                parts.append(piece.get("text", ""))
        return "".join(parts)

    return str(content)

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

def parse_crop_bbox_from_text(text: str):
    """
    从形如 <crop>(Image_0, (10, 20), (110, 100))</crop> 的文本中提取:
      - id_num（整数）
      - top_left，tuple (x1, y1)
      - bottom_right，tuple (x2, y2)
    找不到时返回 (None, None, None)。
    """
    pattern = re.compile(textwrap.dedent(r"""
        <crop>\s*                       # <crop> 标签
        \(\s*                           # 外层左括号
        ([A-Za-z0-9_-]+)_(\d+)          # ① 图名 Image  ② 数字 ID
        \s*,\s*
        \(\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*\)  # ③ x1, ④ y1
        \s*,\s*
        \(\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*\)  # ⑤ x2, ⑥ y2
        \s*\)\s*                        # 外层右括号
        </crop>                         # 结束标签
    """), re.VERBOSE)

    m = pattern.search(text)
    if not m:
        return None, None, None
    
    # 解包并转换类型
    image_name, id_str, x1, y1, x2, y2 = m.groups()
    return int(id_str), (int(x1), int(y1)), (int(x2), int(y2))



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