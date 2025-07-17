import re
import io
import ast
import json
import base64
from PIL import Image, ImageDraw
from agenttrain.tools import crop, extract
from pathlib import Path
from agenttrain.envs.tool_env import ToolEnv
from vllm import LLM, SamplingParams
from agenttrain.parsers import XMLParser
from datasets import Dataset, load_from_disk
from agenttrain.inference.vllm_client import VLLMClient
from typing import List, Dict, Sequence, Any, Union, Tuple
from agenttrain.prompts.system_prompts import CROP_SYSTEM_PROMPT
from agenttrain.prompts.tool_description import CROP_TOOL_DESCRIPTION, EXTRACT_TOOL_DESCRIPTION, FIND_COLOR_TOOL_DESCRIPTION
from agenttrain.prompts.tool_example import CROP_TOOL_EXAMPLE, MERGE_TOOL_EXAMPLE, EXTRACT_TOOL_EXAMPLE, FIND_COLOR_TOOL_EXAMPLE

def encode_image(image_content):
    return base64.b64encode(image_content).decode('utf-8')

def extract_point_answer(content):
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    point_pattern = r'(\d+\.?\d*(?:\s*[,;\s]\s*|\s+)\d+\.?\d*)'
    content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
    if content_answer_match:
        content_answer = content_answer_match.group(1).strip()
        point_match = re.search(point_pattern, content_answer, re.DOTALL)
        if point_match:
            point_str = point_match.group(1)
            point = [float(x) for x in re.findall(r'\d+\.?\d*', point_str)]
            if len(point) >= 2:
                point = point[:2]
            else:
                point = [0, 0]
            return point
    return [0, 0]

def extract_coordinates(result: list[str]):
    text = result[0].strip()

    # 如果有 <answer> 标签，就提取标签内的内容；否则就直接用 text
    answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', text, re.DOTALL)
    if answer_match:
        content = answer_match.group(1)
    else:
        content = text  

    # 按 (x, y) 形式提取
    point_match = re.search(r'\(\s*(\d+)\s*,\s*(\d+)\s*\)', content)
    if point_match:
        x, y = map(int, point_match.groups())
        return (x, y)

    # 如果是 (x1, y1, x2, y2) 形式，取中心点
    box_match = re.search(r'\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)', content)
    if box_match:
        x1, y1, x2, y2 = map(int, box_match.groups())
        return ((x1 + x2)//2, (y1 + y2)//2)

    return None

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

class OSS_LLM:
    def __init__(self, args):
        self.args = args
        self.model = args.model_name
        self.tokenizer = args.model_name
        self.oss_llm = None
        self.oss_llm_init()
    
    def oss_llm_init(self):
        if self.oss_llm is None:
            self.oss_llm = LLM(
                model=self.model,
                tokenizer=self.model,
                tensor_parallel_size=4,
                gpu_memory_utilization=0.9,
                enforce_eager=True,
                max_model_len=19264,
                disable_custom_all_reduce=True,
                enable_prefix_caching=False,
                trust_remote_code=True,
            )
            
    def oss_llm_completion(self, messages, stop=None):
        sampling_params = SamplingParams(
                    n=1,
                    max_tokens=9632,
                    temperature=0,
                    top_p=1.0,
                    frequency_penalty=0,
                    presence_penalty=0
                )  
        sampling_params.stop = stop
        request_output = self.oss_llm.chat(messages, sampling_params)
        response_list = []
        for response in request_output[0].outputs:
            response_list.append(response.text)
        return response_list

    def _ask_llm(self, image_bytes: bytes, text: str) -> tuple[int,int]:
        b64: str = encode_image(image_bytes)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": text},
                ]
            }
        ]
        result = self.oss_llm_completion(messages)
        return result 

def _prepare_multimodal_chat_template(prompts: List[str], images: List[Image.Image]) -> List[dict]:
    '''
    Prepare the multimodal chat template for vLLM inference.
    This function takes a list of prompts and a list of images, and returns a list of dictionaries
    that can be used as input to the vLLM model.
    '''
    multimodal_inputs = []
    for prompt, image in zip(prompts, images):
        initial_prompts = CROP_SYSTEM_PROMPT.format(
        tool_descriptions=CROP_TOOL_DESCRIPTION+EXTRACT_TOOL_DESCRIPTION+FIND_COLOR_TOOL_DESCRIPTION,
        tool_example=CROP_TOOL_EXAMPLE+ EXTRACT_TOOL_EXAMPLE + FIND_COLOR_TOOL_EXAMPLE
        ) + f"\nNow Let's work on the real case:\n[Image_0 is displayed below]\nplease help me to identify the coordinate of the following element: \n{prompt}"
        if image is not None:
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            initial_message = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": initial_prompts},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                        ]
                    }
                ]
        else:
            initial_message = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": initial_prompts},
                        ]
                    }
                ]
        multimodal_inputs.append(initial_message)
    return multimodal_inputs

def vg_reward_func(
    parser: XMLParser,
    completions: List[Any],
    answer: List[Tuple[int, int, int, int]],
    all_images, 
    all_images_offset,
) -> List[float | None]:
    """
    Reward function that checks if the predicted point lies within the ground-truth bounding box.
    
    Args:
        completions: 模型的所有输出
        answer:      每个样本的 (x1, y1, x2, y2) ground-truth 框
        task:        每个样本对应的任务类型，"vg" 表示视觉定位
    
    Returns:
        List[float | None]: 
        - 若 task=="vg"，返回 1.0（在框内）或 0.0（不在框内或解析失败）
        - 否则返回 None
    """
    rewards: List[float | None] = []
    
    for completion, box, images, images_offset in zip(completions, answer, all_images, all_images_offset):
        # parser ground-truth to tuple
        # 1. 取出模型最后一句回答并清洗
        raw = str(get_last_answer(parser,completion)).strip()
        raw = f'<answer>{raw}</answer>'
        
        try:
            # 2. 用正则提取两个整数（支持负数）
            pattern = re.compile(
                r"""
                    <answer>                # 起始标签
                    \s*\(\s*
                    Image_(\d+)             # ① id_num
                    \s*,\s*
                    \(\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*\)  # ②③ x, y
                    \s*\)\s*
                    </answer>               # 结束标签
                """,
                re.VERBOSE
            )
            m = pattern.search(raw)
            if m:
                id_num, x, y = m.groups()
                id_num = int(id_num)
                x, y   = int(x), int(y)
                x = x + images_offset[id_num][0]
                y = y + images_offset[id_num][1]
            
            # 3. 拆箱 ground-truth
            if isinstance(box, str):
                try:
                    box = tuple(ast.literal_eval(box))
                except Exception:
                    nums2 = re.findall(r"-?\d+", box)
                    box = tuple(map(int, nums2))
            x1, y1, x2, y2 = box
            
            # 4. 判断并打分
            reward = 1.0 if (x1 <= x <= x2 and y1 <= y <= y2) else 0.0

        except Exception:
            reward = 0.0
        
        rewards.append(reward)
    
    return rewards

def get_last_answer(parser, trajectory: List[Dict[str, str]]) -> str | None:
    """Extract the last answer from a trajectory."""
    for msg in reversed(trajectory):
        if msg['role'] == 'assistant':
            if parser is None:
                raise ValueError("Parser is not set")
            parsed = parser.parse(msg['content'][0]['text'])
            if hasattr(parsed, 'answer') and parsed.answer is not None:
                return parsed.answer
    return None

import json, re, copy

# ----------------------------------------------------------------------
# 1) 预处理函数：去掉首条 user 的 <image_url>，删除中间 user 的冗余提示
# ----------------------------------------------------------------------
_PROMPT_RE = re.compile(
    r"Please describe Image_\d+ and check if the .*?provide the coordinate in the <answer>...</answer> tag\.",
    re.DOTALL
)

def clean_sample(sample: list) -> list:
    out = copy.deepcopy(sample)
    first_user_done = False

    for msg in out:
        if msg.get("role") != "user":
            continue

        # 只针对首条 user 额外处理
        if not first_user_done:
            text_keep = ""
            image_item = None

            # 从原 content 里提取出 initial_prompt 对应的 text + image_url
            for item in msg.get("content", []):
                if item.get("type") == "text" and not text_keep:
                    text_keep = _extract_initial_prompt(item["text"])
                if item.get("type") == "image_url" and image_item is None:
                    image_item = item

            # 构造新的 content：先放 text，再放 image_url（顺序可根据需要调整）
            new_content = []
            if text_keep:
                new_content.append({"type": "text", "text": text_keep})
            if image_item:
                new_content.append(image_item)

            msg["content"] = new_content
            first_user_done = True
            continue

        # 其余 user 按之前逻辑处理……
        new_content = []
        for item in msg.get("content", []):
            if item.get("type") != "text":
                new_content.append(item)
                continue
            text = _PROMPT_RE.sub("", item["text"]).strip()
            if text:
                new_content.append({"type": "text", "text": text})
        msg["content"] = new_content

    return out


# ----------------------------------------------------------------------
# 2) 打印用：把所有 "image_url": {...} 替成 "<IMAGE>" 方便阅读
# ----------------------------------------------------------------------
def _mask_images(obj):
    if isinstance(obj, dict):
        if "image_url" in obj:
            return "<IMAGE>"
        return {k: _mask_images(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_mask_images(v) for v in obj]
    return obj

def _extract_initial_prompt(text: str) -> str:
    """从长 prompt 中截出以 [Image_0 is displayed below] 开头的部分"""
    start_tag = "[Image_0 is displayed below]"
    idx = text.find(start_tag)
    return text[idx:].strip() if idx != -1 else ""

def main():
    
    try:
        PROCESSED_DATA_PATH = "/mnt/data1/processed_datasets/uground_processed_20000_30000"
        dataset = load_processed_dataset(PROCESSED_DATA_PATH)
    except Exception as e:
        print(f"加载数据失败: {e}")
        print("请先运行 agenttrain/utils/data_collection_save.py 生成预处理数据")
        return  # 或者 raise e 来停止程序
    
    # 2. 随机打乱并按比例分割数据集
    print("2. 分割训练集和验证集...")
    split = dataset.shuffle(seed=0).train_test_split(test_size=0.01, seed=0)
    
    train_dataset = split["train"]    # 99% 用于训练
    # print(f"Fist record in train dataset: {train_dataset[0]}")
    eval_dataset = split["test"]      # 1% 用于评估
    
    # 随机打乱，取前 50 条（如果不足 50，则取全部）
    debug_root = Path("debug")
    debug_root.mkdir(parents=True, exist_ok=True)
    subset = train_dataset.shuffle(seed=50).select(range(min(50, len(train_dataset))))

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

    # vllm_client = VLLMClient("0.0.0.0", 8888)
    # vllm_client.init_communicator()
    model = 'Qwen/Qwen2.5-VL-72B-Instruct'
    vllm = LLM(
        model=model,
        tokenizer=model,
        tensor_parallel_size=4,
        gpu_memory_utilization=0.95,
        enforce_eager=True,
        max_model_len=10000,
        disable_custom_all_reduce=True,
        enable_prefix_caching=False,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(
        max_tokens=8192,
        temperature=1,
        top_k=-1,
        min_p=0.0 
    )
    parser: XMLParser = XMLParser(fields=["reasoning", ("tool", "answer")])
    out_dir   = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "good_completions.jsonl"

    # 如果想重新开始而不是在旧文件后追加，先清空：
    # jsonl_path.unlink(missing_ok=True)
    
    debug_printed = False
    batch_size = 64
    for start in range(20000, len(train_dataset), batch_size):
        end   = min(start + batch_size, len(train_dataset))
        batch = train_dataset.select(range(start, end))   # ← 改这行

        prompts  = batch["question"]                      # ← 直接按列取
        answers  = batch["answer"] if "answer" in batch.column_names else None
        images   = [Image.open(io.BytesIO(b)).convert("RGB") for b in batch["image"]]

        multimodal_inputs = _prepare_multimodal_chat_template(prompts, images)
        env_result  = tool_env.generate(
            prompts          = multimodal_inputs,
            llm              = vllm,
            sampling_params  = sampling_params,
        )
        completions = env_result["all_messages"]
        all_images = env_result['images'] # calculate log_pb
        all_images_offset = env_result["images_offset"]
        
        rewards = vg_reward_func(
            parser = parser,
            completions  = completions,
            answer       = answers,
            all_images  = all_images,
            all_images_offset = all_images_offset
        )
        print(f"Rewards: {rewards}")
        
        good_cnt = 0
        # ----------------------------------------------------------------------
        # 3) 主循环：reward==1.0 时清洗并写入；同时打印一条 debug
        # ----------------------------------------------------------------------
        with jsonl_path.open("a", encoding="utf-8") as f:       # ① 追加
            for idx, r in enumerate(rewards):
                if r != 1.0:
                    continue

                raw_sample = completions[idx]
                sample     = clean_sample(raw_sample)

                # ---- DEBUG：首例前/后对比 -------------------------------------
                if not debug_printed:
                    print("=== RAW SAMPLE ===")
                    print(json.dumps(_mask_images(raw_sample), ensure_ascii=False, indent=2))
                    print("=== CLEANED SAMPLE ===")
                    print(json.dumps(_mask_images(sample),   ensure_ascii=False, indent=2))
                    debug_printed = True

                # ---- 写入 jsonl ----------------------------------------------
                json.dump(sample, f, ensure_ascii=False)
                f.write("\n")
                good_cnt += 1


        print(f"Batch {start//batch_size:4d}: kept {good_cnt}/{len(batch)}")

if __name__ == "__main__":
    main()