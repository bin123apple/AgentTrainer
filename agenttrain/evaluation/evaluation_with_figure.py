import matplotlib.pyplot as plt
import os
import shutil
from pathlib import Path
from PIL import Image, ImageDraw
import re
import io
import ast
import json
import base64
import argparse
from PIL import Image
from pydantic import BaseModel
from agenttrain.tools import crop
from pathlib import Path
from agenttrain.envs.tool_env import ToolEnv
from vllm import LLM, SamplingParams
from agenttrain.parsers import XMLParser
from agenttrain.utils.data_utils import sanitize_dialogs
from datasets import Dataset, load_from_disk
from agenttrain.inference.vllm_client import VLLMClient
from typing import List, Dict, Sequence, Any, Union, Tuple
from agenttrain.prompts.system_prompts import CROP_SYSTEM_PROMPT
from agenttrain.prompts.tool_description import CROP_TOOL_DESCRIPTION, EXTRACT_TOOL_DESCRIPTION, FIND_COLOR_TOOL_DESCRIPTION
from agenttrain.prompts.tool_example import CROP_TOOL_EXAMPLE, EXTRACT_TOOL_EXAMPLE, FIND_COLOR_TOOL_EXAMPLE, TOOL_PROMPT
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="/mnt/data1/home/lei00126/AgentTrainer/outputs/VG-grpo_sft/checkpoint-1200", help="Path to the pretrained model")
    return parser.parse_args()

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
    content = answer_match.group(1) if answer_match else text

    # 1) 先尝试匹配 Image_id + 坐标对
    img_coord_match = re.search(
        r'Image_(\d+)\s*,\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)',
        content
    )
    if img_coord_match:
        img_id = int(img_coord_match.group(1))
        x = int(img_coord_match.group(2))
        y = int(img_coord_match.group(3))
        return img_id, x, y

    # 2) 兼容只有 (x, y) 形式
    point_match = re.search(r'\(\s*(\d+)\s*,\s*(\d+)\s*\)', content)
    if point_match:
        x, y = map(int, point_match.groups())
        return None, x, y

    # 3) 兼容 (x1, y1, x2, y2) 形式，取中心点
    box_match = re.search(r'\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)', content)
    if box_match:
        x1, y1, x2, y2 = map(int, box_match.groups())
        return None, (x1 + x2)//2, (y1 + y2)//2

    return None, None, None

def load_processed_dataset(data_path: str) -> Dataset:
    """
    加载预处理好的数据集
    数据集格式:
    {
        "image": List[bytes],
        "width": List[int],
        "height": List[int],
        "question": List[str],
        "answer": List[str]
    }
    
    Args:
        data_path: 预处理数据的路径
    
    Returns:
        Dataset: 加载的数据集
    """
    print(f"从 {data_path} 加载预处理数据...")
    dataset = load_from_disk(data_path)
    print(f"数据集加载完成，大小: {len(dataset)}")
    return dataset

def _prepare_multimodal_chat_template(prompts: List[str], images: List[Image.Image]) -> List[dict]:
    '''
    Prepare the multimodal chat template for vLLM inference.
    This function takes a list of prompts and a list of images, and returns a list of dictionaries
    that can be used as input to the vLLM model.
    '''
    multimodal_inputs = []
    for prompt, image in zip(prompts, images):
        # initial_prompts = CROP_SYSTEM_PROMPT.format(
        # tool_descriptions=CROP_TOOL_DESCRIPTION+EXTRACT_TOOL_DESCRIPTION+FIND_COLOR_TOOL_DESCRIPTION,
        # tool_example=CROP_TOOL_EXAMPLE+ EXTRACT_TOOL_EXAMPLE + FIND_COLOR_TOOL_EXAMPLE
        # ) + f"\nNow Let's work on the real case:\n[Image_0 is displayed below]\nplease help me to identify the coordinate of the following element: \n{prompt}"
        initial_prompts = TOOL_PROMPT + f"please help me to identify the coordinate of the following element: \n{prompt}"
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

def parse_crop_bbox_from_text(text: str):
    m_img = re.search(
        r"\[Image_(\d+)[^\]]*offset[:：]?\s*\(\s*(\d+),\s*(\d+)\s*\)",
        text
    )
    if not m_img:
        return None
    dx, dy = int(m_img.group(2)), int(m_img.group(3))
    
    # 4. extract bounding box of the cropped region
    m_size = re.search(r"Cropped a region of size\s*(\d+)×(\d+)", text)
    if not m_size:
        raise ValueError("Cannot parse crop size or coords.")
    w, h = map(int, m_size.groups())
    x1, y1, x2, y2 = 0, 0, w, h
    real_tl = (x1 + dx, y1 + dy)
    real_br = (x2 + dx, y2 + dy)
    return (real_tl[0], real_tl[1], real_br[0], real_br[1]) if m_size else None

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
                gpu_memory_utilization=0.95,
                enforce_eager=True,
                max_model_len=30000,
                disable_custom_all_reduce=True,
                enable_prefix_caching=False,
                trust_remote_code=True,
            )
            
    def oss_llm_completion(self, messages, stop=None):
        sampling_params = SamplingParams(
                    n=1,
                    max_tokens=19263,
                    temperature=0,
                    top_p=1.0,
                    frequency_penalty=0,
                    presence_penalty=0
                )  
        sampling_params.stop = stop
        request_output = self.oss_llm.chat(messages, sampling_params)
        return request_output

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

def save_case_analysis(batch_num, case_num, original_img, cropped_imgs, final_click, crop_bboxs, gt_bbox, save_dir):
    case_dir = save_dir / f"batch_{batch_num}_case_{case_num}"
    case_dir.mkdir(parents=True, exist_ok=True)

    # 保存原图
    original_img.save(case_dir / "original.png")

    # 保存被裁剪的图
    for index, cropped_img in enumerate(cropped_imgs):
        cropped_img.save(case_dir / f"cropped_{index}.png")

    # 标记最终点击位置、crop bbox 和 ground truth bbox 的图
    marked_img = original_img.copy()
    draw = ImageDraw.Draw(marked_img)
    
    # 画出所有 crop bbox
    for crop_bbox in crop_bboxs:
        print(f'crop_bbox: {crop_bbox}')
        draw.rectangle(crop_bbox, outline="blue", width=3)
        
    print(f'gt_bbox: {gt_bbox}, final_click: {final_click}')
    draw.rectangle(gt_bbox, outline="green", width=3)
    if final_click:
        draw.ellipse((final_click[0]-5, final_click[1]-5, final_click[0]+5, final_click[1]+5), fill="red")
    marked_img.save(case_dir / "marked.png")


def main(multiturn_tools: bool = True):
    try:
        PROCESSED_DATA_PATH = "/mnt/data1/home/lei00126/datasets/screenspot_arrow"
        dataset = load_processed_dataset(PROCESSED_DATA_PATH)
        print(f"数据集大小: {len(dataset)}")
    except Exception as e:
        print(f"加载数据失败: {e}")
        return

    tool_env = ToolEnv(
        dataset=dataset,
        eval_dataset=None,
        system_prompt=CROP_SYSTEM_PROMPT,
        few_shot=[],
        tools=[crop],
        max_steps=5
    )

    args = parse_args()
    model_name = args.model_name if hasattr(args, 'model_name') else "model_results"
    model_name = model_name.split("/")[-1]

    # 使用一个独立的根目录存放结果
    results_dir = Path(model_name)
    results_dir.mkdir(parents=True, exist_ok=True)

    tester = OSS_LLM(args)

    if multiturn_tools:
        llm = tester.oss_llm
        sampling_params = SamplingParams(
            n=1,
            max_tokens=19263,
            temperature=0,
            top_p=1.0,
            frequency_penalty=0,
            presence_penalty=0
        )

    parser = XMLParser(fields=["reasoning", ("tool", "answer")])

    batch_size = 32
    total_correct = 0
    batch_correct_list = []

    for start in range(0, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))
        batch = dataset.select(range(start, end))

        prompts = batch["question"]
        answers = batch["answer"] if "answer" in batch.column_names else None
        images = [Image.open(io.BytesIO(b)).convert("RGB") for b in batch["image"]]

        multimodal_inputs = _prepare_multimodal_chat_template(prompts, images)
        env_result = tool_env.generate(
            prompts=multimodal_inputs,
            llm=llm,
            sampling_params=sampling_params,
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

        good_cnt = rewards.count(1)
        total_correct += good_cnt
        batch_correct_list.append(good_cnt)
        
        print(f"Batch {start//batch_size:4d}: kept {good_cnt}/{len(batch)}")
        
        # Save analysis for correct and wrong cases
        for case_type, predicate in [("correct", lambda r: r == 1),
                                    ("wrong",   lambda r: r != 1)]:
            for idx, reward in enumerate(rewards):
                if predicate(reward):
                    msgs = env_result["all_messages"][idx]
                    image_offset = env_result["images_offset"][idx]
                    
                    sanitize_message = sanitize_dialogs([msgs])[0]
                    print(f'len(msgs): {len(msgs)}')
                    print(f"sanitize_message: {json.dumps(sanitize_message, indent=2, ensure_ascii=False)}")
                    # 1) 提取所有裁剪图
                    cropped_images = [
                        Image.open(io.BytesIO(base64.b64decode(item["image_url"]["url"]
                                    .split("base64,")[1]))).convert("RGB")
                        for msg in msgs[1:] if msg.get("role") == "user"
                        for item in msg.get("content", [])
                        if item.get("type") == "image_url"
                    ]

                    # 2) 提取所有 crop bbox
                    crop_bboxs = [
                        parse_crop_bbox_from_text(item["text"])
                        for msg in msgs[1:] if msg.get("role") == "user"
                        for item in msg.get("content", [])
                        if item.get("type") == "text"
                        if parse_crop_bbox_from_text(item["text"]) is not None
                    ]

                    # 3) 提取最终点击
                    raw = str(get_last_answer(parser, msgs)).strip()
                    img_id, x, y = extract_coordinates([raw])
                    if img_id:
                        dx, dy  = image_offset[img_id]
                        final_click = (x + dx, y + dy) if x is not None and y is not None else None
                    else:
                        final_click = None
                    if isinstance(final_click, str):
                        try:
                            final_click = tuple(ast.literal_eval(final_click))
                        except Exception:
                            nums = re.findall(r"-?\\d+", final_click)
                            final_click = tuple(map(int, nums))
                            
                    # Convert gt_bbox to tuple if it's a string
                    box = answers[idx]
                    if isinstance(box, str):
                        try:
                            box = tuple(ast.literal_eval(box))
                        except Exception:
                            nums2 = re.findall(r"-?\d+", box)
                            box = tuple(map(int, nums2))

                    # 4) 保存到不同子文件夹
                    save_case_analysis(
                        batch_num=start//batch_size,
                        case_num=f"{case_type}_{idx}",
                        original_img=images[idx],
                        cropped_imgs=cropped_images,
                        final_click=final_click,
                        crop_bboxs=crop_bboxs,
                        gt_bbox=box,
                        save_dir=results_dir
                    )
                    print(f"Saved {case_type} case {idx} of batch {start//batch_size}")
                    break


    # 绘制并保存折线图
    plt.figure(figsize=(10, 6))
    plt.plot(batch_correct_list, marker='o')
    plt.title('Correct Counts per Batch')
    plt.xlabel('Batch Number')
    plt.ylabel('Correct Count')
    plt.grid(True)
    plt.savefig(results_dir / "batch_accuracy.png")
    plt.close()

    print(f"\n✅ Total correct: {total_correct} / {len(dataset)} ({total_correct / len(dataset):.2%})")

if __name__ == "__main__":
    main()
