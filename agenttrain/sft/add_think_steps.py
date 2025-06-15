import re
import io
import ast
import json
import random
import base64
import argparse
from PIL import Image
from datetime import datetime
from agenttrain.tools import crop
from pathlib import Path
from agenttrain.envs.tool_env import ToolEnv
from vllm import LLM, SamplingParams
from agenttrain.parsers import XMLParser
from datasets import Dataset, load_from_disk
from agenttrain.sft.crop_prompts import crop_prompts
from agenttrain.inference.vllm_client import VLLMClient
from typing import List, Dict, Sequence, Any, Union, Tuple, Iterable
from agenttrain.prompts.system_prompts import CROP_SYSTEM_PROMPT
from agenttrain.prompts.tool_description import CROP_TOOL_DESCRIPTION
from agenttrain.prompts.tool_example import CROP_TOOL_EXAMPLE

BBox = Tuple[int, int, int, int]
NormBBox = Tuple[int, int, int, int]

__all__ = [
    "denorm_bbox",
    "map_answer_bboxes",
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-VL-72B-Instruct", help="Path to the pretrained model")
    return parser.parse_args()

# 把 0–999 归一化坐标映射到像素空间 -----------------------------
def denorm_bbox_to_pixel(bbox, w, h):
    x1, y1, x2, y2 = bbox          # 已经是 int
    return (
        round(x1 * w / 999),
        round(y1 * h / 999),
        round(x2 * w / 999),
        round(y2 * h / 999),
    )

# 如果 answer 可能是字符串，就先按你的老办法转成 int 元组 -------
def parse_answer(ans_str):
    try:                                   # 先试 ast.literal_eval
        res = ast.literal_eval(ans_str)
        if isinstance(res, (list, tuple)):
            return tuple(map(int, res[:4]))    # 只取前 4 个
    except Exception:
        pass
    nums = re.findall(r"-?\\d+", ans_str)   # 退而求其次：正则抽数字
    return tuple(map(int, nums[:4]))


def highlight_and_save_region(image: Image.Image, center: tuple[int, int],
                              half_size_x: int = 600, half_size_y: int = 250) -> tuple[tuple[int, int], bytes]:
    """ 
    以 center 为中心，上下左右各 half_size 像素（超出边界则自动裁剪），
    在原图副本上画红色矩形，并保存：
      1. 带标注的整图
      2. 裁剪出的矩形区域
    
    返回 (annotated_path, cropped_path)
    """
    # 1. 计算边界
    width, height = image.size
    x, y = center
    left = max(0, x - half_size_x)
    top = max(0, y - half_size_y)
    right = min(width, x + half_size_x)
    bottom = min(height, y + half_size_y)
    
    if left >= right or top >= bottom:
        raise ValueError(f"Invalid region: {(left, top, right, bottom)}")
    cropped = image.crop((left, top, right, bottom))

    buffer = io.BytesIO()
    cropped.save(buffer, format="PNG") 
    cropped_bytes = buffer.getvalue()
    cropped_b64 = base64.b64encode(cropped_bytes).decode("utf-8")
    
    return (left, top, right, bottom), cropped_b64

def encode_image(image_content):
    return base64.b64encode(image_content).decode('utf-8')

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

def generate_crop_data(generator, prompts: str, images: Image.Image):
    # Initialize crop dialogue
    crop_dialogue = []
    
    crop_prompts_list = [
        line.strip()
        for line in crop_prompts.strip().splitlines()
        if line.strip()
    ]
    # generate crop data
    multimodal_inputs = [] # input messages
    for prompt, image in zip(prompts, images):
        system_prompt = (
            "Which area is the element {instruction} located in?\n"
            "Please describe the area, DO NOT provide the coordinates."
        )
        
        # generate crop range
        instruction_format = system_prompt.format(instruction=prompt)
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": instruction_format},
                ]
            }
        ]
        multimodal_inputs.append(message)
    llm_responses = generator.oss_llm_completion(
        multimodal_inputs,
    )
    print(f"Received {len(llm_responses)} responses from vLLM.")
    
    for llm_response in llm_responses:
        text = llm_response.outputs[0].text
        
        # Append the crop prompt
        extra_prompt = random.choice(crop_prompts_list)
        text = f"{text}\n{extra_prompt}"
        
        # DEBUG: Print the responses
        print(f"Crop Response: {text}")
        
        crop_dialogue.append(text)
    
    return crop_dialogue

def generate_answer_dialogue(generator, prompts: str, crop_commands):
    # Initialize answer dialogue
    answer_dialogue = []
    
    # generate answer dialogue
    multimodal_inputs = [] # input messages
    for prompt, crop_command in zip(prompts, crop_commands):
        system_prompt = (
            "Which area is the element {instruction} located in?\n"
            "Please describe the area, DO NOT provide the coordinates."
        )

        instruction_format = system_prompt.format(instruction=prompt)
        cropped_b64 = crop_command['cropped_b64']
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{cropped_b64}"}},
                    {"type": "text", "text": instruction_format},
                ]
            }
        ]
        multimodal_inputs.append(message)
    llm_responses = generator.oss_llm_completion(
        multimodal_inputs,
    )
    
    for llm_response in llm_responses:
        text = llm_response.outputs[0].text
        
        # DEBUG: Print the responses
        print(f"Answer Response: {text}")
        
        answer_dialogue.append(text)
    
    return answer_dialogue

def generate_crop_commands(answers, images: List[Image.Image]):
    crop_commands = []
    for answer, image in zip(answers, images):
        if isinstance(answer, str):
            try:
                answer = tuple(ast.literal_eval(answer))
            except Exception:
                nums2 = re.findall(r"-?\d+", answer)
                answer = tuple(map(int, nums2))
        x1, y1, x2, y2 = answer
        mid_x = int((x1 + x2) // 2)
        mid_y = int((y1 + y2) // 2)
        point = (mid_x, mid_y)
        assert point is not None and isinstance(point, tuple) and len(point) == 2, "point must be a (x, y) tuple"
        assert all(isinstance(p, (int)) for p in point), "coordinates must be int"
        
        # generate random crop size
        img_w, img_h = image.size          # PIL: (width, height)
        min_x, max_x = int(0.1 * img_w), int(0.4 * img_w)
        min_y, max_y = int(0.1 * img_h), int(0.4 * img_h)
        while True:
            half_size_x = random.randint(min_x, max_x)
            half_size_y = random.randint(min_y, max_y)
            if half_size_x > half_size_y:      # 满足最后一个条件
                break
            
        # crop region
        crop_bbox, cropped_b64 = highlight_and_save_region(
            image, point,
            half_size_x=half_size_x, half_size_y=half_size_y
        ) # (左, 上, 右, 下), data_url
        crop_commands.append({
            "crop_bbox": crop_bbox,
            "cropped_b64": cropped_b64
        })
    return crop_commands

def merge_and_save_dialogues(prompts, images, answers, crop_dialogue, 
                             crop_commands, answer_dialogue, out_dir, image_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dialogues = []
    for prompt, image_original, answer, crop_text, crop_command, answer_text in zip(prompts, images, answers, crop_dialogue, 
                                                                           crop_commands, answer_dialogue):
        dialogue = dict()
        dialogue['messages'] = []
        dialogue['images'] = []
        
        # add first turn mnessage to dialogue
        system_prompt = (
            "Output only the coordinate of one point in your response. "
            "What element matches the following task: {instruction}"
        ) 
        instruction_format = system_prompt.format(instruction=prompt)
        user_initial_message = {
            "content": f"<image>{instruction_format}",
            "role": "user"
        }
        dialogue['messages'].append(user_initial_message)
        
        # Save first turn image to file and add to dialogue['images']
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")  # 精确到毫秒
        file_name = f"image_{ts}.png"
        file_path = image_dir / file_name
        image_original.save(file_path, format="PNG")
        dialogue['images'].append(str(file_path))
        
        # add crop text and crop command to dialogue
        crop_bbox = crop_command['crop_bbox']
        left, top, right, bottom = crop_bbox
        crop_bbox = f"<crop>{((left, top), (right, bottom))}</crop>"
        crop_text = crop_text + f"\n{crop_bbox}"
        assistant_crop_message = {
            "content": f"{crop_text}",
            "role": "assistant"
        }
        dialogue['messages'].append(assistant_crop_message)
        
        # Save cropped image and add it to dialogue['images']
        cropped_b64 = crop_command['cropped_b64']
        img_bytes = base64.b64decode(cropped_b64)
        image_cropped = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")  # 精确到毫秒
        file_name = f"image_{ts}.png"
        file_path = image_dir / file_name
        image_cropped.save(file_path, format="PNG")
        dialogue['images'].append(str(file_path))
        
        # add user reminder message and crop information to dialogue
        new_w, new_h = image_cropped.size
        width, height = image_original.size
        reminder_message = (
            f"<image>Cropped a region of size {new_w}×{new_h} pixels "
            f"from the original image ({width}×{height}), "
            f"located at top-left ({left}, {top}) and bottom-right ({right}, {bottom})."
        )
        crop_reminder = {
            "content": reminder_message,
            "role": "user"
        }
        dialogue['messages'].append(crop_reminder)
        
        # Add answer text to dialogue
        if isinstance(answer, str):
            try:
                answer = tuple(ast.literal_eval(answer))
            except Exception:
                nums2 = re.findall(r"-?\d+", answer)
                answer = tuple(map(int, nums2))
        x1, y1, x2, y2 = answer
        print(f"Answer coordinates: {answer}")
        mid_x = int((x1 + x2) // 2)
        mid_y = int((y1 + y2) // 2)
        point = (mid_x, mid_y)
        assistant_answer_message = {
            "content": answer_text + f"\n<answer>{point}</answer>",
            "role": "assistant"
        }
        dialogue['messages'].append(assistant_answer_message)
        
        # Save it to dialogues
        dialogues.append(dialogue)
    return dialogues

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
                tensor_parallel_size=2,
                gpu_memory_utilization=0.95,
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

def main():
    out_dir   = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "dialogues.json"
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        PROCESSED_DATA_PATH = "/mnt/data1/processed_datasets/uground_processed_10000"
        dataset = load_processed_dataset(PROCESSED_DATA_PATH)
    except Exception as e:
        print(f"加载数据失败: {e}")
        print("请先运行 agenttrain/utils/data_collection_save.py 生成预处理数据")
        return  # 或者 raise e 来停止程序
    
    args = parse_args()
    generator = OSS_LLM(args)
    
    batch_size = 32
    # Whole dataset
    for start in range(0, len(dataset), batch_size):
    # for start in range(0, 8, batch_size):
        
        end   = min(start + batch_size, len(dataset))
        batch = dataset.select(range(start, end))   # ← 改这行

        prompts  = batch["question"]                      # ← 直接按列取
        answers  = batch["answer"] if "answer" in batch.column_names else None
        images   = [Image.open(io.BytesIO(b)).convert("RGB") for b in batch["image"]]
        answers_pixel = []      # ← 存回像素坐标（字符串形式）

        for ans_str, img_bytes in zip(batch["answer"], batch["image"]):
            bbox_norm = parse_answer(ans_str)              # (x1,y1,x2,y2) 全是 int
            img = Image.open(io.BytesIO(img_bytes))        # 打开图片
            w, h = img.size                                # 宽、高
            bbox_pixel = denorm_bbox_to_pixel(bbox_norm, w, h)
            answers_pixel.append(str(bbox_pixel))          # 再转回 str 保存
        answers = answers_pixel if answers is not None else None
        
        # generate crop dialogue (without crop commands)
        crop_dialogue = generate_crop_data(generator, prompts, images) # List[text]
        
        # generate crop commands
        crop_commands = generate_crop_commands(answers, images) # List[dict((左, 上, 右, 下), cropped_b64)]
        
        # generate answers dialogue
        answer_dialogue = generate_answer_dialogue(generator, prompts, crop_commands) # List[text]
        
        # merge and save dialogues
        dialogues = merge_and_save_dialogues(prompts, images, answers, crop_dialogue,
                                 crop_commands, answer_dialogue, out_dir, image_dir)
        
        # save dialogues to json file
        # 1. 读取旧内容
        if json_path.exists() and json_path.stat().st_size > 0:
            with json_path.open("r", encoding="utf-8") as f:
                try:
                    old_dialogues = json.load(f)
                    if not isinstance(old_dialogues, list):
                        raise ValueError("JSON 根对象必须是数组")
                except json.JSONDecodeError as e:
                    raise ValueError(f"无法解析现有 JSON 文件: {e}") from None
        else:
            old_dialogues = []

        # 2. 合并
        merged = old_dialogues + dialogues

        # 3. 写回
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)

        print(f"✅ 已写入 {len(merged)} 条对话到 {json_path}")        
        

if __name__ == "__main__":
    main()