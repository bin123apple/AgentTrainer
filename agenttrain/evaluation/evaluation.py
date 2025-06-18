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
from datasets import Dataset, load_from_disk
from agenttrain.inference.vllm_client import VLLMClient
from typing import List, Dict, Sequence, Any, Union, Tuple
from agenttrain.prompts.system_prompts import CROP_SYSTEM_PROMPT
from agenttrain.prompts.tool_description import CROP_TOOL_DESCRIPTION
from agenttrain.prompts.tool_example import CROP_TOOL_EXAMPLE
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="/home/uconn/BinLei/LLaMA-Factory/saves/UI-TARS-1.5-7B/full/sft", help="Path to the pretrained model")
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
        initial_prompts = CROP_SYSTEM_PROMPT.format(
        tool_descriptions=CROP_TOOL_DESCRIPTION,
        tool_example=CROP_TOOL_EXAMPLE
        ) + f'\nNow please help me to identify the coordinate of the following element : \n{prompt}'
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
    parser,
    completions: List[Any],
    answer: List[Tuple[int, int, int, int]],
    task: List[str],
    **kwargs
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
    
    for completion, box, t in zip(completions, answer, task):
        # parser ground-truth to tuple
        if t == "vg":
            # 1. 取出模型最后一句回答并清洗
            raw = str(get_last_answer(parser, completion)).strip()
            
            try:
                # 2. 用正则提取两个整数（支持负数）
                nums = re.findall(r"-?\d+", raw)
                x, y = int(nums[0]), int(nums[1])
                # print(f"Extracted coordinates: ({x}, {y}).")
                
                # 3. 拆箱 ground-truth
                if isinstance(box, str):
                    try:
                        box = tuple(ast.literal_eval(box))
                    except Exception:
                        nums2 = re.findall(r"-?\d+", box)
                        box = tuple(map(int, nums2))
                x1, y1, x2, y2 = box
                # print(f"Ground-truth box: ({x1}, {y1}), ({x2}, {y2}).")
                
                # 4. 判断并打分
                reward = 1.0 if (x1 <= x <= x2 and y1 <= y <= y2) else 0.0
            except Exception:
                reward = 0.0
        else:
            reward = None
        
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
                gpu_memory_utilization=0.7,
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

def main(multiturn_tools: bool = True):
    
    try:
        PROCESSED_DATA_PATH = "/home/uconn/datasets/screenspot_arrow"
        dataset = load_processed_dataset(PROCESSED_DATA_PATH) # 
        print(f"数据集大小: {len(dataset)}")
    except Exception as e:
        print(f"加载数据失败: {e}")
        return  # 或者 raise e 来停止程序
        
    # 3. 设置工具环境
    print("3. 初始化工具环境...")
    tool_env = ToolEnv(
        dataset=dataset,
        eval_dataset=None,
        system_prompt=CROP_SYSTEM_PROMPT,
        few_shot=[],
        tools=[crop],
        max_steps=5
    )
    
    args = parse_args()
    tester = OSS_LLM(args)
    if multiturn_tools:
        llm = tester.oss_llm
        sampling_params = SamplingParams(
                    n=1,
                    max_tokens=9632,
                    temperature=0,
                    top_p=1.0,
                    frequency_penalty=0,
                    presence_penalty=0
                )  
    
    parser: XMLParser = XMLParser(fields=["reasoning", ("tool", "answer")])
    
    out_dir   = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 如果想重新开始而不是在旧文件后追加，先清空：
    # jsonl_path.unlink(missing_ok=True)

    batch_size    = 32
    total_correct = 0          # ★ 全局计数

    for start in range(0, len(dataset), batch_size):
        end   = min(start + batch_size, len(dataset))
        batch = dataset.select(range(start, end))

        prompts = batch["question"]
        answers = batch["answer"] if "answer" in batch.column_names else None
        print(f"anwers: {answers}")
        images  = [Image.open(io.BytesIO(b)).convert("RGB") for b in batch["image"]]

        if multiturn_tools: # multi-turns + tools
            multimodal_inputs = _prepare_multimodal_chat_template(prompts, images)
            env_result = tool_env.generate(
                prompts         = multimodal_inputs,
                llm             = llm,
                sampling_params = sampling_params,
            )
            completions = env_result["all_messages"]
        else: # zero-shot
            system_prompt = (
                "Output only the coordinate of one point in your response. "
                "What element matches the following task: {instruction}"
            )
            multimodal_inputs = [] # input messages
            for prompt, image in zip(prompts, images):
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
                
            llm_responses = tester.oss_llm_completion(
                multimodal_inputs,
            )
            print(f"Received {len(llm_responses)} responses from vLLM.")
            
            # Collect output messages
            completions = []
            for llm_response in llm_responses:
                text = llm_response.outputs[0].text
                coord_pat = re.compile(r"\(\s*\d+\s*[,\uFF0C]\s*\d+\s*\)")
                wrapped = coord_pat.sub(lambda m: f"<answer>{m.group(0)}</answer>",
                                        text,
                                        count=1) 
                completions.append([{"role": "assistant", "content": [{'type': 'text', 'text': wrapped}]}])

        print(f"completions last elements: {[c[-1] for c in completions]}")
        
        rewards = vg_reward_func(
            parser      = parser,
            completions = completions,
            answer      = answers,
            task        = ["vg"] * len(batch),
        )

        good_cnt       = rewards.count(1)   # ★ 本批命中
        total_correct += good_cnt           # ★ 累计

        print(f"Batch {start//batch_size:4d}: kept {good_cnt}/{len(batch)}")

    # ★ 结束后打印整体命中率
    print(f"\n✅ Total correct: {total_correct} / {len(dataset)} "
          f"({total_correct / len(dataset):.2%})")


if __name__ == "__main__":
    main()