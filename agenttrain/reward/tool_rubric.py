import re
import ast
import json
import base64
from io import BytesIO
from PIL import Image
from typing import List, Dict, Callable, Any, Tuple
from agenttrain.parsers import XMLParser # rewrite this one
from agenttrain.reward.rubric import Rubric
from agenttrain.reward.math_grader import grade
from agenttrain.utils.data_utils import parse_crop_bbox_from_text

def extract_first_image(conv_list):
    for msg in conv_list:
        if msg.get("role") == "user":
            for part in msg.get("content", []):
                if part.get("type") == "image_url":
                    url = part["image_url"]["url"]
                    prefix = "data:image/png;base64,"
                    if url.startswith(prefix):
                        b64_str = url[len(prefix):]
                        image_data = base64.b64decode(b64_str)
                        return Image.open(BytesIO(image_data))
    return None

def compute_iou(pred_bbox, gt_bbox, img_size):
    """
    pred_bbox, gt_bbox: (x1, y1, x2, y2)
    img_size: (width, height)
    返回:
      - 正常情况下：IoU ∈ [0,1]
      - 边界无效 / 尺寸过小 / 无交集：0.0
    """
    width, height = img_size

    def is_valid(bbox):
        x1, y1, x2, y2 = bbox
        # 1) 越界检查
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            return False
        # 2) 坐标逻辑检查
        if x2 <= x1 or y2 <= y1:
            return False
        # 3) 最小尺寸检查（至少 28×28）
        if (x2 - x1) < 28 or (y2 - y1) < 28:
            return False
        return True

    # 任意一个 bbox 不合法，则直接返回 0.0
    if not is_valid(pred_bbox):
        return 0.0

    # 下面是标准的 IoU 计算
    xA = max(pred_bbox[0], gt_bbox[0])
    yA = max(pred_bbox[1], gt_bbox[1])
    xB = min(pred_bbox[2], gt_bbox[2])
    yB = min(pred_bbox[3], gt_bbox[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0

    pred_area = (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1])
    gt_area   = (gt_bbox[2]   - gt_bbox[0]) * (gt_bbox[3]   - gt_bbox[1])
    union = pred_area + gt_area - inter_area
    if union <= 0:
        return 0.0

    return inter_area / union

def average_crop_reward(crop_bboxs, gt_bbox, img_size, weights=None, ):
    """
    crop_bboxs: List[Tuple[int,int,int,int]]，模型每条 <crop> 指令对应的 box
    gt_bbox:   Tuple[int,int,int,int]，ground-truth box
    weights:   Optional[List[float]]，与 crop_bboxs 等长，若为 None 则默认等权重
    返回值:    float，位于 [0,1] 之间，根据覆盖度 (intersection/gt_area) 的加权平均
    """
    n = len(crop_bboxs)
    if n == 0:
        return 0.0

    # 1) 计算每条 crop 的IoU：交集面积 /并集面积
    coverages = [compute_iou(pred, gt_bbox, img_size) for pred in crop_bboxs]

    # 2) 准备权重，默认等权
    if weights is None:
        weights = [1.0 / n] * n
    else:
        s = sum(weights)
        weights = [w / s for w in weights]

    # 3) 加权平均
    reward = sum(w * c for w, c in zip(weights, coverages))
    return reward

class ToolRubric(Rubric):
    def __init__(self,
                 parser: XMLParser = XMLParser(fields=["reasoning", ("tool", "answer")]),
                 env_parser: XMLParser = XMLParser(fields=["result"]),
                 tools: List[Callable] = []):
        self.parser = parser
        self.env_parser = env_parser
        self.tools = {tool.__name__: tool for tool in tools}
        self.reward_funcs = [
            self.mc_reward_func,
            self.math_reward_func,
            self.code_reward_func,
            self.vg_reward_func,
            self.correct_answer_reward_func,
            self.correct_crop_func,
            self.tool_execution_reward_func,
            self.parser.get_format_reward_func(),
            self.parser.get_xml_reward_func(),
        ] # TODO: add tool feedbacks here.
        self.reward_weights = [
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.7,
            0.4,
            0.1,
            0.1,
        ]
        for tool_name in self.tools.keys():
            self.reward_funcs.append(self.get_named_tool_reward_func(tool_name))
            self.reward_weights.append(0.2)
            # FIXME: Do we still need these two rewards?
            # self.reward_funcs.append(self.get_named_tool_count_reward_func(tool_name))
            # self.reward_weights.append(0.0)
            # self.reward_funcs.append(self.get_named_tool_attempt_reward_func(tool_name))
            # self.reward_weights.append(0.0)

    def evaluate_code(self, code_str, answer, **kwargs) -> float:
        import io
        import sys
        import signal
        from contextlib import redirect_stdout
        
        try:
            test_cases = json.loads(answer)['test_cases']
        except:
            return 0.0
        # strip ```python and ``` if present at the beginning and end of the code
        code_str = code_str.strip()
        if code_str.startswith('```python'):
            code_str = code_str[9:]
        elif code_str.startswith('```'):
            code_str = code_str[3:]
        if code_str.endswith('```'):
            code_str = code_str[:-3]
        code_str = code_str.strip()

        def timeout_handler(signum, frame):
            raise TimeoutError("Code execution timed out")

        def normalize_output(output):
            # Normalize line endings and whitespace
            return '\n'.join(line.strip() for line in output.splitlines())
        
        total_cases = 0
        passed = 0
        
        for test in test_cases:
            output = io.StringIO()
            sys.stdin = io.StringIO(test['input'])
            try:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(10)
                with redirect_stdout(output):
                    exec(code_str)
                signal.alarm(0)
                actual = normalize_output(output.getvalue())
                expected = normalize_output(test['output'])
                
                # Compare each line individually
                actual_lines = actual.splitlines()
                expected_lines = expected.splitlines()
                total_cases += len(expected_lines)
                for a, e in zip(actual_lines, expected_lines):
                    if a == e:
                        passed += 1
                    
            except Exception as e:
                sys.stdin = sys.__stdin__
                return 0.0
            sys.stdin = sys.__stdin__
        
        return passed / total_cases if total_cases else 0.0
        

    def code_reward_func(self, completions, answer, task, **kwargs) -> List[float | None]:
        """Reward function that checks if the final answer matches the expected answer."""
        rewards = []
        for completion, ans, t in zip(completions, answer, task):
            if t == "code":
                response = str(self.get_last_answer(completion))
                reward = self.evaluate_code(response, ans, **kwargs)
            else:
                reward = None
            rewards.append(reward)
        return rewards
    
    def mc_reward_func(self, completions, answer, task, **kwargs) -> List[float | None]:
        """Reward function that checks if the final answer matches the expected answer."""
        rewards = []
        for completion, ans, t in zip(completions, answer, task):
            if t == "mc":
                response = str(self.get_last_answer(completion)) #[0]
                if len(response.strip()) > 0 and isinstance(response, str):
                    response = response.strip()[0]
                reward = 1.0 if response == ans.strip() else 0.0
            else:
                reward = None
            rewards.append(reward)
        return rewards

    def vg_reward_func(
        self,
        completions: List[Any],
        answer: List[Tuple[int, int, int, int]],
        task: List[str],
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
        
        for completion, box, t, images, images_offset in zip(completions, answer, task, all_images, all_images_offset):
            # parser ground-truth to tuple
            if t == "vg":
                # 1. 取出模型最后一句回答并清洗
                raw = str(self.get_last_answer(completion)).strip()
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
                        # print(f"VG id_num: {id_num}, x: {x}, y: {y}.")
                        # print(f'images_offset: {images_offset}')
                        x, y   = int(x), int(y)
                        x = x + images_offset[id_num][0]
                        y = y + images_offset[id_num][1]
                        # print(f"VG adjusted x: {x}, y: {y}.")
                    
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
                    # print(f"VG reward: {reward}.")

                except Exception:
                    # print(f"Error parsing VG response: {raw}, reward set to 0.0.")
                    reward = 0.0
            else:
                # print(f"Task type '{t}' is not 'vg', reward set to None.")
                reward = None
            
            rewards.append(reward)
        
        return rewards

    def math_reward_func(self, completions, answer, task, **kwargs) -> List[float | None]:
        """Reward function that checks if the final answer matches the expected answer."""
        rewards = []
        for completion, ans, t in zip(completions, answer, task):
            if t == "math":
                response = str(self.get_last_answer(completion))
                try:
                    reward = 1.0 if grade(response, ans) else 0.0
                except:
                    reward = 0.0
            else:
                reward = None
            rewards.append(reward)
        return rewards
    
    def correct_answer_reward_func(self, completions, answer, task, all_images, all_images_offset, **kwargs) -> List[float | None]:
        """Reward function that checks if the final answer matches the expected answer."""
        rewards = []
        for completion, ans, t, images, images_offset, in zip(completions, answer, task, all_images, all_images_offset):
            reward = None
            if t == "mc":
                try:
                    reward = self.mc_reward_func([completion], [ans], [t], **kwargs)[0]
                except:
                    reward = None
            elif t == "math":
                try:
                    reward = self.math_reward_func([completion], [ans], [t], **kwargs)[0]
                except:
                    reward = None
            elif t == "code":
                try:
                    reward = self.code_reward_func([completion], [ans], [t], **kwargs)[0]
                except:
                    reward = None
            elif t == "vg":
                try:
                    reward = self.vg_reward_func(
                    completions=[completion],
                    answer=[ans],
                    task=[t],
                    all_images=[images],
                    all_images_offset=[images_offset],
                    )[0]
                except:
                    # print(f"Error in vg reward function for task {t}. reward set to None.")
                    reward = None
            else:
                reward = None
            rewards.append(reward)
        return rewards

    def correct_crop_func(self, completions, answer, all_images, all_images_offset, **kwargs) -> List[float | None]:
        rewards: List[float | None] = []
        
        for completion, box, images, images_offset in zip(completions, answer, all_images, all_images_offset):
            # 1. extract the crop bbox from assistant messages
            crop_bboxs: List[Tuple[int,int,int,int]] = []
            try:
                for msg in completion:
                    if msg.get("role") != "assistant":
                        continue
                    for item in msg.get("content", []):
                        if item.get("type") != "text":
                            continue
                        text = item["text"]
                        id_num, top_left, bottom_right = parse_crop_bbox_from_text(text)
                        if top_left is not None:
                            dx, dy = images_offset[id_num]
                            x1, y1 = top_left
                            x2, y2 = bottom_right
                            
                            # Adjust coordinates based on image offset
                            parsed = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)
                            crop_bboxs.append(parsed)
                
                    # 2. 拆箱 ground-truth
                    if isinstance(box, str):
                        try:
                            box = tuple(ast.literal_eval(box))
                        except Exception:
                            nums2 = re.findall(r"-?\d+", box)
                            box = tuple(map(int, nums2))

                    # extract image size from completion
                    first_image = extract_first_image(completion)
                    if first_image is None:
                        raise ValueError("No image found in the conversation.")
                    img_size = first_image.size # (width, height)
                    
                    reward = average_crop_reward(crop_bboxs, box, img_size)
                    # print(f"Crop reward: {reward}.")

            except Exception:
                reward = 0.0
            
            rewards.append(reward)
        
        return rewards
    
    def tool_execution_reward_func(self, completions: List[List[Dict[str, str]]], **kwargs) -> List[float]:
        """
        Reward function that checks tool execution success.

        Uses XMLParser to identify proper tool calls.
        """
        def check_execution(trajectory):
            tool_attempts = 0
            successful_executions = 0
            
            # Find assistant messages with tools and their responses
            for i, msg in enumerate(trajectory):
                if msg['role'] == 'assistant':
                    # Use parser to check for tool tag
                    parsed = self.parser.parse(msg['content'][0]["text"])
                    if hasattr(parsed, 'crop') and parsed.crop is not None:
                        # Found a properly formatted tool message
                        if i + 1 < len(trajectory) and trajectory[i + 1]['role'] == 'user':
                            tool_attempts += 1
                            multiplier = 1.0 
                            response = str(parsed.crop)
                            if (("sympy" in response) or ("numpy" in response)) and len(response) > 100:
                                multiplier = 1.5
                            else:
                                multiplier = 1.0
                                
                            # Extract tool response text
                            tool_response = None
                            for elem in trajectory[i + 1]['content']:
                                # 确保 elem 是个 dict 并包含 'text' 键
                                if isinstance(elem, dict) and "text" in elem:
                                    tool_response = elem["text"]
                                    break
                                
                            if '<|Tool_Error|>' not in tool_response:
                                successful_executions += 1 * multiplier
            
            # Calculate reward
            if tool_attempts == 0:
                return 0.0
            return (successful_executions / tool_attempts)
        
        return [check_execution(c) for c in completions]
    
    def get_named_tool_reward_func(self, tool_name: str) -> Callable:
        """
        Returns a reward function that checks tool execution success for a specific tool.

        Uses XMLParser to identify proper tool calls.
        """
        def tool_reward_func(completions: List[List[Dict[str, str]]], **kwargs) -> List[float]:
            """
            Reward function that checks execution success for the {tool_name} tool.
            
            Check Whether the tool was executed successfully.
            For example, crop tool -> image should be included in the next message.
            """
            import json
            
            def check_tool_execution(trajectory: List[Dict[str, str]]) -> float:
                tool_attempts = 0
                successful_executions = 0
                
                # Find assistant messages with the specific tool and their responses
                for i, msg in enumerate(trajectory):
                    if msg['role'] == 'assistant':
                        # Use parser to check for tool tag
                        parsed = self.parser.parse(msg['content'][0]["text"])
                        if hasattr(parsed, 'crop') and parsed.crop is not None:
                            try:
                                command = parsed.crop
                                if isinstance(command, str):
                                    # 从第一个 '(' 前面提取函数名
                                    func_name = command.split('(', 1)[0].strip()
                                    if func_name == 'crop':
                                        # Found a properly formatted tool message for the specific tool
                                        if i + 1 < len(trajectory) and trajectory[i+1]['role'] == 'user':
                                            tool_attempts += 1

                                            next_msg = trajectory[i+1]
                                            content = next_msg.get('content', [])

                                            # 如果 content 是列表，且其中有一项的 "type" 恰好是 "image_url"，就算成功
                                            if isinstance(content, list) and any(
                                                isinstance(item, dict) and item.get("type") == "image_url"
                                                for item in content
                                            ):
                                                successful_executions += 1
                            except json.JSONDecodeError:
                                pass
                
                # Calculate reward
                if tool_attempts == 0:
                    return 0.0
                return (successful_executions / tool_attempts)
            
            return [check_tool_execution(c) for c in completions]
        
        # Create a function with the dynamic name based on tool_name
        tool_reward_func.__name__ = f"{tool_name}_reward_func"
        return tool_reward_func
    
    def get_named_tool_count_reward_func(self, tool_name: str) -> Callable:
        """
        Returns a reward function that counts the number of times the {tool_name} tool is used.
        """
        def tool_count_reward_func(completions: List[List[Dict[str, str]]], **kwargs) -> List[float]:
            """
            Reward function that counts the number of times the {tool_name} tool is used.
            """
            import json

            def count_tool_executions(trajectory: List[Dict[str, str]]) -> float:
                successful_executions = 0.0
                for i, msg in enumerate(trajectory):
                    if msg['role'] == 'assistant':
                        parsed = self.parser.parse(msg['content'])
                        if hasattr(parsed, 'tool') and parsed.tool is not None:
                            try:
                                command = json.loads(parsed.tool)
                                if isinstance(command, dict) and command.get("name") == tool_name:
                                    # Found a properly formatted tool message for the specific tool
                                    if i + 1 < len(trajectory) and trajectory[i + 1]['role'] == 'user':
                                        parsed_response = self.env_parser.parse(trajectory[i + 1]['content'])
                                        if hasattr(parsed_response, 'result') and parsed_response.result is not None and not parsed_response.result.startswith("Error:"):
                                            successful_executions += 1
                            except json.JSONDecodeError:
                                pass
                return successful_executions
            
            return [count_tool_executions(c) for c in completions]
        
        tool_count_reward_func.__name__ = f"{tool_name}_count_reward_func"
        return tool_count_reward_func

    def get_named_tool_attempt_reward_func(self, tool_name: str) -> Callable:
        """
        Returns a reward function that counts the number of times the {tool_name} tool is used.
        """
        def tool_attempt_reward_func(completions: List[List[Dict[str, str]]], **kwargs) -> List[float]:
            """
            Reward function that counts the number of times the {tool_name} tool is used.
            """
            import json

            def count_tool_executions(trajectory: List[Dict[str, str]]) -> float:
                attempted_executions = 0.0
                for i, msg in enumerate(trajectory):
                    if msg['role'] == 'assistant':
                        parsed = self.parser.parse(msg['content'])
                        if hasattr(parsed, 'tool') and parsed.tool is not None:
                            try:
                                command = json.loads(parsed.tool)
                                if isinstance(command, dict) and command.get("name") == tool_name:
                                    attempted_executions += 1
                            except json.JSONDecodeError:
                                pass
                return attempted_executions
            
            return [count_tool_executions(c) for c in completions]
            
        tool_attempt_reward_func.__name__ = f"{tool_name}_attempt_reward_func"
        return tool_attempt_reward_func