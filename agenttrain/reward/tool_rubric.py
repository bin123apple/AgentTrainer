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
            self.correct_answer_reward_func,
            self.correct_crop_func,
            self.correct_extract_func,
            self.correct_find_color,
            self.tool_execution_reward_func,
            self.parser.get_format_reward_func(),
            self.parser.get_xml_reward_func(),
        ] # TODO: add tool feedbacks here.
        self.reward_weights = [
            1.0,
            0.7,
            0.7,
            0.7,
            0.4,
            0.1,
            0.1,
        ]

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
        debug: bool = kwargs.get("debug", False)
        rewards: List[float | None] = []
        if debug:
            print(f'images_offset: {all_images_offset}')
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
    
    def correct_extract_func(
        self,
        completions,              # List[List[MessageDict]]
        answer,                   # List[GT bbox | str]
        all_images,               # List[List[PIL.Image]]
        all_images_offset,        # List[List[Tuple[int,int]]]
        **kwargs,
    ) -> list[float | None]:
        """
        Evaluate whether the assistant’s <extract>(...) calls correctly cover the
        ground-truth bbox for each sample.

        Returns
        -------
        List[float | None]
            One reward per sample: 1.0 if any extract region fully contains the GT
            bbox, otherwise 0.0. (None is never returned here, kept for API parity.)
        """
        import re, ast
        from typing import Optional, Tuple, List

        debug: bool = kwargs.get("debug", False)

        # ------------------------------------------------------------------
        # === 1. 内部工具 ===
        # ------------------------------------------------------------------
        _EXTRACT_RE = re.compile(
            r"<extract>\s*\(\s*['\"]?Image_(\d+)['\"]?\s*,\s*['\"]?"
            r"(left|center|right)['\"]?\s*,\s*['\"]?(top|center|bottom)['\"]?\s*\)\s*</extract>",
            re.I,
        )

        def _parse_extract(text: str) -> Tuple[Optional[int], Optional[str], Optional[str]]:
            """从文本片段中抽取 (image_id, x_pos, y_pos)。若不存在返回 (None, None, None)。"""
            m = _EXTRACT_RE.search(text)
            if not m:
                return None, None, None
            return int(m.group(1)), m.group(2).lower(), m.group(3).lower()

        def _quadrant_bbox(w: int, h: int, x_pos: str, y_pos: str) -> Tuple[int, int, int, int]:
            """将 (x_pos, y_pos) 转成覆盖 ¼ 图像的 bbox。"""
            half_w, half_h = w // 2, h // 2
            x0 = 0 if x_pos == "left" else (w - half_w) // 2 if x_pos == "center" else w - half_w
            y0 = 0 if y_pos == "top"  else (h - half_h) // 2 if y_pos == "center" else h - half_h
            return (x0, y0, x0 + half_w, y0 + half_h)

        def _contains(outer: Tuple[int, int, int, int], inner: Tuple[int, int, int, int]) -> bool:
            """inner bbox 是否被 outer bbox 完全包住？"""
            ox1, oy1, ox2, oy2 = outer
            ix1, iy1, ix2, iy2 = inner
            return ox1 <= ix1 and oy1 <= iy1 and ox2 >= ix2 and oy2 >= iy2

        def _extract_reward(
            bboxes: List[Tuple[int, int, int, int]],
            gt: Tuple[int, int, int, int]
        ) -> float:
            """
            对 assistant 的每一个 <extract> 框：
                - 若完全包住 ground-truth → 得 1 分
                - 否则                       → 得 0 分
            返回所有框得分的平均值。
            """
            if not bboxes:
                return 0.0

            hits = sum(1 for b in bboxes if _contains(b, gt))
            return hits / len(bboxes)

        # ------------------------------------------------------------------
        # === 2. 主循环 ===
        # ------------------------------------------------------------------
        rewards: List[float] = []
        if debug:
            print(f'images_offset: {all_images_offset}')
        for completion, gt_box, images, offsets in zip(
            completions, answer, all_images, all_images_offset
        ):
            try:
                # 2-1. 收集所有 <extract> 产生的 bbox
                extract_bboxes: list[Tuple[int, int, int, int]] = []

                for idx, msg in enumerate(completion):
                    if msg.get("role") != "assistant":
                        continue
                    for part in msg.get("content", []):
                        if part.get("type") != "text":
                            continue
                        img_id, x_pos, y_pos = _parse_extract(part["text"])
                        print(f"Parsed extract: img_id={img_id}, x_pos={x_pos}, y_pos={y_pos}")
                        if x_pos is None:        # 该片段无 <extract>
                            continue
                        if debug:
                                print(f"images: {images}")
                        w, h = images[img_id].size
                        bbox = _quadrant_bbox(w, h, x_pos, y_pos)   # 原图坐标
                        if debug:
                            print(f"Image_{img_id} size: ({w}, {h}), Extract bbox (pre-offset): {bbox}, offset: {offsets}")
                        dx, dy = offsets[img_id]                    # 全景图偏移
                        x1, y1, x2, y2 = bbox
                        extract_bboxes.append((x1 + dx, y1 + dy, x2 + dx, y2 + dy))
                        
                        if debug:
                            # 打印计算出的真实 bbox
                            print(f"[DEBUG] Extract real_bbox: {(x1 + dx, y1 + dy, x2 + dx, y2 + dy)}")

                            # 从当前 idx 向后找第一条 user 消息
                            for nxt in completion[idx+1:]:
                                if nxt.get("role") == "user":
                                    user_text = "\n".join(
                                        c["text"] for c in nxt.get("content", []) if c.get("type")=="text"
                                    )
                                    print(f"[DEBUG] Corresponding user message:\n{user_text}")
                                    m_img = re.search(
                                        r"\[Image_(\d+)[^\]]*offset[:：]?\s*\(\s*(\d+),\s*(\d+)\s*\)",
                                        user_text
                                    )
                                    dx_debug, dy_debug = int(m_img.group(2)), int(m_img.group(3))
                                    m_size = re.search(r"Cropped a region of size\s*(\d+)×(\d+)", user_text)
                                    w, h = map(int, m_size.groups())
                                    x1_debug, y1_debug, x2_debug, y2_debug = 0, 0, w, h
                                    if (x1_debug + dx_debug, y1_debug + dy_debug, x2_debug + dx_debug, y2_debug + dy_debug) != (x1 + dx, y1 + dy, x2 + dx, y2 + dy):
                                        print(f"[ERROR] Offset mismatch: expected ({x1_debug + dx}, {y1_debug + dy}, {x2_debug + dx}, {y2_debug + dy}), got ({x1 + dx}, {y1 + dy}, {x2 + dx}, {y2 + dy})")
                                    else:
                                        print(f"[DEBUG] Offset matches: ({x1 + dx}, {y1 + dy}, {x2 + dx}, {y2 + dy}) == ({x1_debug + dx_debug}, {y1_debug + dy_debug}, {x2_debug + dx_debug}, {y2_debug + dy_debug})")
                                    break
                        

                # 2-2. 解析 ground-truth
                if isinstance(gt_box, str):
                    try:
                        gt_box = tuple(ast.literal_eval(gt_box))
                    except Exception:
                        nums = [int(n) for n in re.findall(r"-?\d+", gt_box)]
                        gt_box = tuple(nums)

                # 2-3. 计算奖励
                reward = _extract_reward(extract_bboxes, gt_box)

                if debug:
                    print(f"GT: {gt_box} | Extracts: {extract_bboxes} | Reward={reward}")

            except Exception as e:
                if debug:
                    print(f"[correct_extract_func] Error: {e}")
                reward = 0.0

            rewards.append(reward)

        return rewards

    def correct_find_color(
        self,
        completions: List[List[dict]],
        answer: List[Tuple[int, ...]],
        all_images: List[List],
        all_images_offset: List[List[Tuple[int, int]]],
        **kwargs
    ) -> List[float]:
        """
        Reward = 1 if ground-truth coordinate/box is inside the region returned by find_color.
        """
        debug: bool = kwargs.get("debug", False)
        rewards: List[float] = []
        
        for completion, box, images, images_offset in zip(completions, answer, all_images, all_images_offset):
            if debug:
                print(f"[DEBUG] images_offset: {images_offset}")
            reward = 0.0
            
            try:
                # 1. locate the assistant message with a find_color tool call
                for idx, msg in enumerate(completion):
                    if msg.get("role") != "assistant":
                        continue
                    # check for closing tag
                    if not any(item.get("type") == "text" and "</find_color>" in item["text"] 
                            for item in msg.get("content", [])):
                        continue
                    
                    # 2. find the following user message (tool output)
                    user_msg = None
                    for nxt in completion[idx+1:]:
                        if nxt.get("role") == "user":
                            user_msg = nxt
                            break
                    if user_msg is None:
                        raise ValueError("No user message after find_color call.")
                    
                    # 3. extract image id and offset from user message text
                    combined_text = "\n".join(
                        c["text"] for c in user_msg.get("content", []) if c.get("type") == "text"
                    )
                    if debug:
                        print(f"[DEBUG] Find find_color command, the User message text is: {combined_text}")
                    m_img = re.search(
                        r"\[Image_(\d+)[^\]]*offset[:：]?\s*\(\s*(\d+),\s*(\d+)\s*\)",
                        combined_text
                    )
                    if not m_img:
                        raise ValueError("Cannot parse Image ID and offset.")
                    dx, dy = int(m_img.group(2)), int(m_img.group(3))
                    if debug:
                        img_id = int(m_img.group(1))
                        offset_from_input = images_offset[img_id]
                        if (dx, dy) != offset_from_input:
                            print(f"[ERROR] Offset mismatch: expected {offset_from_input}, got ({dx}, {dy})")
                        else:
                            print(f"[DEBUG] Offset matches: ({dx}, {dy})")
                    
                    # 4. extract bounding box of the cropped region
                    m_size = re.search(r"Cropped a region of size\s*(\d+)×(\d+)", combined_text)
                    if not m_size:
                        raise ValueError("Cannot parse crop size or coords.")
                    w, h = map(int, m_size.groups())
                    x1, y1, x2, y2 = 0, 0, w, h
                    real_tl = (x1 + dx, y1 + dy)
                    real_br = (x2 + dx, y2 + dy)
                    
                    # 6. normalize ground-truth
                    if isinstance(box, str):
                        try:
                            box = tuple(ast.literal_eval(box))
                        except Exception:
                            box = tuple(map(int, re.findall(r"-?\d+", box)))
                    
                    # 7. check inclusion
                    if len(box) == 2:
                        x_gt, y_gt = box
                        if real_tl[0] <= x_gt <= real_br[0] and real_tl[1] <= y_gt <= real_br[1]:
                            reward = 1.0
                    else:
                        x1_gt, y1_gt, x2_gt, y2_gt = box
                        if (real_tl[0] <= x1_gt <= real_br[0] and
                            real_tl[1] <= y1_gt <= real_br[1] and
                            real_tl[0] <= x2_gt <= real_br[0] and
                            real_tl[1] <= y2_gt <= real_br[1]):
                            reward = 1.0
                    
                    break  # done for this example
                    
            except Exception as e:
                if debug:
                    print(f"[ERROR] correct_find_color: {e}")
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