from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration, AutoProcessor, AutoModelForCausalLM
from typing import Dict, Any, Union
from trl.data_utils import maybe_apply_chat_template
import torch
import re

from agenttrain.vlm_modules.vlm_module import VLMBaseModule

class Qwen2VLModule(VLMBaseModule):
    def __init__(self):
        super().__init__()

    def get_vlm_key(self):
        return "qwen"

    def get_model_class(self, model_id: str, model_init_kwargs: dict):
        if "Qwen2-VL" in model_id:
            model_cls = Qwen2VLForConditionalGeneration
        elif "Qwen2.5-VL" in model_id:
            model_cls = Qwen2_5_VLForConditionalGeneration
        elif "qwen2_5vl" in model_id:
            model_cls = Qwen2_5_VLForConditionalGeneration
        else:
            # raise ValueError(f"Unsupported model: {model_id}")
            model_cls = AutoModelForCausalLM
        return model_cls
    
    def post_model_init(self, model, processing_class):
        pass
    
    def get_processing_class(self):
        return AutoProcessor
    
    def get_vision_modules_keywords(self):  
        return ['visual']
    
    def get_custom_multimodal_keywords(self):
        return ['pixel_values', 'image_grid_thw']

    def get_non_generate_params(self):
        return []
    
    def get_custom_processing_keywords(self):
        return ['max_pixels', 'min_pixels']
    
    def prepare_prompt(self, processing_class, inputs: dict[str, Union[torch.Tensor, Any]]):
        prompts_text = [maybe_apply_chat_template(example, processing_class)["prompt"] for example in inputs]
        return prompts_text
    
    def prepare_model_inputs(self, processing_class, prompts_text, images, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False):
        # FIXME
        # This could only process pure-multimodal or pure-text inputs
        # factor=processor.image_processor.patch_size * processor.image_processor.merge_size,
        # min_pixels=processor.image_processor.min_pixels,
        # max_pixels=processor.image_processor.max_pixels,
        # print(f'------ patch_size, merge_size, min_pixels, max_pixels: {processing_class.image_processor.patch_size}, {processing_class.image_processor.merge_size}, {processing_class.image_processor.min_pixels}, {processing_class.image_processor.max_pixels}')
        # raise 123
        if len(images) > 0:
            prompt_inputs = processing_class(
                text=prompts_text,
                images=images,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens)
        else:
            prompt_inputs = processing_class(
                text=prompts_text,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens)
        return prompt_inputs
    
    @staticmethod
    def get_question_template(task_type: str):
        # wtk
        prefix = 'Please provide the point coordinates [x, y] of a specific element based on this sentence: '
        suffix = ' First, think about the reasoning process in the mind within <think> </think> tags. Then, output the point coordinates within <answer> </answer> tags.'
        return prefix + "{Question}" + suffix

        # prefix = 'Please provide the point coordinates [x, y] of the center point of a specific element based on this sentence: '
        # suffix = ' First, think about the reasoning process in the mind within <think> </think> tags. Then, output the center point coordinates within <answer> </answer> tags.'
        # return prefix + "{Question}" + suffix


        # prefix = 'Please provide the bounding box coordinates [x1, y1, x2, y2] of a specific element based on this sentence: '
        # suffix = ' First, think about the reasoning process in the mind within <think> </think> tags. Then, output the bounding box coordinates within <answer> </answer> tags.'
        # strict_suffix = ' First, think about the reasoning process in the mind within <think> </think> tags. Then, output the bounding box coordinates in JSON format within <answer> </answer> tags.'
        # return prefix + "{Question}" + suffix

        # prefix = 'Please provide the bounding box coordinates [x1, y1, x2, y2] and the point coordinates [x, y] of a specific element based on this sentence: '
        # suffix = ' First, think about the reasoning process in the mind within <think> </think> tags. Then, output the bounding box coordinates within <box> </box> tags. Finally, output the point coordinates within <answer> </answer> tags.'
        # return prefix + "{Question}" + suffix

    # wtk
    @staticmethod
    def format_reward_rec(prompts, completions, **kwargs):
        """Check if the model output has <think>...</think> followed by <answer> with exactly four numbers."""
        completion_contents = [completion[0]["content"] for completion in completions]

        matches = []
        
        current_score = 0.0
        num_format = 2
        for content in completion_contents:
            
            # 检查 <think> 标签
            # if more than one <think> in the string, then it is not valid
            # if content.count("<think>") > 1 or content.count("</think>") > 1 or (content.count("<think>") == 0 and content.count("</think>") == 0):
            #     pass
            # else:
            if "<think>" in content:
                current_score += 0.5
            if "</think>" in content:
                current_score += 0.5

            # 提取 <answer> 标签
            # if more than one <answer> in the string, then it is not valid
            # if content.count("<answer>") > 1 or content.count("</answer>") > 1:
            #     pass
            # else:
            anwer_match = re.search(r"<answer>.*?</answer>", content, re.DOTALL)
            if not anwer_match:
                if "<answer>" in content:
                    current_score += 1/3
                elif "</answer>" in content:
                    current_score += 1/3
            else:
                current_score += 2/3
                answer_content = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL).group(1)
                numbers = re.findall(r'\d+\.?\d*', answer_content)
                if len(numbers) == 2:
                # if len(numbers) == 4:
                    current_score += 1/3

            matches.append(current_score / num_format)
            current_score = 0.0

        return matches

    # @staticmethod
    # def format_reward_rec(completions, **kwargs):
    #     """Check if the Qwen model output matches a specific format."""
    #     import re
    #     pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    #     completion_contents = [completion[0]["content"] for completion in completions]
    #     matches = [re.search(pattern, content, re.DOTALL) is not None for content in completion_contents]
    #     return [1.0 if match else 0.0 for match in matches]
    
    @staticmethod
    def iou_reward(prompts, completions, solution_box, solution_point, **kwargs):
        """Calculate distance-based reward between predicted bounding box and ground truth point."""
        import re
        import os
        from datetime import datetime
        import math
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

        for content, prompt, sol_box, sol_p in zip(contents, prompts, solution_box, solution_point):
            reward = 0.0
            dist = float('inf')  # 初始化为无穷大
            
            # # 检查 <box> 标签
            # try:
            #     content_answer_match = re.search(r'<box>(.*?)</box>', content, re.DOTALL)
            #     if content_answer_match:
            #         # 提取匹配到的字符串，并分割成数字
            #         box_str = content_answer_match.group(1)
            #         bbox = [float(x) for x in re.findall(r'\d+\.?\d*', box_str)]
            #         assert len(bbox) == 4
            #         iou_res = iou(bbox, sol_box)
            #         if iou_res > 0.5:
            #             reward += 1.0
            # except Exception:
            #     pass  # 如果出错，继续下一个

            # if (content.count("<think>") == 0 and content.count("</think>") == 0) or \
            #     (content.count("<answer>") == 0 and content.count("</answer>") == 0):
            #     # 如果没有 <think> <answer> 标签，直接跳过
            #     pass
            # else:
            # 检查 <answer> 标签
            try:
                content_answer_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
                if content_answer_match:
                    # 提取匹配到的字符串，并分割成数字
                    point_str = content_answer_match.group(1)
                    point = [float(x) for x in re.findall(r'\d+\.?\d*', point_str)]
                    assert len(point) == 2
                    # if we have gt bbox
                    if sol_box[0] != -1:
                        # if point is within the bbox
                        if sol_box[0] <= point[0] <= sol_box[2] and sol_box[1] <= point[1] <= sol_box[3]:
                            dist = 0.0
                    else:
                        dist = math.sqrt((point[0] - sol_p[0]) ** 2 + (point[1] - sol_p[1]) ** 2)

                    # if dist <= 20: reward += 1.0
                    if dist <= 80: reward += 1.0
                    # elif 80 < dist <= 120: reward += 0.5
            except Exception:
                pass  # 如果出错，继续下一个

            rewards.append(reward)
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                with open(log_path, "a", encoding='utf-8') as f:
                    f.write(f"------------- {current_time} Point Distance: {dist}, reward: {reward} -------------\n")
                    f.write(f"Prompt: {prompt[0]['content'][1]['text']}\n")
                    f.write(f"Completion: {content}\n")
                    f.write(f"Solution: {sol_p}\n")
        return rewards
    
    # @staticmethod
    # def iou_reward(completions, solution_box, **kwargs):
    #     """Calculate IoU reward between predicted bounding box from Qwen model and ground truth bounding box."""
    #     import re
    #     import os
    #     from datetime import datetime

    #     contents = [completion[0]["content"] for completion in completions]
    #     rewards = []
    #     current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    #     answer_tag_pattern = r'<answer>(.*?)</answer>'
    #     # bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)]'
    #     for content, sol in zip(contents, solution_box):
    #         reward = 0.0
    #         iou_res = 0.0
    #         # Try symbolic verification first
    #         try:
    #             content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
    #             if content_answer_match:
    #                 content_answer = content_answer_match.group(1).strip()
    #                 # bbox_match = re.search(bbox_pattern, content_answer)
    #                 bbox = [float(x) for x in re.findall(r'\d+\.?\d*', content_answer)]
    #                 assert len(bbox) == 4
    #                 # if bbox_match:
    #                 #     bbox = [int(bbox_match.group(1)), int(bbox_match.group(2)), int(bbox_match.group(3)), int(bbox_match.group(4))]
    #                 iou_res = iou(bbox, sol)
    #                 if iou_res > 0.5:
    #                     reward = 1.0
    #                     # reward = iou(bbox, sol)
    #         except Exception:
    #             pass  # Continue to next verification method if this fails
                    
    #         rewards.append(reward)
    #         if os.getenv("DEBUG_MODE") == "true":
    #             log_path = os.getenv("LOG_PATH")
    #             # local_rank = int(os.getenv("LOCAL_RANK", 0))
    #             with open(log_path, "a", encoding='utf-8') as f:
    #                 f.write(f"------------- {current_time} IoU: {iou_res},  Accuracy reward: {reward} -------------\n")
    #                 f.write(f"Content: {content}\n")
    #                 f.write(f"Solution: {sol}\n")
    #     return rewards


def iou(box1, box2):
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2]-1, box2[2]-1)
    inter_y2 = min(box1[3]-1, box2[3]-1)
    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
    else:
        inter = 0
    union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
    return float(inter)/union