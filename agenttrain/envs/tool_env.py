import io
import re
import ast
import base64
import inspect
from typing import List, Dict, Any, Callable, Union

from datasets import Dataset
from agenttrain.tools import crop, locate
from PIL import Image
from agenttrain.trainers.grpo_env_trainer import RewardFunc
from agenttrain.envs.multiturn_env import MultiTurnEnv
from agenttrain.parsers import XMLParser # rewrite this one
from agenttrain.prompts.tool_description import CROP_TOOL_DESCRIPTION
from agenttrain.prompts.tool_example import CROP_TOOL_EXAMPLE
from agenttrain.reward.tool_rubric import ToolRubric

def infer_schema_from_function(func: Callable) -> Dict[str, Any]:
    """Infers a tool schema from a function's signature and docstring."""
    sig = inspect.signature(func)
    doc = inspect.getdoc(func) or ""
    
    # Parse docstring sections
    doc_parts = doc.split("\n\n")
    description = doc_parts[0].strip()
    
    # Extract examples if present
    examples = []
    return_description = ""
    for part in doc_parts:
        if part.startswith("Examples:"):
            examples = [line.strip() for line in part.split("\n")[1:] if line.strip()]
        elif part.startswith("Returns:"):
            return_description = part.split("\n")[1].strip()

    return_type = str(sig.return_annotation.__name__ if sig.return_annotation != inspect.Parameter.empty else "any")

    # print(f"return_description: {return_description} ({return_type})")
    # Build args schema
    args = {}
    for name, param in sig.parameters.items():
        param_doc = ""
        for part in doc_parts:
            if part.strip().startswith("Args:"):
                for line in part.split("\n")[1:]:
                    if line.strip().startswith(f"{name}:"):
                        param_doc = line.strip()[len(name)+1:].strip()
        
        args[name] = {
            "type": str(param.annotation.__name__ if param.annotation != inspect.Parameter.empty else "any"),
            "description": param_doc,
        }
        if param.default != inspect.Parameter.empty:
            args[name]["default"] = param.default
    
    return {
        "name": func.__name__,
        "description": description,
        "args": args,
        "returns": return_description + f" ({return_type})",
        "examples": examples
    }

def format_tool_descriptions(schemas: List[Dict[str, Any]]) -> str:
    """Formats tool schemas into a user-friendly description string."""
    descriptions = []
    for schema in schemas:
        desc = [f"{schema['name']}: {schema['description']}"]
        
        desc.append("\nArguments:")
        for arg_name, arg_info in schema['args'].items():
            default = f" (default: {arg_info['default']})" if 'default' in arg_info else ""
            desc.append(f"  - {arg_name}: {arg_info['description']}{default}")
        
        if schema['examples']:
            desc.append("\nExamples:")
            for example in schema['examples']:
                desc.append(f"  {example}")
        
        if schema['returns']:
            desc.append(f"\nReturns: {schema['returns']}")
        
        descriptions.append("\n".join(desc))
    
    return "\n\n".join(descriptions)

class ToolEnv(MultiTurnEnv):
    def __init__(self,
                 dataset: Dataset | None = None,
                 eval_dataset: Dataset | None = None,
                 tools: List[Callable] = [],
                 few_shot: List[Dict[str, str]] = [],
                 llm_fields: List[str | tuple[str, str]] = [("crop", "answer", "locate")],
                 env_fields: List[str | tuple[str, str]] = ["result"],
                 sampling_args={
                     "stop": ["</crop>", "</answer>", "</locate>"],
                     "include_stop_str_in_output": True
                 },
                 mask_env_response: bool = True,
                 max_steps: int = 10, **kwargs):
        # Infer schemas from tool functions
        self.tool_schemas = [infer_schema_from_function(tool) for tool in tools]
        self.tools = {tool.__name__: tool for tool in tools}
        
        super().__init__(
            dataset=dataset,
            eval_dataset=eval_dataset,
            few_shot=few_shot,
            mask_env_response=mask_env_response,
            max_steps=max_steps,
            sampling_args=sampling_args,
            **kwargs
        )
        self.dataset_name = dataset
        self.max_steps = max_steps
        self.llm_parser = XMLParser(fields=llm_fields)
        self.env_parser = XMLParser(fields=env_fields)
        self.rubric = ToolRubric(tools=tools, parser=self.llm_parser, env_parser=self.env_parser)

    def get_reward_funcs(self, **kwargs: Any) -> List[RewardFunc]:
        return self.rubric.get_reward_funcs()
    
    def get_reward_weights(self, **kwargs: Any) -> List[float]:
        return self.rubric.get_reward_weights()

    def _get_step_count(self, messages: List[Dict[str, str]]) -> int:
        """Count the number of tool uses in the message history, excluding few-shot examples."""
        step_count = 0
        
        # Skip messages that are part of few-shot examples
        # We need to determine where the actual conversation starts
        # System message + few-shot examples + user query = start of actual conversation
        conversation_start = 1  # Start after system message
        if self.few_shot:
            # Account for all few-shot messages
            conversation_start += len(self.few_shot)
        
        # Only count tool uses from the actual conversation
        for message in messages[conversation_start:]:
            if message.get("role") == "assistant":
                step_count += 1
        return step_count
    
    def is_completed(self, messages: List[dict[str, Union[str, List[dict]]]], **kwargs: Any) -> bool:
        try:
            # Check if we've hit max steps by counting tool uses in the message history
            step_count = self._get_step_count(messages)
            if step_count > self.max_steps:
                return True
            
            parsed = self.llm_parser.parse(messages[-1]["content"][0]["text"])
            # Check if we got a valid answer field (not just None from failed parsing)
            return hasattr(parsed, 'answer') and parsed.answer is not None
        except Exception:
            return False

    def call_crop(self, tool_cmd: str, images: List[Image.Image]):
        """
        1. Convert Crop usage: 
        (Image_0, (10, 20), (110, 100)) ->
        crop(
            image: Image.Image,
            top_left: Tuple[int, int],
            bottom_right: Tuple[int, int]
        )
        2. Perform the crop operation on the specified image.
        Args:
            tool_cmd (str): The command string containing image name, id_num, and coordinates.
            images (List[Image.Image]): List of images available for cropping.
        Returns:
            Tuple[bytes, str]:
                cropped image as PNG bytes,
                message -> As text describing the crop operation
                offset -> For calculating the coordinates relative to the original image
                imgae id_num -> The index of the image in the images list
        """
        try:
            # extract image name, id_num, and coordinates from the tool command
            pattern = re.compile(
                r"""\(?\s*
                    ([A-Za-z0-9_-]+)      # ① image_name  诸如 Image
                    _(\d+)                # ② id_num      必须带下划线和数字
                    \s*,\s*
                    \(\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*\)   # ③④ x1,y1
                    \s*,\s*
                    \(\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*\)   # ⑤⑥ x2,y2
                    \s*\)?
                """,
                re.VERBOSE,
            )
            m = pattern.fullmatch(tool_cmd)
            
            if m:
                image_name, id_num, x1, y1, x2, y2 = m.groups()
                id_num = int(id_num)
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            top_left = (x1, y1)
            bottom_right = (x2, y2)
            img = images[id_num]
            # 4. 真正调用 crop，并返回结果
            data, message, off_set = crop(img, top_left, bottom_right)
            return data, message, off_set, id_num

        except Exception as e:
            usage = (
                f"<crop>(Image_0,(x1, y1), (x2, y2))</crop>, "
                "The first argument is a string, which record the image name with an ID, e.g. Image_0 "
                "and the second and third arguments are tuples representing the top-left and bottom-right coordinates."
            )
            return None, f"<|Tool_Error|>, correct usage: {usage}", None, None

    def call_locate(self, tool_cmd: str, images: List[Image.Image]) -> bytes:
        """
        1. Convert Locate usage:
        (Image_0, query) ->
        locate(
            image: Image.Image,
            query: str
        )
        2. Perform the locate operation on the specified image.
        Args:
            tool_cmd (str): The command string containing image name and query.
            img (Image.Image): The image to search within.
        Returns:
            Tuple[bytes, str]:
                cropped image as PNG bytes,
                message -> As text describing the locate operation
                offset -> For calculating the coordinates relative to the original image
        """
        try:
            # 1. Extract image name and query from the tool command
            pattern = re.compile(
                r"""\(?\s*                 # 可选左括号 + 空格
                    [A-Za-z]+_(\d+)        # ① 数字 ID（捕获）  e.g. 0 / 123
                    \s*,\s*                # 逗号分隔
                    (.*?)                  # ② query（捕获，非贪婪）
                    \s*\)?\s*$             # 可选右括号 + 尾部空白
                """,
                re.VERBOSE,
            )
            m = pattern.fullmatch(tool_cmd)
            if m:
                id_num   = int(m.group(1))   # → 0
                query    = m.group(2)        # → "what is shown here?"
            
            # 2. 真正调用 crop，并返回结果
            img = images[id_num]
            data, message, off_set = locate(img, query)
            return data, message, off_set, id_num

        except Exception as e:
            usage = (
                f"<locate>(Image_0, query)</locate>, "
                "The first argument is a string, which record the image name with an ID, e.g. Image_0 "
                "and the second argument is a string representing the element to locate in the image."
            )
            return None, f"<|Tool_Error|>, correct usage: {usage}", None, None
        

    def call_tool(self, category: str ,tool_cmd: str, images: List[Image.Image]) -> Any:
        """
        Call different tools
        Returns:
            Tuple[bytes, str]:
                image as PNG bytes,
                message -> As text describing the crop operation
                offset -> For calculating the coordinates relative to the original image
                id_num -> The index of the image in the images list
        """
        if category == "crop":
            return self.call_crop(tool_cmd, images)
        elif category == "locate":
            return self.call_locate(tool_cmd, images)
        else:
            return None, f"<|Tool_Error|>, Unsupported tool category: {category}", None, None


    def env_response(self, messages: List[dict[str, Union[str, List[dict]]]], images: List[Image.Image] , images_offset: List[tuple], **kwargs: Any) -> Dict[str, Any]:
        try:
            parsed = self.llm_parser.parse(messages[-1]["content"][0]["text"])
            # print(f"Parsed content: {parsed}")
            if hasattr(parsed, 'crop') and parsed.crop is not None:
                category = 'crop'
                tool_cmd = parsed.crop
            elif hasattr(parsed, 'locate') and parsed.locate is not None:
                category = 'locate'
                tool_cmd = parsed.locate
            else: # No valid tool command found
                tool_feedback = {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "No valid tool command found in the last message."}
                    ]
                }
                messages.append(tool_feedback)
                return
            
            crop_bytes, info_message, off_set, id_num = self.call_tool(category, tool_cmd, images)
            if crop_bytes:
                crop_b64 = base64.b64encode(crop_bytes).decode('utf-8')
                cropped_img = Image.open(io.BytesIO(crop_bytes)).convert("RGB")
                images.append(cropped_img) # add to images list for further processing
                
                # Calculate and add offset
                dx = off_set[0]
                dy = off_set[1]
                x_off_set = images_offset[id_num][0] + dx
                y_off_set = images_offset[id_num][1] + dy
                off_set = (x_off_set, y_off_set)
                images_offset.append(off_set)
                
                info_message = f"[Image_{len(images)-1} is displayed above, offset: {off_set}]\n{info_message}"
                multimodal_message = [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{crop_b64}"}
                        },
                        {
                            "type": "text",
                            "text": info_message
                        }
                    ]
                tool_feedback =  {"role": "user", "content": multimodal_message}
                messages.append(tool_feedback)
                return
                
            else:
                tool_feedback = {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Error: {info_message}"}
                    ]
                }
                messages.append(tool_feedback)
                return

        except Exception as e:
            tool_feedback = {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Unexpected error in env_response: {e}"}
                ]
            }
            messages.append(tool_feedback)
            return