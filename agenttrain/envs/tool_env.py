import io
import re
import ast
import base64
import inspect
from typing import List, Dict, Any, Callable, Union

from datasets import Dataset
from agenttrain.tools import crop
from PIL import Image
from agenttrain.trainers.grpo_env_trainer import RewardFunc
from agenttrain.envs.multiturn_env import MultiTurnEnv
from agenttrain.parsers import XMLParser # rewrite this one
from agenttrain.prompts.system_prompts import CROP_TOOL_PROMPT_TEMPLATE 
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
                 system_prompt: str = CROP_TOOL_PROMPT_TEMPLATE,
                 few_shot: List[Dict[str, str]] = [],
                 llm_fields: List[str | tuple[str, str]] = [("crop", "answer")],
                 env_fields: List[str | tuple[str, str]] = ["result"],
                 sampling_args={
                     "stop": ["</crop>", "</answer>"],
                     "include_stop_str_in_output": True
                 },
                 mask_env_response: bool = True,
                 max_steps: int = 10, **kwargs):
        # Infer schemas from tool functions
        self.tool_schemas = [infer_schema_from_function(tool) for tool in tools]
        self.tools = {tool.__name__: tool for tool in tools}
        
        # Format the system prompt with tool descriptions
        # tool_descriptions = format_tool_descriptions(self.tool_schemas)
        formatted_prompt = system_prompt.format(tool_descriptions=CROP_TOOL_DESCRIPTION,
                                                tool_example = CROP_TOOL_EXAMPLE)
        super().__init__(
            dataset=dataset,
            eval_dataset=eval_dataset,
            system_prompt=formatted_prompt,
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

    # def call_tool(self, tool_json: str, **kwargs: Any) -> str:
    #     """Call a tool based on JSON command."""
    #     try:
    #         command = json.loads(tool_json)
    #         if not isinstance(command, dict):
    #             return "Error: Tool command must be a JSON object, e.g. '{\"name\": \"tool_name\", \"args\": {\"arg1\": \"value1\", \"arg2\": \"value2\"}}'"
            
    #         tool_name = command.get("name")
    #         if not tool_name:
    #             return "Error: Tool command must specify 'name', e.g. '{\"name\": \"tool_name\", \"args\": {\"arg1\": \"value1\", \"arg2\": \"value2\"}}'"
            
    #         if tool_name not in self.tools:
    #             return f"Error: Unknown tool '{tool_name}. " + "Please format your tool call as '{\"name\": \"tool_name\", \"args\": {\"arg1\": \"value1\", \"arg2\": \"value2\"}}'"
            
    #         tool_func = self.tools[tool_name]
    #         tool_args = command.get("args", {})
    #         if isinstance(tool_args, str):
    #             tool_schema = next((schema['args'] for schema in self.tool_schemas if schema['name'] == tool_name), None)
    #             return f"Error: Arguments for {tool_name} must be a JSON object with schema {tool_schema}, not a string." 
            
    #         # Call the tool function with arguments
    #         result = tool_func(**tool_args)
    #         return str(result)
    #     except json.JSONDecodeError:
    #         return "Error: Invalid JSON format. Please format your tool call as '{\"name\": \"tool_name\", \"args\": {\"arg1\": \"value1\", \"arg2\": \"value2\"}}'"
    #     except Exception as e:
    #         return f"Error: {str(e)}. " + "Please format your tool call as '{\"name\": \"tool_name\", \"args\": {\"arg1\": \"value1\", \"arg2\": \"value2\"}}'"

    def call_tool(self, tool_cmd: str, img: Image.Image) -> Any:
        """
        仅支持调用 crop，格式必须是：
            crop((x1, y1), (x2, y2))
        img 参数直接由调用者通过关键字参数传入。

        示例：
            call_tool([crop], "crop((10, 20), (110, 100))", img=your_image)
        """
        name = "crop"
        usage = f"{name}((x1, y1), (x2, y2))"
        try:
            nums = re.findall(r"-?\d+", tool_cmd)
            if len(nums) != 4:
                raise ValueError(f"Extracted {len(nums)} numbers, expected 4")
            x1, y1, x2, y2 = map(int, nums)
            top_left = (x1, y1)
            bottom_right = (x2, y2)
            
            # 4. 真正调用 crop，并返回结果
            return crop(img, top_left, bottom_right)

        except Exception as e:
            # print(f"Error calling tool {e}")
            return None, f"Wrong Format:{usage}"

    def env_response(self, messages: List[dict[str, Union[str, List[dict]]]], images: List[Image.Image] , **kwargs: Any) -> Dict[str, Any]:
        try:
            parsed = self.llm_parser.parse(messages[-1]["content"][0]["text"])
            # print(f"Parsed content: {parsed}")
            if hasattr(parsed, 'crop') and parsed.crop is not None:
                image_entry = next(item for item in messages[0]["content"] if item["type"] == "image_url")
                data_uri = image_entry["image_url"]["url"]
                b64_data = data_uri.split(",", 1)[1]
                img_bytes = base64.b64decode(b64_data)
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                crop_bytes, info_message = self.call_tool(parsed.crop, img)
                if crop_bytes:
                    crop_b64 = base64.b64encode(crop_bytes).decode('utf-8')
                    cropped_img = Image.open(io.BytesIO(crop_bytes)).convert("RGB")
                    images.append(cropped_img) # add to images list for further processing
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
            else:
                tool_feedback = {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "No valid tool command found in the last message.\ncrop tool Usage: crop((x1, y1), (x2, y2))"}
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